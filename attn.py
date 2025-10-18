"""
Attention-Contact Correlation Analysis for Protein Structures
Computes the proportion of high-attention pairs that correspond to spatial contacts
"""
import io
import urllib
from typing import List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import torch
from Bio.PDB import PDBParser
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd
import os
from pdb_ids import pdb_list


# ===================== CONSTANTS =====================
MODEL_ID = "microsoft/Dayhoff-170m-UR90"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_THREE_TO_ONE = {
    'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F', 'GLY':'G', 'HIS':'H', 'ILE':'I',
    'LYS':'K', 'LEU':'L', 'MET':'M', 'ASN':'N', 'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
    'THR':'T', 'VAL':'V', 'TRP':'W', 'TYR':'Y'
}


# ===================== MODEL INITIALIZATION =====================
print(f"Loading model {MODEL_ID}...")
cfg = AutoConfig.from_pretrained(MODEL_ID)
cfg.output_attentions = True
cfg.return_dict = True

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    config=cfg, 
    device_map=DEVICE
).to(DEVICE).eval()
print(f"✓ Model loaded on {DEVICE}\n")


# ===================== PDB PROCESSING =====================
def get_structure(pdb_id: str):
    """Download and parse PDB structure from RCSB"""
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    with urllib.request.urlopen(url) as resource:
        content = resource.read().decode('utf8')
    parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_id, io.StringIO(content))


def extract_chain_seq_and_ca_coords(structure, chain_id: Optional[str] = None) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Extract amino acid sequence and Cα coordinates from structure.
    
    Args:
        structure: Bio.PDB Structure object
        chain_id: Specific chain ID (None = first valid chain)
    
    Returns:
        sequence: One-letter amino acid string (or None if failed)
        coords: (L, 3) array of Cα coordinates (or None if failed)
    """
    model = next(structure.get_models())
    chains = list(model.get_chains())
    
    # Filter by chain_id if specified
    if chain_id:
        chains = [c for c in chains if c.id == chain_id]
        if not chains:
            return None, None
    
    # Find first valid chain with ≥5 residues
    for chain in chains:
        residues = [r for r in chain.get_residues() if r.get_resname() in _THREE_TO_ONE]
        if len(residues) < 5:
            continue
        
        seq, coords = [], []
        for res in residues:
            if 'CA' not in res:
                continue
            seq.append(_THREE_TO_ONE[res.get_resname()])
            coords.append(res['CA'].coord.astype(np.float32))
        
        if len(seq) >= 5:
            return "".join(seq), np.vstack(coords)
    
    return None, None


def get_contact_map(structure, chain_id: Optional[str] = None, 
                   thresh: float = 8.0) -> Tuple[Optional[str], Optional[torch.Tensor]]:
    """
    Compute contact map from Cα-Cα distances.
    
    Args:
        structure: Bio.PDB Structure object
        chain_id: Specific chain ID
        thresh: Distance threshold (Å) for defining contacts
    
    Returns:
        sequence: Amino acid sequence (or None)
        contact_map: (L, L) bool tensor (or None)
    """
    seq, coords = extract_chain_seq_and_ca_coords(structure, chain_id)
    if seq is None or coords is None:
        return None, None
    
    # Vectorized pairwise distance computation
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    
    # Contact = distance < threshold (exclude diagonal)
    cm = (dists < thresh) & (dists > 0.0)
    
    return seq, torch.from_numpy(cm).bool()


# ===================== ATTENTION PROCESSING =====================
@torch.no_grad()
def get_attn_data(sequence: str,
                  layer_indices: Optional[List[int]] = None,
                  drop_special_tokens: bool = True,
                  renormalize_rows: bool = False) -> Tuple[torch.Tensor, List[int]]:
    """
    Extract attention tensors for a protein sequence.
    
    Args:
        sequence: Amino acid sequence string
        layer_indices: Which layers to extract (None = all)
        drop_special_tokens: Remove special tokens from attention
        renormalize_rows: Renormalize attention after filtering
    
    Returns:
        attn: (L_layers, H, T, T) attention tensor on CPU
        keep_idx: Indices of non-special tokens
    """
    # Tokenize
    batch = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    
    # Get attention
    out = model(**batch, output_attentions=True, use_cache=False)
    atts = out.attentions
    
    # Select layers
    if layer_indices is None:
        layer_indices = list(range(len(atts)))
    
    # Filter special tokens
    input_ids = batch["input_ids"][0]
    if drop_special_tokens:
        special = set(tokenizer.all_special_ids or [])
        keep_idx = [i for i, tok in enumerate(input_ids.tolist()) if tok not in special]
    else:
        keep_idx = list(range(input_ids.shape[0]))
    
    # Process each layer
    processed = []
    for i in layer_indices:
        A = atts[i][0]  # (H, T, T)
        if renormalize_rows:
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        processed.append(A.float().cpu())
    
    return torch.stack(processed, dim=0), keep_idx


# ===================== ANALYSIS =====================
@torch.no_grad()
def attn_corr(pdb_items: List[str],
              theta: float = 0.10,
              layer_indices: Optional[List[int]] = None,
              chain_id: Optional[str] = None,
              contact_thresh: float = 8.0,
              log_file: str = "failed_pdb_ids.log") -> torch.Tensor:
    """
    Compute proportion of high-attention pairs that are contacts across dataset.
    
    Args:
        pdb_items: List of PDB IDs to analyze
        theta: Attention threshold for "high attention"
        layer_indices: Layers to analyze (None = all)
        chain_id: Specific chain ID (None = first valid)
        contact_thresh: Distance threshold (Å) for contacts
        log_file: Path to log failed PDB IDs
    
    Returns:
        prop_matrix: (num_heads, num_layers) tensor of proportions
    """
    if layer_indices is None:
        layer_indices = list(range(model.config.num_hidden_layers))
    
    num_heads = model.config.num_attention_heads
    num_layers = len(layer_indices)
    
    # Accumulators
    prop_sum = np.zeros((num_heads, num_layers), dtype=np.float32)
    counts = np.zeros((num_heads, num_layers), dtype=np.int32)
    
    failed_count = 0
    
    for pdb_id in tqdm(pdb_items, desc="Processing structures"):
        try:
            # Get structure
            structure = get_structure(pdb_id)
            
            # Get contact map
            seq, C = get_contact_map(structure, chain_id=chain_id, thresh=contact_thresh)
            if seq is None or C is None:
                failed_count += 1
                with open(log_file, "a") as f:
                    f.write(f"{pdb_id}\tFailed to extract sequence/contacts\n")
                continue
            
            # Get attention
            A, keep_idx = get_attn_data(seq, layer_indices=layer_indices,
                                       drop_special_tokens=True, renormalize_rows=False)
            
            # Validate shapes
            if len(keep_idx) != len(seq):
                failed_count += 1
                with open(log_file, "a") as f:
                    f.write(f"{pdb_id}\tToken mismatch: {len(keep_idx)} tokens vs {len(seq)} residues\n")
                continue
            
            # Analyze each layer and head
            for layer_i, layer in enumerate(layer_indices):
                for head in range(num_heads):
                    # Extract attention submatrix
                    attn_matrix = A[layer_i, head][keep_idx][:, keep_idx]
                    high_attn = attn_matrix >= theta
                    
                    if high_attn.shape != C.shape:
                        failed_count += 1
                        with open(log_file, "a") as f:
                            f.write(f"{pdb_id}\tShape mismatch: {high_attn.shape} vs {C.shape}\n")
                        break  # Skip to next PDB
                    
                    # Compute proportion
                    num_high = high_attn.sum().item()
                    if num_high > 0:
                        num_contact_and_high = (C & high_attn).sum().item()
                        prop_sum[head, layer_i] += num_contact_and_high / num_high
                        counts[head, layer_i] += 1
        
        except Exception as e:
            failed_count += 1
            with open(log_file, "a") as f:
                f.write(f"{pdb_id}\t{str(e)}\n")
            continue
    
    # Report summary
    if failed_count > 0:
        print(f"\n⚠️  Failed: {failed_count}/{len(pdb_items)} structures (see {log_file})")
    
    # Average across structures
    prop_matrix = np.divide(prop_sum, counts, out=np.zeros_like(prop_sum), where=counts > 0)
    
    return torch.from_numpy(prop_matrix)


# ===================== MAIN =====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute attention-contact correlation for protein structures"
    )
    parser.add_argument("--precision", type=str, default="bf16",
                        help="Computation precision (fp32, bf16, etc.)")
    parser.add_argument("--theta", type=float, default=0.1,
                        help="Threshold for high-attention pairs")
    parser.add_argument("--data_sample", type=int, default=None,
                        help="Limit number of proteins to process (None = all)")
    parser.add_argument("--save_path", type=str, default="results/prop_matrix.csv",
                        help="Where to save the final proportion matrix (CSV)")
    parser.add_argument("--layer_indices", type=int, nargs="+", default=[0, 1, 2],
                        help="Indices of attention layers to analyze")
    parser.add_argument("--contact_thresh", type=float, default=8.0,
                        help="Distance threshold (Å) for residue contacts")
    parser.add_argument("--chain_id", type=str, default=None,
                        help="Optional chain ID (default: first valid chain)")
    
    args = parser.parse_args()
    
    # Sample PDB list
    sample_list = pdb_list[:args.data_sample] if args.data_sample else pdb_list
    print(f"Analyzing {len(sample_list)} PDB structures\n")
    
    # Run analysis
    prop_matrix = attn_corr(
        sample_list,
        theta=args.theta,
        layer_indices=args.layer_indices,
        chain_id=args.chain_id,
        contact_thresh=args.contact_thresh
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    
    df = pd.DataFrame(
        prop_matrix.numpy(),
        index=[f"head_{i}" for i in range(prop_matrix.shape[0])],
        columns=[f"layer_{j}" for j in range(prop_matrix.shape[1])]
    )
    df.to_csv(args.save_path)
    
    print(f"\n✅ Saved results to {args.save_path}")
    print(f"   Shape: {prop_matrix.shape}")
    print(f"   Mean proportion: {prop_matrix.mean():.4f}")
    print(f"   Std: {prop_matrix.std():.4f}")