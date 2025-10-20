"""
Attention-Contact Correlation Analysis for Protein Structures
GPU-optimized version with efficient tensor operations
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
MODEL_IDS = ["microsoft/Dayhoff-3b-GR-HM-c", "microsoft/Dayhoff-3b-GR-HM", "microsoft/Dayhoff-3b-UR90", "microsoft/Dayhoff-170m-UR50-BRn", "microsoft/Dayhoff-170m-UR50-BRq", "microsoft/Dayhoff-170m-UR50-BRu", "microsoft/Dayhoff-170m-GR", "microsoft/Dayhoff-170m-UR90", "microsoft/Dayhoff-170m-UR50"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_THREE_TO_ONE = {
    'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F', 'GLY':'G', 'HIS':'H', 'ILE':'I',
    'LYS':'K', 'LEU':'L', 'MET':'M', 'ASN':'N', 'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
    'THR':'T', 'VAL':'V', 'TRP':'W', 'TYR':'Y'
}



# ===================== PDB PROCESSING =====================
from pathlib import Path
CACHE = Path(".pdb_cache"); CACHE.mkdir(exist_ok=True)

def get_structure(pdb_id: str):
    path = CACHE / f"{pdb_id}.pdb"
    if not path.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        with urllib.request.urlopen(url) as r:
            path.write_bytes(r.read())
    parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_id, str(path))


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
        contact_map: (L, L) bool tensor on GPU (or None)
    """
    seq, coords = extract_chain_seq_and_ca_coords(structure, chain_id)
    if seq is None or coords is None:
        return None, None
    
    # Move to GPU immediately for distance computation
    coords_gpu = torch.from_numpy(coords).to(DEVICE)
    
    # Vectorized pairwise distance computation on GPU
    diffs = coords_gpu[:, None, :] - coords_gpu[None, :, :]
    dists = torch.norm(diffs, dim=-1)
    
    # Contact = distance < threshold (exclude diagonal)
    cm = (dists < thresh) & (dists > 0.0)
    
    return seq, cm  # Return GPU tensor


# ===================== ATTENTION PROCESSING =====================
@torch.no_grad()
def get_attn_data(sequence: str,
                  layer_indices: Optional[List[int]] = None,
                  drop_special_tokens: bool = True,
                  renormalize_rows: bool = True) -> Tuple[torch.Tensor, List[int]]:
    """
    Extract attention tensors for a protein sequence.
    
    Args:
        sequence: Amino acid sequence string
        layer_indices: Which layers to extract (None = all)
        drop_special_tokens: Remove special tokens from attention
        renormalize_rows: Renormalize attention after filtering
    
    Returns:
        attn: (L_layers, H, T, T) attention tensor on GPU
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
    
    # Process each layer - keep on GPU
    processed = []
    for i in layer_indices:
        A = atts[i][0]  # (H, T, T)
        if renormalize_rows:
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        processed.append(A.float())  # Keep on GPU

    return torch.stack(processed, dim=0), keep_idx


@torch.no_grad()
def get_attn_data_batch(sequences: List[str],
                        layer_indices: Optional[List[int]] = None,
                        drop_special_tokens: bool = True,
                        renormalize_rows: bool = True) -> List[Tuple[torch.Tensor, List[int]]]:
    """
    Extract attention tensors for multiple protein sequences in a batch.
    
    Args:
        sequences: List of amino acid sequence strings
        layer_indices: Which layers to extract (None = all)
        drop_special_tokens: Remove special tokens from attention
        renormalize_rows: Renormalize attention after filtering
    
    Returns:
        List of (attn, keep_idx) tuples for each sequence
    """
    if not sequences:
        return []
    
    # Tokenize all sequences with padding
    batch = tokenizer(
        sequences, 
        return_tensors="pt", 
        add_special_tokens=True,
        padding=True,
        truncation=True
    )
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    
    # Get attention for entire batch
    out = model(**batch, output_attentions=True, use_cache=False)
    atts = out.attentions
    
    # Select layers
    if layer_indices is None:
        layer_indices = list(range(len(atts)))
    
    # Process each sequence in the batch
    results = []
    for seq_idx in range(len(sequences)):
        input_ids = batch["input_ids"][seq_idx]
        attention_mask = batch["attention_mask"][seq_idx]
        
        # Find actual sequence length (excluding padding)
        actual_len = attention_mask.sum().item()
        
        # Filter special tokens
        if drop_special_tokens:
            special = set(tokenizer.all_special_ids or [])
            keep_idx = [i for i in range(actual_len) 
                       if input_ids[i].item() not in special]
        else:
            keep_idx = list(range(actual_len))
        
        # Process each layer for this sequence
        processed = []
        for i in layer_indices:
            A = atts[i][seq_idx, :, :actual_len, :actual_len]  # (H, T, T)
            if renormalize_rows:
                A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            processed.append(A.float())
        
        results.append((torch.stack(processed, dim=0), keep_idx))
    
    return results


# ===================== ANALYSIS =====================
@torch.no_grad()
def attn_corr(pdb_items: List[str],
              threshold: float = 0.01,
              layer_indices: Optional[List[int]] = None,
              chain_id: Optional[str] = None,
              contact_thresh: float = 8.0,
              batch_size: int = 8,
              log_file: str = "failed_pdb_ids.log") -> torch.Tensor:
    """
    Compute proportion of high-attention pairs that are contacts across dataset.
    
    Args:
        pdb_items: List of PDB IDs to analyze
        threshold: Top-k percent threshold for high-attention pairs
        layer_indices: Layers to analyze (None = all)
        chain_id: Specific chain ID (None = first valid)
        contact_thresh: Distance threshold (Å) for contacts
        batch_size: Number of sequences to process in parallel
        log_file: Path to log failed PDB IDs
    
    Returns:
        prop_matrix: (num_heads, num_layers) tensor of proportions
    """

    if layer_indices is None:
        layer_indices = list(range(model.config.num_hidden_layers))
    
    num_heads = model.config.num_attention_heads
    num_layers = len(layer_indices)
    
    # Use GPU tensors for accumulation
    prop_sum = torch.zeros((num_heads, num_layers), dtype=torch.float32, device=DEVICE)
    counts = torch.zeros((num_heads, num_layers), dtype=torch.int32, device=DEVICE)
    
    failed_count = 0
    
    # Process structures in batches
    num_batches = (len(pdb_items) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(pdb_items))
        batch_pdb_ids = pdb_items[start_idx:end_idx]
        
        # Load and preprocess all structures in batch
        batch_data = []
        for pdb_id in batch_pdb_ids:
            try:
                structure = get_structure(pdb_id)
                seq, C = get_contact_map(structure, chain_id=chain_id, thresh=contact_thresh)
                
                if seq is None or C is None:
                    failed_count += 1
                    with open(log_file, "a") as f:
                        f.write(f"{pdb_id}\tFailed to extract sequence/contacts\n")
                    continue
                
                batch_data.append({
                    'pdb_id': pdb_id,
                    'sequence': seq,
                    'contact_map': C
                })
                
            except Exception as e:
                failed_count += 1
                with open(log_file, "a") as f:
                    f.write(f"{pdb_id}\t{str(e)}\n")
                continue
        
        if not batch_data:
            continue
        
        # Extract sequences for batch processing
        sequences = [item['sequence'] for item in batch_data]
        
        # Get attention for all sequences in one forward pass
        try:
            attention_results = get_attn_data_batch(
                sequences,
                layer_indices=layer_indices,
                drop_special_tokens=True,
                renormalize_rows=False
            )
        except Exception as e:
            # Fallback to individual processing if batch fails
            print(f"\nBatch processing failed, falling back to individual: {e}")
            attention_results = []
            for seq in sequences:
                try:
                    attn, keep_idx = get_attn_data(
                        seq,
                        layer_indices=layer_indices,
                        drop_special_tokens=True,
                        renormalize_rows=False
                    )
                    attention_results.append((attn, keep_idx))
                except Exception as e2:
                    attention_results.append((None, None))
        
        # Process each item in the batch
        for idx, item in enumerate(batch_data):
            pdb_id = item['pdb_id']
            seq = item['sequence']
            C = item['contact_map']
            
            if idx >= len(attention_results):
                failed_count += 1
                with open(log_file, "a") as f:
                    f.write(f"{pdb_id}\tAttention extraction failed\n")
                continue
            
            A, keep_idx = attention_results[idx]
            
            if A is None or keep_idx is None:
                failed_count += 1
                with open(log_file, "a") as f:
                    f.write(f"{pdb_id}\tAttention extraction failed\n")
                continue
            
            # Validate shapes
            if len(keep_idx) != len(seq):
                failed_count += 1
                with open(log_file, "a") as f:
                    f.write(f"{pdb_id}\tToken mismatch: {len(keep_idx)} tokens vs {len(seq)} residues\n")
                continue
            
            try:
                # Precompute lower triangle mask once
                T = len(keep_idx)
                tri_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEVICE), diagonal=-1)
                
                # Process all layers and heads
                for layer_i in range(num_layers):
                    # Get attention for all heads: (H, T, T)
                    attention_full = A[layer_i, :, :, :]
                    attention_mat = attention_full[:, keep_idx, :][:, :, keep_idx]
                    
                    # Process each head
                    for head in range(num_heads):
                        vals = attention_mat[head][tri_mask]
                        if vals.numel() == 0:
                            continue
                        
                        # Compute threshold and high-attention mask
                        theta = torch.quantile(vals, 1 - threshold)
                        high_tri = (attention_mat[head] > theta) & tri_mask
                        
                        # Count high attention pairs and contacts
                        num_high = high_tri.sum()
                        if num_high > 0:
                            num_contact_and_high = (C & high_tri).sum()
                            
                            # Accumulate on GPU
                            prop_sum[head, layer_i] += num_contact_and_high.float()
                            counts[head, layer_i] += num_high
            
            except Exception as e:
                failed_count += 1
                with open(log_file, "a") as f:
                    f.write(f"{pdb_id}\tProcessing error: {str(e)}\n")
                continue
        
        # Clear cache periodically
        if torch.cuda.is_available() and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Report summary
    if failed_count > 0:
        print(f"\n⚠️  Failed: {failed_count}/{len(pdb_items)} structures (see {log_file})")
    
    # Compute proportions on GPU then move to CPU
    prop_matrix = torch.where(
        counts > 0,
        prop_sum / counts.float(),
        torch.zeros_like(prop_sum)
    ).cpu()

    return prop_matrix


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute attention-contact correlation for multiple protein models"
    )
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--top_k_percent", type=float, default=0.01)
    parser.add_argument("--data_sample", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save per-model CSVs")
    parser.add_argument("--layer_indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--contact_thresh", type=float, default=8.0)
    parser.add_argument("--chain_id", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_id", type=str, default=MODEL_IDS[0])
    args = parser.parse_args()

    # PDB subset
    sample_list = pdb_list[:args.data_sample] if args.data_sample else pdb_list
    os.makedirs(args.save_dir, exist_ok=True)

    MODEL_ID = args.model_id if args.model_id else MODEL_IDS[0]

    print(f"Running model: {MODEL_ID}")

    # ---- Load model ----
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    cfg.output_attentions = True
    cfg.return_dict = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        config=cfg,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(DEVICE).eval()

    print(f"✓ Loaded model on {DEVICE}")

    # ---- Run analysis ----
    prop_matrix = attn_corr(
        sample_list,
        threshold=args.top_k_percent,
        layer_indices=args.layer_indices,
        chain_id=args.chain_id,
        contact_thresh=args.contact_thresh,
        batch_size=args.batch_size
    )

    # ---- Save results ----
    model_name = MODEL_ID.split("/")[-1]
    save_path = os.path.join(args.save_dir, f"{model_name}_prop_matrix.csv")
    df = pd.DataFrame(
        prop_matrix.numpy(),
        index=[f"head_{i}" for i in range(prop_matrix.shape[0])],
        columns=[f"layer_{j}" for j in range(prop_matrix.shape[1])]
    )
    df.to_csv(save_path)
    print(f"✅ Saved {save_path} | mean={prop_matrix.mean():.4f}, std={prop_matrix.std():.4f}")

    # Free memory
    del prop_matrix, df, tokenizer, model, cfg
    torch.cuda.empty_cache()

    print("\n🎯model processed successfully!")
