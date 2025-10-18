import io, urllib
from typing import Iterable, List, Optional, Tuple
import numpy as np
import torch
from Bio.PDB import PDBParser, PPBuilder
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd
import os
import mamba_ssm

# ----------------- Model setup -----------------
model_id = "microsoft/Dayhoff-170m-UR90"
device = "cuda"

cfg = AutoConfig.from_pretrained(model_id)
cfg.output_attentions = True
cfg.return_dict = True

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, config=cfg, device_map="cuda").to(device).eval()

# ----------------- PDB helpers -----------------
def get_structure(pdb_id: str):
    resource = urllib.request.urlopen(f'https://files.rcsb.org/download/{pdb_id}.pdb')
    content = resource.read().decode('utf8')
    handle = io.StringIO(content)
    parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_id, handle)

_three_to_one = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I',
    'LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S',
    'THR':'T','VAL':'V','TRP':'W','TYR':'Y'
}

def extract_chain_seq_and_ca_coords(structure, chain_id: Optional[str] = None):
    """
    Extracts amino acid sequence and corresponding Cα coordinates
    from a Biopython Structure object.

    Args:
        structure: Bio.PDB.Structure.Structure
        chain_id: optional chain identifier (if None, use first valid chain)

    Returns:
        sequence: str of one-letter amino acids
        coords: numpy array of shape (L, 3)
    """
    model = next(structure.get_models())
    chains = [c for c in model.get_chains()]

    # pick chain
    if chain_id:
        chains = [c for c in chains if c.id == chain_id]
        if not chains:
            raise ValueError(f"Chain {chain_id} not found in structure.")
    chosen = None
    for c in chains:
        residues = [r for r in c.get_residues() if r.get_resname() in _three_to_one]
        if len(residues) >= 5:
            chosen = c
            break
    if chosen is None:
        raise ValueError("No suitable chain with >=5 standard residues found.")

    seq, coords = [], []
    for res in chosen.get_residues():
        resname = res.get_resname()
        if resname not in _three_to_one:
            continue
        # safely check for CA atom
        if 'CA' not in res:
            continue
        ca_atom = res['CA']
        seq.append(_three_to_one[resname])
        coords.append(ca_atom.coord.astype(float))

    if len(seq) < 2:
        raise ValueError("Too few residues with Cα atoms to compute distances.")

    import numpy as np
    return "".join(seq), np.vstack(coords)


def get_contact_map(structure, chain_id: Optional[str]=None, thresh: float = 8.0) -> Tuple[str, torch.Tensor]:
    """
    Returns (sequence, contact_map) where contact_map is (L,L) bool tensor,
    using Cα–Cα distance < thresh Å as contact.
    """
    seq, coords = extract_chain_seq_and_ca_coords(structure, chain_id)
    # pairwise distances
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diffs * diffs).sum(axis=-1))
    cm = (dists < thresh) & (dists > 0.0)  # exclude self
    return seq, torch.from_numpy(cm).to(torch.bool)

# ----------------- Attention extraction -----------------
@torch.no_grad()
def get_attn_data(sequence: str,
                  layer_indices: Optional[List[int]] = None,
                  drop_special_tokens: bool = True,
                  renormalize_rows: bool = False) -> Tuple[torch.Tensor, List[int]]:
    """
    Returns attention tensor of shape (L_layers, H, T, T) on CPU by default.
    """
    batch = tokenizer(sequence,
                      return_tensors="pt",
                      add_special_tokens=True,
                      return_token_type_ids=False)
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(**batch, output_attentions=True, use_cache=False)
    atts = out.attentions  # tuple of len = number of attention layers emitted

    if layer_indices is None:
        layer_indices = list(range(len(atts)))
    selected = [atts[i] for i in layer_indices]  # each: (B, H, T, T)

    input_ids = batch["input_ids"][0]
    if drop_special_tokens:
        special = set(tokenizer.all_special_ids or [])
        keep_idx = [i for i, tok in enumerate(input_ids.tolist()) if tok not in special]
    else:
        keep_idx = list(range(input_ids.shape[0]))

    processed = []
    for A in selected:
        A = A[0]  # (H, T, T)
        if len(keep_idx) != A.shape[-1]:
            A = A[:, keep_idx, :]
            A = A[:, :, keep_idx]
            if renormalize_rows:
                denom = A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                A = A / denom
        processed.append(A.float().cpu())  # remove .cpu() to keep on GPU
    attn = torch.stack(processed, dim=0)  # (L_layers, H, T, T)
    return attn, keep_idx

# ----------------- Aggregation over dataset -----------------
@torch.no_grad()
def attn_corr(pdb_items: Iterable,
              theta: float = 0.10,
              layer_indices: Optional[List[int]] = None,
              chain_id: Optional[str] = None,
              contact_thresh: float = 8.0) -> torch.Tensor:
    """
    Aggregate p_α(f) over a dataset X as in the figure.

    pdb_items: iterable of either PDB IDs (str) or Bio.PDB Structure objects.
    Returns: matrix (H, L_layers) = proportion of high-attention pairs that are contacts.
    """
    # We’ll infer H and L_layers on the first item
    running_num = None   # (L_layers, H)
    running_den = None   # (L_layers, H)

    for item in pdb_items:
        structure = get_structure(item) if isinstance(item, str) else item

        # contact map (L_res, L_res)
        seq_res, C = get_contact_map(structure, chain_id=chain_id, thresh=contact_thresh)  # bool
        if C.size(0) < 2:
            continue

        # attentions on tokenized sequence (T,T)
        A, keep_idx = get_attn_data(seq_res, layer_indices=layer_indices,
                                    drop_special_tokens=True, renormalize_rows=False)  # (L_layers,H,T,T)

        L_layers, H, T, _ = A.shape
        L = min(T, C.shape[0])          # align lengths
        if L < 2:
            continue

        A = A[:, :, :L, :L]            # crop attentions
        Cm = C[:L, :L]                  # crop contact map
        Cm = Cm.to(A.device) if A.is_cuda else Cm  # dtype=bool

        # Build high-attention mask
        high = (A > theta)              # (L_layers,H,L,L) bool
        # Numerator/denominator per (layer, head)
        num = (high & Cm)               # broadcast Cm over layers/heads
        # sum over i,j
        num = num.sum(dim=(-2, -1)).to(torch.float64)   # (L_layers,H)
        den = high.sum(dim=(-2, -1)).clamp_min(1).to(torch.float64)  # (L_layers,H)

        if running_num is None:
            running_num = num
            running_den = den
        else:
            # make sure shapes match
            if num.shape != running_num.shape:
                raise ValueError("Layer/head shape mismatch across items. "
                                 "Ensure consistent layer_indices.")
            running_num += num
            running_den += den

    if running_num is None:
        raise ValueError("No valid items processed.")

    prop = (running_num / running_den).cpu()   # (L_layers, H)
    # return as (H x L_layers) per your spec (heads rows, layers cols)
    return prop.transpose(0, 1).contiguous()

# ----------------- Example -----------------
# Example: aggregate over a few PDB IDs (replace with yours)


if __name__ == "__main__":
    import argparse, os
    import pandas as pd
    import torch

    parser = argparse.ArgumentParser(description="Compute attention-contact correlation matrix")

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

    # ----------------------- Define PDB list here -----------------------
    # Option 1: hardcode a few test proteins
    pdb_list = ["1crn"]  # replace or extend this list

    # Option 2 (alternative): load from file 'pdb_ids.txt', one ID per line
    # with open("pdb_ids.txt") as f:
    #     pdb_list = [line.strip() for line in f if line.strip()]
    # --------------------------------------------------------------------

    # dtype handling
    if args.precision.lower() == "bf16":
        torch.set_default_dtype(torch.bfloat16)
    elif args.precision.lower() == "fp32":
        torch.set_default_dtype(torch.float32)

    print(f"Running attention-contact correlation:")
    print(f"  θ = {args.theta}")
    print(f"  Layers = {args.layer_indices}")
    print(f"  PDBs = {len(pdb_list)} proteins")
    print(f"  Save path = {args.save_path}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # If user specified data_sample, limit the list
    if args.data_sample is not None:
        pdb_list = pdb_list[:args.data_sample]

    # Run computation
    prop_matrix = attn_corr(
        pdb_list,
        theta=args.theta,
        layer_indices=args.layer_indices,
        chain_id=args.chain_id,
        contact_thresh=args.contact_thresh
    )

    # Save to CSV
    df = pd.DataFrame(
        prop_matrix.numpy(),
        index=[f"head_{i}" for i in range(prop_matrix.shape[0])],
        columns=[f"layer_{j}" for j in range(prop_matrix.shape[1])]
    )
    df.to_csv(args.save_path)
    print(f"✅ Saved matrix to {args.save_path}")


