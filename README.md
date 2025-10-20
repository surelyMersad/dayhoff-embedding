# Attention analysis in Dayhoff models

analyzing how attention patterns in dayhoffs protein language models align with structural contacts in protein 3D structures.


## 📊 Methodology

1. **Structure Processing**: Downloads PDB files and extracts Cα coordinates
2. **Contact Map Generation**: Computes binary contact maps based on distance thresholds (default: 8Å)
3. **Attention Extraction**: Extracts attention patterns from specified layers of protein language models
4. **Correlation Analysis**: Measures what proportion of high-attention pairs correspond to structural contacts

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/attention-contact-correlation.git
cd attention-contact-correlation
```

2. **Create conda environment**
```bash
conda create -n prot python=3.10
conda activate prot
```

3. **Install PyTorch with CUDA support**

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Create PDB list file**

Create a file `pdb_ids.py` with your PDB IDs by running parse_data.py script (you can change the number of parsed sequences in parse_data.py directly:
```python
python parse_data.py
```

## Analysis 

To caclulate the proportion of high attention token pairs that correspond to contact, the following steps were taken : 

1) Extract the high attention pairs (can specify with --top_k_percent arg)
2) Calulate the contact maps (residues that are closer than 8.0 A)
3) Report the proportion of high attention pairs that are also in contact


## 💻 Usage

### Basic Usage

```bash
python gp.py \
    --top_k_percent 0.01 \
    --data_sample 100 \
    --save_dir results \
    --batch_size 32
```

### Using Accelerate (Recommended for Multi-GPU)

```bash
accelerate launch gp.py \
    --top_k_percent 0.01 \
    --data_sample 100 \
    --save_dir results \
    --batch_size 32 \
    --model_id microsoft/Dayhoff-3b-GR-HM-c
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_id` | str | `microsoft/Dayhoff-3b-GR-HM-c` | Model identifier from HuggingFace |
| `--top_k_percent` | float | 0.01 | Top-k percent threshold for high-attention pairs |
| `--data_sample` | int | None | Number of PDB structures to analyze (None = all) |
| `--save_dir` | str | `results` | Directory to save output CSVs |
| `--layer_indices` | int+ | [0, 1, 2] | Which model layers to analyze |
| `--contact_thresh` | float | 8.0 | Distance threshold (Å) for defining contacts |
| `--chain_id` | str | None | Specific chain ID to analyze (None = first valid) |
| `--batch_size` | int | 8 | Number of sequences to process in parallel |
| `--precision` | str | `fp16` | Model precision (fp16/fp32) |

### Example Commands

**Analyze specific layers with custom contact threshold:**
```bash
python gp.py \
    --layer_indices 0 5 10 15 \
    --contact_thresh 10.0 \
    --batch_size 16
```

**Process a small sample for testing:**
```bash
python gp.py \
    --data_sample 10 \
    --batch_size 4
```

**Analyze a different model:**
```bash
python gp.py \
    --model_id microsoft/Dayhoff-170m-UR50 \
    --save_dir results/dayhoff_170m
```

## 📁 Output

### Results File

The script generates a CSV file for each model:
```
results/
└── Dayhoff-3b-GR-HM-c_prop_matrix.csv
```

**CSV Structure:**
- Rows: Attention heads (e.g., `head_0`, `head_1`, ...)
- Columns: Model layers (e.g., `layer_0`, `layer_1`, ...)
- Values: Proportion of high-attention pairs that are structural contacts (0.0 to 1.0)

### Failed PDB Log

Failed structures are logged to `failed_pdb_ids.log`:
```
1ABC    Failed to extract sequence/contacts
2DEF    Processing error: Invalid chain
```

## 🎯 Supported Models

The following Microsoft Dayhoff models are supported:

- `microsoft/Dayhoff-3b-GR-HM-c`
- `microsoft/Dayhoff-3b-GR-HM`
- `microsoft/Dayhoff-3b-UR90`
- `microsoft/Dayhoff-170m-UR50-BRn`
- `microsoft/Dayhoff-170m-UR50-BRq`
- `microsoft/Dayhoff-170m-UR50-BRu`
- `microsoft/Dayhoff-170m-GR`
- `microsoft/Dayhoff-170m-UR90`
- `microsoft/Dayhoff-170m-UR50`

## 🔧 Technical Details

### Architecture

```
Input: PDB IDs → Structure Download → Sequence & Contact Extraction
                                              ↓
                                    Attention Extraction (Batch)
                                              ↓
                                    Correlation Computation
                                              ↓
                                    Output: Proportion Matrix
```

### Memory Optimization

- FP16 precision on GPU for memory efficiency
- Batch processing of sequences
- Periodic CUDA cache clearing
- GPU tensor operations throughout pipeline

### Contact Map Definition

A contact is defined as two Cα atoms within a distance threshold (default 8Å), excluding:
- Self-interactions (diagonal)
- Immediate neighbors (optional)



### CUDA Out of Memory
```bash
# Reduce batch size
python gp.py --batch_size 4

# Use smaller model
python gp.py --model_id microsoft/Dayhoff-170m-UR50
```

### PDB Download Failures
- Check internet connection
- Verify PDB IDs are valid (4-character codes)
- Check RCSB PDB server status

### Token Mismatch Errors
- Some proteins may have non-standard residues
- Check `failed_pdb_ids.log` for details
- These structures are automatically skipped


## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 📧 Contact

For questions or issues, please open an issue on GitHub

## 🙏 Acknowledgments

- Microsoft Research for the Dayhoff protein language models
