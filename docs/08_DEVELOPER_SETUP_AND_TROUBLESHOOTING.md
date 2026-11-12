# Developer Setup, Installation & Troubleshooting Guide

## 1. System Requirements & Prerequisites

GeneWeaver requires an NVIDIA GPU with compute capability $\ge 7.0$ (Turing architecture or newer: RTX 2000 series, RTX 3000/4000 series, A100, H100, L40S).

### 1.1. Hardware Requirements
- **GPU:** NVIDIA GPU with minimum 6 GB VRAM (16 GB - 24 GB recommended for full human genome scans).
- **CPU:** Quad-core x86_64 CPU (AMD Ryzen or Intel Core i7/i9 or Xeon/Threadripper).
- **RAM:** Minimum 16 GB host memory (32 GB - 64 GB recommended for chunk buffering).
- **Storage:** NVMe SSD with at least 15 GB free space for human genome FASTA files (`hg38.fa`).

### 1.2. Software Requirements
- **Operating System:** Windows 10/11 (64-bit) or Ubuntu 20.04/22.04 LTS.
- **NVIDIA GPU Driver:** Version $\ge 535.xx$ (Windows) or $\ge 535.54$ (Linux).
- **CUDA Toolkit:** Version 12.0 – 12.4.
- **Python Runtime:** Python 3.10 or 3.11 (Python 3.12 has partial Numba CUDA support).

---

## 2. Step-by-Step Installation

### 2.1. Conda Environment Setup (Recommended)
Using Miniconda or Mamba ensures binary compatibility for CUDA libraries without compiling from source:

```bash
# 1. Clone or navigate to the repository
cd d:\internship-project-2

# 2. Create dedicated Python 3.11 environment
conda create -n geneweaver python=3.11 -y
conda activate geneweaver

# 3. Install CUDA Toolkit and Numba from Conda-Forge
conda install -c conda-forge -c nvidia cuda-toolkit numba cudatoolkit -y

# 4. Install Bioinformatics and UI Dependencies
pip install biopython textual textual-dev rich pynvml dask distributed dask-cuda pytest
```

### 2.2. Verify CUDA Installation with Numba
Run the following verification script to confirm Numba correctly binds to your NVIDIA GPU:

```python
# check_cuda.py
from numba import cuda
import sys

print(f"CUDA Available: {cuda.is_available()}")
if cuda.is_available():
    device = cuda.get_current_device()
    print(f"Detected GPU: {device.name}")
    print(f"Compute Capability: {device.compute_capability}")
    print(f"Total VRAM: {device.total_memory / (1024**3):.2f} GB")
else:
    print("ERROR: CUDA hardware or driver not detected!", file=sys.stderr)
    sys.exit(1)
```

Execute:
```bash
python check_cuda.py
```
*Expected Output:*
```
CUDA Available: True
Detected GPU: b'NVIDIA GeForce RTX 4090'
Compute Capability: (8, 9)
Total VRAM: 24.00 GB
```

---

## 3. Configuration & Sample Config File

GeneWeaver is configured via `config/geneweaver.yaml` or CLI arguments:

```yaml
# config/geneweaver.yaml
engine:
  mode: "gpu"                  # "gpu", "multi_gpu", or "cpu_fallback"
  device_ids: [0]              # List of GPU devices to utilize
  threads_per_block: 256       # CUDA block size (256 or 512)
  chunk_size_mb: 50            # Size of genomic slices dispatched to VRAM
  overlap_bp: 22               # Boundary seam overlap (Guide_len + PAM_len - 1)
  stream_buffers: 2            # Double buffering with asynchronous CUDA streams

biology:
  target_guide: "GAGTCCGAGCAGAAGAAGAA" # 20-nt target sgRNA
  pam_motif: "NGG"             # Protospacer adjacent motif
  max_mismatches: 4            # Mismatch cutoff threshold (1-6)
  scoring_model: "hsu_zhang"   # "hsu_zhang" or "cfd"
  score_threshold: 0.01        # Minimum cleavage probability to report

tui:
  refresh_rate_hz: 10          # Hardware telemetry polling frequency
  color_mismatches: true       # Highlight mutated base pairs in bold red
```

---

## 4. Troubleshooting Runbook & Common Pitfalls

### Pitfall 1: `NumbaSupportError: CUDA driver library not found`
- **Symptom:** Numba throws an error claiming no CUDA driver or `nvcuda.dll` / `libcuda.so` cannot be located.
- **Root Cause:** NVIDIA GPU drivers are either not installed, or Windows `PATH` / Linux `LD_LIBRARY_PATH` does not reference the CUDA driver folder.
- **Remedy:**
  - *On Windows:* Ensure `C:\Windows\System32\nvcuda.dll` exists. If using Conda, run:
    ```powershell
    $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
    ```
  - *On Linux:*
    ```bash
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    ```

### Pitfall 2: `CUDA_ERROR_OUT_OF_MEMORY (Code 2)`
- **Symptom:** Kernel launch crashes during whole-genome scanning when allocating buffers.
- **Root Cause:** Chunk size is set too large for available free VRAM, or background processes (e.g. browser, model inference) consume memory.
- **Remedy:**
  1. Inspect active processes via `nvidia-smi`.
  2. Reduce `chunk_size_mb` from `100` down to `25` or `50` in `geneweaver.yaml`.
  3. Ensure `d_out_coords` candidate buffer size is bounded (e.g., maximum $500,000$ hits per chunk).

### Pitfall 3: `pynvml.NVMLError_LibraryNotFound` in TUI
- **Symptom:** Textual crashes on launch when initializing the GPU telemetry monitor.
- **Root Cause:** Outdated `pynvml` package looking for older `nvml.dll` paths.
- **Remedy:**
  Reinstall the official NVIDIA Management Library wrapper:
  ```bash
  pip uninstall -y pynvml nvidia-ml-py
  pip install nvidia-ml-py
  ```
  In Python, import as `import pynvml` (compatible with both names).

### Pitfall 4: Textual UI Renders Corrupted Characters on Windows
- **Symptom:** Border boxes or progress bars appear as garbled characters (e.g., `â`).
- **Root Cause:** Legacy Windows Command Prompt (`cmd.exe`) defaults to CP-1252 instead of UTF-8.
- **Remedy:**
  1. Use modern **Windows Terminal** (or VS Code integrated terminal).
  2. Set terminal encoding to UTF-8 before launching:
     ```powershell
     $OutputEncoding = [System.Text.Encoding]::UTF8
     [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
     $env:PYTHONIOENCODING = "utf-8"
     ```

### Pitfall 5: BioPython `MemoryError` when Reading `.fasta`
- **Symptom:** System RAM exhausts immediately when opening `hg38.fa`.
- **Root Cause:** Calling `SeqIO.read()` or `SeqIO.to_dict()` attempts to load all 3.2 billion characters into an uncompressed Python heap object.
- **Remedy:**
  Always use GeneWeaver's streaming generator `SeqIO.parse()` combined with byte-offset chunking (`src/geneweaver/parser/fasta_streamer.py`).
