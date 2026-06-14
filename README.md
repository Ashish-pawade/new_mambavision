# MambaVision + PoM: Polynomial Mixing as a Drop-in Attention Replacement in a Hybrid Mamba-Transformer Vision Backbone

> **M.Tech Research Project** — Investigating whether the Polynomial Mixer (PoM), a linear-time token-mixing operator, can replace self-attention in MambaVision while preserving accuracy and improving efficiency at high resolution.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Dataset Setup](#dataset-setup)
6. [Running the Project](#running-the-project)
7. [Reproducing Results](#reproducing-results)
8. [Configuration](#configuration)
9. [Output Files](#output-files)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)
12. [Future Improvements](#future-improvements)
13. [Citation](#citation)
14. [License](#license)

---

## Overview

### Problem Statement

Modern vision transformers apply self-attention with O(N²) time complexity over N tokens. At standard image resolutions, this is manageable. At high resolution (1024–2048 px), the attention matrix becomes the dominant bottleneck. MambaVision mitigates this with *windowed* attention that caps N at 49–196 tokens regardless of input resolution — but this also prevents any linear-time mixer from demonstrating its efficiency advantage at standard scales.

### Motivation

[MambaVision](https://arxiv.org/abs/2407.08083) (CVPR 2025, NVIDIA) is a state-of-the-art hierarchical backbone that interleaves Mamba (SSM) blocks in early stages with a small number of attention blocks in its final stages. [PoM](https://arxiv.org/abs/2604.06129) (CVPR 2026 Findings) is a linear-time token mixer that achieves the same universal-approximation property as attention by aggregating tokens into a compact polynomial state. The natural hypothesis: *replace MambaVision's attention blocks with PoM and gain efficiency at high resolution with matched accuracy.*

### Techniques Used

| Component | Description |
|-----------|-------------|
| **MambaVision** | 4-stage hierarchical backbone. Stages 0–1: pure convolution. Stages 2–3: interleaved Mamba + Attention (or PoM) blocks with windowed mixing. |
| **Mamba / SSM** | Linear-time selective state-space model. Used in the first half of blocks in Stages 2–3. |
| **Attention** | Standard multi-head self-attention with PyTorch SDPA (Flash-Attention-compatible). Default mixer type. |
| **PoM** | Polynomial Mixer with degree k=2, expansion factor 2, replacing attention in a matched-budget swap. |
| **Global Mixer Mode** | Optional `global_mixer=True` flag removes window partitioning, exposing the full spatial sequence to the mixer. Required to observe PoM's efficiency advantage. |

### Key Results (from research paper)

| Experiment | Finding |
|-----------|---------|
| Standard resolution (224–768 px), windowed | **No difference** between PoM and Attention. Windowing caps N ≤ 196 tokens. |
| Standard resolution, global mode | **Still no difference**. N ≤ 576 tokens; mixer is not the bottleneck. |
| Extreme resolution (768–2048 px), global mode | **PoM wins on throughput**: 1.30× at N=576 → **3.87× at N=4096**. Memory is identical. |
| Parameter count | PoM and Attention differ by **< 0.1%** across all model sizes. |
| Gradient/learning check | PoM variant trains correctly, loss → 0.0001 in 200 steps on a fixed batch. |

The core finding: PoM delivers a real, theory-consistent **compute/throughput advantage** in the long-sequence (high-resolution, global-mixing) regime. Memory is identical because PyTorch's SDPA kernel is already memory-efficient. The efficiency gain is confined to tasks operating at high token counts: semantic segmentation, object detection, high-resolution classification.

---

## Repository Structure

```
NNDL/
├── MambaVision/                        # Core model and training framework
│   ├── mambavision/
│   │   ├── models/
│   │   │   ├── mamba_vision.py         # ★ Main model: MambaVision, MambaVisionLayer, Block, PoM integration
│   │   │   ├── __init__.py
│   │   │   └── registry.py             # Model registration (timm-compatible)
│   │   ├── configs/
│   │   │   ├── mambavision_tiny_1k.yaml    # Full training config for MambaVision-T
│   │   │   ├── mambavision_tiny2_1k.yaml
│   │   │   ├── mambavision_small_1k.yaml
│   │   │   ├── mambavision_base_1k.yaml
│   │   │   ├── mambavision_large_1k.yaml
│   │   │   └── mambavision_large2_1k.yaml
│   │   ├── scheduler/                  # Custom LR schedulers (cosine, plateau, poly, etc.)
│   │   ├── utils/                      # Dataset utilities including LMDB loader
│   │   ├── assets/                     # Architecture diagrams and paper poster
│   │   ├── train.py                    # Multi-GPU ImageNet training script (torchrun)
│   │   ├── train.sh                    # Example 8-GPU training launch script
│   │   ├── validate.py                 # ImageNet validation with local checkpoint
│   │   ├── validate.sh                 # Example validation launch script
│   │   ├── validate_pip_model.py       # Validation using pretrained HuggingFace weights
│   │   ├── validate_pip.sh             # Example pip-model validation script
│   │   ├── throughput_measure.py       # GPU throughput and FLOPs benchmarking
│   │   ├── dummy_test.py               # Quick sanity check: load model, run on random image
│   │   └── tensorboard.py              # TensorboardX logging wrapper
│   ├── object_detection/
│   │   ├── configs/mamba_vision/       # Cascade Mask R-CNN configs for T/S/B
│   │   ├── tools/                      # MMDetection train/test scripts
│   │   └── README.md                   # Detection results, dataset setup, commands
│   ├── semantic_segmentation/
│   │   ├── configs/mamba_vision/       # UPerNet configs for T/S/B/L3
│   │   ├── tools/                      # MMSegmentation train/test scripts
│   │   └── README.md                   # Segmentation results, dataset setup, commands
│   ├── Dockerfile                      # Container: pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
│   ├── requirements.txt                # Python dependencies
│   ├── setup.py                        # pip-installable package (mambavision 1.2.0)
│   └── setup.cfg
│
├── PoM/                                # Polynomial Mixer library
│   ├── pom/
│   │   ├── __init__.py                 # Exports PoM class
│   │   ├── pom.py                      # Core PoM implementation
│   │   ├── pom_rope.py                 # PoM with Rotary Position Embedding
│   │   ├── pom_triton.py               # Triton-optimized PoM kernel
│   │   ├── pom_triton_causal.py        # Causal (autoregressive) Triton variant
│   │   ├── pom_triton_masked.py        # Masked Triton variant
│   │   └── pom_triton_rope.py          # Triton + RoPE variant
│   ├── bench_attn_vs_pom.py            # Benchmark: Flash Attn vs PoM (N sweep)
│   ├── bench_causal_attn_vs_pom.py     # Benchmark: causal variants
│   ├── bench_masked_pom.py             # Benchmark: masked PoM
│   ├── bench_rope_triton.py            # Benchmark: RoPE + Triton
│   ├── test_triton_correctness.py      # Unit test: Triton kernel correctness
│   ├── test_triton_masked_correctness.py
│   ├── test_triton_rope_correctness.py
│   ├── setup.py                        # pip-installable package (pom 0.1.0)
│   └── README.md                       # PoM usage guide and citation
│
├── docs/                               # Research documentation
│   ├── MambaVision_PoM_Research_Review.pdf    # Main research report (canonical)
│   ├── MambaVision_PoM_Research_Review.tex    # LaTeX source
│   ├── drafts/                                # Earlier draft of the report
│   └── figures/                               # Benchmark plot PDFs
│       ├── fig_standard_throughput.pdf        # Standard-res throughput sweep
│       ├── fig_extreme_throughput.pdf         # Extreme-res divergence + speedup ratio
│       ├── fig_extreme_memory.pdf             # Memory parity at extreme resolution
│       ├── fig_train_memory.pdf               # Training memory parity
│       └── fig_params.pdf                     # Parameter count comparison
│
├── notebooks/
│   └── mambavision_pom_validation.ipynb   # Interactive verification and benchmarking
│
├── results/                            # Benchmark output CSVs
│   ├── benchmark_results.csv           # Standard-res: T/S/B × attn/pom × 224–768 px
│   ├── benchmark_global.csv            # Global mode: all models, all resolutions
│   ├── benchmark_extreme.csv           # Extreme-res: T global × 768–2048 px
│   └── benchmark_train_memory.csv      # Training memory: global mode, 1024–2048 px
│
├── scripts/                            # Standalone verification and test scripts
│   ├── test_pom_cpu.py                 # CPU-only smoke test: PoM forward + backward
│   └── verify_pom.py                   # Full A–E verification suite (requires GPU)
│
├── wheels/
│   └── torch-2.4.1+cu121-cp312-cp312-linux_x86_64.whl   # Offline PyTorch wheel
│
└── README.md                           # This file
```

---

## Prerequisites

### Hardware

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | NVIDIA GPU with 8 GB VRAM | 16 GB+ (e.g., T4, A100, RTX 3090) |
| RAM | 32 GB | 64 GB+ |
| Storage | 200 GB (ImageNet) | 500 GB |
| GPU count (training) | 1 (reduced batch) | 8 (matches paper configs) |

> **Note:** Benchmarks in this project were run on a single NVIDIA Tesla T4 (16 GB) via Kaggle. All verification and benchmarking scripts work on a single GPU.

### Software

| Component | Required Version |
|-----------|-----------------|
| OS | Linux (Ubuntu 20.04/22.04 recommended) |
| Python | ≥ 3.9 (3.12 used in experiments) |
| CUDA | 12.1–12.8 (12.4 or 12.6 recommended) |
| cuDNN | ≥ 8.0 |
| PyTorch | ≥ 2.6.0 (with matching CUDA build) |

### Python Dependencies

```
torch>=2.6.0+cu121        # Core deep learning
mamba-ssm==2.2.4          # Mamba SSM CUDA kernels (must compile from source)
causal-conv1d             # Required by mamba-ssm (must compile from source)
timm==1.0.15              # Model utilities and training infrastructure
tensorboardX==2.6.2.2     # Experiment logging
einops==0.8.1             # Tensor manipulation
transformers==4.50.0      # Hugging Face utilities
Pillow==11.1.0            # Image I/O
requests==2.32.3          # Pretrained weight downloads
```

> **Critical:** `mamba-ssm` and `causal-conv1d` pre-built wheels are frequently incompatible with recent PyTorch builds due to C++ ABI mismatches. **Always compile from source.** See [Installation](#installation) for exact commands.

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ashish-pawade/new_mambavision.git
cd new_mambavision
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Or with conda:

```bash
conda create -n mambavision python=3.12
conda activate mambavision
```

### Step 3: Install PyTorch

**Option A — Online (recommended):**

```bash
pip install torch==2.6.0+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Option B — Offline wheel (pre-downloaded):**

```bash
pip install wheels/torch-2.4.1+cu121-cp312-cp312-linux_x86_64.whl
```

> Replace `cu121` with your CUDA version (e.g., `cu124` for CUDA 12.4). Run `nvcc --version` to confirm.

### Step 4: Install CUDA Kernels from Source

> **This step is mandatory.** Do not attempt to use pre-built wheels for `mamba-ssm` or `causal-conv1d` — they will almost certainly produce `undefined symbol` errors due to C++ ABI mismatches between their compiled `.so` and your PyTorch binary.

```bash
# Install causal-conv1d (required by mamba-ssm)
pip install causal-conv1d --no-build-isolation

# Install mamba-ssm
pip install mamba-ssm --no-build-isolation
```

> This compilation takes approximately 10–20 minutes. Ensure you have GCC ≥ 9 and NVCC matching your CUDA version.

### Step 5: Install Remaining Python Dependencies

```bash
pip install timm==1.0.15 tensorboardX==2.6.2.2 einops==0.8.1 \
            transformers==4.50.0 Pillow==11.1.0 requests==2.32.3
```

### Step 6: Install the PoM Library

```bash
cd PoM
pip install -e .
cd ..
```

### Step 7: Install MambaVision

```bash
cd MambaVision
pip install -e .
cd ..
```

### Step 8: Verify the Installation

**A. CPU smoke test for PoM (no GPU required):**

```bash
cd NNDL
PYTHONPATH=PoM python scripts/test_pom_cpu.py
```

Expected output:
```
forward PASS  input=(2, 49, 64)  output=(2, 49, 64)
backward PASS  params with grad: [...]

PASS
```

**B. Full A–E verification suite (requires GPU):**

```bash
cd NNDL
PYTHONPATH=MambaVision/mambavision:PoM python scripts/verify_pom.py
```

Expected output:
```
A PASS: import OK
B PASS: output shape (1, 1000)
C mamba_vision_T attn: total=31,794,248  last_mixer=1,640,960
C mamba_vision_T pom : total=31,827,592  last_mixer=1,651,856
C mamba_vision_B attn: total=97,685,288  last_mixer=4,198,400
C mamba_vision_B pom : total=97,743,216  last_mixer=4,215,824
C PASS
D bad grads: []
D PASS: all grads finite
  E step  50  loss=0.0044
  E step 100  loss=0.0003
  E step 150  loss=0.0001
  E step 200  loss=0.0001
E PASS: final loss 0.0001 < 0.1
```

**C. Quick dummy test (GPU required):**

```bash
cd MambaVision/mambavision
python dummy_test.py --model mamba_vision_T
```

Expected output:
```
mamba_vision_T model succesfully created !
Inference succesfully completed on dummy input !
```

### Docker Alternative

A Dockerfile is provided for a fully reproducible environment:

```bash
cd MambaVision
docker build -t mambavision:latest .
docker run --gpus all -it mambavision:latest bash
```

> The Dockerfile uses `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel` as base. You will still need to compile `mamba-ssm` and `causal-conv1d` from source inside the container.

---

## Dataset Setup

### ImageNet (for Classification Training and Validation)

MambaVision is trained and evaluated on ImageNet-1K (ILSVRC 2012).

#### Download

```bash
# Register and download from https://image-net.org/challenges/LSVRC/2012/
# Requires an ImageNet account.
mkdir -p /datasets/imagenet
cd /datasets/imagenet

# After downloading ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar:
mkdir train val
tar -xf ILSVRC2012_img_train.tar -C train/
tar -xf ILSVRC2012_img_val.tar -C val/

# Extract per-class archives in train/
find train/ -name "*.tar" | while read f; do
    dir=$(basename "$f" .tar)
    mkdir -p "train/$dir"
    tar -xf "$f" -C "train/$dir"
    rm "$f"
done

# Organize val/ using the official script
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
bash valprep.sh
```

#### Expected Directory Structure

```
/datasets/imagenet/
├── train/
│   ├── n01440764/    # 1,281,167 images total across 1,000 classes
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/    # 50,000 images total
    ├── n01443537/
    └── ...
```

#### LMDB Format (used in training configs)

The training config files reference `/datasets/imagenet_lmdb`. To convert ImageNet to LMDB for faster data loading (optional but recommended for multi-GPU training):

```bash
# Assumed: a conversion script generates train.lmdb and val.lmdb
# Point --data-dir in train.py to the LMDB root, or update data_dir in the YAML config.
```

> If you use the standard folder format instead of LMDB, pass `--train-split /datasets/imagenet/train --val-split /datasets/imagenet/val` to `train.py` and remove `data_dir` from the YAML config.

### COCO (for Object Detection)

```bash
cd MambaVision/object_detection
mkdir -p data/coco && cd data/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip && unzip val2017.zip && unzip annotations_trainval2017.zip
```

Expected structure:
```
data/coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

### ADE20K (for Semantic Segmentation)

```bash
cd MambaVision/semantic_segmentation
mkdir -p data/ade20k && cd data/ade20k
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
```

Expected structure:
```
data/ade20k/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

---

## Running the Project

All classification commands are run from inside `MambaVision/mambavision/` unless otherwise noted.

```bash
cd MambaVision/mambavision
```

### Available Model Variants

| Model | Depths | Dim | Heads | Params | Input |
|-------|--------|-----|-------|--------|-------|
| `mamba_vision_T` | [1,3,8,4] | 80 | [2,4,8,16] | 31.8 M | 224² |
| `mamba_vision_T2` | [1,3,11,4] | 80 | [2,4,8,16] | ~34 M | 224² |
| `mamba_vision_S` | [3,3,7,5] | 96 | [2,4,8,16] | 50.2 M | 224² |
| `mamba_vision_B` | [3,3,10,5] | 128 | [2,4,8,16] | 97.7 M | 224² |
| `mamba_vision_L` | varies | 196 | varies | ~218 M | 224² |
| `mamba_vision_L2` | varies | 196 | varies | ~241 M | 224² |

Add `mixer_type="pom"` or `mixer_type="attn"` (default) to any factory function call to switch the mixer.

---

### Training

`train.py` is a full-featured ImageNet training script supporting distributed training, mixed precision, EMA, Mixup/CutMix, cosine LR, LAMB optimizer, and LMDB datasets.

**Multi-GPU training (8 GPUs, matches paper config):**

```bash
torchrun --nproc_per_node=8 train.py \
    --config configs/mambavision_tiny_1k.yaml \
    --train-split /datasets/imagenet/train \
    --val-split /datasets/imagenet/val
```

**From the provided shell script:**

```bash
bash train.sh
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `mamba_vision_T` | Model variant |
| `--config` | — | Path to YAML config (overrides individual flags) |
| `--train-split` | — | Path to training data root |
| `--val-split` | — | Path to validation data root |
| `--batch-size` | 256 | Per-GPU batch size |
| `--lr` | 5e-4 | Peak learning rate |
| `--weight-decay` | 0.05 | Weight decay |
| `--drop-path` | 0.2 | Stochastic depth rate |
| `--amp` | — | Enable automatic mixed precision |
| `--epochs` | 310 | Total training epochs |
| `--warmup-epochs` | 20 | LR warmup epochs |
| `--output` | `./` | Root path for checkpoint output |
| `--tag` | `my_experiment` | Sub-folder name for this run |
| `--resume` | — | Resume from checkpoint path |

**Single-GPU training (for debugging; reduce batch size):**

```bash
python train.py \
    --model mamba_vision_T \
    --train-split /datasets/imagenet/train \
    --val-split /datasets/imagenet/val \
    --batch-size 64 \
    --lr 5e-4 \
    --drop-path 0.2 \
    --amp \
    --epochs 310 \
    --warmup-epochs 20 \
    --tag debug_run
```

---

### Validation with a Local Checkpoint

```bash
python validate.py \
    --model mamba_vision_T \
    --checkpoint /path/to/mambavision_tiny_1k.pth.tar \
    --data-dir /datasets/imagenet \
    --batch-size 128 \
    --input-size 3 224 224
```

Or using the shell script (edit paths first):

```bash
bash validate.sh
```

**Key arguments:**

| Argument | Description |
|----------|-------------|
| `--model` | Model name (e.g., `mamba_vision_T`) |
| `--checkpoint` | Path to `.pth.tar` checkpoint |
| `--data-dir` | ImageNet root (must contain `val/`) |
| `--batch-size` | Validation batch size (128 fits on 16 GB) |
| `--input-size 3 224 224` | Input resolution |
| `--crop-pct` | Center crop fraction (model-specific; see config) |

---

### Validation with Pretrained HuggingFace Weights

This script downloads pretrained weights automatically — no local checkpoint needed.

```bash
python validate_pip_model.py \
    --model mamba_vision_T \
    --data-dir /datasets/imagenet \
    --batch-size 128
```

Or:

```bash
bash validate_pip.sh
```

Weights are cached to `/tmp/mamba_vision_<NAME>.pth.tar` on first run.

---

### Inference on a Single Image (Quick Sanity Check)

```bash
python dummy_test.py --model mamba_vision_T
# With a checkpoint:
python dummy_test.py --model mamba_vision_T --checkpoint /path/to/checkpoint.pth.tar
```

This runs a forward pass on a random `(1, 3, 754, 234)` tensor and prints the output logit shape `(1, 1000)`.

---

### Throughput Benchmarking

`throughput_measure.py` measures GPU throughput (images/second) and parameter counts using `ptflops`.

```bash
# Install ptflops first
pip install ptflops

python throughput_measure.py \
    --model mamba_vision_T \
    --resolution 224 \
    --bs 128 \
    --channel_last
```

| Argument | Description |
|----------|-------------|
| `--model` | One of `mamba_vision_{T,T2,S,B,L,L2}` |
| `--resolution` | Input resolution (int, e.g., 224, 384, 512) |
| `--bs` | Batch size |
| `--channel_last` | Use `torch.channels_last` memory format (faster on A100/H100) |

---

### PoM-Specific Benchmarking

**Attention vs PoM throughput sweep (sequence length N sweep):**

```bash
cd PoM
python bench_attn_vs_pom.py
```

This benchmarks JIT-attention, Flash-attention (SDPA), and PoM with degrees 2–5 across N = 64 to 2048, measuring forward + backward throughput on GPU.

**Causal and masked variants:**

```bash
python bench_causal_attn_vs_pom.py
python bench_masked_pom.py
python bench_rope_triton.py
```

---

### Object Detection (COCO)

From `MambaVision/object_detection/`:

```bash
# Install MMDetection dependencies first
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless \
            mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0

# Multi-GPU training (8 GPUs)
srun --gres=gpu:8 python tools/train.py \
    configs/mamba_vision/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.py

# Single-GPU training
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/mamba_vision/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.py

# Evaluation
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/mamba_vision/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.py \
    /path/to/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.pth \
    --eval bbox segm
```

Detection results (Cascade Mask R-CNN, 3× schedule):

| Backbone | box mAP | mask mAP | Params (M) | FLOPs (G) |
|----------|---------|----------|------------|-----------|
| MambaVision-T | 51.1 | 44.3 | 86 | 740 |
| MambaVision-S | 52.3 | 45.2 | 108 | 828 |
| MambaVision-B | 52.8 | 45.7 | 145 | 964 |

---

### Semantic Segmentation (ADE20K)

From `MambaVision/semantic_segmentation/`:

```bash
# Multi-GPU training (8 GPUs)
srun --gres=gpu:8 python tools/train.py \
    configs/mamba_vision/mamba_vision_160k_ade20k-512x512_tiny.py

# Single-GPU evaluation
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/mamba_vision/mamba_vision_160k_ade20k-512x512_tiny.py \
    /path/to/checkpoint.pth
```

Segmentation results (UPerNet, 160K schedule):

| Backbone | mIoU | Params (M) | FLOPs (G) |
|----------|------|------------|-----------|
| MambaVision-T | 46.0 | 55 | 945 |
| MambaVision-S | 48.2 | 84 | 1135 |
| MambaVision-B | 49.1 | 126 | 1342 |
| MambaVision-L3-512-21K | 53.2 | 780 | 3670 |

---

## Reproducing Results

### Verification Suite (A–E)

```bash
cd /path/to/NNDL
PYTHONPATH=MambaVision/mambavision:PoM python scripts/verify_pom.py
```

- **Runtime:** ~5 minutes on a single GPU (includes 200-step overfitting check)
- **Expected output:** All 5 checks PASS (see [Installation Step 8](#step-8-verify-the-installation))
- **Output location:** stdout only

---

### Standard-Resolution Benchmark (reproduces `results/benchmark_results.csv`)

This benchmark was run from the validation notebook. To replicate manually:

```bash
cd MambaVision/mambavision
python -c "
import torch
from models.mamba_vision import *

models = [('T', mamba_vision_T), ('S', mamba_vision_S), ('B', mamba_vision_B)]
mixers = ['attn', 'pom']
resolutions = [224, 384, 512, 768]
bs = 128

for mname, factory in models:
    for mixer in mixers:
        for res in resolutions:
            model = factory(pretrained=False, mixer_type=mixer).cuda().eval()
            x = torch.randn(bs, 3, res, res).cuda()
            for _ in range(5):  # warmup
                with torch.no_grad(): model(x)
            torch.cuda.synchronize()
            import time
            t0 = time.time()
            for _ in range(20):
                with torch.no_grad(): model(x)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            print(f'{mname},{mixer},{res},{bs*20/elapsed:.1f}')
"
```

**Expected results (from `results/benchmark_results.csv`):**

| Model | Mixer | 224 px | 384 px | 512 px | 768 px |
|-------|-------|--------|--------|--------|--------|
| T | attn | 341.1 | 99.0 | 48.5 | 24.4 |
| T | pom  | 335.3 | 96.5 | 47.3 | 23.3 |
| S | attn | 214.7 | 60.9 | 29.1 | 14.1 |
| S | pom  | 187.3 | 52.9 | 25.1 | 13.6 |
| B | attn | 99.7  | 31.3 | 13.8 | 7.6  |
| B | pom  | 101.6 | 31.4 | 14.0 | 7.9  |

*Values are images/second. Differences are within measurement noise — this is the null result.*

---

### Extreme-Resolution Benchmark (reproduces `results/benchmark_extreme.csv`)

```bash
cd MambaVision/mambavision
python -c "
import torch
from models.mamba_vision import *

resolutions = [768, 1024, 1280, 1536, 2048]
bs = 4

for res in resolutions:
    for mixer in ['attn', 'pom']:
        model = mamba_vision_T(pretrained=False, mixer_type=mixer,
                               global_mixer=True).cuda().eval()
        x = torch.randn(bs, 3, res, res).cuda()
        try:
            for _ in range(5):
                with torch.no_grad(): model(x)
            torch.cuda.synchronize()
            import time
            t0 = time.time()
            for _ in range(20):
                with torch.no_grad(): model(x)
            torch.cuda.synchronize()
            print(f'{mixer},{res},{bs*20/(time.time()-t0):.2f}')
        except torch.cuda.OutOfMemoryError:
            print(f'{mixer},{res},OOM')
"
```

**Expected results (from `results/benchmark_extreme.csv`):**

| Resolution | N (tokens) | Attn (img/s) | PoM (img/s) | Speedup |
|------------|------------|-------------|------------|---------|
| 768 px  | 576  | 18.80 | 24.46 | **1.30×** |
| 1024 px | 1024 | 8.59  | 13.17 | **1.53×** |
| 1280 px | 1600 | 4.33  | 8.57  | **1.98×** |
| 1536 px | 2304 | 2.58  | 6.39  | **2.48×** |
| 2048 px | 4096 | 0.91  | 3.52  | **3.87×** |

- **Runtime:** ~15–30 minutes per resolution sweep on a T4 (synchronization overhead)
- **Output location:** `results/benchmark_extreme.csv`

---

## Configuration

### YAML Config Files

All training hyperparameters are stored in `MambaVision/mambavision/configs/`. The YAML is loaded via `-c configs/mambavision_tiny_1k.yaml` and any command-line flag overrides the YAML value.

Key hyperparameters from `mambavision_tiny_1k.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model` | `mamba_vision_T` | Model variant |
| `epochs` | 310 | Total training epochs |
| `batch_size` | 128 | Per-GPU batch size |
| `opt` | `lamb` | Optimizer (LAMB) |
| `lr` | 0.005 | Peak LR (for 8×128 = 1024 total batch) |
| `weight_decay` | 0.05 | AdamW/LAMB weight decay |
| `warmup_epochs` | 20 | Linear LR warmup |
| `cooldown_epochs` | 10 | LR cooldown |
| `sched` | `cosine` | LR schedule |
| `drop_path` | `null` | Set per-model in command line |
| `mixup` | 0.8 | Mixup alpha |
| `cutmix` | 1.0 | CutMix alpha |
| `smoothing` | 0.1 | Label smoothing |
| `model_ema` | `true` | EMA of model weights |
| `model_ema_decay` | 0.9998 | EMA decay rate |
| `amp` | `true` | Automatic mixed precision |
| `channels_last` | `true` | NHWC memory layout |
| `data_dir` | `/datasets/imagenet_lmdb` | **Change this to your dataset path** |
| `log_dir` | `./log_dir/` | TensorboardX log output |

### Mixer Type and Global Mode (code-level flags)

These are not in the YAML and must be passed programmatically or added to `train.py`:

```python
# In train.py, add to the argument parser:
parser.add_argument('--mixer-type', default='attn', choices=['attn', 'pom'])
parser.add_argument('--global-mixer', action='store_true')

# Then pass to model creation:
model = create_model(args.model, mixer_type=args.mixer_type,
                     global_mixer=args.global_mixer)
```

### PoM Hyperparameters

Set in `Block.__init__` when `mixer_type="pom"`:

```python
self.mixer = PoM(
    dim=dim,
    degree=2,       # polynomial degree k (paper default; higher = more expressive, more compute)
    expand=2,       # internal dimension D = expand × dim
    n_groups=1,     # token grouping (1 = no grouping)
    n_sel_heads=num_heads,  # selection heads (matches attention head count)
    bias=qkv_bias,
)
```

### Environment Variables

```bash
# Distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Suppress CUDA warnings
export CUDA_LAUNCH_BLOCKING=0

# For source compilation of mamba-ssm
export MAX_JOBS=4   # limit parallel compile jobs to avoid OOM during build
```

---

## Output Files

### Training Outputs

| File/Directory | Description |
|---------------|-------------|
| `<output>/<tag>/last.pth.tar` | Latest checkpoint (weights + optimizer state) |
| `<output>/<tag>/model_best.pth.tar` | Best validation accuracy checkpoint |
| `<output>/<tag>/checkpoint-<epoch>.pth.tar` | Per-epoch checkpoints (controlled by `--checkpoint-hist`) |
| `<output>/<tag>/args.yaml` | Saved training config for reproducibility |
| `./log_dir/` | TensorboardX event files |

View TensorBoard logs:
```bash
tensorboard --logdir MambaVision/mambavision/log_dir/
```

### Pretrained Weights (HuggingFace)

Pretrained weights are auto-downloaded to `/tmp/` when calling model factories with `pretrained=True`:

| Model | HuggingFace Repo | File |
|-------|-----------------|------|
| MambaVision-T | nvidia/MambaVision-T-1K | `mambavision_tiny_1k.pth.tar` |
| MambaVision-T2 | nvidia/MambaVision-T2-1K | `mambavision_tiny2_1k.pth.tar` |
| MambaVision-S | nvidia/MambaVision-S-1K | `mambavision_small_1k.pth.tar` |
| MambaVision-B | nvidia/MambaVision-B-1K | `mambavision_base_1k.pth.tar` |
| MambaVision-B (21K) | nvidia/MambaVision-B-21K | `mambavision_base_21k.pth.tar` |
| MambaVision-L | nvidia/MambaVision-L-1K | `mambavision_large_1k.pth.tar` |
| MambaVision-L2 | nvidia/MambaVision-L2-1K | `mambavision_large2_1k.pth.tar` |

### Benchmark Results

All benchmark outputs are CSV files in `results/`:

| File | Columns | Description |
|------|---------|-------------|
| `benchmark_results.csv` | `model,mixer,resolution,img_per_sec,peak_mem_MB,params` | Standard-res sweep |
| `benchmark_global.csv` | `model,variant,resolution,img_per_sec,peak_mem_MB,params` | Global-mode sweep |
| `benchmark_extreme.csv` | `variant,resolution,seq_len_N,img_per_sec,peak_mem_MB` | Extreme-res sweep |
| `benchmark_train_memory.csv` | `variant,resolution,seq_len_N,train_peak_mem_MB` | Training memory |

### Benchmark Figures

Generated figures are stored in `docs/figures/`:

| File | Description |
|------|-------------|
| `fig_standard_throughput.pdf` | Throughput vs resolution for T/S/B (null result) |
| `fig_extreme_throughput.pdf` | Throughput divergence + speedup ratio at extreme N |
| `fig_extreme_memory.pdf` | Memory parity at extreme resolution |
| `fig_train_memory.pdf` | Training memory parity (forward + backward) |
| `fig_params.pdf` | Parameter count comparison across model sizes |

---

## Troubleshooting

### `undefined symbol` when importing `mamba_ssm` or `causal_conv1d`

```
ImportError: .../causal_conv1d_cuda...so: undefined symbol: _ZN3c106detail14torchCheckFail...
```

**Cause:** C++ ABI mismatch between the pre-built wheel and your PyTorch binary.

**Fix:** Remove the pre-built installation and compile from source:
```bash
pip uninstall mamba-ssm causal-conv1d -y
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

---

### Compilation hangs or runs out of memory during `pip install mamba-ssm`

**Fix:** Limit parallel compilation jobs:
```bash
MAX_JOBS=2 pip install mamba-ssm --no-build-isolation
```

---

### `ModuleNotFoundError: No module named 'pom'`

**Fix:** The PoM package is a local install. Either install it or set `PYTHONPATH`:
```bash
# Option A: install
cd PoM && pip install -e . && cd ..

# Option B: set path for a single command
PYTHONPATH=/path/to/NNDL/PoM python scripts/verify_pom.py
```

---

### CUDA out of memory during training

**Cause:** Batch size too large for available VRAM.

**Fixes:**
```bash
# Reduce per-GPU batch size
--batch-size 64   # instead of 128

# Enable gradient checkpointing
--grad-checkpointing

# Use fewer workers
--workers 4
```

For the extreme-resolution benchmark, use a smaller batch size (`--bs 1` or `--bs 4`).

---

### CUDA out of memory during extreme-resolution benchmarking

At resolutions ≥ 2048 px with global mode and batch size > 1, a 16 GB GPU may OOM for the attention variant. PoM typically survives longer (lower peak memory at equal settings). This is an expected result and is recorded as `OOM` in the CSV.

**Workaround:** Reduce batch size to 1 and/or run inference only (disable gradients with `torch.no_grad()`).

---

### `torch.cuda.is_available()` returns False

**Diagnose:**
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
nvcc --version
nvidia-smi
```

**Common fixes:**
- Ensure your PyTorch build matches your CUDA version (`+cu121`, `+cu124`, etc.)
- Reinstall PyTorch with the correct CUDA index: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Verify NVIDIA driver is installed: `nvidia-smi`

---

### `FileNotFoundError` for dataset path

The training configs hardcode `/datasets/imagenet_lmdb`. Override the path on the command line:
```bash
python train.py --train-split /your/imagenet/train --val-split /your/imagenet/val ...
```

Or edit the `data_dir` key in the relevant YAML config file.

---

### Pretrained weight download fails

Weights are fetched from HuggingFace. If the network is unavailable:
1. Download manually from the HuggingFace repos listed in [Output Files](#output-files).
2. Place the `.pth.tar` file at the path specified by `model_path`:

```python
model = mamba_vision_T(pretrained=True, model_path="/my/local/path/mambavision_tiny_1k.pth.tar")
```

---

### Validation accuracy is lower than expected

1. Verify `--crop-pct` matches the model's config (T: 1.0, T2: 0.98, S: 0.93, B: 1.0).
2. Use `--input-size 3 224 224` explicitly.
3. Ensure the validation split is organized correctly (`val/<classname>/image.JPEG`).
4. Check that you are using the correct checkpoint (EMA vs. non-EMA weights).

---

### PoM is slower than Attention at 224 px

This is **expected and correct behavior**. PoM's linear-time advantage only manifests when the sequence length N is large (N ≥ ~600). At 224 px with windowed mixing, N = 49, and PoM is marginally slower due to constant-factor overhead in the polynomial state computation. See the research report (`docs/MambaVision_PoM_Research_Review.pdf`) for a full explanation.

---

## FAQ

**Q: Do I need to fine-tune a pretrained checkpoint to use `mixer_type="pom"`?**

A: For benchmarking and verification purposes, `pretrained=False` is sufficient. For accuracy evaluation, the current codebase supports loading the pretrained *attention* weights into the non-mixer layers — the PoM layers will be randomly initialized. Up-training (fine-tuning from such a partially-loaded checkpoint) is the recommended next step to obtain a competitive accuracy number. This has not yet been completed in the current project.

---

**Q: Does PoM use the same number of parameters as Attention?**

A: Yes, within < 0.1%. For MambaVision-T: Attention uses 1,640,960 parameters in the last-stage mixer, PoM uses 1,651,856 — a 0.07% increase. Total model parameters: 31,794,248 (attn) vs 31,827,592 (pom).

---

**Q: What is `global_mixer` and when should I use it?**

A: `global_mixer=True` disables MambaVision's window partitioning in the transformer stages, so the full spatial feature map is passed as a single token sequence to the mixer. This is **required** to observe PoM's throughput advantage. At standard resolutions (224–768 px), windowing caps N at 49–196 tokens and masks any linear-vs-quadratic difference. Use `global_mixer=True` only for research experiments; it does not apply to standard classification as trained.

---

**Q: Why does memory not improve with PoM despite the attention being O(N²)?**

A: MambaVision's attention uses `F.scaled_dot_product_attention` (SDPA), which is PyTorch's Flash-Attention-compatible kernel. SDPA computes attention in memory-efficient tiles and never materializes the full N×N score matrix, keeping memory O(N) even though arithmetic is O(N²). Therefore PoM (intrinsically O(N) in both memory and compute) matches SDPA on memory but wins on wall-clock time at large N.

---

**Q: Can I use PoM with the 21K pretrained models?**

A: Yes. The `mixer_type` argument is accepted by all model factory functions, including `mamba_vision_B_21k`, `mamba_vision_L_21k`, etc. However, the pretrained 21K checkpoints contain attention weights in the mixer layers; loading them into a PoM model will load all non-mixer weights and randomly initialize the PoM layers. Fine-tuning is needed for meaningful accuracy.

---

**Q: What is the `global_mixer` flag doing to Mamba blocks?**

A: `global_mixer` operates at the `MambaVisionLayer` level and affects the entire stage, including the first-half Mamba blocks. In global mode, the full spatial sequence is passed through all blocks (Mamba and transformer alike). Since both `attn_global` and `pom_global` share this treatment identically, it cancels from their head-to-head comparison — the only difference between the two is the mixer module.

---

**Q: How do I add `mixer_type` support to the training script?**

A: `train.py` currently does not expose `mixer_type` as a CLI flag. Add it as follows:

```python
# In train.py, in the argument parser section:
parser.add_argument('--mixer-type', default='attn', choices=['attn', 'pom'],
                    help='Token mixer type for transformer stages')
parser.add_argument('--global-mixer', action='store_true',
                    help='Disable windowing and use full-sequence mixing')

# Then in the model creation call:
model = create_model(
    args.model,
    mixer_type=args.mixer_type,
    global_mixer=args.global_mixer,
    ...
)
```

---

## Future Improvements

The following research directions are prioritized based on the findings in this project (see `docs/MambaVision_PoM_Research_Review.pdf` Section 9 for full analysis):

1. **Accuracy validation via up-training (Priority: High)** — Load a pretrained attention checkpoint, swap in PoM for the mixer layers, and fine-tune for a short schedule (e.g., 10–15 epochs on ImageNet-100 as proxy, then full ImageNet). This is the single most important next step: an efficiency advantage is only publishable with a matched accuracy result.

2. **High-resolution dense prediction (Priority: High)** — Evaluate PoM as backbone in semantic segmentation (ADE20K) and object detection (COCO) at high input resolution, where N is genuinely large and the 3.87× throughput advantage translates to real training speedups.

3. **Representational benefit of global mixing (Priority: Medium)** — Windowed attention cannot cross window boundaries; global PoM can, at linear cost. This may yield accuracy improvements on tasks requiring long-range spatial context, independent of efficiency.

4. **PoM hyperparameter ablations (Priority: Medium)** — The defaults (degree k=2, expand=2) come from non-vision domains. Ablating degree and expansion factor within MambaVision may further improve the accuracy-efficiency trade-off.

5. **Per-block-type windowing (Priority: Low)** — Apply windowing to Mamba blocks and global mixing to PoM blocks within the same stage. This requires per-block shape juggling in `MambaVisionLayer.forward` but provides a cleaner ablation and potentially better accuracy.

6. **Memory profiling against eager attention (Priority: Low)** — For completeness, force the eager (non-SDPA) attention path and confirm that PoM also wins on memory in that setting, providing a complete characterization of the memory story.

---

## Citation

If you use this codebase or build on the findings of this research, please cite the two foundational works:

**MambaVision:**
```bibtex
@inproceedings{hatamizadeh2025mambavision,
  title={MambaVision: A Hybrid Mamba-Transformer Vision Backbone},
  author={Hatamizadeh, Ali and Kautz, Jan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  pages={25261--25270},
  year={2025}
}
```

**Polynomial Mixer (PoM):**
```bibtex
@inproceedings{picard2026pom,
  title={{PoM}: {E}fficient Image and Video Generation with the Polynomial Mixer},
  author={David Picard and Nicolas Dufour and Lucas Degeorge and Arijit Ghosh and
          Davide Allegro and Tom Ravaud and Yohann Perron and Corentin Sautier and
          Zeynep Sonat Baltaci and Fei Meng and Syrine Kalleli and Marta López-Rauhut
          and Thibaut Loiseau and Ségolène Albouy and Raphael Baena and
          Elliot Vincent and Loic Landrieu},
  booktitle={CVPR Findings},
  year={2026}
}
```

**Supporting works:**
```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@inproceedings{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Daniel Y and Ermon, Stefano and Rudra, Atri and Ré, Christopher},
  booktitle={NeurIPS},
  year={2022}
}
```

---

## License

**MambaVision code and pretrained models:**
- Source code: [NVIDIA Source Code License-NC](MambaVision/LICENSE)
- Pretrained weights: [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Copyright © 2025, NVIDIA Corporation. All rights reserved.

**PoM library:**
- MIT License (see `PoM/LICENSE`)
- Copyright © David Picard and contributors

**This repository (integration and research contributions):**
- M.Tech Research Project — not yet licensed for redistribution. Contact the author for permissions.

> The pre-trained models are for **non-commercial research use only** per the NVIDIA Source Code License-NC. For commercial use, contact NVIDIA.
