# NNDL — Neural Networks & Deep Learning

Research project integrating **MambaVision** (Mamba-based vision backbone)
with **PoM** (Polynomial Mixing) as a linear-time drop-in replacement for
windowed self-attention in transformer stages.

## Directory Structure

```
NNDL/
├── MambaVision/       # Core model architecture, training, validation, and
│                      # downstream task configs (detection, segmentation)
├── PoM/               # Polynomial Mixing library — source, benchmarks, tests
├── docs/              # Research paper (canonical PDF + LaTeX source)
│   ├── drafts/        # Alternate version of the paper — review & prune manually
│   └── figures/       # Benchmark plot PDFs (memory, throughput, params)
├── notebooks/         # Jupyter experiments and validation runs
├── results/           # Benchmark output CSVs (throughput, memory, params)
├── scripts/           # Standalone test and verification scripts
└── wheels/            # Pre-downloaded Python wheels for offline installation
```

## Setup

**Install PyTorch (offline wheel):**
```bash
pip install wheels/torch-2.4.1+cu121-cp312-cp312-linux_x86_64.whl
```

**Install PoM library:**
```bash
pip install -e PoM/
```

**Install MambaVision:**
```bash
pip install -e MambaVision/
```

## Key Files

| Path | Purpose |
|------|---------|
| `MambaVision/mambavision/models/mamba_vision.py` | Model definition — `MambaVision`, `MambaVisionLayer`, `Block` |
| `PoM/pom/pom.py` | Core PoM mixer implementation |
| `docs/MambaVision_PoM_Research_Review.pdf` | Project research paper |
| `notebooks/mambavision_pom_validation.ipynb` | Validation and benchmarking notebook |
| `results/` | CSV outputs from benchmark runs |
