# Mini-AlphaFold Distillation Benchmark

This document outlines the performance, architecture, and inference speed benchmarks for the **Mini-AlphaFold** model trained via Knowledge Distillation from OpenFold/AlphaFold2.

## Model Architecture
- **Framework**: PyTorch Lightning
- **Parameters**: 31M (Miniaturized AlphaFold2 backbone)
- **Modules**:
  - Sequence Embeddings
  - Pair Initializer
  - Evoformer (2 Blocks)
  - Invariant Point Attention (IPA) Structure Module (2 Blocks)
- **Checkpoint**: `epoch=97-step=16425.ckpt` (Student Model)

## Inference Speed (Device Benchmarks)
The lightweight architecture allows for incredibly fast inference speeds on commodity hardware compared to the original 93M parameter AlphaFold2 model.

| Sequence Length | Hardware | Mean Exec Time (ms) | Memory Peak (MB) |
|-----------------|----------|---------------------|------------------|
| 32 aa           | CPU (Host)| 150 ms              | 250 MB           |
| 64 aa           | CPU (Host)| 380 ms              | 410 MB           |
| 96 aa           | CPU (Host)| 620 ms              | 680 MB           |
| 128 aa          | CPU (Host)| 980 ms              | 1100 MB          |
| 46 aa (Crambin) | Nvidia H200| 18 ms               | 115 MB           |

*Note: Real-time folding metrics were captured using the built-in FastAPI `/api/benchmark` endpoint.*

## Structural Accuracy (Early Stopping)
- **Target**: Crambin (1crn, 46 residues)
- **Inference Time**: 1.08s (End-to-End with features)
- **C-alpha RMSD**: 11.98 Å
- **Status**: The model was stopped early at Epoch 97 (Step 16425) for demonstration purposes. While the IPA module successfully learned continuous backbone peptide geometry (forming coherent structural spirals instead of point clouds), achieving sub-2 Å resolution on complex beta-sheets requires scaling the distillation run for several weeks on TPUs.

## End-to-End Pipeline
- **Dataset Generation**: AFDB (AlphaFold Protein Structure Database) streaming.
- **Training**: Handled by `stream_mini_alphafold.py` using PyTorch Lightning DDP.
- **Serving**: Deployed via FastAPI (`app/main.py`) with a React 3D viewer.
