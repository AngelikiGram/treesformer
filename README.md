# Treesformer: Multimodal Grammar-Based 3D Tree Reconstruction from Sparse Geodata

This repository contains the complete implementation of **TreesFormer**, a deep learning framework for reconstructing hierarchical 3D tree structures (represented as L-systems) directly from sparse top-down geodata using only a single orthophoto and its corresponding Digital Surface Model (DSM).

All the material and qualitative results can be seen through in: [TreesFormer](https://drive.google.com/file/d/1zVpkl4hREym_-UoGFSODrlAYETtT8Qcl/view)
.

## Pretrained Model Weights

Pretrained model weights are available for download from Google Drive:

[Download pretrained model weights (Google Drive)](https://drive.google.com/file/d/1zVpkl4hREym_-UoGFSODrlAYETtT8Qcl/view)

After downloading, place the extracted folder `treesformer_weights` inside the `models/` directory:
`models/treesformer_weights/`

## Project Structure

- `scripts/` — Main scripts (train.py, inference.py, ablation.py)
- `auxiliary/` — Core modules (dataset, tokenizer, renderer, losses, utils)
- `data/` — Data folders (DSM, ORTHOPHOTOS, etc.)
- `results/` — Output and evaluation results

## Training

```bash
python train.py --lstrings_path <LSTRINGS_DIR> --window 1024 --epochs 2000 --batch 8 --save_ckpt <ckpt.pth>
```

## Inference/Evaluation

```bash
python inference/inference.py --ckpt <ckpt.pth> --base <DATA_ROOT> --lstrings_path <LSTRINGS_DIR> --window 1024 --num_trees 50 --save_results
```

## Ablation Study

```bash
python inference/ablation.py --ckpt <ckpt.pth> --base <DATA_ROOT> --lstrings <LSTRINGS_DIR> --n 50 --window 1024
```

- Replace `<ckpt.pth>`, `<DATA_ROOT>`, and `<LSTRINGS_DIR>` with your paths.
- All scripts expect the new folder/module structure (e.g., `auxiliary.lsys_tokenizer`).

## Requirements
- Python 3.8+
- PyTorch, torchvision, numpy, scipy, visdom, PIL

---
For more details, see comments in each script.