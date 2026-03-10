
# Treesformer: L-System Tree Generation

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