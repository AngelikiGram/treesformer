#!/usr/bin/env python3
"""
ablation.py — Treesformer Ablation Study
=========================================
Evaluates a trained checkpoint under multiple conditions and reports the
three most diagnostic metrics:

  1. Chamfer Distance  (geometry quality,   lower  = better)
  2. Coverage@r        (geometry recall,    higher = better)
  3. Validity          (bracket balance,    higher = better)

Ablation groups
───────────────
A. Modality     : full / DSM-only / Ortho-only / LM-baseline (no visual)
                  Uses one checkpoint, masks modalities at inference.

B. Temperature  : 0.5 / 1.0 / 1.5
                  Same checkpoint, tests the quality-diversity trade-off.

C. Checkpoint   : compare loss / architecture variants
                  Pass any number of labelled checkpoint paths with
                  --extra_ckpts label1:path1 label2:path2 ...

Suggested loss-ablation checkpoints to train separately
────────────────────────────────────────────────────────
   no_state_loss  : set state_loss weight 5.0 → 0.0 in train.py
   no_soft_angle  : set soft_angle_loss weight 2.0 → 0.0 in train.py
   no_vis_dropout : FullMultimodalEncoder(visual_dropout=0.0)
   high_species   : set species_loss weight 0.02 → 0.2 in train.py
   no_anchor_det  : remove global_feat.detach() in FullMultimodalEncoder

Usage
─────
  # Quick modality + temperature ablation
  python ablation.py --ckpt p3_effsymbolic_L.pth

  # With extra loss-ablation checkpoints
  python ablation.py --ckpt p3_effsymbolic_L.pth \\
      --extra_ckpts no_state:ckpts/no_state.pth high_sp:ckpts/high_sp.pth

  # Full options
  python ablation.py --ckpt p3_effsymbolic_L.pth \\
      --base /home/grammatikakis1/TREES_DATASET_SIDE \\
      --lstrings SYMBOLIC_LSTRINGS_d3 \\
      --n 100 --window 1024 --dim 512 --heads 16 --layers 8 \\
      --f_bins 10 --theta_bins 6 --phi_bins 6
"""

import os
import sys
import re
import math
import random
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from train import LSystemModel
from auxiliary.lsys_dataset import LSystemDataset
from auxiliary.lsys_tokenizer import LSystemTokenizerV2, TokenType
from auxiliary.lsys_renderer import render_lsystem
from visualize import LSystemVisdom

# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

COVERAGE_RADIUS = 0.10   # 10 % of GT bounding-box diagonal


def bracket_validity(lstring: str):
    """Returns (is_balanced: bool, max_depth: int)."""
    depth = max_depth = 0
    for ch in lstring:
        if ch == '[':
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ']':
            depth -= 1
            if depth < 0:
                return False, 0
    return depth == 0, max_depth


def count_segments(lstring: str) -> int:
    """Count branch-move segments (F tokens) in an L-string."""
    return len(re.findall(r'[BSbs]\d+_\d+[F_]?\d+', lstring))


def subsample(pts: np.ndarray, k: int = 1024) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts
    if pts.shape[0] > k:
        return pts[np.random.choice(pts.shape[0], k, replace=False)]
    return pts


def point_cloud_metrics(pred_pts: np.ndarray, gt_pts: np.ndarray,
                        r: float = COVERAGE_RADIUS) -> dict:
    """
    Compute three point-cloud metrics after normalising into GT bounding box.

    Returns
    -------
    chamfer   : bidirectional average nearest-neighbour distance  (lower = better)
    coverage  : recall@r — fraction of GT points within r of pred  (higher = better)
    precision : precision@r — fraction of pred points within r of GT (higher = better)
    """
    if pred_pts.shape[0] == 0:
        return dict(chamfer=float('inf'), coverage=0.0, precision=0.0)

    pred_pts = subsample(pred_pts.astype(np.float32), 1024)
    gt_pts   = subsample(gt_pts.astype(np.float32),   1024)

    # Normalise by GT bounding box (centred at GT midpoint)
    gt_lo, gt_hi = gt_pts.min(0), gt_pts.max(0)
    gt_ctr = (gt_lo + gt_hi) / 2.0
    gt_scl = (gt_hi - gt_lo).max() + 1e-6

    pred_n = (pred_pts - gt_ctr) / gt_scl
    gt_n   = (gt_pts   - gt_ctr) / gt_scl

    # Pairwise squared L2: (N, M)
    diff   = pred_n[:, None, :] - gt_n[None, :, :]   # (N, M, 3)
    dists  = np.sqrt((diff ** 2).sum(-1) + 1e-12)     # (N, M)

    s2t = dists.min(1).mean()                          # pred → gt
    t2s = dists.min(0).mean()                          # gt  → pred

    return dict(
        chamfer   = float(s2t + t2s),
        coverage  = float((dists.min(0) < r).mean()),  # gt→pred recall
        precision = float((dists.min(1) < r).mean()),  # pred→gt precision
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, tokenizer, dataset, dim: int,
               layers: int, heads: int, device: str,
               visual_bottleneck: int = 1024) -> LSystemModel:
    """Load a checkpoint exactly as inference.py does, with full auto-detection."""
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)

    # ── Auto-detect architecture from checkpoint (mirrors inference.py) ────────
    f_bins     = tokenizer.f_bins
    theta_bins = tokenizer.theta_bins
    phi_bins   = tokenizer.phi_bins
    num_species = dataset.num_species

    if "val_emb_length.weight" in state:
        f_bins_ckpt = state["val_emb_length.weight"].shape[0]
        if f_bins_ckpt != f_bins:
            print(f"    [INFO] Auto-adjusting f_bins: {f_bins} -> {f_bins_ckpt}")
            f_bins = f_bins_ckpt

    if "val_emb_theta.weight" in state:
        theta_bins_ckpt = state["val_emb_theta.weight"].shape[0]
        if theta_bins_ckpt != theta_bins:
            print(f"    [INFO] Auto-adjusting theta_bins: {theta_bins} -> {theta_bins_ckpt}")
            theta_bins = theta_bins_ckpt

    if "val_emb_phi.weight" in state:
        phi_bins_ckpt = state["val_emb_phi.weight"].shape[0]
        if phi_bins_ckpt != phi_bins:
            print(f"    [INFO] Auto-adjusting phi_bins: {phi_bins} -> {phi_bins_ckpt}")
            phi_bins = phi_bins_ckpt

    if "species_head.weight" in state:
        num_species_ckpt = state["species_head.weight"].shape[0]
        if num_species_ckpt != num_species:
            print(f"    [INFO] Auto-adjusting num_species: {num_species} -> {num_species_ckpt}")
            num_species = num_species_ckpt

    if "type_emb.weight" in state:
        dim_ckpt = state["type_emb.weight"].shape[1]
        if dim_ckpt != dim:
            print(f"    [INFO] Auto-adjusting dim: {dim} -> {dim_ckpt}")
            dim = dim_ckpt

    block_keys = [k for k in state.keys() if "blocks." in k]
    if block_keys:
        layers_ckpt = max(int(k.split(".")[1]) for k in block_keys) + 1
        if layers_ckpt != layers:
            print(f"    [INFO] Auto-adjusting layers: {layers} -> {layers_ckpt}")
            layers = layers_ckpt

    # CRITICAL: auto-detect heads from RoPE inv_freq dimension
    if "rope.inv_freq" in state:
        inv_len = state["rope.inv_freq"].shape[0]
        heads_ckpt = dim // (inv_len * 2)
        if heads_ckpt != heads:
            print(f"    [INFO] Auto-adjusting heads: {heads} -> {heads_ckpt}")
            heads = heads_ckpt

    # Re-init tokenizer if any bins changed
    local_tokenizer = LSystemTokenizerV2(
        f_bins=f_bins, theta_bins=theta_bins, phi_bins=phi_bins
    )

    model = LSystemModel(
        f_bins            = local_tokenizer.f_bins,
        theta_bins        = local_tokenizer.theta_bins,
        phi_bins          = local_tokenizer.phi_bins,
        num_species       = num_species,
        dim               = dim,
        max_window        = dataset.window,
        cross_attn_window = dataset.window,
        heads             = heads,
        layers            = layers,
        visual_bottleneck = visual_bottleneck,
    ).to(device)

    # Robust load: filter mismatched shapes (then strict=False for the rest)
    curr_sd = model.state_dict()
    filtered_sd = {k: v for k, v in state.items()
                   if k in curr_sd and v.shape == curr_sd[k].shape}

    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)

    n_skipped = len(state) - len(filtered_sd)
    if n_skipped:
        mismatched = [k for k in state if k in curr_sd and state[k].shape != curr_sd[k].shape]
        print(f"    [WARN] Skipped {n_skipped} mismatched keys "
              f"(e.g. {mismatched[0] if mismatched else 'none'}). "
              f"Loaded {len(filtered_sd)}/{len(state)}. Missing: {len(missing)}")

    model.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"    Loaded epoch {epoch} from {ckpt_path}  "
          f"(dim={dim} heads={heads} layers={layers} "
          f"f={f_bins} θ={theta_bins} φ={phi_bins})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model,
    dataset,
    tokenizer,
    sample_indices: list,
    modality: str,
    temperature: float,
    max_gen: int,
    device: str,
    seed: int = 42,
    verbose: bool = False,
    export_dir: str = None,
    viz: LSystemVisdom = None,
) -> dict:
    """
    Run generation on each sample index and return aggregated metrics.

    Parameters
    ----------
    modality   : "both" | "dsm" | "ortho" | "none"
    export_dir : If provided, saves each generated L-string to {export_dir}/{tid}.txt
    viz        : If provided, visualizes results in Visdom
    """
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    f_bins     = tokenizer.f_bins
    theta_bins = tokenizer.theta_bins
    phi_bins   = tokenizer.phi_bins

    chamfers, coverages, precisions = [], [], []
    validities, seg_ratios, depths  = [], [], []

    t_start = time.time()

    from tqdm import tqdm
    for i, idx in enumerate(tqdm(sample_indices, desc=f"Evaluating {modality} T={temperature}", leave=False)):
        sample = dataset[idx]
        tid    = sample["tid"]

        dsm   = sample["dsm"].unsqueeze(0).to(device).float()
        ortho = sample["ortho"].unsqueeze(0).to(device).float()

        # ── Modality masking ───────────────────────────────────────────────
        if modality == "dsm":
            ortho = torch.zeros_like(ortho)
        elif modality == "ortho":
            dsm = torch.zeros_like(dsm)
        elif modality == "none":
            dsm   = torch.zeros_like(dsm)
            ortho = torch.zeros_like(ortho)

        # ── Generate ───────────────────────────────────────────────────────
        try:
            with torch.no_grad():
                pred_str = model.generate(
                    dsm, ortho, tokenizer,
                    max_len = max_gen,
                    temperature = temperature,
                    temperature_structural = 0.7,
                )
        except Exception as exc:
            if verbose:
                print(f"    [WARN] generate() failed for {tid}: {exc}")
            pred_str = ""

        # ── Optional Export ────────────────────────────────────────────────
        if export_dir:
            out_fn = os.path.join(export_dir, f"{tid}.txt")
            with open(out_fn, "w", encoding="utf-8") as f:
                f.write(pred_str)

        # ── Validity & depth ───────────────────────────────────────────────
        valid, max_depth = bracket_validity(pred_str)
        validities.append(float(valid))
        depths.append(float(max_depth))

        # ── GT L-string ────────────────────────────────────────────────────
        gt_path = os.path.join(dataset.lstring_dir, f"{tid}.txt")
        try:
            with open(gt_path, encoding="utf-8") as fh:
                gt_str = fh.read().strip()
        except OSError:
            # Fallback: decode from stored token arrays
            D = dataset.data.get(tid, {})
            gt_types = D.get("types", []).tolist()
            gt_vals  = D.get("vals",  [])
            gt_str   = tokenizer.decode(gt_types,
                                        [[int(v[0]), int(v[1]), int(v[2])]
                                         for v in gt_vals])

        # ── Segment count ratio ────────────────────────────────────────────
        pred_segs = count_segments(pred_str)
        gt_segs   = count_segments(gt_str)
        seg_ratios.append(pred_segs / (gt_segs + 1e-9))

        # ── Point-cloud metrics ────────────────────────────────────────────
        scale_factor = 2.0 / (sample["dsm_scale"].item() + 1e-9)

        pred_pts = render_lsystem(
            pred_str, step_scale=scale_factor,
            num_bins_theta=theta_bins, num_bins_phi=phi_bins, num_bins_f=f_bins,
        )
        gt_pts = render_lsystem(
            gt_str, step_scale=scale_factor,
            num_bins_theta=theta_bins, num_bins_phi=phi_bins, num_bins_f=f_bins,
        )

        if viz is not None:
            viz.visualize_inference(
                tid,
                ortho[0].cpu(),
                dsm[0].cpu(),
                gt_pts,
                pred_pts
            )

        m = point_cloud_metrics(pred_pts, gt_pts)
        chamfers.append(m["chamfer"])
        coverages.append(m["coverage"])
        precisions.append(m["precision"])

        if verbose and (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(sample_indices)}  "
                  f"cd={np.nanmean([c for c in chamfers if math.isfinite(c)]):.4f}  "
                  f"cov={np.nanmean(coverages)*100:.1f}%  "
                  f"val={np.nanmean(validities)*100:.1f}%")

    elapsed = time.time() - t_start

    def safe_mean(lst):
        finite = [v for v in lst if math.isfinite(v)]
        return float(np.mean(finite)) if finite else float('nan')

    return dict(
        chamfer   = safe_mean(chamfers),
        coverage  = safe_mean(coverages),
        precision = safe_mean(precisions),
        validity  = safe_mean(validities),
        seg_ratio = safe_mean(seg_ratios),
        max_depth = safe_mean(depths),
        n_valid   = int(sum(v > 0.5 for v in validities)),
        n_total   = len(sample_indices),
        elapsed   = elapsed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_f(v, w=8):
    return f"{v:.4f}".rjust(w) if math.isfinite(v) else "   —   ".rjust(w)

def _fmt_p(v, w=8):
    return f"{v*100:.1f}%".rjust(w) if math.isfinite(v) else "   —   ".rjust(w)

def _fmt_r(v, w=8):
    return f"{v:.2f}".rjust(w) if math.isfinite(v) else "   —   ".rjust(w)


def print_ablation_table(rows: list, title: str = ""):
    """
    rows : list of (label: str, metrics: dict)
    Primary columns: Chamfer↓, Coverage↑, Validity↑
    Secondary:       Precision, Seg.Ratio, MaxDepth
    """
    W_NAME = max(28, max(len(r[0]) for r in rows))
    HDR = (f"{'Condition':<{W_NAME}}  "
           f"{'Chamfer↓':>9}  {'Coverage↑':>9}  {'Validity↑':>9}  "
           f"{'Precision':>9}  {'Seg.Ratio':>9}  {'MaxDepth':>8}  "
           f"{'N':>6}")
    SEP = "─" * len(HDR)

    bar = "═" * len(HDR)
    print(f"\n{bar}")
    if title:
        print(f"  {title}")
        print(bar)
    print(HDR)
    print(SEP)

    # Find best (finite) value per column to highlight
    best_cd  = min((m["chamfer"]   for _, m in rows if math.isfinite(m["chamfer"])),  default=None)
    best_cov = max((m["coverage"]  for _, m in rows if math.isfinite(m["coverage"])), default=None)
    best_val = max((m["validity"]  for _, m in rows if math.isfinite(m["validity"])), default=None)

    for label, m in rows:
        star_cd  = "*" if best_cd  is not None and abs(m["chamfer"]  - best_cd)  < 1e-6 else " "
        star_cov = "*" if best_cov is not None and abs(m["coverage"] - best_cov) < 1e-6 else " "
        star_val = "*" if best_val is not None and abs(m["validity"] - best_val) < 1e-6 else " "

        row = (f"{label:<{W_NAME}}  "
               f"{star_cd}{_fmt_f(m['chamfer'],8)}  "
               f"{star_cov}{_fmt_p(m['coverage'],8)}  "
               f"{star_val}{_fmt_p(m['validity'],8)}  "
               f" {_fmt_p(m['precision'],8)}  "
               f" {_fmt_r(m['seg_ratio'],8)}  "
               f" {_fmt_r(m['max_depth'],7)}  "
               f"{m['n_valid']:>3}/{m['n_total']:<3}")
        print(row)

    print(SEP)
    print("  * = best in column   |   Coverage/Validity: higher is better   |   Chamfer: lower is better")


def print_training_commands(args):
    """Print train.py commands for each suggested loss ablation."""
    base_cmd = (
        f"python train.py"
        f" --lstrings_path {args.lstrings}"
        f" --base {args.base}"
        f" --window {args.window}"
        f" --dim {args.dim}"
        f" --heads {args.heads}"
        f" --layers {args.layers}"
        f" --f_bins {args.f_bins}"
        f" --theta_bins {args.theta_bins}"
        f" --phi_bins {args.phi_bins}"
        f" --epochs 500 --batch 16"
    )
    ablations = {
        "no_state_loss  ": "  # Edit train.py: loss = loss + 0.0 * loss_state  (was 5.0)",
        "no_soft_angle  ": "  # Edit train.py: loss = loss + 0.0 * loss_soft_angle  (was 2.0)",
        "no_vis_dropout ": "  # Edit FullMultimodalEncoder.__init__: visual_dropout=0.0",
        "high_species   ": "  # Edit train.py: 0.02 * loss_sp → 0.2 * loss_sp",
        "no_anchor_detach": " # Edit FullMultimodalEncoder.forward: global_feat.detach() → global_feat",
    }
    print("\n" + "═" * 72)
    print("  TRAINING COMMANDS FOR LOSS ABLATION")
    print("  (run these, then pass the checkpoints via --extra_ckpts)")
    print("═" * 72)
    for label, comment in ablations.items():
        ckpt = f"ckpts/{label.strip()}.pth"
        print(f"\n  # {label.strip()}")
        print(f"  {base_cmd} --id_cache {label.strip()} {comment}")
        print(f"  # Then evaluate with:  --extra_ckpts {label.strip()}:{ckpt}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Treesformer Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Checkpoint
    parser.add_argument("--ckpt", required=True,
                        help="Path to main trained checkpoint (.pth)")
    parser.add_argument("--extra_ckpts", nargs="*", default=[],
                        help="Extra checkpoints for loss/arch ablation. "
                             "Format: label:path  (e.g. no_state:ckpts/no_state.pth)")
    parser.add_argument("--modality_ckpts", nargs="*", default=[],
                        help="Checkpoints trained with a specific modality, evaluated with "
                             "that same modality.  "
                             "Format: label:path:modality  where modality ∈ {both,dsm,ortho}. "
                             "Example: 'Full:p3_effL_small_norm.pth:both' "
                             "'DSM:p3_effL_small_norm_dsm.pth:dsm' "
                             "'Ortho:p3_effL_small_norm_ortho.pth:ortho'")
    # Dataset
    parser.add_argument("--base",     default="/home/grammatikakis1/TREES_DATASET_SIDE")
    parser.add_argument("--lstrings", default="SYMBOLIC_LSTRINGS_d3",
                        help="Sub-folder inside --base containing .txt L-strings")
    parser.add_argument("--n",        type=int, default=100,
                        help="Number of evaluation samples")
    parser.add_argument("--window",   type=int, default=1024)
    # Model architecture (must match checkpoint)
    parser.add_argument("--dim",        type=int, default=512)
    parser.add_argument("--heads",      type=int, default=16)
    parser.add_argument("--layers",     type=int, default=8)
    parser.add_argument("--f_bins",     type=int, default=10)
    parser.add_argument("--theta_bins", type=int, default=6)
    parser.add_argument("--phi_bins",   type=int, default=6)
    # Evaluation settings
    parser.add_argument("--dsm_dir",    default="DSM_ALIGNED",
                        help="Sub-folder inside --base for DSM .mat files (must match training)")
    parser.add_argument("--ortho_dir",  default="ORTHOPHOTOS",
                        help="Sub-folder inside --base for orthophoto images (must match training)")
    parser.add_argument("--visual_bottleneck", type=int, default=1024,
                        help="Visual encoder bottleneck dim (must match checkpoint)")
    parser.add_argument("--max_gen",    type=int, default=1500,
                        help="Max tokens to generate per sample")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose",    action="store_true")
    # Which groups to run
    parser.add_argument("--skip_modality",         action="store_true")
    parser.add_argument("--skip_temperature",      action="store_true")
    parser.add_argument("--skip_checkpoint",       action="store_true")
    parser.add_argument("--skip_modality_ckpts",   action="store_true")
    parser.add_argument("--tid", nargs="*", default=[],
                        help="Specific Tree ID(s) to evaluate. If provided, skips random selection.")
    parser.add_argument("--print_cmds",            action="store_true",
                        help="Print training commands for loss ablations and exit")
    parser.add_argument("--export_dir", type=str, default=None,
                        help="Directory to export predicted .txt L-strings for qualitative study")
    parser.add_argument("--temp_modality", type=str, default="both",
                        help="Modality to use for the temperature ablation (Section B)")
    parser.add_argument("--temp_sweep_all", action="store_true",
                        help="If set, runs 0.5, 1.0, 1.5 temperatures for ALL checkpoints (Main + Modality Checkpoints)")
    # Visdom
    parser.add_argument("--visdom", action="store_true")
    parser.add_argument("--visdom_env", default="ablation_study")

    args = parser.parse_args()

    if args.print_cmds:
        print_training_commands(args)
        return

    # ── Reproducibility ────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    print(f"\nDevice : {device}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = LSystemTokenizerV2(
        f_bins=args.f_bins, theta_bins=args.theta_bins, phi_bins=args.phi_bins
    )

    # ── Dataset ────────────────────────────────────────────────────────────────
    lstring_dir = os.path.join(args.base, args.lstrings)
    all_ids = sorted([f[:-4] for f in os.listdir(lstring_dir) if f.endswith(".txt")])

    # Selection logic
    if args.tid:
        print(f"[INFO] Using explicitly provided Tree IDs: {args.tid}")
        eval_ids = [tid for tid in args.tid if tid in all_ids]
        if not eval_ids:
            print(f"[ERROR] None of the provided TIDs exist in the dataset folder!")
            return
    else:
        # Use the same 90/10 split as training (seed=42), take the held-out 10%
        random.seed(42)
        shuffled = list(all_ids)
        random.shuffle(shuffled)
        split = max(1, int(len(shuffled) * 0.1))
        test_ids = shuffled[:split]   # ~10% held-out

        # If too few test IDs, fall back to all IDs
        if len(test_ids) < args.n:
            print(f"[INFO] Only {len(test_ids)} held-out IDs; using all {len(all_ids)} IDs.")
            test_ids = shuffled

        # Draw evaluation subset
        eval_ids = test_ids[:args.n]

    print(f"\nDataset : {args.base}")
    print(f"L-strings: {lstring_dir}")
    if not args.tid:
        print(f"All IDs : {len(all_ids)}   |   Held-out: {len(test_ids)}   |   Eval: {len(eval_ids)}")
    else:
        print(f"All IDs : {len(all_ids)}   |   Targeted Eval: {len(eval_ids)}")
    print(f"\nLoading dataset ...")

    dataset = LSystemDataset(
        base_path    = args.base,
        lstring_dir  = args.lstrings,
        tokenizer    = tokenizer,
        ids          = eval_ids,
        window       = args.window,
        overlap      = 0,
        preload      = True,
        dsm_dirname  = args.dsm_dir,
        ortho_dirname= args.ortho_dir,
    )

    # Map: tid → first sample index (we want one window per tree)
    tid_to_idx = {}
    for idx, (tid, start) in enumerate(dataset.samples):
        if tid not in tid_to_idx:
            tid_to_idx[tid] = idx
    eval_indices = [tid_to_idx[tid] for tid in eval_ids if tid in tid_to_idx]

    print(f"Eval samples : {len(eval_indices)}")

    # ── Visdom ────────────────────────────────────────────────────────────────
    viz = None
    if args.visdom:
        viz = LSystemVisdom(env=args.visdom_env, port=8099)
        print(f"Visdom enabled. Env: {args.visdom_env}")

    # ── Load main model ────────────────────────────────────────────────────────
    print(f"\nLoading main checkpoint: {args.ckpt}")
    model = load_model(args.ckpt, tokenizer, dataset,
                       args.dim, args.layers, args.heads, device,
                       visual_bottleneck=args.visual_bottleneck)

    all_results = {}   # label → metrics

    # ══════════════════════════════════════════════════════════════════════════
    # A. MODALITY ABLATION
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_modality:
        print("\n" + "─" * 60)
        print("A. MODALITY ABLATION")
        print("─" * 60)

        modality_conditions = [
            ("Full  (DSM + Ortho)",     "both"),
            ("DSM  only",               "dsm"),
            ("Ortho only",              "ortho"),
            ("LM Baseline  (no visual)","none"),
        ]
        rows = []
        for label, mod in modality_conditions:
            print(f"\n  [{label}] generating ...")
            edir = os.path.join(args.export_dir, f"mod_{mod}") if args.export_dir else None
            m = evaluate(
                model, dataset, tokenizer, eval_indices,
                modality=mod, temperature=1.0,
                max_gen=args.max_gen, device=device,
                seed=args.seed, verbose=args.verbose,
                export_dir=edir, viz=viz,
            )
            rows.append((label, m))
            all_results[f"mod:{label.strip()}"] = m
            print(f"    Chamfer={m['chamfer']:.4f}  "
                  f"Coverage={m['coverage']*100:.1f}%  "
                  f"Validity={m['validity']*100:.1f}%  "
                  f"({m['elapsed']:.0f}s)")

        print_ablation_table(rows, title="A. MODALITY ABLATION  (same checkpoint)")

    # ══════════════════════════════════════════════════════════════════════════
    # B. TEMPERATURE ABLATION
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_temperature:
        print("\n" + "─" * 60)
        print("B. TEMPERATURE ABLATION  (modality=full)")
        print("─" * 60)

        temp_conditions = [
            ("Temp 0.5  (greedy-ish)", 0.5),
            ("Temp 1.0  (default)",    1.0),
            ("Temp 1.5  (diverse)",    1.5),
        ]
        rows = []
        for label, temp in temp_conditions:
            print(f"\n  [{label} (modality={args.temp_modality})] generating ...")
            edir = os.path.join(args.export_dir, f"temp_{temp}") if args.export_dir else None
            m = evaluate(
                model, dataset, tokenizer, eval_indices,
                modality=args.temp_modality, temperature=temp,
                max_gen=args.max_gen, device=device,
                seed=args.seed, verbose=args.verbose,
                export_dir=edir, viz=viz,
            )
            rows.append((label, m))
            all_results[f"temp:{label.strip()}"] = m
            print(f"    Chamfer={m['chamfer']:.4f}  "
                  f"Coverage={m['coverage']*100:.1f}%  "
                  f"Validity={m['validity']*100:.1f}%  "
                  f"({m['elapsed']:.0f}s)")

        print_ablation_table(rows, title="B. TEMPERATURE ABLATION  (same checkpoint)")

    # ══════════════════════════════════════════════════════════════════════════
    # C. CHECKPOINT ABLATION  (loss / architecture comparisons)
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_checkpoint and args.extra_ckpts:
        extra_map = {}
        for item in args.extra_ckpts:
            parts = item.split(":", 1)
            if len(parts) == 2:
                extra_map[parts[0]] = parts[1]
            else:
                print(f"[WARN] Skipping malformed --extra_ckpts entry: {item!r}  "
                      f"(expected label:path)")

        if extra_map:
            print("\n" + "─" * 60)
            print("C. CHECKPOINT ABLATION  (loss / architecture variants)")
            print("─" * 60)

            rows = []

            # Main checkpoint as baseline
            print("\n  [Main (full loss)] generating ...")
            m_main = evaluate(
                model, dataset, tokenizer, eval_indices,
                modality="both", temperature=1.0,
                max_gen=args.max_gen, device=device,
                seed=args.seed, verbose=args.verbose,
            )
            rows.append(("Main  (full loss)", m_main))
            all_results["ckpt:main"] = m_main
            print(f"    Chamfer={m_main['chamfer']:.4f}  "
                  f"Coverage={m_main['coverage']*100:.1f}%  "
                  f"Validity={m_main['validity']*100:.1f}%  "
                  f"({m_main['elapsed']:.0f}s)")

            for label, ckpt_path in extra_map.items():
                print(f"\n  [{label}] loading {ckpt_path} ...")
                try:
                    m_extra = load_model(ckpt_path, tokenizer, dataset,
                                         args.dim, args.layers, args.heads, device,
                                         visual_bottleneck=args.visual_bottleneck)
                    print(f"  [{label}] generating ...")
                    edir = os.path.join(args.export_dir, f"ckpt_{label}") if args.export_dir else None
                    m = evaluate(
                        m_extra, dataset, tokenizer, eval_indices,
                        modality="both", temperature=1.0,
                        max_gen=args.max_gen, device=device,
                        seed=args.seed, verbose=args.verbose,
                        export_dir=edir, viz=viz,
                    )
                    rows.append((label, m))
                    all_results[f"ckpt:{label}"] = m
                    print(f"    Chamfer={m['chamfer']:.4f}  "
                          f"Coverage={m['coverage']*100:.1f}%  "
                          f"Validity={m['validity']*100:.1f}%  "
                          f"({m['elapsed']:.0f}s)")
                    del m_extra
                    torch.cuda.empty_cache()
                except Exception as exc:
                    print(f"  [ERROR] {label}: {exc}")

            print_ablation_table(rows, title="C. CHECKPOINT ABLATION  (loss / architecture)")

    elif not args.skip_checkpoint and not args.extra_ckpts:
        print("\n[INFO] No --extra_ckpts provided → skipping checkpoint ablation.")
        print("       Run with --print_cmds to see how to train loss-ablation checkpoints.")

    # ══════════════════════════════════════════════════════════════════════════
    # D. MODALITY-CHECKPOINT ABLATION
    #    Each checkpoint is evaluated with the modality it was trained on.
    #    Format: --modality_ckpts "label:path:modality" ...
    #    where modality ∈ {both, dsm, ortho}
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_modality_ckpts and args.modality_ckpts:
        mod_map = []   # list of (label, path, modality)
        for item in args.modality_ckpts:
            parts = item.split(":", 2)
            if len(parts) == 3:
                label_m, path_m, mod_m = parts
                if mod_m not in ("both", "dsm", "ortho"):
                    print(f"[WARN] Unknown modality {mod_m!r} for {label_m} — "
                          f"must be one of: both / dsm / ortho.  Skipping.")
                    continue
                mod_map.append((label_m, path_m, mod_m))
            else:
                print(f"[WARN] Skipping malformed --modality_ckpts entry: {item!r}  "
                      f"(expected label:path:modality, e.g. 'Full:p3_eff.pth:both')")

        if mod_map:
            print("\n" + "─" * 60)
            print("D. MODALITY-CHECKPOINT ABLATION")
            print("   Each checkpoint evaluated with the modality it was trained on.")
            print("─" * 60)

            rows = []
            for label_m, path_m, mod_m in mod_map:
                print(f"\n  [{label_m}  (modality={mod_m})]  loading {path_m} ...")
                try:
                    m_mod = load_model(path_m, tokenizer, dataset,
                                       args.dim, args.layers, args.heads, device,
                                       visual_bottleneck=args.visual_bottleneck)
                    
                    temps = [0.5, 1.0, 1.5] if args.temp_sweep_all else [1.0]
                    
                    for t in temps:
                        t_lbl = f"_temp_{t}" if len(temps) > 1 else ""
                        print(f"  [{label_m}{t_lbl}] generating ...")
                        
                        edir = os.path.join(args.export_dir, f"modckpt_{label_m}{t_lbl}") if args.export_dir else None
                        m = evaluate(
                            m_mod, dataset, tokenizer, eval_indices,
                            modality=mod_m, temperature=t,
                            max_gen=args.max_gen, device=device,
                            seed=args.seed, verbose=args.verbose,
                            export_dir=edir, viz=viz,
                        )
                        display = f"{label_m}{t_lbl}  [{mod_m}]"
                        rows.append((display, m))
                        all_results[f"modckpt:{label_m}{t_lbl}"] = m
                        print(f"    Chamfer={m['chamfer']:.4f}  "
                              f"Coverage={m['coverage']*100:.1f}%  "
                              f"Validity={m['validity']*100:.1f}%  "
                              f"({m['elapsed']:.0f}s)")
                    del m_mod
                    torch.cuda.empty_cache()
                except Exception as exc:
                    print(f"  [ERROR] {label_m}: {exc}")

            if rows:
                print_ablation_table(
                    rows,
                    title="D. MODALITY-CHECKPOINT ABLATION  "
                          "(each model evaluated with its trained modality)",
                )

    elif not args.skip_modality_ckpts and not args.modality_ckpts:
        print("\n[INFO] No --modality_ckpts provided → skipping modality-checkpoint ablation.")
        print("       Example:  --modality_ckpts "
              "'Full:p3_effL_small_norm.pth:both' "
              "'DSM:p3_effL_small_norm_dsm.pth:dsm' "
              "'Ortho:p3_effL_small_norm_ortho.pth:ortho'")

    # ══════════════════════════════════════════════════════════════════════════
    # COMBINED SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    if len(all_results) > 1:
        summary_rows = [(k, v) for k, v in all_results.items()]
        print_ablation_table(summary_rows, title="COMBINED SUMMARY — ALL CONDITIONS")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
