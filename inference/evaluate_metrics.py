"""
evaluate_metrics.py

Evaluate a trained TreesFormer checkpoint on N trees and report:
  - Chamfer RMSE  (rendered point clouds, normalized by GT bounding box)
  - Normalized Chamfer Distance
  - F1 Score  (precision / recall at a distance threshold)
  - Token-level Type Accuracy
  - Token-level Value (length-bin) Accuracy
  - Structural metrics: bracket balance error, sequence-length diff, F-token count diff
  - Diversity: fraction of unique closest-GT matches across predictions
  - Species classification accuracy (when species head is present)

Usage:
    python inference/evaluate_metrics.py \\
        --checkpoint path/to/checkpoint.pth \\
        --base /path/to/TREES_DATASET_SIDE \\
        --num_trees 20
"""

import sys
import os
import inspect
import argparse
import time
import importlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# Allow imports from parent directory (train.py, train_nospecies.py, auxiliary/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auxiliary.lsys_tokenizer import LSystemTokenizerV2, TokenType
from auxiliary.lsys_dataset import LSystemDataset
from auxiliary.lsys_renderer import render_lsystem

# Deferred: filled by load_training_logic()
LSystemModel = None


def load_training_logic(use_no_species: bool):
    global LSystemModel
    module_name = "train_nospecies" if use_no_species else "train"
    try:
        mod = importlib.import_module(module_name)
        LSystemModel = mod.LSystemModel
        print(f"[INFO] Loaded model architecture from {module_name}.py")
    except ImportError as e:
        print(f"[ERROR] Could not import {module_name}.py: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Small utilities
# ─────────────────────────────────────────────────────────────────────────────

def natural_sort_key(s):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)]


def render_to_pts(lstring: str, f_bins: int, theta_bins: int, phi_bins: int):
    """Render an L-string to a float32 (N,3) numpy array; returns None if empty."""
    pts = render_lsystem(
        lstring, step_scale=1.0,
        num_bins_f=f_bins, num_bins_theta=theta_bins, num_bins_phi=phi_bins,
    )
    if pts is None or len(pts) == 0:
        return None
    return np.array(pts, dtype=np.float32)


def pad_or_trim(pts: np.ndarray, n: int) -> np.ndarray:
    """Return exactly n rows from pts, padding by repeating if too short."""
    if len(pts) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if len(pts) >= n:
        return pts[:n]
    reps = (n + len(pts) - 1) // len(pts)
    return np.tile(pts, (reps, 1))[:n]


def chamfer_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Bidirectional Chamfer distance (RMSE) between two (N,3) arrays."""
    d2 = np.sum((pred[:, None] - gt[None, :]) ** 2, axis=-1)  # (N,M)
    fwd = d2.min(axis=1).mean()
    bwd = d2.min(axis=0).mean()
    return float(np.sqrt((fwd + bwd) / 2.0))


def f1_score_pts(pred: np.ndarray, gt: np.ndarray, threshold: float):
    """Precision / recall / F1 between two point clouds."""
    knn_gt   = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(gt)
    d_p2g, _ = knn_gt.kneighbors(pred)
    knn_pred = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(pred)
    d_g2p, _ = knn_pred.kneighbors(gt)
    prec  = float(np.mean(d_p2g.flatten() < threshold))
    rec   = float(np.mean(d_g2p.flatten() < threshold))
    f1    = 2 * prec * rec / (prec + rec + 1e-9)
    return f1, prec, rec


def bracket_balance_err(types) -> int:
    """Count unmatched brackets in a type sequence."""
    depth, unmatched = 0, 0
    for t in types:
        if t == TokenType.LBR:
            depth += 1
        elif t == TokenType.RBR:
            if depth > 0:
                depth -= 1
            else:
                unmatched += 1
    return unmatched + depth


def count_f(types) -> int:
    return sum(1 for t in types if t == TokenType.F)


# ─────────────────────────────────────────────────────────────────────────────
# ID selection helpers (mirrors inference.py logic)
# ─────────────────────────────────────────────────────────────────────────────

def _id_variants(tid: str) -> list:
    """Return a list of zero-padding variants for a tree ID like 'tree_0013'."""
    variants = [tid]
    if "_" in tid:
        prefix, num_str = tid.split("_", 1)
        try:
            n = int(num_str)
            for fmt in (f"{prefix}_{n}", f"{prefix}_{n:04d}", f"{prefix}_{n:05d}"):
                if fmt not in variants:
                    variants.append(fmt)
        except ValueError:
            pass
    return variants


def _has_dsm(tid: str, dsm_root: str) -> bool:
    for t in _id_variants(tid):
        if os.path.exists(os.path.join(dsm_root, f"{t}.mat")):
            return True
    return False


def _has_ortho(tid: str, ortho_root: str) -> bool:
    for t in _id_variants(tid):
        if os.path.exists(os.path.join(ortho_root, f"{t}.png")):
            return True
        folder = os.path.join(ortho_root, t)
        if os.path.isdir(folder):
            if any(f.lower().endswith(".png") for f in os.listdir(folder)):
                return True
            sub = os.path.join(folder, "rendering")
            if os.path.isdir(sub) and any(f.lower().endswith(".png") for f in os.listdir(sub)):
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    base_path: str,
    num_trees: int = 20,
    lstrings_path: str = "SYMBOLIC_LSTRINGS_d3",
    dsm_dir: str = "DSM",
    ortho_dir: str = "ORTHOPHOTOS",
    window: int = 700,
    device: str = "cuda",
    f_bins: int = 10,
    theta_bins: int = 6,
    phi_bins: int = 6,
    dim: int = 512,
    heads: int = 16,
    layers: int = 8,
    visual_bottleneck: int = 1024,
    modality: str = "both",
    temp: float = 1.0,
    normalize: bool = False,
    f1_threshold: float = 0.05,
    max_pts: int = 1000,
):
    # ── 1. Checkpoint & architecture detection ──────────────────────────────
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    sd_keys  = set(model_sd.keys())

    is_no_species = any("dsm_pooler" in k for k in sd_keys) or not any("species_head" in k for k in sd_keys)
    load_training_logic(use_no_species=is_no_species)

    # Auto-adjust hyperparameters from checkpoint weights
    if "val_emb_length.weight" in model_sd:
        f_bins = model_sd["val_emb_length.weight"].shape[0]
    if "val_emb_theta.weight"  in model_sd:
        theta_bins = model_sd["val_emb_theta.weight"].shape[0]
    if "val_emb_phi.weight"    in model_sd:
        phi_bins = model_sd["val_emb_phi.weight"].shape[0]
    if "type_emb.weight"       in model_sd:
        dim = model_sd["type_emb.weight"].shape[1]
    if "mm.fusion.weight"      in model_sd:
        visual_bottleneck = model_sd["mm.fusion.weight"].shape[0]

    block_keys = [k for k in sd_keys if "blocks." in k]
    if block_keys:
        layers = max(int(k.split(".")[1]) for k in block_keys) + 1
    if "rope.inv_freq" in model_sd:
        heads = dim // (model_sd["rope.inv_freq"].shape[0] * 2)

    num_species = 13
    if "species_head.weight" in model_sd:
        num_species = model_sd["species_head.weight"].shape[0]

    print(f"[INFO] Arch: f_bins={f_bins} theta={theta_bins} phi={phi_bins} "
          f"dim={dim} heads={heads} layers={layers} vb={visual_bottleneck}")

    tokenizer = LSystemTokenizerV2(f_bins=f_bins, theta_bins=theta_bins, phi_bins=phi_bins)

    # ── 2. Select tree IDs ───────────────────────────────────────────────────
    lstrings_full = os.path.join(base_path, lstrings_path)
    dsm_root      = os.path.join(base_path, dsm_dir)
    ortho_root    = os.path.join(base_path, ortho_dir)

    print(f"[DEBUG] base_path     : {base_path}")
    print(f"[DEBUG] lstrings_full : {lstrings_full}  exists={os.path.exists(lstrings_full)}")
    print(f"[DEBUG] dsm_root      : {dsm_root}  exists={os.path.exists(dsm_root)}")
    print(f"[DEBUG] ortho_root    : {ortho_root}  exists={os.path.exists(ortho_root)}")

    if os.path.exists(lstrings_full):
        all_ids = sorted([f[:-4] for f in os.listdir(lstrings_full) if f.endswith(".txt")],
                         key=natural_sort_key)
        print(f"[DEBUG] IDs from lstrings dir: {len(all_ids)} total. First 5: {all_ids[:5]}")
    elif os.path.exists(dsm_root):
        all_ids = sorted([f[:-4] for f in os.listdir(dsm_root) if f.endswith(".mat")],
                         key=natural_sort_key)
        print(f"[DEBUG] IDs from DSM dir: {len(all_ids)} total. First 5: {all_ids[:5]}")
    else:
        print(f"[ERROR] Neither lstrings dir nor DSM dir found. Check --base, --lstrings, --dsm_dir.")
        return {}

    # Count failures per check for diagnostics
    no_dsm, no_ortho = 0, 0
    filtered_ids = []
    for tid in all_ids:
        ok_dsm   = _has_dsm(tid, dsm_root)
        ok_ortho = _has_ortho(tid, ortho_root)
        if not ok_dsm:
            no_dsm += 1
        if not ok_ortho:
            no_ortho += 1
        if ok_dsm and ok_ortho:
            filtered_ids.append(tid)

    print(f"[DEBUG] Filter results: {len(all_ids)} total → "
          f"{no_dsm} missing DSM, {no_ortho} missing ortho, "
          f"{len(filtered_ids)} complete pairs.")
    if len(filtered_ids) == 0:
        print(f"[ERROR] No complete pairs found.")
        print(f"        Sample IDs tried: {all_ids[:5]}")
        print(f"        DSM files in dir: {os.listdir(dsm_root)[:5] if os.path.exists(dsm_root) else 'N/A'}")
        print(f"        Ortho files in dir: {os.listdir(ortho_root)[:5] if os.path.exists(ortho_root) else 'N/A'}")
        return {}

    ids = filtered_ids[:num_trees]
    print(f"[INFO] Evaluating {len(ids)} trees: {ids[:5]}{'...' if len(ids) > 5 else ''}\n")

    # ── 3. Dataset ───────────────────────────────────────────────────────────
    dataset = LSystemDataset(
        base_path=base_path,
        lstring_dir=lstrings_path,
        tokenizer=tokenizer,
        ids=ids,
        window=window,
        overlap=0,
        dsm_dirname=dsm_dir,
        ortho_dirname=ortho_dir,
        normalize=normalize,
        training=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ── 4. Model ─────────────────────────────────────────────────────────────
    model_args = dict(
        f_bins=f_bins, theta_bins=theta_bins, phi_bins=phi_bins,
        dim=dim, max_window=window, cross_attn_window=window,
        heads=heads, layers=layers, visual_bottleneck=visual_bottleneck,
    )
    if "num_species" in inspect.signature(LSystemModel.__init__).parameters:
        model_args["num_species"] = num_species

    model = LSystemModel(**model_args).to(device)
    try:
        model.load_state_dict(model_sd, strict=True)
        print("[INFO] Weights loaded (strict).")
    except RuntimeError as e:
        print(f"[WARN] Strict load failed – attempting partial load. ({e})")
        curr_sd = model.state_dict()
        filtered_sd = {k: v for k, v in model_sd.items()
                       if k in curr_sd and v.shape == curr_sd[k].shape}
        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        print(f"[INFO] Non-strict load: {len(missing)} missing, {len(unexpected)} unexpected")
    model.eval()

    # ── 5. Inference & metric accumulation ───────────────────────────────────
    all_pred_pts_sub: list[np.ndarray] = []
    all_gt_pts_sub:   list[np.ndarray] = []

    chamfer_list:     list[float] = []
    norm_chamfer_list: list[float] = []
    f1_list:          list[float] = []
    prec_list:        list[float] = []
    rec_list:         list[float] = []
    type_acc_list:    list[float] = []
    val_acc_list:     list[float] = []
    bracket_err_list: list[int]   = []
    len_diff_list:    list[int]   = []
    f_diff_list:      list[int]   = []
    species_hits:     list[int]   = []
    gen_times:        list[float] = []
    tids_done:        list[str]   = []

    for batch in tqdm(loader, desc="Evaluating"):
        dsm   = batch["dsm"].to(device).float()
        ortho = batch["ortho"].to(device).float()
        tid   = batch["tid"][0]
        tids_done.append(tid)

        if modality == "dsm":
            ortho = torch.zeros_like(ortho)
        elif modality == "ortho":
            dsm = torch.zeros_like(dsm)

        # Ground-truth token sequence (trim at EOS/PAD)
        t_tgt = batch["type_tgt"][0].cpu().tolist()
        v_tgt = batch["val_tgt"][0].cpu().tolist()
        trim  = next((i for i, t in enumerate(t_tgt) if t in (TokenType.EOS, TokenType.PAD)), len(t_tgt))
        gt_types = t_tgt[:trim]
        gt_vals  = v_tgt[:trim]

        # Species prediction
        mm_out = model.mm(dsm[0:1], ortho[0:1])
        if isinstance(mm_out, (list, tuple)) and len(mm_out) == 3:
            _, _, species_logits = mm_out
            pred_sp = torch.argmax(species_logits, dim=-1).item()
            gt_sp   = batch["species"][0].item()
            species_hits.append(int(pred_sp == gt_sp))

        # Autoregressive generation
        t0 = time.time()
        try:
            final_types, final_vals = model.pure_inference(
                dsm[0:1], ortho[0:1], tokenizer, max_len=window,
                temperature=temp, temperature_structural=max(0.3, temp * 0.7),
            )
            pred_str       = tokenizer.decode(final_types[0], final_vals[0])
            pred_type_list = [t for t in final_types[0] if t not in (TokenType.EOS, TokenType.PAD)]
            pred_val_list  = final_vals[0][:len(pred_type_list)]
        except AttributeError:
            pred_str = model.generate(
                dsm[0:1], ortho[0:1], tokenizer, max_len=window,
                temperature=temp, temperature_structural=max(0.3, temp * 0.7),
            )
            enc_types, enc_vals = tokenizer.encode(pred_str)
            pred_type_list = [t for t in enc_types if t not in (TokenType.EOS, TokenType.PAD)]
            pred_val_list  = enc_vals[:len(pred_type_list)]

        gen_times.append(time.time() - t0)

        # Token-level accuracy (on the overlapping prefix)
        cmp_len = min(len(pred_type_list), len(gt_types))
        if cmp_len > 0:
            type_acc_list.append(
                sum(p == g for p, g in zip(pred_type_list[:cmp_len], gt_types[:cmp_len])) / cmp_len
            )
            # Value accuracy: length bin on correctly matched F tokens
            f_match, f_total = 0, 0
            for i in range(cmp_len):
                if gt_types[i] == TokenType.F and pred_type_list[i] == TokenType.F:
                    if pred_val_list[i][0] == gt_vals[i][0]:
                        f_match += 1
                    f_total += 1
            if f_total > 0:
                val_acc_list.append(f_match / f_total)

        # GT point cloud: read the full L-string file directly so we render the
        # complete tree (the batch window may be shorter than the full sequence).
        gt_lstring_file = os.path.join(base_path, lstrings_path, f"{tid}.txt")
        if os.path.exists(gt_lstring_file):
            with open(gt_lstring_file, "r", encoding="utf-8") as _f:
                gt_str_full = _f.read().strip()
            # Full GT sequence for structural metrics
            gt_types_full, gt_vals_full = tokenizer.encode(gt_str_full)
            gt_types_full = [t for t in gt_types_full if t not in (TokenType.EOS, TokenType.PAD)]
        else:
            # Fallback: reconstruct from the (possibly windowed) batch tokens
            gt_str_full   = tokenizer.decode(gt_types, gt_vals)
            gt_types_full = gt_types

        # Structural quality (use full GT sequence length)
        bracket_err_list.append(bracket_balance_err(pred_type_list))
        len_diff_list.append(abs(len(pred_type_list) - len(gt_types_full)))
        f_diff_list.append(abs(count_f(pred_type_list) - count_f(gt_types_full)))

        # Render both to point clouds (full GT vs predicted)
        gt_pts   = render_to_pts(gt_str_full, f_bins, theta_bins, phi_bins)
        pred_pts = render_to_pts(pred_str,    f_bins, theta_bins, phi_bins)

        if gt_pts is None:
            print(f"  [{tid}] WARN: empty GT point cloud – skipping geometry metrics.")
            all_gt_pts_sub.append(np.zeros((max_pts, 3), dtype=np.float32))
            all_pred_pts_sub.append(np.zeros((max_pts, 3), dtype=np.float32))
            continue

        if pred_pts is None or len(pred_pts) == 0:
            print(f"  [{tid}] WARN: empty predicted point cloud.")
            pred_pts = np.zeros((1, 3), dtype=np.float32)

        # Normalize into GT bounding box (shape comparison, not absolute position)
        gt_min, gt_max = gt_pts.min(0), gt_pts.max(0)
        gt_scale  = max(float((gt_max - gt_min).max()), 1e-6)
        gt_center = (gt_min + gt_max) / 2.0
        tree_height_norm = float((gt_max - gt_min)[2]) / gt_scale  # normalized Z extent

        gt_norm   = (gt_pts   - gt_center) / gt_scale
        pred_norm = (pred_pts - gt_center) / gt_scale

        gt_sub   = pad_or_trim(gt_norm,   max_pts)
        pred_sub = pad_or_trim(pred_norm, max_pts)

        cd = chamfer_rmse(pred_sub, gt_sub)
        chamfer_list.append(cd)
        norm_chamfer_list.append(cd / (tree_height_norm + 1e-6))

        f1, prec, rec = f1_score_pts(pred_sub, gt_sub, threshold=f1_threshold)
        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)

        all_gt_pts_sub.append(gt_sub)
        all_pred_pts_sub.append(pred_sub)

        t_acc_str = f"{type_acc_list[-1]:.3f}" if type_acc_list else "n/a"
        print(f"  [{tid}]  CD={cd:.4f}  F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}  "
              f"TypeAcc={t_acc_str}  BrackErr={bracket_err_list[-1]}")

    # ── 6. Diversity: for each pred, find closest GT (inter-sample) ──────────
    closest_matches = []
    for pred in all_pred_pts_sub:
        dists = [
            float(np.sum((pred[:, None] - gt[None, :]) ** 2, axis=-1).min(axis=1).mean())
            for gt in all_gt_pts_sub
        ]
        closest_matches.append(int(np.argmin(dists)))

    unique_matches = len(set(closest_matches))
    variance_score = unique_matches / max(len(closest_matches), 1) * 100.0

    # ── 7. Print results ──────────────────────────────────────────────────────
    W = 70
    print(f"\n{'='*W}")
    print("  TREESFORMER EVALUATION RESULTS")
    print(f"{'='*W}")
    print(f"  Trees evaluated         : {len(tids_done)}")
    print(f"  Avg generation time     : {np.mean(gen_times):.2f}s")

    print(f"\n  GEOMETRIC ACCURACY")
    if chamfer_list:
        print(f"  {'Chamfer RMSE (normalized):':<34} {np.mean(chamfer_list):.4f}  ±{np.std(chamfer_list):.4f}")
        print(f"  {'Normalized Chamfer:':<34} {np.mean(norm_chamfer_list):.4f}")
    else:
        print("  (no geometry data)")

    print(f"\n  COMPLETENESS & PRECISION")
    if f1_list:
        print(f"  {'F1 Score:':<34} {np.mean(f1_list):.4f}")
        print(f"  {'Precision:':<34} {np.mean(prec_list):.4f}")
        print(f"  {'Recall:':<34} {np.mean(rec_list):.4f}")

    print(f"\n  TOKEN-LEVEL ACCURACY")
    if type_acc_list:
        print(f"  {'Type Accuracy:':<34} {np.mean(type_acc_list):.4f}")
    if val_acc_list:
        print(f"  {'Value (length-bin) Accuracy:':<34} {np.mean(val_acc_list):.4f}")

    print(f"\n  STRUCTURAL QUALITY")
    print(f"  {'Avg bracket balance error:':<34} {np.mean(bracket_err_list):.2f}")
    print(f"  {'Avg |sequence length diff|:':<34} {np.mean(len_diff_list):.1f} tokens")
    print(f"  {'Avg |F-token count diff|:':<34} {np.mean(f_diff_list):.1f}")

    print(f"\n  DIVERSITY")
    print(f"  {'Unique closest-GT matches:':<34} {unique_matches} / {len(closest_matches)}")
    print(f"  {'Variance Score:':<34} {variance_score:.1f}%")

    if species_hits:
        print(f"\n  SPECIES CLASSIFICATION")
        print(f"  {'Species Accuracy:':<34} {np.mean(species_hits) * 100:.1f}%")

    print(f"\n{'='*W}")
    print("  EVALUATION COMPLETE")
    print(f"{'='*W}\n")

    return {
        "num_trees":           len(tids_done),
        "chamfer_rmse":        np.mean(chamfer_list)       if chamfer_list else None,
        "chamfer_std":         np.std(chamfer_list)        if chamfer_list else None,
        "normalized_chamfer":  np.mean(norm_chamfer_list)  if norm_chamfer_list else None,
        "f1_score":            np.mean(f1_list)            if f1_list else None,
        "precision":           np.mean(prec_list)          if prec_list else None,
        "recall":              np.mean(rec_list)            if rec_list else None,
        "type_accuracy":       np.mean(type_acc_list)      if type_acc_list else None,
        "val_accuracy":        np.mean(val_acc_list)       if val_acc_list else None,
        "bracket_error":       float(np.mean(bracket_err_list)),
        "seq_len_diff":        float(np.mean(len_diff_list)),
        "f_count_diff":        float(np.mean(f_diff_list)),
        "variance_score":      variance_score,
        "species_accuracy":    np.mean(species_hits) * 100 if species_hits else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a TreesFormer checkpoint")
    parser.add_argument("--checkpoint",        type=str,   required=True,
                        help="Path to .pth checkpoint file")
    parser.add_argument("--base",              type=str,   default="/home/grammatikakis1/TREES_DATASET_SIDE",
                        help="Dataset root directory")
    parser.add_argument("--lstrings",          type=str,   default="SYMBOLIC_LSTRINGS_d3",
                        help="Sub-directory containing ground-truth L-string .txt files")
    parser.add_argument("--dsm_dir",           type=str,   default="DSM",
                        help="Sub-directory containing DSM .mat files")
    parser.add_argument("--ortho_dir",         type=str,   default="ORTHOPHOTOS",
                        help="Sub-directory containing orthophoto images")
    parser.add_argument("--num_trees",         type=int,   default=20,
                        help="Number of trees to evaluate")
    parser.add_argument("--window",            type=int,   default=700,
                        help="Token sequence window length")
    parser.add_argument("--dim",               type=int,   default=512)
    parser.add_argument("--heads",             type=int,   default=16)
    parser.add_argument("--layers",            type=int,   default=8)
    parser.add_argument("--f_bins",            type=int,   default=10)
    parser.add_argument("--theta_bins",        type=int,   default=6)
    parser.add_argument("--phi_bins",          type=int,   default=6)
    parser.add_argument("--visual_bottleneck", type=int,   default=1024)
    parser.add_argument("--device",            type=str,   default="cuda")
    parser.add_argument("--modality",          type=str,   default="both",
                        choices=["both", "dsm", "ortho"],
                        help="Which input modalities to use")
    parser.add_argument("--temp",              type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--normalize",         action="store_true",
                        help="Enable strict unit-cube DSM normalization")
    parser.add_argument("--f1_threshold",      type=float, default=0.05,
                        help="Distance threshold for F1 score (normalized space)")
    parser.add_argument("--max_pts",           type=int,   default=1000,
                        help="Points sampled per cloud for Chamfer / F1")
    args = parser.parse_args()

    evaluate(
        checkpoint_path  = args.checkpoint,
        base_path        = args.base,
        num_trees        = args.num_trees,
        lstrings_path    = args.lstrings,
        dsm_dir          = args.dsm_dir,
        ortho_dir        = args.ortho_dir,
        window           = args.window,
        device           = args.device,
        f_bins           = args.f_bins,
        theta_bins       = args.theta_bins,
        phi_bins         = args.phi_bins,
        dim              = args.dim,
        heads            = args.heads,
        layers           = args.layers,
        visual_bottleneck= args.visual_bottleneck,
        modality         = args.modality,
        temp             = args.temp,
        normalize        = args.normalize,
        f1_threshold     = args.f1_threshold,
        max_pts          = args.max_pts,
    )
