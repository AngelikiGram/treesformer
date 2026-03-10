import os
import torch
import numpy as np
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader

# Import evaluation and loss functions from train.py and lsys_losses.py

from train import forward_with_truncated_bptt, compute_position_weights, compute_rotation_smoothness_loss, compute_soft_angle_loss, compute_differentiable_turtle_positions, LSystemModel
from auxiliary.lsys_losses import chamfer_distance, ChamferLoss
from auxiliary.lsys_tokenizer import LSystemTokenizerV2, TokenType
from auxiliary.lsys_dataset import LSystemDataset
from auxiliary.lsys_renderer import render_lsystem
from visualize import LSystemVisdom

@torch.no_grad()
def run_inference(
    checkpoint_path,
    num_trees=10,
    base_path=".",
    lstrings_path="SYMBOLIC_LSTRINGS_d3",
    window=1024,
    device="cuda",
    batch_size=1,
    f_bins=10,
    theta_bins=6,
    phi_bins=6,
    dim=512,
    heads=16,
    layers=8,
    save_results=True,
    custom_data=False,
    save_dir="inference_results",
    dsm_dir="DSM_ALIGNED",
    ortho_dir="ORTHOPHOTOS",
    modality="both", # "both", "dsm", "ortho"
    visual_bottleneck=1024,
    id_cache=None,
    target_tid=None,
    temp=1.0,
    normalize=False,
    eval_mode="autoregressive", # "autoregressive", "teacher", "student"
    eval_metrics=True,
    bptt_chunk_size=None,
    chamfer_weight=1.0,
    max_points_chamfer=1000,
    rotation_smoothness_weight=0.01,
):
    # 1. Initialize Tokenizer
    tokenizer = LSystemTokenizerV2(f_bins=f_bins, theta_bins=theta_bins, phi_bins=phi_bins)
    
    # 2. Select IDs (Same way as during training)
    # Check for memory of held-out ids from training
    held_out_ids = set()
    if id_cache:
        hf = f"{id_cache}_held_out_ids.txt"
        if os.path.exists(hf):
            with open(hf, "r", encoding="utf-8") as f:
                held_out_ids = {line.strip() for line in f if line.strip()}
            print(f"[INFO] Found {len(held_out_ids)} held-out IDs from training memory.")

    dsm_dirname = dsm_dir if custom_data else "DSM"
    if lstrings_path:
        full_lstrings_dir = os.path.join(base_path, lstrings_path)
        if not os.path.exists(full_lstrings_dir):
            print(f"[WARNING] L-strings directory not found: {full_lstrings_dir}. Falling back to {dsm_dirname}.")
            full_ids_dir = os.path.join(base_path, dsm_dirname)
            ext = ".mat"
        else:
            full_ids_dir = full_lstrings_dir
            ext = ".txt"
    else:
        full_ids_dir = os.path.join(base_path, dsm_dirname)
        ext = ".mat"

    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', s)]

    all_raw_ids = sorted([f[:-len(ext)] for f in os.listdir(full_ids_dir) if f.endswith(ext)], key=natural_sort_key)
    
    if target_tid:
        # User specified a specific ID - Try to find it in the raw list
        if target_tid in all_raw_ids:
            all_ids = [target_tid]
        else:
            # Try fuzzy match (e.g. user gave "13" but it's "tree_13" or "tree_0013")
            matches = [r for r in all_raw_ids if target_tid in r or (target_tid.isdigit() and str(int(target_tid)) in r)]
            if matches:
                all_ids = [matches[0]]
                print(f"[INFO] fuzzy matched '{target_tid}' to '{all_ids[0]}'")
            else:
                print(f"[ERROR] Specified TID '{target_tid}' not found in {full_ids_dir}.")
                print(f"Available IDs (first 20): {all_raw_ids[:20]}")
                return
        print(f"[INFO] Running inference for SPECIFIC tree ID: {all_ids[0]}")
    elif held_out_ids:
        all_ids = [tid for tid in all_raw_ids if tid in held_out_ids]
        print(f"[INFO] Filtered to {len(all_ids)} held-out trees (out of {len(all_raw_ids)} total).")
    else:
        all_ids = all_raw_ids

    # 2b. Filter for IDs that also have orthophotos
    ortho_path_base = os.path.join(base_path, ortho_dir)
    print(f"[INFO] Filtering {len(all_ids)} IDs for orthophoto existence in {ortho_path_base}...")
    
    filtered_ids = []
    for tid in all_ids:
        # Check DSM first (in case IDs came from lstrings_path)
        dsm_path = os.path.join(base_path, dsm_dirname, f"{tid}.mat")
        if not os.path.exists(dsm_path):
            continue
            
        # Check Orthophoto
        # Normalization logic from dataset (tree_1 -> tree_0001)
        norm_tid = tid
        if "_" in tid:
            parts = tid.split("_")
            try:
                num = int(parts[1])
                if num < 1000: norm_tid = f"{parts[0]}_{num:04d}"
            except: pass
            
        found_ortho = False
        # Check direct files
        for t in [tid, norm_tid]:
            if os.path.exists(os.path.join(ortho_path_base, f"{t}.png")):
                found_ortho = True; break
        
        if not found_ortho:
            # Check folders
            for t in [tid, norm_tid]:
                p = os.path.join(ortho_path_base, t)
                if os.path.isdir(p):
                    # Check for any png in folder or rendering subfolder
                    if any(f.lower().endswith(".png") for f in os.listdir(p)):
                        found_ortho = True; break
                    sub = os.path.join(p, "rendering")
                    if os.path.isdir(sub) and any(f.lower().endswith(".png") for f in os.listdir(sub)):
                        found_ortho = True; break
        
        if found_ortho:
            filtered_ids.append(tid)

    ids = filtered_ids[:num_trees]
    print(f"[INFO] Found {len(filtered_ids)} complete pairs. Proceeding with {len(ids)} trees.")

    # 3. Load Dataset
    dataset = LSystemDataset(
        base_path=base_path,
        lstring_dir=lstrings_path if lstrings_path else dsm_dir, # dummy if empty
        tokenizer=tokenizer,
        ids=ids,
        window=window,
        overlap=0,
        dsm_dirname=dsm_dirname,
        ortho_dirname=ortho_dir,
        normalize=normalize,
        training=False,
    )
    print(f"[INFO] Loaded dataset using {dsm_dir}.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 4. Load Checkpoint first to check shapes if necessary
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    file_size = os.path.getsize(checkpoint_path)
    print(f"[INFO] Loading checkpoint: {checkpoint_path} ({file_size / 1024**2:.2f} MB)")
    
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to load checkpoint: {checkpoint_path}")
        print(f"The file appears to be corrupted or not a valid PyTorch save.")
        print(f"Error: {e}")
        if file_size == 0:
            print("ALERT: The checkpoint file is 0 bytes!")
        return

    # Detect base properties
    num_species = dataset.num_species
    model_sd = ckpt["model"]

    print("[DEBUG] Dataset num_species:", num_species)
    if "species_head.weight" in model_sd:
        print("[DEBUG] Checkpoint species_head.weight shape:", model_sd["species_head.weight"].shape)
    print("[DEBUG] inv_species_map:", getattr(dataset, 'inv_species_map', None))

    # 4b. Auto-detect bin sizes from checkpoint if they differ from args
    if "val_emb_length.weight" in model_sd:
        f_bins_ckpt = model_sd["val_emb_length.weight"].shape[0]
        if f_bins_ckpt != f_bins:
            print(f"[INFO] Auto-adjusting f_bins: {f_bins} -> {f_bins_ckpt}")
            f_bins = f_bins_ckpt
    
    if "val_emb_theta.weight" in model_sd:
        theta_bins_ckpt = model_sd["val_emb_theta.weight"].shape[0]
        if theta_bins_ckpt != theta_bins:
            print(f"[INFO] Auto-adjusting theta_bins: {theta_bins} -> {theta_bins_ckpt}")
            theta_bins = theta_bins_ckpt

    if "val_emb_phi.weight" in model_sd:
        phi_bins_ckpt = model_sd["val_emb_phi.weight"].shape[0]
        if phi_bins_ckpt != phi_bins:
            print(f"[INFO] Auto-adjusting phi_bins: {phi_bins} -> {phi_bins_ckpt}")
            phi_bins = phi_bins_ckpt

    if "species_head.weight" in model_sd:
        num_species_ckpt = model_sd["species_head.weight"].shape[0]
        if num_species_ckpt != num_species:
            print(f"[INFO] Auto-adjusting num_species: {num_species} -> {num_species_ckpt}")
            num_species = num_species_ckpt

    if "type_emb.weight" in model_sd:
        dim_ckpt = model_sd["type_emb.weight"].shape[1]
        if dim_ckpt != dim:
            print(f"[INFO] Auto-adjusting dim: {dim} -> {dim_ckpt}")
            dim = dim_ckpt

    # Auto-adjust heads and layers (heuristic from state dict keys)
    block_keys = [k for k in model_sd.keys() if "blocks." in k]
    if block_keys:
        # Each block has multiple layers, assume numbering matches
        layers_ckpt = max([int(k.split(".")[1]) for k in block_keys]) + 1
        if layers_ckpt != layers:
            print(f"[INFO] Auto-adjusting layers: {layers} -> {layers_ckpt}")
            layers = layers_ckpt
            
    # CRITICAL: Auto-detect heads using RoPE dimension
    # inv_freq length is head_dim // 2
    if "rope.inv_freq" in model_sd:
        inv_len = model_sd["rope.inv_freq"].shape[0]
        # inv_len = (dim // heads) // 2  => heads = dim // (inv_len * 2)
        heads_ckpt = dim // (inv_len * 2)
        if heads_ckpt != heads:
            print(f"[INFO] Auto-adjusting heads: {heads} -> {heads_ckpt}")
            heads = heads_ckpt
    elif "blocks.0.self_attn.q_proj.weight" in model_sd:
        # Fallback heuristic if RoPE isn't there
        pass 

    # Re-init tokenizer if bins changed
    tokenizer = LSystemTokenizerV2(f_bins=f_bins, theta_bins=theta_bins, phi_bins=phi_bins)

    # 5. Initialize Model
    model = LSystemModel(
        f_bins=f_bins,
        theta_bins=theta_bins,
        phi_bins=phi_bins,
        num_species=num_species,
        dim=dim,
        max_window=window,
        cross_attn_window=window,
        heads=heads,
        layers=layers,
        visual_bottleneck=visual_bottleneck,
    ).to(device)
    # Print species head shape from the multimodal encoder
    if hasattr(model.mm, 'species_head'):
        print("[DEBUG] Model.mm.species_head.weight shape:", model.mm.species_head[-1].weight.shape)

    # 6. Load weights with same fallbacks as train.py
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as e:
        print(f"[WARNING] Strict load failed: {e}")
        print("[INFO] Attempting non-strict load with weight filtering...")
        curr_sd = model.state_dict()
        filtered_sd = {k: v for k, v in ckpt["model"].items() 
                       if k in curr_sd and v.shape == curr_sd[k].shape}
        missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
        print(f"[INFO] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    model.eval()

    # 6. Setup Visualization
    viz = LSystemVisdom(env=f"inference_results_{save_dir.split('/')[-1]}", port=8099)
    viz.viz.close()
    results_dir = save_dir
    if save_results:
        os.makedirs(results_dir, exist_ok=True)

    # 7. Inference Loop
    # Metrics accumulators
    total_type_acc = 0.0
    total_val_acc = 0.0
    total_chamfer = 0.0
    total_rot_smooth = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Inference")):
        dsm = batch["dsm"].to(device).float()
        ortho = batch["ortho"].to(device).float()
        # ── MODALITY MASKING ──
        if modality == "dsm":
            ortho = torch.zeros_like(ortho)
        elif modality == "ortho":
            dsm = torch.zeros_like(dsm)
        t_in = batch["type_in"].to(device)
        v_in = batch["val_in"].to(device)
        t_tgt = batch["type_tgt"].to(device)
        v_tgt = batch["val_tgt"].to(device)
        tid = batch["tid"][0]
        # --- Species Prediction ---
        with torch.no_grad():
            v_mem_seq, global_feat, species_logits = model.mm(dsm[0:1], ortho[0:1])
            species_id = torch.argmax(species_logits, dim=-1).item()
            species_name = dataset.inv_species_map.get(species_id, f"ID_{species_id}")
            gt_species_id = batch["species"][0].item()
            gt_species_name = dataset.inv_species_map.get(gt_species_id, "N/A")
        print(f"[INFERENCE] [{tid}] Predicted Species: {species_name} (GT: {gt_species_name})")

        # --- Evaluation Modes ---
        if eval_mode == "teacher":
            # Use teacher-forcing with chunked BPTT
            chunk_size = bptt_chunk_size or window
            tlog_chunks, vlog_chunks, _, _ = forward_with_truncated_bptt(
                model, t_in, v_in, dsm, ortho, chunk_size=chunk_size
            )
            t_logits = torch.cat(tlog_chunks, dim=1)
            v_logits = torch.cat(vlog_chunks, dim=1)
            # Argmax predictions
            pred_types = torch.argmax(t_logits, dim=-1)
            pred_vals = torch.argmax(v_logits[..., :f_bins], dim=-1).unsqueeze(-1)
            # Metrics
            valid_mask = (t_tgt != TokenType.PAD)
            type_acc = (pred_types == t_tgt)[valid_mask].float().mean().item()
            val_acc = (pred_vals[..., 0] == v_tgt[..., 0])[valid_mask].float().mean().item()
            total_type_acc += type_acc
            total_val_acc += val_acc
            # Geometry-aware metrics (optional)
            if eval_metrics:
                # Compute predicted positions (differentiable turtle)
                pred_pos = compute_differentiable_turtle_positions(
                    v_logits, batch["states"][0:1], t_tgt, f_bins, theta_bins, phi_bins
                )
                gt_pos = batch["states_tgt"][0:1, :, :3]
                chamfer = chamfer_distance(pred_pos, gt_pos, num_points=max_points_chamfer, normalize=normalize).item()
                total_chamfer += chamfer
                # Rotation smoothness
                rot_smooth = compute_rotation_smoothness_loss(v_logits, v_tgt, t_tgt, valid_mask, theta_bins, phi_bins).item()
                total_rot_smooth += rot_smooth
            total_samples += 1
            print(f"[EVAL][{tid}] Type Acc: {type_acc:.3f} | Val Acc: {val_acc:.3f} | Chamfer: {chamfer:.4f} | RotSmooth: {rot_smooth:.4f}")
            pred_str = tokenizer.decode(pred_types[0].tolist(), v_tgt[0].tolist())
            pred_pts = compute_differentiable_turtle_positions(v_logits, batch["states"][0:1], t_tgt, f_bins, theta_bins, phi_bins)[0].cpu().numpy()
        elif eval_mode == "autoregressive":
            # --- Autoregressive Generation (default) ---
            pred_str = model.generate(
                dsm[0:1], ortho[0:1], tokenizer, max_len=window,
                temperature=temp, temperature_structural=max(0.3, temp * 0.7)
            )
            # Geometry for visualization
            center = batch["dsm_center"][0].cpu().numpy()
            scale  = batch["dsm_scale"][0].item()
            s_factor = 2.0 / (scale + 1e-9)
            base_world = batch["states_tgt"][0, 0, :3].cpu().numpy()
            offset = (base_world - center) / (scale + 1e-9) * 2.0
            pred_pts = render_lsystem(pred_str, step_scale=s_factor, num_bins_theta=theta_bins, num_bins_phi=phi_bins, num_bins_f=f_bins)
            pred_pts += offset
        else:
            raise NotImplementedError(f"eval_mode {eval_mode} not implemented")

        # --- Visualization ---
        gt_str = None
        gt_pts = None
        if lstrings_path and os.path.exists(os.path.join(base_path, lstrings_path, f"{tid}.txt")):
            gt_types = t_tgt[0].cpu().tolist()
            trim_idx = len(gt_types)
            if TokenType.EOS in gt_types:
                trim_idx = gt_types.index(TokenType.EOS)
            elif TokenType.PAD in gt_types:
                trim_idx = gt_types.index(TokenType.PAD)
            gt_vals = v_tgt[0].cpu().tolist()
            gt_str = tokenizer.decode(gt_types[:trim_idx], gt_vals[:trim_idx])
            gt_pts = render_lsystem(gt_str, step_scale=s_factor, num_bins_theta=theta_bins, num_bins_phi=phi_bins, num_bins_f=f_bins)
            gt_pts += offset
        viz.visualize_inference(
            tid=tid,
            ortho=batch["ortho"][0],
            dsm=batch["dsm"][0],
            gt_pts=gt_pts,
            pred_pts=pred_pts
        )
        if save_results:
            # 1. Save Predicted L-string to its own file
            res_path = os.path.join(results_dir, f"{tid}.txt")
            with open(res_path, "w", encoding="utf-8") as f:
                f.write(pred_str)
            # 2. Append Species Prediction to global summary file
            summary_path = os.path.join(results_dir, "species_summary.txt")
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(f"{tid}: {species_name} (GT: {gt_species_name})\n")

    print(f"[OK] Inference complete. Results saved in {results_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pth file")
    parser.add_argument("--num_trees", type=int, default=10, help="Number of trees to evaluate")
    parser.add_argument("--base", type=str, default="/home/grammatikakis1/TREES_DATASET_SIDE", help="Base path for data")
    parser.add_argument("--lstrings", type=str, default="SYMBOLIC_LSTRINGS_d3")
    parser.add_argument("--window", type=int, default=700)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--f_bins", type=int, default=10)
    parser.add_argument("--theta_bins", type=int, default=6)
    parser.add_argument("--phi_bins", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--custom_data", action="store_true")
    parser.add_argument("--save_dir", type=str, default="inference_results")
    parser.add_argument("--dsm_dir", type=str, default="DSM_ALIGNED")
    parser.add_argument("--ortho_dir", type=str, default="ORTHOPHOTOS")
    parser.add_argument("--modality", type=str, default="both", choices=["both", "dsm", "ortho"])
    parser.add_argument("--visual_bottleneck", type=int, default=1024)
    parser.add_argument("--id_cache", type=str, default=None, help="Memory file ID from training to use held-out trees")
    parser.add_argument("--tid", type=str, default=None, help="Specific tree ID to run inference on (e.g. 0013)")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--normalize", action="store_true", help="Enable strict unit-cube and local-ortho normalization")
    parser.add_argument("--eval_mode", type=str, default="autoregressive", choices=["autoregressive", "teacher"], help="Evaluation mode: autoregressive or teacher-forcing")
    parser.add_argument("--eval_metrics", action="store_true", help="Compute evaluation metrics (accuracy, chamfer, etc)")
    parser.add_argument("--bptt_chunk_size", type=int, default=None, help="Chunk size for teacher-forcing eval")
    parser.add_argument("--chamfer_weight", type=float, default=1.0, help="Weight for Chamfer loss (geometry)")
    parser.add_argument("--max_points_chamfer", type=int, default=1000, help="Number of points for Chamfer loss")
    parser.add_argument("--rotation_smoothness_weight", type=float, default=0.01, help="Weight for rotation smoothness loss")
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        num_trees=args.num_trees,
        base_path=args.base,
        lstrings_path=args.lstrings,
        window=args.window,
        device=args.device,
        f_bins=args.f_bins,
        theta_bins=args.theta_bins,
        phi_bins=args.phi_bins,
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        custom_data=args.custom_data,
        save_dir=args.save_dir,
        dsm_dir=args.dsm_dir,
        ortho_dir=args.ortho_dir,
        modality=args.modality,
        visual_bottleneck=args.visual_bottleneck,
        id_cache=args.id_cache,
        target_tid=args.tid,
        temp=args.temp,
        normalize=args.normalize,
        eval_mode=args.eval_mode,
        eval_metrics=args.eval_metrics,
        bptt_chunk_size=args.bptt_chunk_size,
        chamfer_weight=args.chamfer_weight,
        max_points_chamfer=args.max_points_chamfer,
        rotation_smoothness_weight=args.rotation_smoothness_weight,
    )

# python inference.py --checkpoint p3_effsymbolic_L_small_norm.pth --num_trees 100 --lstrings "SYMBOLIC_LSTRINGS_d3" --window 700 --dim 256 --heads 8 --layers 8 --f_bins 10 --theta_bins 6 --phi_bins 6 --device cpu --save_dir results_test

# python inference.py --checkpoint p3_effsymbolic_L_small_norm.pth --num_trees 72 --custom_data --lstrings "" --window 700 --dim 256 --heads 8 --layers 8 --f_bins 10 --theta_bins 6 --phi_bins 6 --device cpu --save_dir results_landmarks --base "./inference_data"