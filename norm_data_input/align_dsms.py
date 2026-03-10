import os
import numpy as np
from scipy.io import loadmat, savemat
from tqdm import tqdm
import argparse

def get_dsm_data(path):
    """Loads mat file and returns points + the original dictionary."""
    if not os.path.exists(path): return None, None
    try:
        M = loadmat(path)
        pts = None
        key_found = None
        for key in ['points', 'vertices', 'Vertices', 'Points']:
            if key in M:
                pts = M[key]
                key_found = key
                break
        if pts is None:
            for k, v in M.items():
                if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 3:
                    pts = v
                    key_found = k
                    break
        return pts.astype(np.float32), M, key_found
    except: return None, None, None

def compute_stats(pts):
    p_min, p_max = np.min(pts, axis=0), np.max(pts, axis=0)
    return {"center": (p_min + p_max) / 2.0, "radius": np.max(p_max - p_min) / 2.0}

def align_directories(source_dir, target_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    src_files = {f for f in os.listdir(source_dir) if f.endswith(".mat")}
    tgt_files = {f for f in os.listdir(target_dir) if f.endswith(".mat")}
    common_files = sorted(list(src_files.intersection(tgt_files)))
    
    if not common_files:
        print("[ERROR] No common files found to calculate general tendency. 1-1 names required for calibration.")
        return

    print(f"[INFO] Calibrating using {len(common_files)} common files...")
    scales = []
    offsets = []
    
    for fname in common_files:
        s_pts, _, _ = get_dsm_data(os.path.join(source_dir, fname))
        t_pts, _, _ = get_dsm_data(os.path.join(target_dir, fname))
        if s_pts is None or t_pts is None: continue
        
        s_stats = compute_stats(s_pts)
        t_stats = compute_stats(t_pts)
        
        scales.append(t_stats["radius"] / s_stats["radius"])
        offsets.append(t_stats["center"] - s_stats["center"])
    
    # Use Median to find the "General Tendency" (robust to outliers)
    global_scale = np.median(scales)
    global_offset = np.median(offsets, axis=0)
    
    print(f"[INFO] General Tendency Found:")
    print(f"  - Median Scale Factor: {global_scale:.4f}")
    print(f"  - Median Translation Offset: {global_offset}")
    
    print(f"\n[INFO] Applying global transformation to ALL {len(src_files)} files in {source_dir}...")
    for fname in tqdm(sorted(list(src_files))):
        s_pts, s_m, s_key = get_dsm_data(os.path.join(source_dir, fname))
        if s_pts is None: continue
        
        # Apply global tendency transformation:
        # 1. Move to local center
        s_stats = compute_stats(s_pts)
        pts_norm = s_pts - s_stats["center"]
        
        # 2. Scale
        pts_rescaled = pts_norm * global_scale
        
        # 3. Move back to original center + global offset
        # (This preserves the internal relative center but shifts it by the general bias)
        pts_aligned = pts_rescaled + (s_stats["center"] + global_offset)
        
        # 4. Save
        s_m[s_key] = pts_aligned
        savemat(os.path.join(output_dir, fname), s_m)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="inference_data/DSM", help="Directory with high quality points (Dir 1)")
    parser.add_argument("--tgt", type=str, default="inference_data/DSM_TEST_DATASET", help="Directory with desired coordinates (Dir 2)")
    parser.add_argument("--out", type=str, default="inference_data/DSM_ALIGNED")
    args = parser.parse_args()
    align_directories(args.src, args.tgt, args.out)
