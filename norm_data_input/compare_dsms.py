import os
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import argparse

def get_dsm_points(path):
    """Extracts points from a .mat file using the same logic as the dataset."""
    if not os.path.exists(path):
        return None
    try:
        M = loadmat(path)
        pts = None
        for key in ['points', 'vertices', 'Vertices', 'Points']:
            if key in M:
                pts = M[key]
                break
        if pts is None:
            for v in M.values():
                if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 3:
                    pts = v
                    break
        return pts.astype(np.float32) if pts is not None else None
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def compute_stats(pts):
    """Computes geometric stats for a point cloud."""
    if pts is None or len(pts) == 0:
        return None
    
    # 1. Min/Max
    p_min = np.min(pts, axis=0)
    p_max = np.max(pts, axis=0)
    
    # 2. Center (Bounding Box Center)
    center = (p_min + p_max) / 2.0
    
    # 3. Mean Center
    mean_center = np.mean(pts, axis=0)
    
    # 4. Radius (Max extent from center)
    extents = (p_max - p_min)
    radius = np.max(extents) / 2.0
    
    # 5. Max distance from mean center
    max_dist = np.max(np.linalg.norm(pts - mean_center, axis=1))
    
    return {
        "min": p_min,
        "max": p_max,
        "center": center,
        "mean_center": mean_center,
        "radius": radius,
        "max_dist": max_dist,
        "count": len(pts)
    }

def compare_dsms(dir1, dir2):
    print(f"Comparing DSMs in:\n  Dir 1: {dir1}\n  Dir 2: {dir2}\n")
    
    files1 = {f for f in os.listdir(dir1) if f.endswith(".mat")}
    files2 = {f for f in os.listdir(dir2) if f.endswith(".mat")}
    common_files = sorted(list(files1.intersection(files2)))
    
    if not common_files:
        print("No matching .mat files found between the two directories.")
        return

    print(f"Found {len(common_files)} common files. Analyzing...\n")
    
    header = f"{'Filename':<20} | {'Metric':<15} | {'Dir 1':<25} | {'Dir 2':<25} | {'Delta':<10}"
    print(header)
    print("-" * len(header))

    for fname in common_files:
        p1 = os.path.join(dir1, fname)
        p2 = os.path.join(dir2, fname)
        
        pts1 = get_dsm_points(p1)
        pts2 = get_dsm_points(p2)
        
        s1 = compute_stats(pts1)
        s2 = compute_stats(pts2)
        
        if s1 is None or s2 is None:
            continue

        # Compare Centers
        c_delta = np.linalg.norm(s1["center"] - s2["center"])
        r_delta = abs(s1["radius"] - s2["radius"])
        
        print(f"{fname:<20} | {'Box Center':<15} | {str(np.round(s1['center'], 3)):<25} | {str(np.round(s2['center'], 3)):<25} | {c_delta:.4f}")
        print(f"{'':<20} | {'Radius':<15} | {s1['radius']:<25.3f} | {s2['radius']:<25.3f} | {r_delta:.4f}")
        
        # Check for significant tilt/rotation by looking at bounds
        min_delta = np.linalg.norm(s1["min"] - s2["min"])
        print(f"{'':<20} | {'Min Bounds':<15} | {str(np.round(s1['min'], 3)):<25} | {str(np.round(s2['min'], 3)):<25} | {min_delta:.4f}")
        print("-" * len(header))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", type=str, default="inference_data/DSM")
    parser.add_argument("--dir2", type=str, default="inference_data/DSM_TEST_DATASET")
    args = parser.parse_args()
    
    compare_dsms(args.dir1, args.dir2)
