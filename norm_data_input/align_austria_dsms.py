import os
import numpy as np
from scipy.io import loadmat, savemat
from tqdm import tqdm

def process_dsms():
    src_dir = "./inference_data/austria_data/DSM"
    out_dir = "./inference_data/austria_data/DSM_ALIGNED"
    os.makedirs(out_dir, exist_ok=True)
    
    files = [f for f in os.listdir(src_dir) if f.endswith(".mat")]
    print(f"Processing {len(files)} files...")
    
    for f in tqdm(files):
        path = os.path.join(src_dir, f)
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
        
        if pts is None or len(pts) == 0:
            continue
            
        pts = pts.astype(np.float32)
        
        # We need to crop X and Z to match the GT proportion.
        # GT average X max = 0.65, Z max = 0.98. 
        # This implies extent_XZ / extent_Y = 0.66. (Since Y is height in raw mat files)
        
        height_axis = 1 # Y is height based on lsys_dataset mapping (pts_rot[:, 2] = pts_raw[:, 1])
        x_axis = 0
        z_axis = 2
        
        h_min, h_max = pts[:, height_axis].min(), pts[:, height_axis].max()
        height = h_max - h_min
        
        if height < 1e-4:
            continue
            
        # Target radius in X and Z is 0.66 / 0.99 = 0.66 ratio of total bounds, which is
        # a radius of 0.33 * height. Let's make it 0.35 to be safe and match the GT stats.
        max_radius = 0.35 * height
        
        # Best way to find tree center: centroid of top 10% highest points
        cutoff = h_max - 0.1 * height
        top_pts = pts[pts[:, height_axis] >= cutoff]
        if len(top_pts) > 0:
            center_x = top_pts[:, x_axis].mean()
            center_z = top_pts[:, z_axis].mean()
        else:
            center_x = pts[:, x_axis].mean()
            center_z = pts[:, z_axis].mean()
            
        # Compute horizontal distance from trunk center
        dist = np.sqrt((pts[:, x_axis] - center_x)**2 + (pts[:, z_axis] - center_z)**2)
        
        # Filter points (Cropping the bounding box to match the GT proportion)
        mask = dist <= max_radius
        pts_cropped = pts[mask]
        
        # If we cropped too aggressively (unlikely, but just in case), keep original
        if len(pts_cropped) < 100:
            pts_cropped = pts
            
        M[key_found] = pts_cropped
        savemat(os.path.join(out_dir, f), M)

if __name__ == "__main__":
    process_dsms()
