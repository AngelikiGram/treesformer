import os
import torch
import numpy as np
from auxiliary.lsys_tokenizer import LSystemTokenizerV2
from auxiliary.lsys_dataset import LSystemDataset
from tqdm import tqdm

def check_dsm_bounds():
    base_path = "./inference_data/austria_data"
    dsm_dirname = "DSM_ALIGNED"
    window = 1290
    
    # Check what IDs are available
    full_dsm_dir = os.path.join(base_path, dsm_dirname)
    if not os.path.exists(full_dsm_dir):
        print(f"Error: Path {full_dsm_dir} does not exist. Update base_path if needed.")
        return

    num_trees=100

    all_ids = [f[:-4] for f in os.listdir(full_dsm_dir) if f.endswith(".mat")]
    ids = all_ids[:num_trees]
    print(f"Found {len(all_ids)} DSMs in {full_dsm_dir}")

    # Set up tokenizer
    tokenizer = LSystemTokenizerV2(f_bins=12, theta_bins=12, phi_bins=15)
    
    dataset = LSystemDataset(
        base_path=base_path,
        tokenizer=tokenizer,
        ids=ids,
        window=window,
        preload=True,
        lstring_dir="LSTRINGS", # dummy
        dsm_dirname=dsm_dirname,
        ortho_dirname="ORTHOPHOTOS", # dummy
        normalize=True,
        training=False
    )
    
    stats_x = []
    stats_y = []
    stats_z = []
    
    print("Checking DSM bounds as they are yielded by the dataset...")
    for idx in tqdm(range(num_trees)):
        data = dataset[idx]
        dsm = data["dsm"]
        tid = data["tid"]
        
        # dsm shape is expected to be [N, 3]
        if dsm.shape[0] == 0 or (dsm.abs().max() < 1e-5):
            continue
            
        x_min, x_max = dsm[:, 0].min().item(), dsm[:, 0].max().item()
        y_min, y_max = dsm[:, 1].min().item(), dsm[:, 1].max().item()
        z_min, z_max = dsm[:, 2].min().item(), dsm[:, 2].max().item()
        
        stats_x.append((x_min, x_max))
        stats_y.append((y_min, y_max))
        stats_z.append((z_min, z_max))
        
    if len(stats_x) == 0:
        print("No valid DSMs processed.")
        return
        
    def print_axis_stats(name, stats):
        mins = [s[0] for s in stats]
        maxs = [s[1] for s in stats]
        print(f"\n=== {name}-Axis Scale Statistics ===")
        print(f"Global min value: {min(mins):.6f}")
        print(f"Global max value: {max(maxs):.6f}")
        print(f"Average {name} min: {sum(mins)/len(mins):.6f}")
        print(f"Average {name} max: {sum(maxs)/len(maxs):.6f}")
        
        eps = 1e-3
        not_touching_bounds = sum(1 for (mn, mx) in stats if abs(mn - (-1.0)) > eps or abs(mx - 1.0) > eps)
        outside_bounds = sum(1 for (mn, mx) in stats if mn < -1.0 - eps or mx > 1.0 + eps)
        
        if outside_bounds > 0:
            print(f"[FAIL] {outside_bounds} trees exceed the strict [-1.0, 1.0] bounding box!")
        elif not_touching_bounds > 0:
            print(f"[INFO] {not_touching_bounds} trees fit inside [-1.0, 1.0] but do not strictly touch BOTH ends (-1 and 1).")
        else:
            print(f"[OK] ALL {len(stats)} trees span EXACTLY from -1.0 to 1.0!")

    print(f"\nTotal processed: {len(stats_x)}")
    print_axis_stats("X", stats_x)
    print_axis_stats("Y", stats_y)
    print_axis_stats("Z", stats_z)

if __name__ == "__main__":
    check_dsm_bounds()
