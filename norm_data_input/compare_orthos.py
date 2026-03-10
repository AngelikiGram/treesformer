import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

def get_all_pngs(root_dir):
    """
    Robustly find all PNG files, similar to inference.py logic.
    Supports both direct files and those nested in tree folders.
    """
    png_paths = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".png"):
                png_paths.append(os.path.join(root, f))
    return png_paths

def analyze_set(name, paths, target_size=(256, 256)):
    print(f"\n[INFO] Analyzing '{name}' set ({len(paths)} images)...")
    
    means = []
    stds = []
    brightnesses = []
    
    # We accumulate a running sum to create a "Global Average Image"
    # Note: Use float64 to prevent overflow
    avg_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.float64)
    processed_count = 0
    
    for p in tqdm(paths):
        try:
            with Image.open(p).convert("RGB") as img:
                img_resize = img.resize(target_size)
                img_np = np.array(img_resize).astype(np.float64) / 255.0
                
                # Global Stats
                means.append(img_np.mean(axis=(0, 1)))
                stds.append(img_np.std(axis=(0, 1)))
                
                # Luminosity (Perceived brightness)
                lum = (0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]).mean()
                brightnesses.append(lum)
                
                # Running average image
                avg_img += img_np
                processed_count += 1
        except Exception as e:
            print(f"  [WARN] Failed to process {p}: {e}")
            
    if processed_count == 0:
        return None
        
    avg_img /= processed_count
    
    # Final Averages
    final_mean = np.mean(means, axis=0)
    final_std = np.mean(stds, axis=0)
    final_brightness = np.mean(brightnesses)
    
    # Save the "average image" for visual inspection
    out_name = f"average_ortho_{name}.png"
    Image.fromarray((avg_img * 255).astype(np.uint8)).save(out_name)
    
    return {
        "count": processed_count,
        "mean_rgb": final_mean,
        "std_rgb": final_std,
        "brightness": final_brightness,
        "avg_img_file": out_name
    }

def main():
    parser = argparse.ArgumentParser(description="Compare dataset-wide orthophoto stats.")
    parser.add_argument("--dir1", type=str, default="inference_data/ORTHOPHOTOS")
    parser.add_argument("--dir2", type=str, default="inference_data/ORTHOPHOTOS_TEST_DATASET")
    args = parser.parse_args()
    
    dir1 = args.dir1
    dir2 = args.dir2
    
    paths1 = get_all_pngs(dir1)
    paths2 = get_all_pngs(dir2)
    
    stats1 = analyze_set("set1", paths1)
    stats2 = analyze_set("set2", paths2)
    
    if not stats1 or not stats2:
        print("[ERROR] One of the directories produced no valid images.")
        return

    print("\n" + "="*50)
    print("      DATASET DIFFERENCE ANALYSIS (AVERAGE)")
    print("="*50)
    
    print(f"{'Metric':<20} | {'Set 1 (Base)':<15} | {'Set 2 (Test)':<15} | {'Diff %':<10}")
    print("-" * 65)
    
    # Brightness Comparison
    diff_br = (stats2['brightness'] - stats1['brightness']) / stats1['brightness'] * 100
    print(f"{'Brightness':<20} | {stats1['brightness']:<15.4f} | {stats2['brightness']:<15.4f} | {diff_br:>+7.1f}%")
    
    # RGB Channels
    channels = ['Red Mean', 'Green Mean', 'Blue Mean']
    for i, c in enumerate(channels):
        m1 = stats1['mean_rgb'][i]
        m2 = stats2['mean_rgb'][i]
        diff = (m2 - m1) / m1 * 100
        print(f"{c:<20} | {m1:<15.4f} | {m2:<15.4f} | {diff:>+7.1f}%")
        
    # Variability Comparison
    s1 = np.mean(stats1['std_rgb'])
    s2 = np.mean(stats2['std_rgb'])
    diff_s = (s2 - s1) / s1 * 100
    print(f"{'Texture (StdDev)':<20} | {s1:<15.4f} | {s2:<15.4f} | {diff_s:>+7.1f}%")
    
    print("\n[OK] Analysis complete.")
    print(f"  Visual 'Average Tree' images saved to: {stats1['avg_img_file']}, {stats2['avg_img_file']}")
    
    # Threshold check for model performance
    if abs(diff_br) > 15:
        print("\n[WARNING] Set 2 is significantly brighter/darker than Set 1. This may confuse the model.")
    if abs(stats1['mean_rgb'][1] - stats2['mean_rgb'][1]) / stats1['mean_rgb'][1] > 0.15:
        print("[WARNING] Significant 'Green' channel shift detected. This can affect species classification.")

if __name__ == "__main__":
    main()
