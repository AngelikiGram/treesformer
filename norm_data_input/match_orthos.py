import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import random

def histogram_matching(source, reference):
    """
    Matches the cumulative histogram of a source image to a reference image.
    Works per channel.
    """
    matched = np.zeros_like(source)
    for i in range(3): # For B, G, R
        hist_src, bins = np.histogram(source[:,:,i].flatten(), 256, [0,256])
        hist_ref, bins = np.histogram(reference[:,:,i].flatten(), 256, [0,256])
        
        cdf_src = hist_src.cumsum()
        cdf_src = (cdf_src - cdf_src.min()) * 255 / (cdf_src.max() - cdf_src.min())
        cdf_src = cdf_src.astype('uint8')
        
        cdf_ref = hist_ref.cumsum()
        cdf_ref = (cdf_ref - cdf_ref.min()) * 255 / (cdf_ref.max() - cdf_ref.min())
        cdf_ref = cdf_ref.astype('uint8')
        
        # Mapping from src cdf to ref cdf
        im_list = list(source[:,:,i].flatten())
        
        # Use look-up table for speed
        lut = np.zeros(256, dtype='uint8')
        g_i = 0
        for r_i in range(256):
            while g_i < 255 and cdf_ref[g_i] < cdf_src[r_i]:
                g_i += 1
            lut[r_i] = g_i
            
        matched[:,:,i] = cv2.LUT(source[:,:,i], lut)
    return matched

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="inference_data/ORTHOPHOTOS", help="Source folder (Set 1)")
    parser.add_argument("--ref", type=str, default="inference_data/ORTHOPHOTOS_TEST_DATASET", help="Reference folder (Set 2)")
    parser.add_argument("--dst", type=str, default="inference_data/ORTHOPHOTOS_ALIGNED", help="Output folder")
    parser.add_argument("--blur", type=int, default=1, help="Gaussian blur kernel size (1 = almost none)")
    parser.add_argument("--noise", type=float, default=0.01, help="Amount of grain to add")
    args = parser.parse_args()
    
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)
        
    # Get a few reference images from Set 2 to compute a "Master Histogram"
    ref_paths = []
    for root, dirs, files in os.walk(args.ref):
        for f in files:
            if f.lower().endswith(".png"):
                ref_paths.append(os.path.join(root, f))
    
    if not ref_paths:
        print("[ERROR] No reference images found in Set 2!")
        return
        
    # Load and combine a few reference images for a more stable histogram
    print(f"[INFO] Building color profile from {len(ref_paths)} reference images...")
    ref_imgs = []
    for p in random.sample(ref_paths, min(5, len(ref_paths))):
        img = cv2.imread(p)
        if img is not None:
            ref_imgs.append(cv2.resize(img, (256, 256)))
    
    # Create a composite reference image to match against
    master_ref = np.hstack(ref_imgs) 

    print(f"[INFO] Matching {args.src} -> {args.dst}")
    print(f"  [Method] Histogram Matching + Mild Blur ({args.blur}) + Grain")
    
    for root, dirs, files in os.walk(args.src):
        for f in files:
            if f.lower().endswith(".png"):
                src_p = os.path.join(root, f)
                rel_p = os.path.relpath(src_p, args.src)
                dst_p = os.path.join(args.dst, rel_p)
                
                os.makedirs(os.path.dirname(dst_p), exist_ok=True)
                
                img = cv2.imread(src_p)
                if img is None: continue
                
                # 1. Histogram Match (Fixes color/brightness perfectly)
                img_matched = histogram_matching(img, master_ref)
                
                # 2. Add mild Gaussian Blur (if requested)
                if args.blur > 1:
                    img_matched = cv2.GaussianBlur(img_matched, (args.blur, args.blur), 0)
                
                # 3. Add fine grain (helps avoid 'plastic' look after blurring)
                noise = np.random.normal(0, args.noise * 255, img_matched.shape).astype(np.float32)
                img_matched = np.clip(img_matched.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                
                cv2.imwrite(dst_p, img_matched)

    print(f"[OK] Produced domain-aligned dataset in {args.dst}")

if __name__ == "__main__":
    main()
