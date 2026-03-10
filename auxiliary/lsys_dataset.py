import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from scipy.io import loadmat
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from auxiliary.lsys_tokenizer import TokenType, NUM_TYPES, GRAMMAR_MATRIX, compute_grammar_mask, apply_grammar_mask


def normalize_to_unit_cube(points):
    min_coord = points.min(axis=0)
    max_coord = points.max(axis=0)
    center = (max_coord + min_coord) / 2.0
    points_centered = points - center
    extent = (max_coord - min_coord).max()
    scale = float(extent) + 1e-6
    normalized = (points_centered / scale) * 2.0
    return normalized.astype(np.float32), center.astype(np.float32), scale

def pad_to_length(arr, length, pad_value):
    arr = np.array(arr)
    if arr.ndim == 1:
        pad = np.full((length - len(arr),), pad_value, dtype=arr.dtype)
        return np.concatenate([arr, pad])
    else:
        pad_value = np.broadcast_to(pad_value, arr.shape[1:])
        pad = np.full((length - len(arr),) + arr.shape[1:], pad_value, dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)

def build_dsm_heightmap(pts_tensor, resolution=64, margin=0.2):
    pts = pts_tensor.cpu().float().numpy()

    if pts.shape[0] == 0 or (pts == 0).all():
        return (
            torch.zeros(resolution, resolution),
            torch.zeros(2),
            torch.ones(2),
        )

    xy = pts[:, :2]
    z  = pts[:, 2]

    xy_min = xy.min(0);  xy_max = xy.max(0)
    span   = xy_max - xy_min
    span   = np.maximum(span, 1e-3)
    xy_min = xy_min - span * margin
    xy_max = xy_max + span * margin
    span   = xy_max - xy_min + 1e-6

    col = np.clip(((xy[:, 0] - xy_min[0]) / span[0] * (resolution - 1)).astype(int), 0, resolution - 1)
    row = np.clip(((xy[:, 1] - xy_min[1]) / span[1] * (resolution - 1)).astype(int), 0, resolution - 1)

    heightmap = np.full((resolution, resolution), -np.inf, dtype=np.float32)
    np.maximum.at(heightmap, (row, col), z)

    valid = heightmap > -1e9
    if valid.any():
        from scipy.ndimage import grey_dilation, distance_transform_edt
        heightmap = grey_dilation(heightmap, size=3)
        valid     = heightmap > -1e9
        if not valid.all():
            ind       = distance_transform_edt(~valid, return_distances=False, return_indices=True)
            heightmap = heightmap[tuple(ind)]
    else:
        heightmap[:] = 0.0

    return (
        torch.from_numpy(heightmap),
        torch.tensor(xy_min.astype(np.float32)),
        torch.tensor(span.astype(np.float32)),
    )

def compute_depth_sequence(type_ids):
    depth = 0
    out = []
    for t in type_ids:
        if t == TokenType.LBR:
            depth += 1
            out.append(depth)
        elif t == TokenType.RBR:
            out.append(depth)
            depth -= 1
        else:
            out.append(depth)
    return out

def compute_bracket_distance(type_ids):
    stack = []
    dist = [0] * len(type_ids)
    for i, t in enumerate(type_ids):
        if t == TokenType.LBR:
            stack.append(i)
        elif t == TokenType.RBR and stack:
            j = stack.pop()
            d = i - j
            dist[j] = d
            dist[i] = d
    return dist

def compute_states_numpy(types, values, tokenizer, init_pos=None):
    T = len(types)
    states = np.zeros((T, 9), dtype=np.float32)
    pos = np.array(init_pos, dtype=np.float32) if init_pos is not None else np.zeros(3, dtype=np.float32)
    H = np.array([0, 0, 1], dtype=np.float32) 
    L = np.array([1, 0, 0], dtype=np.float32) 
    U = np.array([0, 1, 0], dtype=np.float32) 
    stack = []
    
    f_bins = getattr(tokenizer, 'f_bins', 10)
    theta_bins = getattr(tokenizer, 'theta_bins', 6)
    phi_bins = getattr(tokenizer, 'phi_bins', 6)
    
    deg2rad = np.pi / 180.0
    
    for t in range(T):
        states[t, 0:3] = pos
        states[t, 3:6] = H
        states[t, 6:9] = U
        tok = types[t]
        val = values[t]
        
        if tok == TokenType.F:
            # val = [length, theta, phi]
            f_idx, theta_idx, phi_idx = val[0], val[1], val[2]
            
            length = (f_idx + 0.5) / float(f_bins) * 1.0
            theta = (theta_idx + 0.5) / float(theta_bins) * np.pi
            phi = (phi_idx + 0.5) / float(phi_bins) * (2.0 * np.pi)
            
            st, ct = np.sin(theta), np.cos(theta)
            sp, cp = np.sin(phi), np.cos(phi)
            
            new_H = np.array([st * cp, st * sp, ct], dtype=np.float32)
            H = new_H
            
            if abs(H[0]) > 0.9:
                Lv = np.array([0, 1, 0], dtype=np.float32)
            else:
                Lv = np.array([1, 0, 0], dtype=np.float32)
                
            U = np.cross(H, Lv)
            U /= (np.linalg.norm(U) + 1e-9)
            L = np.cross(U, H)
            
            pos = pos + H * length
            
        elif tok == TokenType.LBR:
            stack.append((pos.copy(), H.copy(), L.copy(), U.copy()))
        elif tok == TokenType.RBR and stack:
            if stack:
                pos, H, L, U = stack.pop()
                
    return states

def load_species_map(base_path):

    base_path = '/home/grammatikakis1/TREES_DATASET_SIDE'
    # Try multiple common locations for the species log
    possible_paths = [
        os.path.join(base_path, "species_log_from_mtl.txt"),
        os.path.join(base_path, "TREES", "species_log_from_mtl.txt"),
        os.path.join(os.path.dirname(base_path), "species_log_from_mtl.txt"),
    ]
    
    species_file = None
    for p in possible_paths:
        if os.path.exists(p):
            species_file = p
            break
            
    if not species_file: 
        print(f"[WARN] species_log_from_mtl.txt not found in {base_path} or subfolders.")
        return {}, {}, {}

    entries, species_names = [], []
    with open(species_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2: continue
            if parts[0].lower() in ["tid", "id"]: continue
            if parts[1].lower() in ["species", "sname", "name"]: continue
            
            entries.append((parts[0], parts[1]))
            species_names.append(parts[1])
            
    # Alphabetical sort is key to matching the model's head indices
    unique_species = sorted(list(set(species_names)))
    species_map = {name: i for i, name in enumerate(unique_species)}
    inv_species_map = {i: name for name, i in species_map.items()}
    
    # Debug: Print the actual classes
    print(f"[INFO] Species Classes ({len(unique_species)}): {unique_species}")

    # Normalize IDs for mapping (tree_0001 -> tree_1) to ensure they match DSM filenames
    def normalize_id(tid):
        if "_" not in tid: return tid
        parts = tid.split("_")
        prefix = parts[0]
        try:
            num = int(parts[1])
            return f"{prefix}_{num}"
        except:
            return tid

    tid_to_species = {}
    for tid, sname in entries:
        norm = normalize_id(tid)
        tid_to_species[tid] = species_map.get(sname, -1)
        tid_to_species[norm] = species_map.get(sname, -1)

    return species_map, inv_species_map, tid_to_species

class LSystemDataset(Dataset):
    def __init__(self, base_path, tokenizer, ids, window=1024, overlap=128, preload=True, lstring_dir="LSTRINGS", dsm_dirname="DSM", ortho_dirname="ORTHOPHOTOS", normalize=False, training=False):
        self.base_path = base_path
        self.window = window
        self.tokenizer = tokenizer
        self.training = training
        self.lstring_dir = os.path.join(base_path, lstring_dir)
        self.dsm_dir = os.path.join(base_path, dsm_dirname)
        self.ortho_dir = os.path.join(base_path, ortho_dirname)
        
        # ✅ ORTHO DROPOUT/ADJUSTMENTS: 
        # Only jitter during training to ensure model sees varied lighting.
        # During inference (training=False), we want deterministic loading.
        if self.training:
            self.img_tf = T.Compose([
                T.Resize((224, 224)),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1), 
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_tf = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        self.preload = preload
        self.normalize = normalize
        
        # ✅ NEW: DSM Augmentation parameters
        self.aug_scale = 0.1 # +/- 10%
        self.aug_noise = 0.005 # Jitter 
        self.aug_rot   = True  # Random Z-rotation

        # Dynamically load species map relative to base path
        self.species_map, self.inv_species_map, self.tid_to_species = load_species_map(base_path)
        self.num_species = len(self.species_map) if self.species_map else 1
        self.data = {}
        self.samples = []
        
        step = window - overlap
        if step <= 0: step = window // 2

        for tid in tqdm(ids, desc="Loading L-strings"):
            path = os.path.join(self.lstring_dir, f"{tid}.txt")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        types, vals = [TokenType.F], [[0,0,0]] # Minimal fallback
                    else:
                        types, vals = tokenizer.encode(content)
                
                self.data[tid] = {
                    "types": np.array(types, dtype=np.int64),
                    "vals": np.array(vals, dtype=np.float32),
                    "depth": np.array(compute_depth_sequence(types), dtype=np.float32),
                    "bdist": np.array(compute_bracket_distance(types), dtype=np.float32),
                    "species": self.tid_to_species.get(tid, -1),
                    "masks": compute_grammar_mask(types),
                    "states": None
                }
            else:
                # No L-string available (Inference mode)
                self.data[tid] = {
                    "types": np.array([TokenType.F], dtype=np.int64),
                    "vals": np.array([[0,0,0]], dtype=np.float32),
                    "depth": np.array([0], dtype=np.float32),
                    "bdist": np.array([0], dtype=np.float32),
                    "species": self.tid_to_species.get(tid, -1),
                    "masks": torch.zeros((1, NUM_TYPES)),
                    "states": None
                }
            
            length = len(self.data[tid]["types"])
            if length <= window:
                self.samples.append((tid, 0))
            else:
                for start in range(0, length - window + 1, step):
                    self.samples.append((tid, start))

        self.dsm_cache, self.ortho_cache = {}, {}
        self.dsm_meta = {} 
        self.heightmap_cache = {}
        if preload:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as exc:
                futures = {exc.submit(self._preload_tid, tid): tid for tid in ids if tid in self.data}
                for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Preloading data"):
                    try:
                        tid, dsm, ortho, init_pos = f.result()
                        self.dsm_cache[tid], self.ortho_cache[tid] = dsm, ortho
                        self.data[tid]["states"] = compute_states_numpy(
                            self.data[tid]["types"], 
                            self.data[tid]["vals"], 
                            self.tokenizer,
                            init_pos=init_pos
                        )
                    except Exception as e:
                        print(f"[ERROR] Failed to preload {futures[f]}: {e}")

        for tid in tqdm(self.data, desc="Building DSM heightmaps"):
            pts = self.dsm_cache.get(tid)
            if pts is None:
                pts, root, center, scale = self._load_dsm_with_root(tid)
                self.dsm_cache[tid] = pts
                self.dsm_meta[tid] = (center, scale)
            
            hm, origin, scale_hm = build_dsm_heightmap(pts, resolution=64)
            self.heightmap_cache[tid] = (hm, origin, scale_hm)

    def _preload_tid(self, tid):
        dsm, init_pos, center, scale = self._load_dsm_with_root(tid)
        self.dsm_meta[tid] = (center, scale)
        return tid, dsm, self._load_ortho(tid), init_pos

    def _load_dsm_with_root(self, tid):
        path = os.path.join(self.dsm_dir, f"{tid}.mat")
        if not os.path.exists(path):
            return torch.zeros((1, 3)), np.zeros(3), np.zeros(3), 1.0
        
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
            
            if pts is not None and len(pts) > 0:
                pts_raw = pts.astype(np.float32)
                root_raw = np.zeros(3, dtype=np.float32)
                
                # Rotation: old (x,y,z) -> new (x, -z, y)
                pts_rot = pts_raw.copy()
                pts_rot[:, 1] = -pts_raw[:, 2]
                pts_rot[:, 2] = pts_raw[:, 1]

                # ✅ ROBUST NORMALIZATION: Use percentiles instead of min/max to ignore outliers
                # 1st and 99th percentiles are safer for 'random' DSMs with sensor noise
                p1 = np.percentile(pts_rot, 1, axis=0)
                p99 = np.percentile(pts_rot, 99, axis=0)
                
                center = (p1 + p99) / 2.0
                extent = (p99 - p1).max()
                scale = float(extent) + 1e-6

                # Important: return raw pts_rot so __getitem__ can handle normalization
                root_norm = (root_raw - center) / (scale + 1e-9) * 2.0

                return torch.tensor(pts_rot, dtype=torch.float32), root_norm, center, scale
        except Exception as e:
            print(f"[DEBUG] Error loading {path}: {e}")
            pass
        return torch.zeros((1, 3)), np.zeros(3), np.zeros(3), 1.0

    def _load_dsm(self, tid):
        pts, root, center, scale = self._load_dsm_with_root(tid)
        self.dsm_meta[tid] = (center, scale)
        return pts

    def _normalize_tid_for_ortho(self, tid):
        if "_" not in tid: return tid
        prefix, num_str = tid.split("_")
        try:
            num = int(num_str)
            if num >= 1000: return f"{prefix}_{num}"
            return f"{prefix}_{num:04d}"
        except ValueError:
            return tid

    def _load_ortho(self, tid):
        norm_tid = self._normalize_tid_for_ortho(tid)
        
        # 1. Check direct file (e.g. tree_1.png)
        for t in [tid, norm_tid]:
            direct_file = os.path.join(self.ortho_dir, f"{t}.png")
            if os.path.exists(direct_file):
                try:
                    img = Image.open(direct_file).convert("RGB")
                    return [self.img_tf(img)]
                except Exception:
                    pass

        # 2. Check depth folder (original behavior)
        folder = os.path.join(self.ortho_dir, norm_tid, "rendering")
        if not os.path.exists(folder):
            folder = os.path.join(self.ortho_dir, norm_tid)
            if not os.path.exists(folder):
                # Fallback to plain TID folder
                folder = os.path.join(self.ortho_dir, tid)
                if not os.path.exists(folder):
                    return [torch.zeros(3, 224, 224)]
            
        pngs = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])
        if not pngs: return [torch.zeros(3, 224, 224)]
        
        imgs = []
        for f in pngs[:8]:
            try:
                img = Image.open(os.path.join(folder, f)).convert("RGB")
                imgs.append(self.img_tf(img))
            except Exception:
                continue
        return imgs if imgs else [torch.zeros(3, 224, 224)]

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        tid, start = self.samples[index]
        D = self.data[tid]
        
        t_in = pad_to_length(D["types"][start:start+self.window], self.window, TokenType.PAD)
        v_in = pad_to_length(D["vals"][start:start+self.window], self.window, np.array([0,0,0]))
        t_tgt = pad_to_length(D["types"][start+1:start+self.window+1], self.window, TokenType.PAD)
        v_tgt = pad_to_length(D["vals"][start+1:start+self.window+1], self.window, np.array([0,0,0]))
        
        dsm = self.dsm_cache.get(tid)
        if dsm is None: dsm = self._load_dsm(tid)
        
        # Consistent sampling: 2500 points matching high-capacity design
        num_points = 2500
        if len(dsm) >= num_points:
            idx = np.random.choice(len(dsm), num_points, replace=False)
            dsm_sub = dsm[idx]
        elif len(dsm) > 0:
            idx = np.random.choice(len(dsm), num_points, replace=True)
            dsm_sub = dsm[idx]
        else:
            dsm_sub = torch.zeros((num_points, 3))
        
        dsm_raw = dsm_sub.float().clone().detach()
        # Ensure exact shape for the encoder
        if dsm_raw.shape[0] != num_points:
            new_dsm = torch.zeros((num_points, 3), dtype=torch.float32)
            n_copy = min(len(dsm_raw), num_points)
            if n_copy > 0:
                new_dsm[:n_copy] = dsm_raw[:n_copy]
            dsm_raw = new_dsm
        
        center, scale = self.dsm_meta.get(tid, (np.zeros(3), 1.0))
        dsm_norm = (dsm_raw - torch.tensor(center, device=dsm_raw.device).float()) / (scale + 1e-9) * 2.0
        
        # ✅ STRICT NORMALIZATION Mode (for cross-sensor consistency)
        if self.normalize and dsm_norm.shape[0] > 0:
            # 1. DSM: Force strictly into [-1, 1] unit cube based on THIS sample's bounds
            # This makes the model invariant to absolute scale errors in foreign datasets
            p_min = dsm_norm.min(dim=0)[0]
            p_max = dsm_norm.max(dim=0)[0]
            p_center = (p_min + p_max) / 2.0
            p_extent = (p_max - p_min).max()
            dsm_norm = (dsm_norm - p_center) / (p_extent / 2.0 + 1e-9)
        
        
        # ✅ DSM AUGMENTATION (Training only)
        if hasattr(self, 'aug_scale') and self.training and dsm_norm.shape[0] > 0:
            # 1. Scaling
            s = 1.0 + (random.random() * 2 - 1) * self.aug_scale
            dsm_norm = dsm_norm * s
            
            # 2. Jitter (Simulate sensor noise)
            noise = torch.randn_like(dsm_norm) * self.aug_noise
            dsm_norm = dsm_norm + noise
            
            # 3. Random Rotation around Vertical (Z) axis
            if self.aug_rot:
                angle = random.random() * 2 * np.pi
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                # In our rot: X is horizontal, Y is depth, Z is vertical
                # Rotate X,Y around Z
                rot_mat = torch.tensor([
                    [cos_a, -sin_a, 0],
                    [sin_a,  cos_a, 0],
                    [0, 0, 1]
                ], dtype=torch.float32)
                dsm_norm = dsm_norm @ rot_mat.T
        
        
        orthos = self.ortho_cache.get(tid)
        if orthos is None: orthos = self._load_ortho(tid)
        ortho = random.choice(orthos)
        
        # 2. Ortho: Per-sample standardization (ignore global lighting/exposure)
        if self.normalize:
            # We already have ImageNet norm, but this replaces it with 
            # "Local Mean/Std" normalization to handle foreign sensors.
            eps = 1e-6
            c, h, w = ortho.shape
            o_flat = ortho.view(c, -1)
            o_mean = o_flat.mean(dim=1, keepdim=True).view(c, 1, 1)
            o_std  = o_flat.std(dim=1, keepdim=True).view(c, 1, 1)
            ortho = (ortho - o_mean) / (o_std + eps)
        
        
        hm, hm_origin, hm_scale = self.heightmap_cache.get(
            tid,
            (torch.zeros(64, 64), torch.zeros(2), torch.ones(2))
        )

        return {
            "type_in": torch.tensor(t_in, dtype=torch.long),
            "val_in": torch.tensor(v_in, dtype=torch.long),
            "type_tgt": torch.tensor(t_tgt, dtype=torch.long),
            "val_tgt": torch.tensor(v_tgt, dtype=torch.long),
            "states": torch.tensor(pad_to_length(D["states"][start:start+self.window], self.window, 0.0), dtype=torch.float32),
            "states_tgt": torch.tensor(pad_to_length(D["states"][start+1:start+self.window+1], self.window, 0.0), dtype=torch.float32),
            "dsm": dsm_norm,
            "dsm_raw": dsm_raw,
            "ortho": ortho,
            "tid": tid,
            "species": torch.tensor(D["species"], dtype=torch.long),
            "heightmap": hm,
            "hm_origin": hm_origin,
            "hm_scale":  hm_scale,
            "dsm_center": torch.tensor(center, dtype=torch.float32),
            "dsm_scale":  torch.tensor(scale, dtype=torch.float32),
        }