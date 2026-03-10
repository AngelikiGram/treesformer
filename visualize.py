# visdom_lsystem_visualization.py

import numpy as np
# Monkeypatch for "module 'numpy' has no attribute 'flexible'" error in older Visdom versions
if not hasattr(np, 'flexible'):
    # np.flexible was removed in NumPy 1.20; it was a superclass for np.character and np.void
    # We define it as a dummy class or alias to prevent AttributeError
    np.flexible = (np.void, np.character) if hasattr(np, 'character') else np.void

import visdom
import torch
import re
from auxiliary.lsys_renderer import render_lsystem

class LSystemVisdom:
    def __init__(self, env="gpt_model_lsystems", port=8099):
        self.env = env
        self.port = port
        self.viz = visdom.Visdom(port=port, env=env)
        # self.viz.close() # Keep windows on restart

        # windows
        self.win_total_loss = None
        self.win_train_detail = None
        self.win_val_detail = None
        self.loss_win = None
        self.tree_win = None
        self.tree_win_full = None
        self.text_compare_win = None
        self.gt_tree_win = None
        self.pred_tree_win = None
        self.dsm_win = None
        self.skel_win = None
        self.ortho_win = None
        self.pc_win = None
        self.loss_geom_win = None
        self.vox_win = None
        self.diff_pc_win = None
        self.diff_pc_win = None
        self.diff_geom_win = None
        self.state_debug_win = None
        self.viz_rotate = False # 🔴 Disabled: Do NOT apply any rotations for visualization


    # ============================================================
    # HELPER — fix any array shape to [N,3]
    # ============================================================
    def ensure_xyz(self, arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        arr = np.array(arr)

        if arr is None or arr.size == 0:
            return np.zeros((0, 3))

        if arr.ndim == 0: # Handle scalar numpy array
             return np.zeros((0, 3))

        if arr.ndim == 1:
            padded = np.zeros(3)
            padded[:min(3, len(arr))] = arr[:3]
            res = padded.reshape(1, 3).astype(np.float32)
        elif arr.ndim == 2:
            N, D = arr.shape
            if D == 3:
                res = arr.astype(np.float32)
            elif D > 3:
                res = arr[:, :3].astype(np.float32)
            else: # D < 3
                pad = np.zeros((N, 3 - D))
                res = np.concatenate([arr, pad], axis=1).astype(np.float32)
        else:
            res = arr.reshape(-1, 3).astype(np.float32)

        # Shift all 3D coordinates soZ=0 is the ground (since model uses center-based norm)
        if res.shape[1] >= 3:
            res[:, 2] += 1.0
            
        return res

    # ============================================================
    # ROTATION HELPER
    # ============================================================
    def rotate_y_to_z(self, pts):
        # Rotates so that old Y becomes new Z
        # (x, y, z) -> (x, -z, y)
        if pts.shape[1] < 3: return pts
        new_pts = np.zeros_like(pts)
        new_pts[:, 0] = pts[:, 0]
        new_pts[:, 1] = -pts[:, 2]
        new_pts[:, 2] = pts[:, 1]
        return new_pts

    def rotate_z_to_y(self, pts):
        # Rotates so that old Z becomes new Y
        # (x, z, y) -> (x, y, -z)
        if pts.shape[1] < 3: return pts
        new_pts = np.zeros_like(pts)
        new_pts[:, 0] = pts[:, 0]
        new_pts[:, 1] = pts[:, 2]
        new_pts[:, 2] = -pts[:, 1]
        return new_pts

    def rotate_90_z(self, pts):
        # Rotate 90 degrees around Z axis: (x, y, z) -> (-y, x, z)
        pts = self.ensure_xyz(pts)
        if pts.shape[0] == 0: return pts
        new_pts = np.zeros_like(pts)
        new_pts[:, 0] = -pts[:, 1]
        new_pts[:, 1] = pts[:, 0]
        new_pts[:, 2] = pts[:, 2]
        return new_pts

    # ============================================================
    # LOSS PLOT
    # ============================================================
    # ============================================================
    # LOSS PLOT (Multi-line)
    # ============================================================
    def plot_losses(self, step, losses_dict):
        """ Divide and conquer losses into 3 windows for better visibility. """
        if not losses_dict: return
        
        # 1. Total (Train vs Val Total) + LR
        total_group = {
            "Train Total": losses_dict.get("Train Total", 0),
            "Val Total": losses_dict.get("Val Total", 0),
        }
        if "LR" in losses_dict:
             total_group["LR (x1000)"] = losses_dict["LR"] * 1000 # Scaling for visibility

        self._plot_subgroup(step, total_group, "win_total_loss", "1. Total Losses & LR")
        
        # 2. Train Details (everything else starting with 'Train')
        train_group = {k.replace("Train ", ""): v for k, v in losses_dict.items() if k.startswith("Train ") and "Total" not in k}
        self._plot_subgroup(step, train_group, "win_train_detail", "2. Train Details")
        
        # 3. Val Details (everything else starting with 'Val')
        val_group = {k.replace("Val ", ""): v for k, v in losses_dict.items() if k.startswith("Val ") and "Total" not in k}
        self._plot_subgroup(step, val_group, "win_val_detail", "3. Val Details")

        if "LR" in losses_dict:
            lr_group = {"Learning Rate": losses_dict["LR"]}
            self._plot_subgroup(step, lr_group, "win_lr", "5. Learning Rate")

    def _plot_subgroup(self, step, group_dict, win_attr, title):
        if not group_dict: return
        
        labels = list(group_dict.keys())
        values = []
        for v in group_dict.values():
            if hasattr(v, 'item'): values.append(v.item())
            else: values.append(float(v))
            
        M = len(labels)
        X = np.column_stack([np.array([step])] * M)
        Y = np.array([values]).reshape(1, M)
        
        win = getattr(self, win_attr, None)
        
        if win is None:
            win = self.viz.line(
                X=X, Y=Y,
                opts=dict(title=title, showlegend=True, legend=labels, xlabel="Epoch", ylabel="Loss")
            )
            setattr(self, win_attr, win)
        else:
            self.viz.line(
                X=X, Y=Y, win=win, update='append',
                opts=dict(legend=labels)
            )

    # ============================================================
    # TEXT — combined GT + Pred + Template window
    # ============================================================
    def show_lstring_triple(self, step, gt_text, pred_text, template_text=None, max_len=500):
        if gt_text is None: gt_text = "N/A"
        if pred_text is None: pred_text = "N/A"
        if template_text is None: template_text = "N/A"

        def clean_text(t):
            if len(t) > max_len:
                return t[:max_len] + " ..."
            return t

        gt_text = clean_text(gt_text)
        pred_text = clean_text(pred_text)
        template_text = clean_text(template_text)

        def get_stats(t):
            if not t or not isinstance(t, str):
                return 0, 0
            try:
                # Count segments: B...F..., S..._..., s..._...
                # We look for the start char followed by numbers and underscores/F
                segs = re.findall(r"[BSbs]\d+_\d+[F_]?\d+", t)
                seg_count = len(segs)
                # Count brackets
                bracket_count = t.count("[") + t.count("]")
                return seg_count + bracket_count, len(t)
            except Exception:
                return 0, len(str(t))

        gt_toks, gt_chars = get_stats(gt_text)
        pred_toks, pred_chars = get_stats(pred_text)
        temp_toks, temp_chars = get_stats(template_text)

        html = f"""
        <h2>L-System Comparison — Step {step}</h2>

        <h3 style=\"color:#4b0082;\">Template L-String (The Prompt) (tokens: {temp_toks}, chars: {temp_chars})</h3>
        <div style=\"background:#f0e6ff; padding:10px; border-radius:6px;
                font-family:monospace; max-height:150px; overflow-y:auto;\">
        {template_text}
        </div>

        <h3 style=\"color:#006400;\">Ground Truth L-String (Target) (tokens: {gt_toks}, chars: {gt_chars})</h3>
        <div style=\"background:#e8ffe8; padding:10px; border-radius:6px;
                font-family:monospace; max-height:150px; overflow-y:auto;\">
        {gt_text}
        </div>

        <h3 style=\"color:#00008b;\">Predicted L-String (Refined) (tokens: {pred_toks}, chars: {pred_chars})</h3>
        <div style=\"background:#e8e8ff; padding:10px; border-radius:6px;
                font-family:monospace; max-height:150px; overflow-y:auto;\">
        {pred_text}
        </div>
        """

        if self.text_compare_win is None:
            self.text_compare_win = self.viz.text(
                html,
                opts=dict(title="Template vs GT vs Predicted", width=900, height=800)
            )
        else:
            self.viz.text(html, win=self.text_compare_win)

    # ============================================================
    # 3D POINT CLOUD — with clickable legend (GT + Pred)
    # ============================================================
    def show_gt_and_pred(self, gt_pts, pred_pts, template_pts=None, title="Structure: GT vs Pred", win=None):
        # Convert to Nx3 arrays
        gt_np = self.ensure_xyz(gt_pts)
        pred_np = self.ensure_xyz(pred_pts)
        
        # Combine arrays
        if template_pts is not None:
            temp_np = self.ensure_xyz(template_pts)
            X = np.vstack([gt_np, pred_np, temp_np])
            Y = np.concatenate([
                np.ones(len(gt_np)),
                np.ones(len(pred_np)) * 2,
                np.ones(len(temp_np)) * 3
            ])
            legend = ["GT (Target)", "Pred (Refined)", "Template (Input)"]
        else:
            X = np.vstack([gt_np, pred_np])
            Y = np.concatenate([
                np.ones(len(gt_np)),
                np.ones(len(pred_np)) * 2
            ])
            legend = ["GT", "Pred"]

        opts = dict(
            title=title,
            markersize=4,
            legend=legend,
            projection='3d',
            xlabel="x",
            ylabel="y",
            zlabel="z"
        )

        if self.viz_rotate:
            X = self.rotate_90_z(X)

        if win is not None:
            self.viz.scatter(X=X, Y=Y, opts=opts, win=win)
            return win

        if self.tree_win is None:
            self.tree_win = self.viz.scatter(X=X, Y=Y, opts=opts)
        else:
            self.viz.scatter(X=X, Y=Y, opts=opts, win=self.tree_win)

    def show_gt_and_pred_full(self, gt_pts, pred_pts, title="GT vs Pred Tree"):
        gt_np = self.ensure_xyz(gt_pts)
        pred_np = self.ensure_xyz(pred_pts)

        # Rotate Y -> Z
        gt_np = self.rotate_90_z_to_y(gt_np)
        pred_np = self.rotate_90_z_to_y(pred_np)

        X = np.vstack([gt_np, pred_np])

        Y = np.concatenate([
            np.ones(len(gt_np)),
            np.ones(len(pred_np)) * 2
        ])

        opts = dict(
            title=title,
            markersize=3,
            legend=["GT", "Pred"],
            xlabel="x",
            ylabel="z (was -y)",
            zlabel="y (was z)"
        )

        if self.tree_win_full is None:
            self.tree_win_full = self.viz.scatter(
                X=X,
                Y=Y,
                opts=opts
            )
        else:
            self.viz.scatter(
                X=X,
                Y=Y,
                win=self.tree_win_full,
                opts=opts
            )

    # ============================================================
    # SIMPLE point cloud (GT or Pred separately)
    # ============================================================
    def show_pointcloud(self, pts, title="Point Cloud", win=None):
        if self.viz_rotate:
            pts = self.rotate_90_z(pts)
            
        if win is not None:
             self.viz.scatter(
                X=pts,
                win=win,
                opts=dict(title=title, markersize=2, projection='3d')
            )
             return win

        if self.pc_win is None:
            self.pc_win = self.viz.scatter(
                X=pts,
                opts=dict(title=title, markersize=2, projection='3d')
            )
        else:
            self.viz.scatter(
                X=pts,
                win=self.pc_win,
                opts=dict(title=title, markersize=2)
            )
        return self.pc_win

    def plot_lsystem(self, lstring, title="L-System Tree"):
        """ Renders an L-string and plots it as a point cloud. """
        pts = render_lsystem(lstring)
        if pts is not None and len(pts) > 0:
            pts = self.ensure_xyz(pts)
            # NO ROTATION: Renderer is already Z-up
            self.viz.scatter(
                X=pts,
                win="lsystem_plot",
                opts=dict(title=title, markersize=2)
            )

    # -----------------------------------------------------------
    # Show orthophotos
    # -----------------------------------------------------------
    def show_orthophotos(self, ortho, title="Orthophotos", win=None):
        if isinstance(ortho, np.ndarray):
            ortho = torch.tensor(ortho)
        ortho = ortho.float()
        
        if ortho.ndim == 4:
            ortho = ortho[0]
        elif ortho.ndim == 3:
            pass
        elif ortho.ndim == 2:
            ortho = ortho.unsqueeze(0).repeat(3,1,1)
        else:
            print("Bad orthophoto shape:", ortho.shape)
            return

        # Check if normalization seems to be ImageNet (mean ~0, std ~1 range [-2..2])
        # If min is significantly < 0, implies normalization.
        if ortho.min() < -0.1:
            mean = torch.tensor([0.485, 0.456, 0.406], device=ortho.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=ortho.device).view(3, 1, 1)
            ortho = ortho * std + mean
            
        if self.viz_rotate:
            ortho = torch.rot90(ortho, k=1, dims=[1, 2])
            
        ortho = ortho.clamp(0, 1)

        # Convert to numpy explicitly for Visdom to avoid type check errors
        img_np = ortho.cpu().detach().numpy()

        if win is not None:
            self.viz.image(img_np, win=win, opts=dict(title=title))
            return win

        if self.ortho_win is None:
            self.ortho_win = self.viz.image(
                img_np,
                opts=dict(title=title)
            )
        else:
            self.viz.image(
                img_np,
                win=self.ortho_win,
                opts=dict(title=title)
            )

    def rotate_90_z_to_y(self, pts):
        """
        Rotate point cloud 90 degrees:
            old (x, y, z) → new (x, z, -y)

        This makes the Z-axis become Y.
        """
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] < 3:
            return pts

        out = np.zeros_like(pts)
        out[:, 0] = pts[:, 0]
        out[:, 1] = pts[:, 2]      # new y = old z
        out[:, 2] = -pts[:, 1]     # new z = -old y (Standard 90 deg rotation around X)
        return out

    # -----------------------------------------------------------
    # Show DSM as a full 3D point cloud
    # -----------------------------------------------------------
    def show_dsm(self, dsm, title="DSM (3D Point Cloud)", win=None):
        pts = self.ensure_xyz(dsm)
        # NO ROTATION: DSM is already in world-space (Z-up)

        opts = dict(
            title=title,
            markersize=2,
            legend=["DSM"],
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            projection='3d'
        )

        if self.viz_rotate:
            pts = self.rotate_90_z(pts)

        if win is not None:
             self.viz.scatter(
                X=pts,
                Y=np.ones(len(pts)),
                win=win,
                opts=opts
            )
             return win

        if self.dsm_win is None:
            self.dsm_win = self.viz.scatter(
                X=pts,
                Y=np.ones(len(pts)),
                opts=opts
            )
        else:
            self.viz.scatter(
                X=pts,
                Y=np.ones(len(pts)),
                win=self.dsm_win,
                opts=opts
            )

    # -----------------------------------------------------------
    # Show text windows
    # -----------------------------------------------------------
    def show_text(self, txt, title="Text"):
        self.viz.text(f"<pre>{txt}</pre>", opts=dict(title=title))
        
    # ============================================================
    # LOSS GEOMETRY — DSM vs Predicted Nodes
    # ============================================================
    def show_loss_geometry(self, dsm_pts, pred_nodes, title="Loss Geometry: DSM (Blue) vs Pred (Red)"):
        # DSM -> Blue (1), Pred -> Red (2)
        dsm_np = self.ensure_xyz(dsm_pts)
        pred_np = self.ensure_xyz(pred_nodes)
        
        # NO ROTATION: standardized on Z-up
        
        X = np.vstack([dsm_np, pred_np])
        Y = np.concatenate([
            np.ones(len(dsm_np)),     # 1 = Blue
            np.ones(len(pred_np)) * 2 # 2 = Red
        ])
        
        opts = dict(
            title=title,
            markersize=3,
            legend=["DSM (Target)", "Pred Nodes (Source)"],
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            projection='3d'
        )
        
        if self.viz_rotate:
            X = self.rotate_90_z(X)
        
        if self.loss_geom_win is None:
            self.loss_geom_win = self.viz.scatter(
                X=X, Y=Y, opts=opts
            )
        else:
            self.viz.scatter(
                X=X, Y=Y, win=self.loss_geom_win, opts=opts
            )

    # ============================================================
    # DIFFERENTIABLE RENDER — visualized during training
    # ============================================================
    def show_diff_vs_gt(self, gt_pts, diff_pts, diff_mask=None, title="Chamfer Loss: GT PC (Blue) vs Soft Pred (Red)"):
        # GT -> Blue (1), Diff -> Red (2)
        gt_np = self.ensure_xyz(gt_pts)
        diff_np = self.ensure_xyz(diff_pts)
        
        if diff_mask is not None:
            # Filter by mask to show what the loss "sees"
            m = self.ensure_xyz(diff_mask).flatten()
            if len(m) == len(diff_np):
                diff_np = diff_np[m > 0.1]
        
        if len(diff_np) == 0:
            diff_np = np.zeros((1, 3))

        X = np.vstack([gt_np, diff_np])
        Y = np.concatenate([
            np.ones(len(gt_np)),     # 1 = Blue
            np.ones(len(diff_np)) * 2 # 2 = Red
        ])
        
        opts = dict(
            title=title,
            markersize=3,
            legend=["GT PC (Target)", "Soft Pred (Source)"],
            xlabel="X", ylabel="Y", zlabel="Z",
            projection='3d'
        )
        
        if self.viz_rotate:
            X = self.rotate_90_z(X)
        
        if self.diff_geom_win is None:
            self.diff_geom_win = self.viz.scatter(X=X, Y=Y, opts=opts)
        else:
            self.viz.scatter(X=X, Y=Y, win=self.diff_geom_win, opts=opts)

    def show_diff_vs_gt_adaptive(self, gt_pts, diff_pts, diff_mask=None, title="Chamfer Loss: GT PC (Blue) vs Soft Pred (Red)"):
        # GT -> Blue (1), Diff -> Red (2)
        gt_np = self.ensure_xyz(gt_pts)
        diff_np = self.ensure_xyz(diff_pts)
        
        if diff_mask is not None:
            # Filter by mask to show what the loss "sees"
            m = self.ensure_xyz(diff_mask).flatten()
            if len(m) == len(diff_np):
                diff_np = diff_np[m > 0.1]
        
        if len(diff_np) == 0:
            diff_np = np.zeros((1, 3))

        X = np.vstack([gt_np, diff_np])
        Y = np.concatenate([
            np.ones(len(gt_np)),     # 1 = Blue
            np.ones(len(diff_np)) * 2 # 2 = Red
        ])
        
        opts = dict(
            title=title,
            markersize=3,
            legend=["GT PC (Target)", "Soft Pred (Source)"],
            xlabel="X", ylabel="Y", zlabel="Z"
        )   

        # Always create a new window per call (unique name from title)
        self.viz.scatter(X=X, Y=Y, win=title, opts=opts)

    # -----------------------------------------------------------
    # FULL VISUALIZATION DASHBOARD (GT + Pred + Template)
    # -----------------------------------------------------------
    def visualize(self, ortho, dsm, gt_lstring, pred_lstring, window_size=3072, step=None, gt_pts=None, pred_pts=None, diff_pts=None, diff_mask=None, show_full=False, skel_gt=None, skel_pred=None, vox_pts=None, **kwargs):
        if step is None:
            step = 0

        # 1. Orthophoto
        try:
            if ortho is not None:
                self.show_orthophotos(ortho, title=f"Orthophoto (Step {step})")
        except Exception as e:
            print(f"Orthophoto viz failed: {e}")

        # 2. DSM (3D World Space)
        try:
            if dsm is not None:
                self.show_dsm(dsm, title=f"DSM (World Space)")
        except Exception as e:
            print(f"DSM viz failed: {e}")

        # 3. L-String Text Comparison
        if pred_lstring is not None:
            # Use the new triple function
            self.show_lstring_triple(step, gt_lstring, pred_lstring, template_text=kwargs.get("template_lstring"), max_len=window_size)

        # 4. Geometric Loss: DSM vs Predicted Nodes
        if dsm is not None and pred_pts is not None:
             self.show_loss_geometry(dsm, pred_pts)

        # 5. Chamfer Loss: Ground Truth PC vs Soft Predicted PC
        if gt_pts is not None and diff_pts is not None:
             self.show_diff_vs_gt(gt_pts, diff_pts, diff_mask=diff_mask)

        # 6. Structural Comparison: Rendered GT vs Rendered Pred (+ Template)
        if gt_pts is not None and pred_pts is not None:
            self.show_gt_and_pred(gt_pts, pred_pts, template_pts=kwargs.get("template_pts"), title=f"Structure: GT vs Pred vs Template")

        # 7. Skeleton Comparison (if absolute coords provided)
        if skel_gt is not None and skel_pred is not None:
            self.show_skeletons(skel_gt, skel_pred, title=f"Skeleton Comparison")

        # 8. Full Recursive Rendering (Optional, expensive)
        if show_full:
            try:
                if gt_lstring is not None and gt_lstring != "N/A":
                    gt_render = render_lsystem(gt_lstring)
                else:
                    gt_render = np.zeros((1,3))

                if pred_lstring is not None:   
                    pred_render = render_lsystem(pred_lstring)
                else:
                    pred_render = np.zeros((1,3))

                self.show_gt_and_pred_full(gt_render, pred_render, title=f"Full Recursive Render")
            except Exception as e:
                print(f"Full render viz failed: {e}")

    def show_skeletons(self, skel_gt, skel_pred, title="Skeleton (GT vs Pred)"):
        def to_np(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            return np.asarray(x)

        gt_np   = to_np(skel_gt)
        pred_np = to_np(skel_pred)

        if gt_np.size > 0 and self.viz_rotate: gt_np = self.rotate_90_z(gt_np)
        if pred_np.size > 0 and self.viz_rotate: pred_np = self.rotate_90_z(pred_np)

        X = np.vstack([gt_np, pred_np]) if (gt_np.size > 0 and pred_np.size > 0) else (gt_np if gt_np.size > 0 else pred_np)

        Y = np.concatenate([
            np.ones(len(gt_np)),
            np.ones(len(pred_np))*2
        ])

        opts = dict(
            title=title,
            markersize=4,
            legend=["GT", "Pred"],
            xlabel="x",
            ylabel="y",
            zlabel="z",
            projection="3d" 
        )

        if self.skel_win is None:
            self.skel_win = self.viz.scatter(
                X=X,
                Y=Y,
                opts=opts
            )
        else:
            self.viz.scatter(
                X=X,
                Y=Y,
                win=self.skel_win,
                opts=opts
            )

            self.viz.scatter(
                X=X,
                Y=Y,
                win=self.skel_win,
                opts=opts
            )

    def show_state_debug(self, gt_pts, pred_pts, mask=None, title="State Loss Debug"):
        gt_np = self.ensure_xyz(gt_pts)
        pred_np = self.ensure_xyz(pred_pts)
        
        if mask is not None:
            m = self.ensure_xyz(mask).flatten()
            # Ensure mask length matches points (handle batch/seq flattening)
            if len(m) == len(gt_np):
                gt_np = gt_np[m > 0.5]
                pred_np = pred_np[m > 0.5]
        
        if len(gt_np) == 0: return

        X = np.vstack([gt_np, pred_np])
        Y = np.concatenate([
            np.ones(len(gt_np)),     # 1 = GT (Blue)
            np.ones(len(pred_np)) * 2 # 2 = Pred (Red)
        ])
        
        opts = dict(
            title=title,
            markersize=4,
            legend=["GT Next State", "Pred Next State"],
            xlabel="x", ylabel="y", zlabel="z",
            projection="3d"
        )
        
        if self.state_debug_win is None:
            self.state_debug_win = self.viz.scatter(X=X, Y=Y, opts=opts)
        else:
            self.viz.scatter(X=X, Y=Y, win=self.state_debug_win, opts=opts)

    # ============================================================
    # PARALLEL DECODE DEBUG — L-String + Rendered Points
    # ============================================================
    def show_parallel_debug(self, lstring, pts, gt_pts=None, title="Parallel Decode Debug"):
        # 1. Show Text
        html = f"""
        <h3>Parallel Decoded L-String</h3>
        <div style="background:#fff0f5; padding:10px; border-radius:6px;
                font-family:monospace; max-height:200px; overflow-y:auto;">
        {lstring}
        </div>
        """
        if not hasattr(self, 'parallel_text_win') or self.parallel_text_win is None:
            self.parallel_text_win = self.viz.text(html, opts=dict(title=f"{title} (Text)"))
        else:
            self.viz.text(html, win=self.parallel_text_win)

        # 2. Show Points (Parallel Rendered vs GT)
        if pts is None or len(pts) == 0:
            return

        pts_np = self.ensure_xyz(pts)
        
        # Combine with GT if provided
        if gt_pts is not None:
             gt_np = self.ensure_xyz(gt_pts)
             X = np.vstack([gt_np, pts_np])
             Y = np.concatenate([np.ones(len(gt_np)), np.ones(len(pts_np)) * 2])
             legend = ["GT", "Parallel Pred"]
        else:
             X = pts_np
             Y = np.ones(len(pts_np)) * 1 # Use label 1, since legend has 1 item
             legend = ["Parallel Pred"]

        opts = dict(
            title=f"{title} (Points)",
            markersize=3,
            legend=legend,
            xlabel="x", ylabel="y", zlabel="z"
        )
        
        # We need a new window attribute for this
        if not hasattr(self, 'parallel_pts_win') or self.parallel_pts_win is None:
            self.parallel_pts_win = self.viz.scatter(X=X, Y=Y, opts=opts)
        else:
             self.viz.scatter(X=X, Y=Y, win=self.parallel_pts_win, opts=opts)

    def show_parallel_debug_adaptive(self, lstring, pts, gt_pts=None, title="Parallel Decode Debug"):
        # 1. Show Text — unique window per title
        html = f"""
        <h3>Parallel Decoded L-String</h3>
        <div style="background:#fff0f5; padding:10px; border-radius:6px;
                font-family:monospace; max-height:200px; overflow-y:auto;">
        {lstring}
        </div>
        """
        self.viz.text(html, win=f"{title}_text", opts=dict(title=f"{title} (Text)"))

        # 2. Show Points — unique window per title
        if pts is None or len(pts) == 0:
            return

        pts_np = self.ensure_xyz(pts)
        
        if gt_pts is not None:
             gt_np = self.ensure_xyz(gt_pts)
             X = np.vstack([gt_np, pts_np])
             Y = np.concatenate([np.ones(len(gt_np)), np.ones(len(pts_np)) * 2])
             legend = ["GT", "Parallel Pred"]
        else:
             X = pts_np
             Y = np.ones(len(pts_np)) * 1
             legend = ["Parallel Pred"]

        opts = dict(
            title=f"{title} (Points)",
            markersize=3,
            legend=legend,
            xlabel="x", ylabel="y", zlabel="z"
        )
        
        self.viz.scatter(X=X, Y=Y, win=f"{title}_pts", opts=opts)

    def visualize_inference(self, tid, ortho, dsm, gt_pts, pred_pts):
        """
        Specialized visualization for inference:
        Shows Orthophoto, DSM, and GT vs Pred structure.
        Uses tid in titles.
        """
        # 1. Orthophoto
        if ortho is not None:
            self.show_orthophotos(ortho, title=f"[{tid}] Orthophoto", win=f"[{tid}] Orthophoto")

        # 2. DSM
        if dsm is not None:
            self.show_dsm(dsm, title=f"[{tid}] DSM (Input)", win=f"[{tid}] DSM (Input)")

        # 3. Structure Plot (GT vs Pred)
        has_gt = gt_pts is not None and (torch.is_tensor(gt_pts) and gt_pts.numel() > 0 or not torch.is_tensor(gt_pts) and len(gt_pts) > 0)
        
        if has_gt:
            self.show_gt_and_pred(
                gt_pts, 
                pred_pts, 
                title=f"[{tid}] Structure: GT (Blue) vs Pred (Red)",
                win=f"[{tid}] Structure: GT (Blue) vs Pred (Red)"
            )
        else:
            self.show_pointcloud(
                pred_pts,
                title=f"[{tid}] Structure: Prediction (AR)",
                win=f"[{tid}] Structure: Prediction (AR)"
            )
  