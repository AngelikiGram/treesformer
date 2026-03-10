# ============================================================
#  tokenizer.py — Unified L-System Tokenizer (Type + Values)
# ============================================================

import re
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from scipy.io import loadmat
from PIL import Image
import random
import torchvision.transforms as T
from tqdm import tqdm
import torchvision.models as models


from auxiliary.lsys_renderer import render_lsystem
from visualize import LSystemVisdom
from auxiliary.lsys_tokenizer import LSystemTokenizerV2, TokenType, NUM_TYPES, GRAMMAR_MATRIX, compute_grammar_mask, apply_grammar_mask
from auxiliary.lsys_dataset import LSystemDataset

# Import Chamfer loss for geometry-aware supervision
try:
    from auxiliary.lsys_losses import ChamferLoss, chamfer_distance, fast_chamfer
    CHAMFER_AVAILABLE = True
except ImportError:
    print("[WARNING] ChamferLoss not available - geometry-aware loss disabled")
    CHAMFER_AVAILABLE = False

# Token types and NUM_TYPES are now imported from lsys_dataset


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

# ============================================================
#  Scheduled Sampling & Position-Weighted Loss
# ============================================================

def get_scheduled_sampling_prob(epoch, ramp_start=50, ramp_end=200, max_prob=0.5):
    """
    Returns probability of using model prediction instead of ground truth.

    Schedule:
    - epoch 0-ramp_start: p = 0.0 (pure teacher forcing)
    - epoch ramp_start-ramp_end: p → 0.3 (linear ramp)
    - epoch ramp_end+: p → max_prob (asymptotic)

    Args:
        epoch: Current training epoch
        ramp_start: When to start ramping (epochs with pure teacher forcing before this)
        ramp_end: When to finish ramping to 0.3
        max_prob: Maximum probability (approached after ramp_end)
    """
    if epoch < ramp_start:
        return 0.0
    elif epoch < ramp_end:
        # Linear ramp from 0 to 0.3
        progress = (epoch - ramp_start) / (ramp_end - ramp_start)
        return 0.3 * progress
    else:
        # Asymptotic approach to max_prob
        # From 0.3 → 0.5 over next 200 epochs
        extra = epoch - ramp_end
        return 0.3 + (max_prob - 0.3) * (1 - np.exp(-extra / 200))


def compute_position_weights(seq_len, alpha=0.002):
    """
    Compute position-dependent weights that decay exponentially.
    
    Early tokens get higher weight because errors compound.
    
    w(t) = exp(-alpha * t)
    
    Args:
        seq_len: Sequence length
        alpha: Decay rate (smaller = slower decay)
    
    Returns:
        weights: (seq_len,) tensor
    """
    positions = torch.arange(seq_len, dtype=torch.float32)
    weights = torch.exp(-alpha * positions)
    # Normalize so mean weight = 1.0
    return weights / weights.mean()


def sample_with_grammar_constraints(logits_t, logits_v, prev_type, bracket_depth, 
                                    f_bins, theta_bins, phi_bins, f_at_depth, temperature=1.0, max_depth=20):
    # Sample next token with grammar and bracket-depth constraints.
    # 🔴 FIX: Scheduled sampling must respect the SAME constraints as generation
    
    # Apply grammar mask
    logits_t = apply_grammar_mask(logits_t.clone(), prev_type) # Clone to avoid mutating original logits
    
    # Apply bracket-depth constraints
    if bracket_depth <= 0:
        logits_t[TokenType.RBR] = float('-inf')
    if bracket_depth >= max_depth:
        logits_t[TokenType.LBR] = float('-inf')
    
    # 🔴 STRUCTURAL RULE: Cannot close branch if no F was emitted in it
    if bracket_depth > 0 and bracket_depth < len(f_at_depth) and not f_at_depth[bracket_depth]:
        logits_t[TokenType.RBR] = float('-inf')
    
    # 🔴 EOS Guard: Must have F at root level (depth 0)
    if not f_at_depth[0]:
        logits_t[TokenType.EOS] = float('-inf')
        
    if bracket_depth > 0:
        logits_t[TokenType.EOS] = float('-inf')
    
    # 🔴 MOVEMENT PRESSURE (Exploration): Boost F in 20% of sampling steps
    if random.random() < 0.2:
        logits_t[TokenType.F] += 2.0
    
    # Sample type
    probs_t = F.softmax(logits_t / max(temperature, 1e-6), dim=-1)
    next_t = torch.multinomial(probs_t, 1).item()
    
    # Sample values based on type
    if next_t == TokenType.F:
        # Sample F-length (v[0])
        f_idx = torch.multinomial(F.softmax(logits_v[:f_bins], dim=-1), 1).item()
        
        # d3 / B_F-token mode: Index 1 = Theta, Index 2 = Phi
        # logits_v layout: [f_bins, theta_bins, phi_bins]
        off1 = f_bins + theta_bins
        off2 = f_bins + theta_bins + phi_bins
        
        v1_idx = torch.multinomial(F.softmax(logits_v[f_bins : off1], dim=-1), 1).item()
        v2_idx = torch.multinomial(F.softmax(logits_v[off1 : off2], dim=-1), 1).item()
        next_v = [f_idx, v1_idx, v2_idx]
    else:
        next_v = [0, 0, 0]
    
    # Update state
    new_depth = bracket_depth
    new_f_at_depth = f_at_depth.copy()
    
    if next_t == TokenType.F:
        new_f_at_depth[new_depth] = True
    elif next_t == TokenType.LBR:
        new_depth += 1
        if new_depth < len(new_f_at_depth):
            new_f_at_depth[new_depth] = False # Reset F-found for new level
    elif next_t == TokenType.RBR:
        new_depth = max(0, bracket_depth - 1)
        
    return next_t, next_v, new_depth, new_f_at_depth


def compute_rotation_smoothness_loss(vlog, v_tgt, tgt_types, valid_mask, theta_bins, phi_bins):
    # DEBUG SHAPES
    if not hasattr(compute_rotation_smoothness_loss, "_printed"):
       # print(f"[DEBUG_LOSS] vlog: {vlog.shape}, v_tgt: {v_tgt.shape}, tgt_types: {tgt_types.shape}, valid_mask: {valid_mask.shape}")
        compute_rotation_smoothness_loss._printed = True
        
    B, T = vlog.shape[:2]
    tgt_types = tgt_types[:, :T]
    v_tgt = v_tgt[:, :T]
    valid_mask = valid_mask[:, :T]
    
    # In d3 mode, F tokens also have rotation. 
    # Check if any token targets has non-zero rotation indices.
    # We use v_tgt[..., 1:] instead of mask as a safer approach to avoid shape mismatch.
    is_Rot = (tgt_types == TokenType.F) & valid_mask
        
    is_Rot_prev = torch.cat([
        torch.zeros(B, 1, dtype=torch.bool, device=tgt_types.device),
        is_Rot[:, :-1]
    ], dim=1)
    
    consecutive_Rot = is_Rot & is_Rot_prev
    
    if not consecutive_Rot.any():
        return torch.tensor(0.0, device=vlog.device)
    
    # Compute EXPECTED rotation
    pred_rot = torch.zeros(B, T, 2, device=vlog.device)
    
    # Offsets and bin counts for [Theta, Phi]
    # layout: [theta_bins, phi_bins]
    offsets = [0, theta_bins]
    bins    = [theta_bins, phi_bins]
    
    for i in range(2):
        # vlog is already rotation part [0..theta_bins+phi_bins]
        logits = vlog[:, :, offsets[i]:offsets[i] + bins[i]]
        probs = F.softmax(logits, dim=-1)
        bin_indices = torch.arange(bins[i], dtype=torch.float32, device=vlog.device)
        pred_rot[:, :, i] = (probs * bin_indices).sum(dim=-1)
    
    rot_diff = pred_rot - torch.cat([
        torch.zeros(B, 1, 2, device=vlog.device),
        pred_rot[:, :-1, :]
    ], dim=1)
    
    rot_change = (rot_diff ** 2).sum(dim=-1)
    loss = rot_change[consecutive_Rot].mean()
    
    return loss


def compute_differentiable_turtle_positions(vlog, st_in, t_tgt, f_bins, theta_bins, phi_bins, scale_factor=None, temperature=0.1):
    device = vlog.device
    B, T, _ = vlog.shape
    
    # Detect d3 mode: F tokens with non-zero rotation indices in ground truth or predicted stats
    # Actually, let's just use the appropriate mapping based on detected format.
    is_d3 = (t_tgt == TokenType.F).any()
    
    if is_d3:
        # k = v0: 10 bins -> length = (v+0.5)/10 * 1.0
        # i = v1: 6 bins -> theta = (v+0.5)/6 * 180
        # j = v2: 6 bins -> phi = (v+0.5)/6 * 360
        f_vals = (torch.arange(f_bins, device=device).float() + 0.5) / float(f_bins) * 1.0
        theta_vals = (torch.arange(theta_bins, device=device).float() + 0.5) / (float(theta_bins)) * (math.pi) # 180 deg
        phi_vals   = (torch.arange(phi_bins, device=device).float() + 0.5) / (float(phi_bins)) * (2.0 * math.pi) # 360 deg

    # Standard Soft Mode (Expected Value via Softmax)
    exp_F = (torch.softmax(vlog[..., :f_bins].float() / temperature, dim=-1) * f_vals).sum(-1)   # (B, T)
    vlog_R = vlog[..., f_bins:].float()
    
    if is_d3:
        # vlog_angles layout: [Theta_bins(0), Phi_bins(1)]
        t_off, p_off = 0, theta_bins
        exp_Theta = (torch.softmax(vlog_R[:, :, t_off:t_off + theta_bins] / temperature, dim=-1) * theta_vals).sum(-1)
        exp_Phi   = (torch.softmax(vlog_R[:, :, p_off:p_off + phi_bins] / temperature, dim=-1) * phi_vals).sum(-1)
    else:
        raise RuntimeError("compute_differentiable_turtle_positions: is_d3=False path is unsupported — no F tokens found in t_tgt")

    if scale_factor is not None:
        sf = scale_factor if isinstance(scale_factor, torch.Tensor) \
             else torch.tensor(scale_factor, device=device, dtype=torch.float32)
        exp_F = exp_F * sf.view(B, 1)

    curr_pos = st_in[:, 0, 0:3].float().detach()
    curr_H   = st_in[:, 0, 3:6].float().detach()
    curr_U   = st_in[:, 0, 6:9].float().detach()
    curr_L   = torch.cross(curr_U, curr_H, dim=-1)
    stacks   = [[] for _ in range(B)]
    pred_positions: list = []

    types_cpu = t_tgt.cpu() if torch.is_tensor(t_tgt) else t_tgt

    for t in range(T):
        tok_type = types_cpu[:, t]

        if is_d3:
            # ── d3 Mode: Absolute Spherical Heading ───────────────────────────────
            mask_F = (tok_type == TokenType.F).unsqueeze(-1).float().to(device)
            if mask_F.any():
                theta = exp_Theta[:, t]
                phi   = exp_Phi[:, t]
                st, ct = torch.sin(theta), torch.cos(theta)
                sp, cp = torch.sin(phi), torch.cos(phi)
                new_H = torch.stack([st * cp, st * sp, ct], dim=-1)
                curr_H = curr_H * (1.0 - mask_F) + new_H * mask_F

                batch_L = torch.tensor([1, 0, 0], device=device, dtype=torch.float32).repeat(B, 1)
                alt_L = torch.tensor([0, 1, 0], device=device, dtype=torch.float32).repeat(B, 1)
                use_alt = (torch.abs(curr_H[:, 0]) > 0.9).unsqueeze(-1).float()
                curr_L = batch_L * (1.0 - use_alt) + alt_L * use_alt
                curr_U = F.normalize(torch.cross(curr_H, curr_L, dim=-1), dim=-1)
                curr_L = torch.cross(curr_U, curr_H, dim=-1)

        # ── Movement ──────────────────────────────────────────────────────────
        is_F  = (tok_type == TokenType.F).float().to(device).unsqueeze(-1)
        new_pos = curr_pos + curr_H * exp_F[:, t].unsqueeze(-1)
        curr_pos = new_pos * is_F + curr_pos * (1.0 - is_F)
        pred_positions.append(curr_pos.clone())

        # ── Bracket stack ─────────────────────────────────────────────────────
        needs_bracket = False
        for b in range(B):
            if tok_type[b] in (TokenType.LBR, TokenType.RBR):
                needs_bracket = True
                break
        
        if needs_bracket:
            pos_list = list(curr_pos.unbind(0))
            H_list   = list(curr_H.unbind(0))
            U_list   = list(curr_U.unbind(0))
            L_list   = list(curr_L.unbind(0))

            for b in range(B):
                tok = tok_type[b].item()
                if tok == TokenType.LBR:
                    stacks[b].append((
                        curr_pos[b].detach(), curr_H[b].detach(),
                        curr_U[b].detach(),   curr_L[b].detach(),
                    ))
                elif tok == TokenType.RBR and stacks[b]:
                    bp, bh, bu, bl = stacks[b].pop()
                    pos_list[b], H_list[b], U_list[b], L_list[b] = bp, bh, bu, bl

            curr_pos = torch.stack(pos_list)
            curr_H   = torch.stack(H_list)
            curr_U   = torch.stack(U_list)
            curr_L   = torch.stack(L_list)

    return torch.stack(pred_positions, dim=1)


def compute_soft_angle_loss(vlog, v_tgt, t_tgt, valid_mask, f_bins, theta_bins, phi_bins):    
    # Ensure all inputs have the same length as vlog
    T = vlog.shape[1]
    v_tgt = v_tgt[:, :T]
    t_tgt = t_tgt[:, :T]
    valid_mask = valid_mask[:, :T]
    
    mask_Rot = (t_tgt == TokenType.F) & valid_mask

    if not mask_Rot.any():
        return torch.tensor(0.0, device=vlog.device)

    vlog_R = vlog[..., f_bins:]
    loss = torch.tensor(0.0, device=vlog.device)
    
    # Offsets and bin counts for [Theta, Phi]
    offsets = [0, theta_bins]
    bins    = [theta_bins, phi_bins]
    
    for i in range(2):
        logits = vlog_R[:, :, offsets[i]:offsets[i] + bins[i]]
        probs = F.softmax(logits, dim=-1)
        bin_indices = torch.arange(bins[i], dtype=torch.float32, device=vlog.device)
        pred_soft = (probs * bin_indices).sum(dim=-1)
        
        # In d3 mode with B_F, rotation target starts at index 1: (Length(0), Theta(1), Phi(2))
        t_idx = i + 1
        
        target = v_tgt[:, :, t_idx].float()
        
        diff = torch.abs(pred_soft - target)
        # Only phi (i=1, azimuthal 0→2π) is periodic and wraps.
        # Theta (i=0, polar 0→π) does NOT wrap — bin 0 (up) and bin 5 (down)
        # are maximally different, not neighbours.  Using circular distance
        # here would collapse that error from 5 bins to 1 bin, inverting
        # the gradient signal for large polar-angle mistakes.
        if i == 1:  # phi: circular wrap
            circ_diff = torch.min(diff, float(bins[i]) - diff)
        else:        # theta: plain L1
            circ_diff = diff

        loss += circ_diff[mask_Rot].mean()
        
    return loss

# ============================================================
#  Dataset
# ============================================================

def pad_to_length(arr, length, pad_value):
    arr = np.array(arr)
    if arr.ndim == 1:
        pad = np.full((length - len(arr),), pad_value, dtype=arr.dtype)
        return np.concatenate([arr, pad])
    else:
        pad = np.full((length - len(arr),) + arr.shape[1:], pad_value, dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)


# ============================================================
#  Model Components
# ============================================================

class AttentionPooler(nn.Module):
    """
    Learned attention-based aggregator for global features.
    Provides a more robust global representation than simple max-pooling.
    """
    def __init__(self, dim, heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x: (B, N, D)
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        # Use cross-attention where the query is a learnable token
        out, _ = self.attn(q, x, x)
        return self.ln(out.squeeze(1))

class PointNetBackbone(nn.Module):
    """
    PointNet-like backbone that extracts per-point and global features.
    Provides significantly better structural grounding than a simple MLP.
    """
    def __init__(self, in_dim=3, out_dim=512):
        super().__init__()
        # 1. Local MLP: Extracts basic geometric features per point
        self.local_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        # 2. Global Aggregator: Condenses all points into a global descriptor
        self.global_mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        # 3. Fusion: Every point now 'knows' the whole tree's shape
        # Concatenate local (256) + global (512) = 768
        self.fuse_proj = nn.Linear(768, out_dim)

    def forward(self, x):
        # x: (B, N, 3)
        local_feat = self.local_mlp(x)   # (B, N, 256)
        
        # Max-pool for global context
        global_context = self.global_mlp(local_feat) # (B, N, 512)
        global_pool = torch.max(global_context, dim=1, keepdim=True)[0] # (B, 1, 512)
        
        # Broadcast global to all points
        global_feat = global_pool.expand(-1, x.shape[1], -1) # (B, N, 512)
        
        # Concat and project
        combined = torch.cat([local_feat, global_feat], dim=-1) # (B, N, 768)
        return self.fuse_proj(combined) # (B, N, out_dim)

class FullMultimodalEncoder(nn.Module):
    """
    Encodes the FULL Orthophoto grid and FULL DSM point set.
    Instead of one vector, it returns a 'Memory Bank' of features.
    
    MEMORY FIX 1: Visual bottleneck to reduce cross-attention cost
    - Downsample from 2549 tokens to ~256 tokens
    - 10x reduction in cross-attention memory
    """
    def __init__(self, dim=512, visual_bottleneck=640, visual_dropout=0.1, num_species=30):
        super().__init__()
        self.visual_dropout = visual_dropout
        self.dim = dim
        
        # 0. Domain Adapter: Learnable Pre-processor (Learnable White Balance/Exposure)
        self.domain_adapter = nn.Conv2d(3, 3, 1)
        with torch.no_grad():
            self.domain_adapter.weight.copy_(torch.eye(3).view(3, 3, 1, 1))
            self.domain_adapter.bias.zero_()

        # 1. Ortho Backbone (High-Res tokens)
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.ortho_backbone = nn.Sequential(*list(base.children())[:-3]) 
        self.ortho_head = nn.Linear(256, dim) # Project to model dim
        # Added scale alignment norm
        self.ortho_norm = nn.LayerNorm(dim)
        
        # 2. DSM Backbone (PointNet implementation)
        self.dsm_backbone = PointNetBackbone(in_dim=3, out_dim=dim)
        # Added scale alignment norm
        self.dsm_norm = nn.LayerNorm(dim)
        
        # 3. Global Aggregators (Attention based)
        # ✅ IMPROVEMENT: Attention pooling captures more nuanced tree features than MaxPool
        self.dsm_pooler = AttentionPooler(dim)
        self.ortho_pooler = AttentionPooler(dim)
        
        # 4. Balanced Global Fusion -> Species Head
        self.species_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2), # Increased dropout for better generalization
            nn.Linear(512, 512), # Added deeper head
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, num_species)
        )
        
        # Redundant bottleneck removed in favor of balanced additive fusion
        # Additive fusion happens in model-dim space (dim)
        self.fuse = nn.Linear(dim, dim)
        self.anchor_proj = nn.Linear(dim, dim)
        
        # Internal normalization buffers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.visual_pool = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.visual_bottleneck = visual_bottleneck
        
        self.visual_bottleneck = visual_bottleneck

    def forward(self, dsm_pts, ortho_img):
        B = ortho_img.shape[0]
        device = ortho_img.device
        
        # 0. Check for missing modalities (for train_dsm / train_ortho)
        # We use a small epsilon to detect intentionally zeroed-out batches
        ortho_is_missing = (ortho_img.abs().max() < 1e-5)
        dsm_is_missing = (dsm_pts.abs().max() < 1e-5)
        
        # ✅ NEW: MODALITY DROPOUT (Training only)
        # Prevents over-dependence on Orthophoto for species identification.
        # Forces model to learn identifying species from geometry alone in 20% of batches.
        if self.training and not ortho_is_missing and not dsm_is_missing:
            r = random.random()
            if r < 0.2:
                ortho_is_missing = True # Drop Ortho
            elif r < 0.4:
                dsm_is_missing = True   # Drop DSM
        

        # 1. Process Orthophoto if available
        if not ortho_is_missing:
            # Robust Internal Normalization
            if ortho_img.max() <= 1.01:
                ortho_img = (ortho_img - self.mean) / self.std
            
            # Learnable Domain Adapter
            ortho_img = self.domain_adapter(ortho_img)

            # ResNet Backbone
            o = self.ortho_backbone(ortho_img)
            o = o.flatten(2).transpose(1, 2)
            o = self.ortho_norm(self.ortho_head(o)) # Added norm
        else:
            # Provide zero tokens (14x14 = 196 tokens)
            o = torch.zeros((B, 196, self.dim), device=device)

        # 2. Process DSM if available
        if not dsm_is_missing:
            d = self.dsm_norm(self.dsm_backbone(dsm_pts)) # Added norm
        else:
            d = torch.zeros((B, dsm_pts.shape[1], self.dim), device=device)
        
        # --- ROBUST GLOBAL AGGREGATION ---
        # Instead of raw max, we use attention poolers which can focus on key structural details
        d_global = self.dsm_pooler(d)
        o_global = self.ortho_pooler(o)
        
        if not ortho_is_missing and not dsm_is_missing:
            # Both present: forced equal contribution
            global_feat = (d_global + o_global) / 2.0
        elif not ortho_is_missing:
            global_feat = o_global
        else:
            global_feat = d_global

        # ✅ NEW: Multi-view species prediction
        # Output fused prediction AND individual predictions for robust supervision
        species_logits = self.species_head(global_feat)
        d_logits = self.species_head(d_global)
        o_logits = self.species_head(o_global)

        # --- LOCAL MEMORY BANK (For Structural Generation) ---
        num_ortho = o.shape[1]
        num_dsm_target = max(1, self.visual_bottleneck - num_ortho)
        stride_d = max(1, d.shape[1] // num_dsm_target)
        d_sub = d[:, ::stride_d, :]
        
        visual_memory = torch.cat([o, d_sub], dim=1)
        visual_memory = self.fuse(visual_memory)
        
        # Prepend a global "anchor token" so the decoder can attend to the overall scene context.
        # ✅ STOCHASTIC ANCHOR DROPOUT: Randomly zero out the species anchor during training.
        # This forces the transformer to extract geometry from raw DSM/Ortho tokens
        # instead of lazily relying on the "Species Template" anchor.
        anchor = self.anchor_proj(global_feat).unsqueeze(1)  # (B, 1, dim)
        
        if self.training:
            # Drop the anchor entirely for 20% of the batch to force geometry-dependence
            anchor_mask = (torch.rand(B, 1, 1, device=device) > 0.2).float()
            anchor = anchor * anchor_mask
            
            # Add a tiny bit of Gaussian noise to the anchor to encourage structural diversity
            # within the same species (prevents template collapse)
            anchor = anchor + torch.randn_like(anchor) * 0.02

        visual_memory = torch.cat([anchor, visual_memory], dim=1)
        visual_memory = visual_memory + self.visual_pool(visual_memory)

        # Visual dropout (training only): drops the ENTIRE memory bank
        if self.training and self.visual_dropout > 0.0:
            keep = (torch.rand(B, device=device) >= self.visual_dropout).float()  # (B,)
            visual_memory = visual_memory * keep.view(B, 1, 1)

        return visual_memory, global_feat, species_logits, d_logits, o_logits


# ============================================================
# Rotary Position Embeddings (RoPE)
# ============================================================

class RotaryEmbedding(nn.Module):
    """
    RoPE: Rotary Position Embeddings
    - Length-agnostic (extrapolates to any sequence length)
    - No learned parameters
    - Applied in attention mechanism
    """
    def __init__(self, dim, max_seq_len=16384, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for cos/sin values
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len, device):
        """Precompute cos/sin up to seq_len. Only recomputes when the table needs to grow."""
        if self._seq_len_cached is None or seq_len > self._seq_len_cached:
            new_len = max(seq_len, self.max_seq_len)
            t = torch.arange(new_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]
            self._seq_len_cached = new_len
        return self._cos_cached, self._sin_cached
    
    def rotate_half(self, x):
        """Helper function to rotate half the hidden dims"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q, k, seq_len, pos_offset=0):
        """
        Apply rotary embeddings to query and key tensors
        q, k: (B, H, T, D_head)
        seq_len: total sequence length (including any cached tokens)
        pos_offset: starting position for the current tokens (for KV caching)
        """
        cos, sin = self._update_cache(seq_len, q.device)
        
        # Extract positions for current tokens only
        # cos/sin shape: (1, seq_len, 1, D_head)
        # If pos_offset=100 and T=5, we want positions [100, 101, 102, 103, 104]
        T = q.shape[2]  # Current sequence length
        
        # Slice along the sequence dimension (dim=1)
        cos_slice = cos[:, pos_offset:pos_offset+T, :, :]  # (1, T, 1, D_head)
        sin_slice = sin[:, pos_offset:pos_offset+T, :, :]  # (1, T, 1, D_head)
        
        # Reshape for proper broadcasting: (1, T, 1, D_head) -> (1, 1, T, D_head)
        # This broadcasts correctly with q, k: (B, H, T, D_head)
        cos_slice = cos_slice.transpose(1, 2)  # (1, 1, T, D_head)
        sin_slice = sin_slice.transpose(1, 2)  # (1, 1, T, D_head)
        
        # Apply rotation
        # q, k: (B, H, T, D_head)
        # cos_slice, sin_slice: (1, 1, T, D_head) -> broadcasts to (B, H, T, D_head)
        q_embed = (q * cos_slice) + (self.rotate_half(q) * sin_slice)
        k_embed = (k * cos_slice) + (self.rotate_half(k) * sin_slice)
        
        return q_embed, k_embed


class CrossAttentionBlock(nn.Module):
    """
    The core block that 're-looks' at the images/points.
    
    FIX 1: Restrict cross-attention to last K tokens only
    - Dramatically reduces memory from O(T × 2500) to O(K × 256)
    - Visual geometry doesn't change, so we don't need full-sequence grounding
    
    FIX 2: Self-attention now uses KVAttention with RoPE
    - Provides positional signal without learned embeddings
    - Enables extrapolation to longer sequences
    
    FIX 3: Supports KV caching for fast generation
    - Can process single token with cached states
    
    MEMORY FIX 2: Cross-attention only every N layers
    - Reduces redundant visual grounding
    - 50% reduction in cross-attention overhead
    
    MEMORY FIX 3: Cache visual K/V once (static across generation)
    - Reuse precomputed visual keys/values
    - No recomputation during generation
    """
    def __init__(self, dim, heads, cross_attn_window=256, rope=None, enable_cross_attn=True):
        super().__init__()
        self.cross_attn_window = cross_attn_window  # Only attend last K tokens
        self.enable_cross_attn = enable_cross_attn
        
        self.norm1 = nn.LayerNorm(dim)
        # CRITICAL FIX: Use KVAttention with RoPE instead of nn.MultiheadAttention
        self.self_attn = KVAttention(dim, heads, rope=rope)
        
        if enable_cross_attn:
            self.norm2 = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # MEMORY FIX 3: Cache visual K/V (computed once, reused forever)
        self.visual_kv_cache = None

    def forward(self, x, visual_memory, self_mask=None, use_visual_cache=False):
        B, T, D = x.shape
        
        # 1. Self-Attention with RoPE (What have I generated so far?)
        # MEMORY FIX 4: Avoid allocating float mask every time
        if self_mask is not None:
            # ✅ FIX: Safe mask conversion (avoid True * -inf → NaN)
            attn_mask = torch.zeros_like(self_mask, dtype=x.dtype)
            attn_mask = attn_mask.masked_fill(self_mask, float('-inf'))
            # Broadcast attn_mask to (B, H, T, T_all) for attention
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T_all)
        else:
            attn_mask = None
        
        h, _ = self.self_attn(self.norm1(x), attn_mask=attn_mask, kv_cache=None)
        x = x + h
        
        # 2. Cross-Attention (How does this match the actual tree shape?)
        # MEMORY FIX 2: Skip if disabled for this layer
        if not self.enable_cross_attn:
            # Skip cross-attention, proceed to FFN
            x = x + self.mlp(self.norm3(x))
            return x
        
        # FIX 1: Only cross-attend with the LAST K tokens
        # - Reduces memory from O(T × 2500) to O(K × 256)
        # - Visual geometry is static, early tokens don't need re-grounding
        K = min(self.cross_attn_window, T)
        
        if K < T:
            # Only compute cross-attention for last K tokens
            q = self.norm2(x[:, -K:, :])  # (B, K, D)
            h2, _ = self.cross_attn(query=q, 
                                    key=visual_memory, 
                                    value=visual_memory)
            # MEMORY FIX 5: Create new tensor (avoid in-place modification)
            # Use concatenation to avoid gradient issues
            x = torch.cat([
                x[:, :-K, :],           # Keep early tokens unchanged
                x[:, -K:, :] + h2       # Add cross-attention to last K tokens
            ], dim=1)
        else:
            # Sequence shorter than window, use full cross-attention
            h2, _ = self.cross_attn(query=self.norm2(x), 
                                    key=visual_memory, 
                                    value=visual_memory)
            x = x + h2
        
        # 3. Feed Forward
        x = x + self.mlp(self.norm3(x))
        return x
    
    def forward_with_cache(self, x, visual_memory, self_mask=None, kv_cache=None, pos_offset=0):
        """
        Forward pass with KV cache support for Truncated BPTT
        
        Args:
            x: (B, T_chunk, D) input embeddings for current chunk
            visual_memory: (B, N, D) visual features
            self_mask: Optional causal mask
            kv_cache: (k_cache, v_cache) from previous chunk, or None
            pos_offset: Starting position for RoPE
            
        Returns:
            x_out: (B, T_chunk, D) output features
            new_cache: (k_cache, v_cache) for next chunk
        """
        B, T, D = x.shape
        # 1. Self-Attention with KV caching (CRITICAL FOR TRUNCATED BPTT)
        # MEMORY FIX 4: Avoid allocating float mask every time
        if self_mask is not None:
            # Determine T_all (total length including cache)
            if kv_cache is not None and kv_cache[0] is not None:
                T_all = kv_cache[0].shape[2] + T  # cached + current
            else:
                T_all = T
            # Create mask of shape (T, T_all)
            attn_mask = torch.zeros((T, T_all), dtype=x.dtype, device=x.device)
            # The rightmost T columns are the usual mask
            attn_mask[:, T_all - T:] = torch.zeros_like(self_mask, dtype=x.dtype)
            attn_mask[:, T_all - T:] = attn_mask[:, T_all - T:].masked_fill(self_mask, float('-inf'))
            # The leftmost (T_all - T) columns (cached tokens) remain zero (no mask)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T_all)
        else:
            attn_mask = None
        
        # ✅ TRUNCATED BPTT: Pass cache and pos_offset for proper RoPE positions
        h, new_cache = self.self_attn(self.norm1(x), attn_mask=attn_mask, 
                                       kv_cache=kv_cache, pos_offset=pos_offset)
        x = x + h
        
        # 2. Cross-Attention (How does this match the actual tree shape?)
        # MEMORY FIX 2: Skip if disabled for this layer
        if not self.enable_cross_attn:
            # Skip cross-attention, proceed to FFN
            x = x + self.mlp(self.norm3(x))
            return x, new_cache
        
        # FIX 1: Only cross-attend with the LAST K tokens
        # For chunked processing, always use full chunk for cross-attention
        # (cross-attention doesn't accumulate state across chunks)
        K = min(self.cross_attn_window, T)
        
        # Patch: Handle inference mode (K=1, T=1) shape
        if T == 1:
            # Single token inference: query shape (B, 1, D)
            q = self.norm2(x)
            # Ensure visual_memory is contiguous and shape is (B, N, D)
            visual_memory = visual_memory.contiguous()
            q = q.contiguous()
            # If visual_memory shape is (N, D), unsqueeze batch dim
            if visual_memory.dim() == 2:
                visual_memory = visual_memory.unsqueeze(0)
            h2, _ = self.cross_attn(query=q, key=visual_memory, value=visual_memory)
            x = x + h2
        elif K < T:
            # Only compute cross-attention for last K tokens
            q = self.norm2(x[:, -K:, :])  # (B, K, D)
            h2, _ = self.cross_attn(query=q, 
                                    key=visual_memory, 
                                    value=visual_memory)
            # MEMORY FIX 5: Create new tensor (avoid in-place modification)
            x = torch.cat([
                x[:, :-K, :],           # Keep early tokens unchanged
                x[:, -K:, :] + h2       # Add cross-attention to last K tokens
            ], dim=1)
        else:
            # Sequence shorter than window, use full cross-attention
            h2, _ = self.cross_attn(query=self.norm2(x), 
                                    key=visual_memory, 
                                    value=visual_memory)
            x = x + h2
        
        # 3. Feed Forward
        x = x + self.mlp(self.norm3(x))
        return x, new_cache

class KVAttention(nn.Module):
    """
    KV-cached attention with RoPE support
    
    FIX 2: Uses Rotary Position Embeddings instead of learned absolute positions
    - Extrapolates to any sequence length (no hard limit)
    - Reduces repetition and late-sequence collapse
    """
    def __init__(self, dim, heads, rope=None):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        assert self.dim % self.heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"

        # Q,K,V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # Output projection
        self.o_proj = nn.Linear(dim, dim)
        
        # Rotary embeddings (shared across all attention layers)
        self.rope = rope
        
        # CRITICAL: Verify RoPE dimension matches head dimension
        if self.rope is not None:
            assert self.rope.dim == self.head_dim, \
                f"RoPE dim ({self.rope.dim}) must match head_dim ({self.head_dim})"

    def forward(self, x, attn_mask=None, kv_cache=None, pos_offset=0):
        """
        x: (B, T, D)
        kv_cache: (k_cache, v_cache) or None
        pos_offset: starting position index for RoPE (for KV-cached generation)
        returns:
            out: (B, T, D)
            new_k: (B, H, T_total, Hd)
            new_v: (B, H, T_total, Hd)
        """

        B, T, D = x.shape
        H = self.heads
        Hd = self.head_dim

        q = self.q_proj(x)                 # (B,T,D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape into heads
        q = q.view(B, T, H, Hd).transpose(1, 2)  # (B,H,T,Hd)
        k = k.view(B, T, H, Hd).transpose(1, 2)
        v = v.view(B, T, H, Hd).transpose(1, 2)

        # Apply RoPE BEFORE caching (if available)
        if self.rope is not None:
            # CRITICAL: always use pos_offset (the true absolute position of the new
            # tokens) for RoPE — NOT cache_len.  cache_len == pos_offset only when the
            # cache has never been trimmed; after sliding-window trimming they diverge.
            T_total = pos_offset + T   # covers positions [pos_offset … pos_offset+T-1]
            actual_offset = pos_offset

            # Apply rotary embeddings with proper offset
            q, k = self.rope.apply_rotary_pos_emb(q, k, T_total, pos_offset=actual_offset)
        
        # If cache exists → append new keys/values
        if kv_cache is not None:
            kc, vc = kv_cache  # kc:(B,H,T_prev,Hd)
            k = torch.cat([kc, k], dim=2)
            v = torch.cat([vc, v], dim=2)

        # save updated cache
        new_cache = (k, v)

        # scaled dot attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (Hd ** 0.5)  # (B,H,T,T_all)

        if attn_mask is not None:
            # Broadcast attn_mask to match scores shape if needed
            # scores: (B, H, T, T_all), attn_mask: (1, 1, T, T_all) or (B, H, T, T_all)
            if attn_mask.shape != scores.shape:
                attn_mask = attn_mask.expand_as(scores)
            scores = scores + attn_mask

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)              # (B,H,T,Hd)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.o_proj(out), new_cache


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, rope=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = KVAttention(dim, heads, rope=rope)   # <── Pass RoPE to attention
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, attn_mask=None, kv_cache=None, pos_offset=0):
        """
        kv_cache: list of layer-wise caches or None
        pos_offset: starting position for RoPE (used by structural decoder layers)
        returns:
            x_out, new_cache
        """

        h = self.norm1(x)

        # pass cache and pos_offset for this layer
        h2, new_cache = self.attn(h, attn_mask=attn_mask, kv_cache=kv_cache,
                                  pos_offset=pos_offset)

        x = x + h2
        h = self.norm2(x)
        x = x + self.mlp(h)

        return x, new_cache

class LSystemModel(nn.Module):
    @torch.no_grad()
    def pure_inference(self, dsm_pts, ortho_img, tokenizer, max_len=256, temperature=1.0, temperature_structural=0.7, max_depth=20):
        """
        Robust BATCHED autoregressive generation with grammar/bracket constraints.
        Returns (types, vals) as lists of lists (excluding the starting Axiom).
        
        This vectorized implementation is 8-16x faster than single-sample iteration.
        """
        self.eval()
        B = dsm_pts.shape[0]
        device = dsm_pts.device
        
        # Encode visual context once
        # Integrated species logits are now returned from the mm encoder
        mm_out = self.mm(dsm_pts, ortho_img)
        visual_memory, global_feat, species_logits = mm_out[0], mm_out[1], mm_out[2]
        self._pooled_visual_cache = mm_out
        
        # Track sequences as tensors
        types_acc = torch.full((B, 1), TokenType.F, dtype=torch.long, device=device)
        vals_acc  = torch.zeros((B, 1, 3), dtype=torch.long, device=device)
        
        # State tracking
        bracket_depth = torch.zeros(B, dtype=torch.long, device=device)
        finished      = torch.zeros(B, dtype=torch.bool, device=device)
        
        # KV cache slots: backbone + structural decoder layers
        n_backbone = len(self.blocks)
        n_struct   = len(self.struct_layers)
        kv_caches = [None for _ in range(n_backbone + n_struct)]

        for step in range(max_len):
            # Only process the last generated token
            t_in = types_acc[:, -1:]
            v_in = vals_acc[:, -1:]

            # Embeddings
            x = self.type_emb(t_in)
            # F Tokens (S-Segments): Length + Theta + Phi
            mask_F = (t_in == TokenType.F).unsqueeze(-1).float()
            x = x + mask_F * (self.val_emb_length(v_in[..., 0].clamp(0, self.f_bins - 1)) +
                              self.val_emb_theta(v_in[..., 1].clamp(0, self.theta_bins - 1)) +
                              self.val_emb_phi(v_in[..., 2].clamp(0, self.phi_bins - 1)))

            # ── Backbone blocks ────────────────────────────────────────────────
            backbone_kv = kv_caches[:n_backbone]
            struct_kv   = kv_caches[n_backbone:]

            new_backbone_kv = []
            for i, blk in enumerate(self.blocks):
                x, new_cache = blk.forward_with_cache(
                    x,
                    visual_memory,
                    kv_cache=backbone_kv[i],
                    pos_offset=step
                )
                new_backbone_kv.append(new_cache)

            h_backbone = self.final_norm(x)  # (B, 1, dim)

            # ── Structural decoder ─────────────────────────────────────────────
            h_struct = self.struct_proj(h_backbone)
            new_struct_kv = []
            for i, blk in enumerate(self.struct_layers):
                h_struct, sc = blk(h_struct, attn_mask=None,
                                   kv_cache=struct_kv[i], pos_offset=step)
                new_struct_kv.append(sc)
            h_struct = self.struct_norm(h_struct)

            # ── Parameter decoder ──────────────────────────────────────────────
            h_param = self.param_norm(
                self.param_proj(torch.cat([h_backbone, h_struct.detach()], dim=-1))
            )  # (B, 1, dim)

            # Sliding-window KV cache (backbone + struct together)
            # Sliding-window KV cache for backbone - tied to model's total capacity (NOT cross-attn window)
            if new_backbone_kv[0][0].shape[2] > self.max_window:
                new_backbone_kv = [(k[:, :, -self.max_window:, :], v[:, :, -self.max_window:, :]) 
                                   for k, v in new_backbone_kv]
            
            # Sliding-window KV cache for structural decoder - tied to model's total capacity
            if new_struct_kv[0][0].shape[2] > self.max_window:
                new_struct_kv = [(k[:, :, -self.max_window:, :], v[:, :, -self.max_window:, :]) 
                                 for k, v in new_struct_kv]

            kv_caches = new_backbone_kv + new_struct_kv

            # Prediction heads
            logits_t = self.type_head(h_struct[:, -1])     # (B, NUM_TYPES)
            lv_F     = self.val_head_length(h_param[:, -1])     # (B, f_bins)
            lv_angles = self.val_head_angles(h_param[:, -1])    # (B, theta_bins + phi_bins)
            
            # --- CONSTRAINTS ---
            prev_hard_t = types_acc[:, -1]
            logits_t = apply_grammar_mask(logits_t, prev_hard_t)

            # Bracket depth constraints (Vectorized)
            logits_t[:, TokenType.RBR] = logits_t[:, TokenType.RBR].masked_fill(bracket_depth == 0, float('-inf'))
            logits_t[:, TokenType.LBR] = logits_t[:, TokenType.LBR].masked_fill(bracket_depth >= max_depth, float('-inf'))
            logits_t[:, TokenType.EOS] = logits_t[:, TokenType.EOS].masked_fill(bracket_depth > 0, float('-inf'))
            
            logits_t = torch.clamp(logits_t, -100, 100)
            
            # Choose temperature per sample
            structural_mask = (prev_hard_t == TokenType.LBR) | (prev_hard_t == TokenType.RBR) 
            temp = torch.where(structural_mask, 
                               torch.tensor(temperature_structural, device=device), 
                               torch.tensor(temperature, device=device)).unsqueeze(-1)
            
            probs_t = torch.softmax(logits_t / temp.clamp(min=1e-6), dim=-1)
            
            # Force PAD for already finished sequences
            probs_t[finished] = 0.0
            probs_t[finished, TokenType.PAD] = 1.0
            
            # Sample next token
            next_t = torch.multinomial(probs_t, 1) # (B, 1)
            
            # Update sequence state
            bracket_depth += (next_t == TokenType.LBR).long().view(-1)
            bracket_depth -= (next_t == TokenType.RBR).long().view(-1)
            finished      |= (next_t == TokenType.EOS).view(-1)
            finished      |= (next_t == TokenType.PAD).view(-1)
            
            # Sample values
            next_vals = torch.zeros((B, 1, 3), dtype=torch.long, device=device)
            # F-length sampling
            f_probs = torch.softmax(torch.clamp(lv_F, -100, 100), dim=-1)
            next_vals[:, 0, 0] = torch.multinomial(f_probs, 1).view(-1)
            
            # Rotation sampling (S-segments use Pitch/Yaw heads for Theta/Phi)
            mask_F = (next_t == TokenType.F).view(-1)
            
            # Offsets for lv_angles: [Theta(0), Phi(1)]
            tb = self.theta_bins
            pb = self.phi_bins
            
            if mask_F.any():
                theta_probs = torch.softmax(torch.clamp(lv_angles[:, :tb], -100, 100), dim=-1)
                phi_probs = torch.softmax(torch.clamp(lv_angles[:, tb:tb+pb], -100, 100), dim=-1)
                
                sampled_theta = torch.multinomial(theta_probs, 1).view(-1)
                sampled_phi = torch.multinomial(phi_probs, 1).view(-1)
                
                next_vals[mask_F, 0, 1] = sampled_theta[mask_F]
                next_vals[mask_F, 0, 2] = sampled_phi[mask_F]
                
            # Accumulate
            types_acc = torch.cat([types_acc, next_t], dim=1)
            vals_acc  = torch.cat([vals_acc, next_vals], dim=1)
            
            if finished.all():
                break
                
        # Post-process tensors back into simple length-trimmed python lists
        final_types, final_vals = [], []
        for b in range(B):
            t_seq = types_acc[b, 1:].tolist()
            v_seq = vals_acc[b, 1:].tolist()
            
            # Trim to EOS/PAD
            if TokenType.EOS in t_seq:
                idx = t_seq.index(TokenType.EOS)
                t_seq, v_seq = t_seq[:idx], v_seq[:idx]
            elif TokenType.PAD in t_seq:
                idx = t_seq.index(TokenType.PAD)
                t_seq, v_seq = t_seq[:idx], v_seq[:idx]
                
            final_types.append(t_seq)
            final_vals.append(v_seq)
            
        return final_types, final_vals

    def __init__(self, type_vocab=5, f_bins=10, theta_bins=6, phi_bins=6, dim=512, layers=8, heads=16, num_species=13, max_window=1024, cross_attn_window=None, visual_bottleneck=None):
        super().__init__()
        # Dynamically tie internal windows to the total window size if not specified
        if cross_attn_window is None: cross_attn_window = max_window
        if visual_bottleneck is None: visual_bottleneck = max_window
        
        self.max_window = max_window
        self.cross_attn_window = cross_attn_window
        self.f_bins = f_bins
        self.theta_bins = theta_bins
        self.phi_bins = phi_bins
        self.dim = dim

        self.type_emb = nn.Embedding(type_vocab, dim)
        self.val_emb_length = nn.Embedding(f_bins, dim)
        self.val_emb_theta  = nn.Embedding(theta_bins, dim)
        self.val_emb_phi    = nn.Embedding(phi_bins, dim)
        # Dropout on token embeddings: prevents the model from relying purely on exact
        # token identity (LM shortcut) and encourages use of visual cross-attention.
        self.emb_drop = nn.Dropout(p=0.1)

        self.rope = RotaryEmbedding(dim=dim // heads, max_seq_len=max_window)
        self.mm = FullMultimodalEncoder(dim=dim, visual_bottleneck=visual_bottleneck, num_species=num_species)
        
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim, heads, 
                cross_attn_window=cross_attn_window, 
                rope=self.rope,
                enable_cross_attn=(i % 2 == 0)
            ) 
            for i in range(layers)
        ])
        
        self.final_norm = nn.LayerNorm(dim)
        
        self.struct_dim   = max(64, dim // 2)
        self.struct_heads = max(2, heads // 2)
        self.struct_rope  = RotaryEmbedding(dim=self.struct_dim // self.struct_heads,
                                            max_seq_len=max_window)
        self.struct_proj   = nn.Linear(dim, self.struct_dim)
        self.struct_layers = nn.ModuleList([
            DecoderBlock(self.struct_dim, self.struct_heads, mlp_ratio=2.0,
                         rope=self.struct_rope)
            for _ in range(2)
        ])
        self.struct_norm = nn.LayerNorm(self.struct_dim)
        self.type_head   = nn.Linear(self.struct_dim, type_vocab)

        self.param_proj = nn.Linear(dim + self.struct_dim, dim)
        self.param_norm = nn.LayerNorm(dim)

        self.val_head_length = nn.Linear(dim, f_bins)
        self.val_head_angles = nn.Linear(dim, theta_bins + phi_bins)

        # Species prediction is now handled inside the multimodal encoder
        # ensuring DSM and Ortho contribute equally to the classification.
        pass
        
        self.state_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 9)
        )
        
        # MEMORY FIX 3: Cache causal mask (allocated once, reused forever)
        self.register_buffer("causal_mask_cache", None, persistent=False)
        self.cached_mask_size = 0
        
        # MEMORY FIX 7: Cache pooled visual features (static per batch)
        self._pooled_visual_cache = None
        
        # ✅ FIX: Initialize weights properly to prevent NaN
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights to prevent NaN values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Reduced gain for stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _get_causal_mask(self, T, device):
        """MEMORY FIX 3: Get cached causal mask or create if needed"""
        if self.causal_mask_cache is None or T > self.cached_mask_size:
            # Allocate larger mask (with headroom)
            new_size = max(T, 1024)
            self.causal_mask_cache = torch.triu(
                torch.ones(new_size, new_size, dtype=torch.bool, device=device), 
                diagonal=1
            )
            self.cached_mask_size = new_size
        
        # Return slice of cached mask
        return self.causal_mask_cache[:T, :T]

    def forward(self, type_in, val_in, dsm_pts=None, ortho_img=None, kv_caches=None, pos_offset=0, visual_memory_cache=None):
        """
        Forward pass with KV cache support for Truncated BPTT
        
        Args:
            type_in: (B, T_chunk) token types for current chunk
            val_in: (B, T_chunk, 3) token values for current chunk
            dsm_pts, ortho_img: visual inputs (only processed if visual_memory_cache is None)
            kv_caches: List of (k, v) tuples per layer, or None
            pos_offset: Starting position for RoPE (for proper positional encoding across chunks)
            visual_memory_cache: Cached visual features (to avoid recomputing for each chunk)
            
        Returns:
            type_logits, val_logits, species_logits, new_kv_caches, visual_memory, pred_state
        """
        B, T = type_in.shape
        
        # ✅ TRUNCATED BPTT: Reuse visual memory across chunks (computed once per sequence)
        if visual_memory_cache is not None:
            # Unpack cached features (handles expanded return signature)
            visual_memory, global_feat, species_logits = visual_memory_cache[0], visual_memory_cache[1], visual_memory_cache[2]
            # Extra logits for supervision are usually index 3 and 4
            d_logits = visual_memory_cache[3] if len(visual_memory_cache) > 3 else None
            o_logits = visual_memory_cache[4] if len(visual_memory_cache) > 4 else None
        else:
            # Returns: visual_memory, global_feat, species_logits, d_logits, o_logits
            mm_out = self.mm(dsm_pts, ortho_img)
            visual_memory, global_feat, species_logits = mm_out[0], mm_out[1], mm_out[2]
            d_logits, o_logits = mm_out[3], mm_out[4]
        
        self._pooled_visual_cache = species_logits  # Cache primary logits for logging
        
        x = self.type_emb(type_in)
        
        # Type-aware value embeddings — safe additive form (no in-place indexing)
        # In-place `x[mask] += emb` can break autograd on advanced-indexed tensors.
        # We build a full-shape delta and add it once instead.
        mask_F = (type_in == TokenType.F).unsqueeze(-1).float()  # (B, T, 1)
        # B_F Segments: Length (0) + Theta (1) + Phi (2)
        emb_F  = (self.val_emb_length(val_in[..., 0].clamp(0, self.f_bins - 1)) +
                  self.val_emb_theta(val_in[..., 1].clamp(0, self.theta_bins - 1)) +
                  self.val_emb_phi(val_in[..., 2].clamp(0, self.phi_bins - 1)))
        x = self.emb_drop(x + mask_F * emb_F)

        # MEMORY FIX 3: Use cached causal mask
        mask = self._get_causal_mask(T, x.device)

        # ── BACKBONE BLOCKS ────────────────────────────────────────────────────
        n_backbone = len(self.blocks)
        n_struct   = len(self.struct_layers)
        if kv_caches is not None:
            backbone_caches = kv_caches[:n_backbone]
            struct_caches   = kv_caches[n_backbone:]
        else:
            backbone_caches = [None] * n_backbone
            struct_caches   = [None] * n_struct

        new_backbone_caches = []
        for i, blk in enumerate(self.blocks):
            x, new_cache = blk.forward_with_cache(x, visual_memory, self_mask=mask,
                                                   kv_cache=backbone_caches[i],
                                                   pos_offset=pos_offset)
            new_backbone_caches.append(new_cache)

        x = self.final_norm(x)
        h_backbone = x  # (B, T, dim)

        # ── STRUCTURAL DECODER ─────────────────────────────────────────────────
        # Operates in a smaller struct_dim space; only type-CE flows through here.
        h_struct = self.struct_proj(h_backbone)

        # Build float causal mask that accounts for cached context from earlier chunks
        T_struct_prev = struct_caches[0][0].shape[2] if struct_caches[0] is not None else 0
        T_struct_all  = T_struct_prev + T
        struct_float_mask = torch.zeros((T, T_struct_all), dtype=h_backbone.dtype,
                                        device=h_backbone.device)
        struct_float_mask[:, T_struct_prev:] = struct_float_mask[:, T_struct_prev:].masked_fill(
            mask, float('-inf'))
        struct_float_mask = struct_float_mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T_all)

        new_struct_caches = []
        for i, blk in enumerate(self.struct_layers):
            h_struct, sc = blk(h_struct, attn_mask=struct_float_mask,
                               kv_cache=struct_caches[i], pos_offset=pos_offset)
            new_struct_caches.append(sc)
        h_struct = self.struct_norm(h_struct)

        # Type logits from structural decoder ONLY
        type_logits = self.type_head(h_struct)  # (B, T, NUM_TYPES)

        # ── PARAMETER DECODER ──────────────────────────────────────────────────
        # Computes length and angle logits for B_F tokens
        h_param = self.param_norm(
            self.param_proj(torch.cat([h_backbone, h_struct.detach()], dim=-1))
        )  # (B, T, dim)

        val_logits_length = self.val_head_length(h_param)  # (B, T, f_bins)
        val_logits_angles = self.val_head_angles(h_param)  # (B, T, theta_bins + phi_bins)

        val_logits = torch.cat([val_logits_length, val_logits_angles], dim=-1)

        # Predict latent state (structural supervision)
        pred_state = self.state_head(h_backbone)  # (B, T, 9)

        all_new_caches = new_backbone_caches + new_struct_caches
        # Return full mm_out for caching if we just computed it
        mm_cache = visual_memory_cache if visual_memory_cache is not None else (visual_memory, global_feat, species_logits, d_logits, o_logits)
        return type_logits, val_logits, species_logits, all_new_caches, mm_cache, pred_state
    
    @torch.no_grad()
    def generate(self, dsm_pts, ortho_img, tokenizer, max_len=1000, temperature=1.0, temperature_structural=0.7, max_depth=20):
        """
        BRACKET-SAFE Generation with KV Caching for 10-30x speedup
        
        FIX 1: Hard bracket-depth constraints during generation
        - Tracks bracket depth online
        - Forbids ] when depth == 0
        - Forces EOS only when depth == 0
        - Caps max depth to prevent explosion
        
        FIX 2: Lower temperature for structural tokens
        - temperature_structural (default 0.7) for [, ], A->
        - temperature (default 1.0) for F, R
        - Reduces catastrophic structural errors
        
        FIX 3: KV Caching for O(T) complexity instead of O(T²)
        - Only processes new token at each step
        - Caches key/value states per layer
        - 10-30x faster for long sequences
        """
        device = dsm_pts.device
        self.eval()

        # Encode multimodal context ONCE
        # Unpack species_logits (now internal to mm)
        mm_out = self.mm(dsm_pts, ortho_img)
        visual_memory, global_feat, species_logits = mm_out[0], mm_out[1], mm_out[2]
        self._pooled_visual_cache = species_logits  # Cache for subsequent chunks if needed

        # 🔴 START FIX: Do NOT hardcode F0. 
        # Start with an empty sequence or a neutral starting state if needed.
        types = [TokenType.F] # Base Axiom is F
        vals  = [[0, 0, 0]]   # But we allow the model to refine this first token
        prev_type = TokenType.F
        # KV cache slots: backbone layers + structural decoder layers
        kv_caches = [None for _ in range(len(self.blocks) + len(self.struct_layers))]

        # Bracket depth tracking
        bracket_depth = 0
        total_f_emitted = 0
        steps_since_f = 0
        f_at_depth = [False] * 33 # Track F presence per level
        
        for step in range(max_len):
            # --------------------------------------------------
            # ONLY PROCESS NEW TOKEN (KV caching optimization)
            # --------------------------------------------------
            if step == 0:
                # First step: process full sequence (just "F->")
                t_in = torch.tensor([types], device=device)
                v_in = torch.tensor([vals], device=device)
            else:
                # Subsequent steps: only process last token
                t_in = torch.tensor([[types[-1]]], device=device)
                v_in = torch.tensor([[vals[-1]]], device=device)
            
            # --------------------------------------------------
            # Embeddings (ONLY NEW TOKEN)
            # --------------------------------------------------
            # Safe additive embedding (mirrors forward()) - no in-place indexing
            x = self.type_emb(t_in)
            mask_F = (t_in == TokenType.F).unsqueeze(-1).float()
            x = x + mask_F * (self.val_emb_length(v_in[..., 0].clamp(0, self.f_bins - 1)) +
                              self.val_emb_theta(v_in[..., 1].clamp(0, self.theta_bins - 1)) +
                              self.val_emb_phi(v_in[..., 2].clamp(0, self.phi_bins - 1)))

            # --------------------------------------------------
            # Process through blocks — EXACTLY mirrors forward_with_cache()
            # --------------------------------------------------
            # Absolute position of current token (for RoPE offset)
            pos = len(types) - 1

            # ── Backbone blocks ────────────────────────────────────────────────
            n_backbone = len(self.blocks)
            backbone_kv = kv_caches[:n_backbone]
            struct_kv   = kv_caches[n_backbone:]

            new_backbone_kv = []
            for i, blk in enumerate(self.blocks):
                x, new_cache = blk.forward_with_cache(
                    x,
                    visual_memory,
                    self_mask=None,   # single-token: no future positions to mask
                    kv_cache=backbone_kv[i],
                    pos_offset=pos
                )
                new_backbone_kv.append(new_cache)

            x = self.final_norm(x)
            h_backbone = x  # (1, 1, dim)

            # ── Structural decoder ─────────────────────────────────────────────
            h_struct = self.struct_proj(h_backbone)
            new_struct_kv = []
            for i, blk in enumerate(self.struct_layers):
                # attn_mask=None: single-token query, no future masking needed
                h_struct, sc = blk(h_struct, attn_mask=None,
                                   kv_cache=struct_kv[i], pos_offset=pos)
                new_struct_kv.append(sc)
            h_struct = self.struct_norm(h_struct)

            # ── Parameter decoder ──────────────────────────────────────────────
            h_param = self.param_norm(
                self.param_proj(torch.cat([h_backbone, h_struct.detach()], dim=-1))
            )  # (1, 1, dim)

            # Sliding-window KV cache (backbone + struct together; same T growth rate)
            new_kv = new_backbone_kv + new_struct_kv
            if new_kv[0][0].shape[2] > self.max_window:
                kv_caches = [
                    (k[:, :, -self.max_window:, :], v[:, :, -self.max_window:, :])
                    for k, v in new_kv
                ]
            else:
                kv_caches = new_kv

            # --------------------------------------------------
            # Predict next token with STRUCTURAL CONSTRAINTS
            # --------------------------------------------------
            logits_t        = self.type_head(h_struct[0, -1])     # (NUM_TYPES,)
            logits_v_length = self.val_head_length(h_param[0, -1]) # (f_bins,)
            logits_v_angles = self.val_head_angles(h_param[0, -1]) # (theta_bins + phi_bins,)

            # Apply grammar mask
            logits_t = apply_grammar_mask(logits_t, prev_type)

            # FIX 1: Hard bracket constraints
            # Forbid ] if depth == 0
            if bracket_depth == 0:
                logits_t[TokenType.RBR] = float('-inf')
            
            # Forbid [ if depth >= max_depth
            if bracket_depth >= max_depth:
                logits_t[TokenType.LBR] = float('-inf')

            # 🔴 MOVEMENT PRESSURE: Force F if stuck in non-moving loops
            if steps_since_f > 5:
                logits_t[TokenType.F] += 5.0
            
            # 🔴 STRUCTURAL RULE: Cannot close branch if no F was emitted in it
            if bracket_depth > 0 and bracket_depth < len(f_at_depth) and not f_at_depth[bracket_depth]:
                logits_t[TokenType.RBR] = float('-inf')
            
            # 🔴 EOS Guard: Must have F at root level
            if not f_at_depth[0]:
                logits_t[TokenType.EOS] = float('-inf')

            # Force EOS only if depth == 0 (balanced brackets)
            if bracket_depth > 0:
                logits_t[TokenType.EOS] = float('-inf')
            
            # Clip logits to prevent numerical instability
            logits_t = torch.clamp(logits_t, min=-100, max=100)
            
            # FIX 2: Lower temperature for structural tokens
            # Determine if we're predicting a structural token
            structural_tokens = {TokenType.LBR, TokenType.RBR}
            
            # Check top prediction
            top_pred = logits_t.argmax().item()
            if top_pred in structural_tokens:
                temp = temperature_structural
            else:
                temp = temperature
            
            probs = torch.softmax(logits_t / max(temp, 1e-6), dim=-1)
            
            # Defensive check: ensure probs are valid
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                print(f"[WARN] Invalid probs at step {step}, using uniform distribution")
                print(f"  logits_t stats: min={logits_t.min():.4f}, max={logits_t.max():.4f}")
                print(f"  probs stats: min={probs.min():.4f}, max={probs.max():.4f}, sum={probs.sum():.4f}")
                # Fallback to uniform distribution over valid tokens
                probs = torch.ones_like(logits_t) / len(logits_t)
                probs[logits_t == float('-inf')] = 0
                probs = probs / probs.sum()
            
            next_t = torch.multinomial(probs, 1).item()

            if next_t == TokenType.EOS:
                # Final safety check: only allow EOS if brackets balanced
                if bracket_depth == 0:
                    break
                else:
                    # Force a valid token instead
                    logits_t[TokenType.EOS] = float('-inf')
                    logits_t = torch.clamp(logits_t, min=-100, max=100)
                    probs = torch.softmax(logits_t / max(temp, 1e-6), dim=-1)
                    
                    # Defensive check again
                    if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                        print(f"[WARN] Invalid probs after EOS rejection at step {step}")
                        probs = torch.ones_like(logits_t) / len(logits_t)
                        probs[logits_t == float('-inf')] = 0
                        probs = probs / probs.sum()
                    
                    next_t = torch.multinomial(probs, 1).item()

            # --------------------------------------------------
            # Predict value bins (MEMORY FIX 6: use appropriate head)
            # --------------------------------------------------
            if next_t == TokenType.F:
                logits_v_length = torch.clamp(logits_v_length, min=-100, max=100)
                probs_F = torch.softmax(logits_v_length, dim=-1)
                f = torch.multinomial(probs_F, 1).item() if not torch.isnan(probs_F).any() else 0
                
                # Also sample rotations for F in d3 format
                logits_v_angles = torch.clamp(logits_v_angles, min=-100, max=100)
                tb = self.theta_bins
                pb = self.phi_bins
                theta = torch.multinomial(torch.softmax(logits_v_angles[0:tb], dim=-1), 1).item()
                phi = torch.multinomial(torch.softmax(logits_v_angles[tb:tb+pb], dim=-1), 1).item()
                
                new_v = [f, theta, phi]
            else:
                new_v = [0, 0, 0]
            
            # 🔴 START FIX: If we just emitted the first token, try to predict its values 
            # (In reality, we should prepend a START token to our dataset, but for now 
            # we just let the sequence begin).

            # Update bracket depth and movement stats
            if next_t == TokenType.F:
                total_f_emitted += 1
                steps_since_f = 0
                f_at_depth[bracket_depth] = True
            elif next_t == TokenType.LBR:
                bracket_depth += 1
                if bracket_depth < 33:
                    f_at_depth[bracket_depth] = False
            elif next_t == TokenType.RBR:
                bracket_depth = max(0, bracket_depth - 1)
            
            if next_t != TokenType.F:
                steps_since_f += 1

            types.append(next_t)
            vals.append(new_v)
            prev_type = next_t

        return tokenizer.decode(types, vals)

def extract_segments_batch(positions, type_ids):
    """
    positions: (B, T, 3)
    type_ids:  (B, T)

    Returns:
        p0: (B, N, 3)
        p1: (B, N, 3)
    """
    B, T, _ = positions.shape

    # previous positions
    p_prev = torch.cat([positions[:, :1], positions[:, :-1]], dim=1)

    is_F = (type_ids == TokenType.F)

    seg_p0 = []
    seg_p1 = []

    for b in range(B):
        mask = is_F[b]
        seg_p0.append(p_prev[b][mask])
        seg_p1.append(positions[b][mask])

    return seg_p0, seg_p1

def render_soft_lines(p0_list, p1_list, image_size=128, sigma=0.02):
    """
    p0_list, p1_list: lists of (N,3) tensors per batch
    Returns: (B, 1, H, W) soft images
    """

    B = len(p0_list)
    device = p0_list[0].device
    images = []

    # 2D grid
    xs = torch.linspace(-1, 1, image_size, device=device)
    ys = torch.linspace(-1, 1, image_size, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([xx, yy], dim=-1)  # (H,W,2)

    for b in range(B):

        if p0_list[b].shape[0] == 0:
            images.append(torch.zeros(1, image_size, image_size, device=device))
            continue

        p0 = p0_list[b][:, :2]  # project to XY
        p1 = p1_list[b][:, :2]

        img = torch.zeros(image_size, image_size, device=device)

        for i in range(p0.shape[0]):

            a = p0[i]
            bpt = p1[i]

            # line vector
            v = bpt - a
            v_norm = torch.dot(v, v) + 1e-8

            # project grid onto segment
            w = grid - a
            t = (w[..., 0]*v[0] + w[..., 1]*v[1]) / v_norm
            t = t.clamp(0, 1)

            proj = a + t.unsqueeze(-1) * v
            dist = torch.norm(grid - proj, dim=-1)

            img = img + torch.exp(-(dist**2) / (2 * sigma**2))

        img = img.clamp(0, 1)
        images.append(img.unsqueeze(0))

    return torch.stack(images, dim=0)
    
# ============================================================
# Truncated BPTT (Backpropagation Through Time) Helper
# ============================================================

def forward_with_truncated_bptt(model, type_in, val_in, dsm_pts, ortho_img, chunk_size=2048, visual_memory_cache=None):
    """
    Process long sequences in chunks with KV cache carry-over
    """
    B, T_total = type_in.shape
    device = type_in.device
    
    # ✅ TRUNCATED BPTT: Reuse visual memory across chunks (computed once per sequence)
    if visual_memory_cache is not None:
        visual_memory_cache_all = visual_memory_cache
    else:
        # Returns: (visual_memory, global_feat, species_logits, d_logits, o_logits)
        visual_memory_cache_all = model.mm(dsm_pts, ortho_img)
    
    # Initialize KV caches (one per layer)
    kv_caches = None
    
    all_type_logits = []
    all_val_logits = []
    all_pred_states = []
    species_logits = None  
    mm_out = None
    
    # Process sequence in chunks
    num_chunks = (T_total + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        # Get chunk boundaries
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, T_total)
        T_chunk = end - start
        
        # Extract chunk
        type_chunk = type_in[:, start:end]
        val_chunk = val_in[:, start:end]
        
        # Forward pass for this chunk
        tlog, vlog, sp_logits, new_kv_caches, mm_cache_current, chunk_pred_state = model(
            type_chunk, val_chunk, dsm_pts, ortho_img,
            kv_caches=kv_caches,
            pos_offset=start,  # Critical: proper RoPE positions
            visual_memory_cache=visual_memory_cache_all  # Reuse visual features
        )
        
        all_type_logits.append(tlog)
        all_val_logits.append(vlog)
        all_pred_states.append(chunk_pred_state)
        # Primary logits
        species_logits = sp_logits
        mm_out = mm_cache_current
        
        # ✅ CRITICAL: Detach caches to prevent gradient flow across chunks
        if new_kv_caches is not None:
            kv_caches = [(k.detach(), v.detach()) for k, v in new_kv_caches]
        else:
            kv_caches = None
    
    return all_type_logits, all_val_logits, species_logits, mm_out, all_pred_states


# ============================================================
# train_model (Cleaned up for Cross-Attention)
# ============================================================

def train_model(
    dataset,
    tokenizer,
    save_path="model_ar2d.pth",
    batch_size=2,
    lr=1e-4,
    dim=256,
    epochs=300,
    device="cuda",
    env="train",
    resume_path=None,
    # NEW: Truncated BPTT parameters
    use_truncated_bptt=True,  # Enable chunked training for long sequences
    bptt_chunk_size=2048,     # Process in 2048-token chunks
    # NEW: Scheduled sampling parameters
    use_scheduled_sampling=True,
    ss_ramp_start=50,
    ss_ramp_end=200,
    ss_max_prob=0.8,   # raised from 0.5 — more exposure to own predictions
    # NEW: Position-weighted loss
    use_position_weights=True,
    position_weight_alpha=0.002,
    # NEW: Geometry-aware loss
    use_chamfer_loss=False,  # Disabled by default (expensive)
    chamfer_weight=0.1,
    chamfer_sample_freq=10,  # Only compute every N batches
    # NEW: Rotation smoothness regularization
    use_rotation_smoothness=True,
    rotation_smoothness_weight=0.01,
    heads=8,
    layers=8,
    max_points_chamfer=1000,
    use_student_forcing=False,
    f_bins=10,
    theta_bins=6,
    phi_bins=6,
    modality="both", # "both", "dsm", "ortho"
    use_species_loss=True, # Whether to use the species classification loss
):
    loader     = DataLoader(dataset,     batch_size=batch_size, shuffle=True,  drop_last=False, num_workers=4, pin_memory=True)

    model = LSystemModel(
        f_bins=tokenizer.f_bins,
        theta_bins=tokenizer.theta_bins,
        phi_bins=tokenizer.phi_bins,
        num_species=dataset.num_species,
        dim=dim,
        max_window=dataset.window,
        cross_attn_window=dataset.window,
        heads=heads,
        layers=layers,
    ).to(device)

    # Optional: Compile model for speed (PyTorch 2.0+)
    # if hasattr(torch, "compile"):
    #     print("[INFO] Compiling model with torch.compile...")
    #     model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Learning rate scheduler - reduces LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='min',
        factor=0.5,
        patience=20,     # 10 was too aggressive; allow enough epochs to learn
        verbose=True,
        min_lr=1e-10      # Lower floor for fine-tuning
    )
    
    scaler = torch.amp.GradScaler('cuda')

    # Initialize Chamfer loss if requested
    chamfer_loss_fn = None
    if use_chamfer_loss and CHAMFER_AVAILABLE:
        chamfer_loss_fn = ChamferLoss().to(device)
        print(f"[INFO] Chamfer loss enabled with weight={chamfer_weight}, sample_freq={chamfer_sample_freq}")
    elif use_chamfer_loss and not CHAMFER_AVAILABLE:
        print("[WARNING] Chamfer loss requested but not available")

    if resume_path is not None and os.path.exists(resume_path):
        print(f"[RESUME] Loading checkpoint from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        # strict=False lets the model load even when architecture changed
        # (e.g. val_emb_R → val_emb_R0/R1/R2).  New parameters start random.
        try:
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        except RuntimeError as _shape_err:
            # type_head changed input dim (dim → struct_dim); filter mismatched tensors.
            print(f"[RESUME] Shape mismatch: {_shape_err}")
            print("[RESUME] Filtering incompatible weights — split-decoder heads start random")
            model_sd = model.state_dict()
            filtered = {k: v for k, v in ckpt["model"].items()
                        if k in model_sd and v.shape == model_sd[k].shape}
            missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            print(f"[RESUME] Missing keys (will be random-init): {missing}")
        if unexpected:
            print(f"[RESUME] Unexpected keys in ckpt (ignored): {unexpected}")
        # Optimizer state is tied to parameter count — skip gracefully if
        # the architecture changed and the counts no longer match.
        try:
            opt.load_state_dict(ckpt["opt"])
        except (ValueError, RuntimeError) as _e:
            print(f"[RESUME] Optimizer state incompatible (architecture changed): {_e}")
            print("[RESUME] Starting with a fresh optimizer state.")
        if "scaler" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                pass
        start_epoch = ckpt.get("epoch", 0)
    else:
        start_epoch = 0

    viz = LSystemVisdom(env=env, port=8099)
    viz.viz.close()

    # GT point cloud cache (render_loop_hard, same scale as model output)
    # Populated lazily on first encounter of each tree ID.
    gt_pointcloud_cache = {}

    # Cache the last chamfer point clouds so the viz step can display them
    _last_chamfer_pred_np = None  # (NUM_PTS, 3) numpy, first batch element
    _last_chamfer_gt_np   = None

    for ep in range(start_epoch, epochs):
        model.train()
        epoch_loss            = torch.tensor(0.0, device=device)
        epoch_loss_type       = torch.tensor(0.0, device=device)
        epoch_loss_val        = torch.tensor(0.0, device=device)
        epoch_loss_chamfer    = torch.tensor(0.0, device=device)
        epoch_loss_rot_smooth = torch.tensor(0.0, device=device)
        epoch_loss_soft_angle = torch.tensor(0.0, device=device)
        epoch_loss_state      = torch.tensor(0.0, device=device)
        
        # Compute scheduled sampling probability for this epoch
        ss_prob = get_scheduled_sampling_prob(
            ep,
            ramp_start=ss_ramp_start,
            ramp_end=ss_ramp_end,
            max_prob=ss_max_prob
        ) if use_scheduled_sampling else 0.0
        
        if ss_prob > 0:
            print(f"[SCHED SAMPLING] Epoch {ep}: p(model_pred) = {ss_prob:.3f}")
        
        # IMPORTANT: use batch_size=1 when testing overfit
        # IMPORTANT: use drop_last=False in DataLoader

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {ep}")):
            dsm   = batch["dsm"].to(device).float()
            ortho = batch["ortho"].to(device).float()
            
            # ── MODALITY MASKING ──
            if modality == "dsm":
                ortho = torch.zeros_like(ortho) # Hide the image
            elif modality == "ortho":
                dsm = torch.zeros_like(dsm)     # Hide the 3D points

            t_in  = batch["type_in"].to(device)
            v_in  = batch["val_in"].to(device)
            t_tgt = batch["type_tgt"].to(device)
            v_tgt = batch["val_tgt"].to(device)
            sp_gt = batch["species"].to(device)
            states_tgt = batch["states_tgt"].to(device)
            states_in  = batch["states"].to(device)

            # --------------------------------------------------
            # SCHEDULED SAMPLING: PREFIX-CONSISTENT suffix replacement
            # 🔴 FIX: Replace entire suffix, not random positions
            # This preserves bracket balance and execution semantics
                       # 🔴 FIX 2: Use grammar-aware sampling (same constraints as generation)
            # MEMORY FIX 5: Model forward already optimized with:
            # --------------------------------------------------
            # 1. Pre-compute Visual Memory (Shared across all passes)
            # --------------------------------------------------
            # We compute with gradients enabled so the ENCODER can learn.
            # We use .detach() later when we don't want backprop (like sampling).
            v_mem = model.mm(dsm, ortho)
            
            # 2. Forcing Mode (Teacher vs Student vs Scheduled)
            forcing_mode = "teacher"
            if use_student_forcing or (use_scheduled_sampling and ss_prob > 0 and random.random() < ss_prob):
                forcing_mode = "student" if use_student_forcing else "scheduled"
                B, T = t_in.shape
                
                with torch.no_grad():
                    if forcing_mode == "student":
                        # 🔴 PURE STUDENT: 1024-token autonomous sequence (no GT leak)
                        pred_types_b = torch.full((B, T), TokenType.PAD, device=device)
                        pred_vals_b = torch.zeros(B, T, 3, dtype=torch.long, device=device)
                        pred_types_b[:, 0] = TokenType.F
                        
                        kv_s = None
                        s_depths = [0] * B
                        s_f_at_depth = [[False]*33 for _ in range(B)]
                        s_prev_t = [TokenType.F] * B
                        
                        for s_ptr in range(T - 1):
                            tl_s, vl_s, _, kv_s, _, _ = model(
                                pred_types_b[:, s_ptr:s_ptr+1], pred_vals_b[:, s_ptr:s_ptr+1],
                                None, None, # dsm_pts, ortho_img
                                kv_caches=kv_s, pos_offset=s_ptr,
                                visual_memory_cache=v_mem
                            )
                            for b in range(B):
                                nt, nv, s_depths[b], s_f_at_depth[b] = sample_with_grammar_constraints(
                                    tl_s[b, 0], vl_s[b, 0], s_prev_t[b], s_depths[b],
                                    tokenizer.f_bins, tokenizer.theta_bins, tokenizer.phi_bins, s_f_at_depth[b]
                                )
                                pred_types_b[b, s_ptr+1] = nt
                                pred_vals_b[b, s_ptr+1] = torch.tensor(nv, device=device)
                                s_prev_t[b] = nt
                        pred_types, pred_vals = pred_types_b, pred_vals_b
                    else:
                        # 🟡 SCHEDULED: Parallel logits from GT context (already teacher-forced in the model call)
                        # Use .detach() on v_mem here because we only want gradients from the final loss pass
                        v_mem_detached = (v_mem[0].detach(), v_mem[1].detach(), v_mem[2].detach())
                        tl_p, vl_p, _, _, _, _ = model(t_in, v_in, None, None, visual_memory_cache=v_mem_detached)
                        tl_cpu, vl_cpu = tl_p.cpu(), vl_p.cpu()
                        
                        pred_types_p = torch.zeros(B, T, dtype=torch.long, device=device)
                        pred_vals_p = torch.zeros(B, T, 3, dtype=torch.long, device=device)
                        for b in range(B):
                            bd, pt, fd = 0, TokenType.F, [False]*33
                            for t_idx in range(T):
                                nt, nv, bd, fd = sample_with_grammar_constraints(
                                    tl_cpu[b, t_idx], vl_cpu[b, t_idx], pt, bd,
                                    tokenizer.f_bins, tokenizer.theta_bins, tokenizer.phi_bins, fd
                                )
                                pred_types_p[b, t_idx] = nt
                                pred_vals_p[b, t_idx] = torch.tensor(nv, device=device)
                                pt = nt
                        pred_types, pred_vals = pred_types_p, pred_vals_p
                
                # Replace suffixes incrementally
                t_in = t_in.clone()
                v_in = v_in.clone()
                for b in range(B):
                    # 🔴 Student Forcing: force cut=1 (whole window is generated)
                    if forcing_mode == "student":
                        cut = 1
                    else:
                        # Random cut point (preserve at least first token TokenType.F)
                        cut = random.randint(1, T - 1)
                    
                    # 🔴 FIX: pred_types[b, t] is predicted from context up to t, so it's the prediction for t+1
                    # To replace t_in[b, cut:], we need pred_types[b, cut-1 : T-1]
                    t_in[b, cut:] = pred_types[b, cut-1 : T-1]
                    v_in[b, cut:] = pred_vals[b, cut-1 : T-1]
            
            # --------------------------------------------------
            # FORWARD (with Truncated BPTT support)
            # --------------------------------------------------
            with torch.amp.autocast('cuda'):
                if use_truncated_bptt and t_in.shape[1] > bptt_chunk_size:
                    # ✅ TRUNCATED BPTT: Process long sequences in chunks (skip redundant encoding)
                    all_tlog, all_vlog, sp_logits, mm_out, all_pred_states = forward_with_truncated_bptt(
                        model, t_in, v_in, dsm, ortho, 
                        chunk_size=bptt_chunk_size,
                        visual_memory_cache=v_mem
                    )
                    
                    # Concatenate chunk outputs
                    tlog = torch.cat(all_tlog, dim=1)  # (B, T_total, num_types)
                    vlog = torch.cat(all_vlog, dim=1)  # (B, T_total, val_dim)
                    pred_state = torch.cat(all_pred_states, dim=1) # (B, T_total, 9)
                    
                else:
                    # Standard forward pass using pre-computed visual memory (NO redundant encoding)
                    tlog, vlog, sp_logits, _, mm_out, pred_state = model(
                        t_in, v_in, None, None, visual_memory_cache=v_mem
                    )
            
            # --- ROBUST SPECIES LOSS ---
            # sp_gt == -1 means species unknown; ignore those.
            valid_sp = sp_gt >= 0
            if use_species_loss and valid_sp.any():
                # 1. Primary Fused Loss
                # Increased label smoothing to 0.15 for better generalization/less over-confidence
                loss_sp_fused = F.cross_entropy(sp_logits[valid_sp], sp_gt[valid_sp], label_smoothing=0.15)
                
                # 2. Modality-Specific Auxiliary Losses (Independent supervision)
                # Unpack modality-specific logits from mm_out
                d_logits, o_logits = mm_out[3], mm_out[4]
                loss_sp_dsm = F.cross_entropy(d_logits[valid_sp], sp_gt[valid_sp], label_smoothing=0.1)
                loss_sp_ortho = F.cross_entropy(o_logits[valid_sp], sp_gt[valid_sp], label_smoothing=0.1)
                
                # 3. Consistency Regularization (KL Divergence)
                # Ensure modalities agree on identity when both are present
                ortho_is_missing_batch = (ortho.abs().max() < 1e-5)
                dsm_is_missing_batch = (dsm.abs().max() < 1e-5)
                
                loss_sp_consist = torch.tensor(0.0, device=device)
                if not ortho_is_missing_batch and not dsm_is_missing_batch:
                    # Symmetric KL Divergence between modality predictions
                    p_d = F.softmax(d_logits[valid_sp].detach(), dim=-1)
                    p_o = F.softmax(o_logits[valid_sp].detach(), dim=-1)
                    log_p_d = F.log_softmax(d_logits[valid_sp], dim=-1)
                    log_p_o = F.log_softmax(o_logits[valid_sp], dim=-1)
                    
                    kl_ovd = F.kl_div(log_p_d, p_o, reduction='batchmean')
                    kl_dvo = F.kl_div(log_p_o, p_d, reduction='batchmean')
                    loss_sp_consist = 0.5 * (kl_ovd + kl_dvo)
                
                # Combined species loss
                loss_sp = 5.0 * (loss_sp_fused + 0.5 * loss_sp_dsm + 0.5 * loss_sp_ortho + 0.2 * loss_sp_consist)
            else:
                loss_sp = torch.tensor(0.0, device=device, dtype=sp_logits.dtype)

            # --------------------------------------------------
            # POSITION-WEIGHTED TYPE LOSS
            # --------------------------------------------------
            valid_mask = (t_tgt != TokenType.PAD)
            B, T = t_tgt.shape
            
            # Compute position weights
            if use_position_weights:
                pos_weights = compute_position_weights(T, alpha=position_weight_alpha).to(device)
                pos_weights = pos_weights.unsqueeze(0).expand(B, -1)  # (B, T)
            else:
                pos_weights = torch.ones(B, T, device=device)

            # --------------------------------------------------
            # POSITION-WEIGHTED TYPE LOSS
            # 🔴 FIX: Use class weights to favor F tokens (preventing "mostly rotations")
            # --------------------------------------------------
            tlog_flat = tlog.reshape(-1, NUM_TYPES)
            tgt_flat  = t_tgt.reshape(-1)
            valid_flat = valid_mask.reshape(-1)
            pos_weights_flat = pos_weights.reshape(-1)

            type_weights = torch.ones(NUM_TYPES, device=device, dtype=tlog.dtype)
            type_weights[TokenType.F] = 1.0   # 🔴 NEUTRAL: Let the geometric loss drive the segments
            type_weights[TokenType.LBR] = 1.0 
            type_weights[TokenType.RBR] = 1.0

            loss_per_token = F.cross_entropy(
                tlog_flat[valid_flat],
                tgt_flat[valid_flat],
                weight=type_weights,
                reduction='none'
            )
            
            # Apply position weights
            weighted_loss = loss_per_token * pos_weights_flat[valid_flat]
            loss_type = 10*weighted_loss.mean()

            # --------------------------------------------------
            # POSITION-WEIGHTED VALUE LOSS
            # --------------------------------------------------
            loss_val = 0.0

            # vlog now has shape (B, T, f_bins + 2*theta_bins + phi_bins)
            rot_bins_total = tokenizer.theta_bins + tokenizer.phi_bins
            vlog_flat = vlog.reshape(-1, tokenizer.f_bins + rot_bins_total)
            vtgt_flat = v_tgt.reshape(-1, 3)
            tgt_type_flat = tgt_flat

            mask_F = (tgt_type_flat == TokenType.F) & valid_flat
            
            if mask_F.any():
                # B_F tokens have 3 valid parameters: Length, Theta, Phi
                # Length: first f_bins
                loss_length = F.cross_entropy(
                    vlog_flat[mask_F, :tokenizer.f_bins],
                    vtgt_flat[mask_F, 0],
                    reduction='none'
                )
                
                # Theta: next theta_bins
                tb = tokenizer.theta_bins
                loss_theta = F.cross_entropy(
                    vlog_flat[mask_F, tokenizer.f_bins : tokenizer.f_bins + tb],
                    vtgt_flat[mask_F, 1],
                    reduction='none'
                )
                
                # Phi: last phi_bins
                pb = tokenizer.phi_bins
                loss_phi = F.cross_entropy(
                    vlog_flat[mask_F, tokenizer.f_bins + tb : tokenizer.f_bins + tb + pb],
                    vtgt_flat[mask_F, 2],
                    reduction='none'
                )
                
                weighted_loss_F = (loss_length + loss_theta + loss_phi) / 3.0 * pos_weights_flat[mask_F]
                loss_val += weighted_loss_F.mean()

            # --------------------------------------------------
            # ROTATION SMOOTHNESS REGULARIZATION
            # 🔴 FIX: Penalize high-frequency rotation oscillation
            # Prevents spiral artifacts, exploding crowns, branch jitter
            # --------------------------------------------------
            loss_rot_smooth = torch.tensor(0.0, device=device)
            
            if use_rotation_smoothness:
                # vlog_R extracted only for rotation heads
                vlog_R = vlog[..., tokenizer.f_bins:]
                loss_rot_smooth = compute_rotation_smoothness_loss(
                    vlog_R, v_tgt, t_tgt, valid_mask, tokenizer.theta_bins, tokenizer.phi_bins
                )

            # --- Differentiable geometry losses (every batch) ---
            loss_soft_angle = compute_soft_angle_loss(
                vlog, v_tgt, t_tgt, valid_mask, tokenizer.f_bins, tokenizer.theta_bins, tokenizer.phi_bins
            )
            gt_states = batch["states_tgt"].to(device)
            center = batch["dsm_center"].to(device)
            scale  = batch["dsm_scale"].to(device)
            if scale.dim() == 1: scale = scale.unsqueeze(-1)

            # Normalize GT turtle positions by the tree's OWN bounding radius.
            # compute_states_numpy starts at [0,0,0] in tree-local space, so the
            # DSM world-space center/scale is the WRONG frame and produces a
            # constant offset that the model cannot learn.  Instead we normalise
            # each sequence by its own centroid + max-radius so positions land
            # roughly in [-1, 1] regardless of tree size.
            with torch.no_grad():
                gt_pos = gt_states[..., :3].float()          # (B, T, 3) tree-local
                valid3 = valid_mask.unsqueeze(-1).expand_as(gt_pos)
                n_valid = valid_mask.float().sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
                gt_pos_v = gt_pos.clone();  gt_pos_v[~valid3] = 0.0
                tree_centroid = gt_pos_v.sum(dim=1, keepdim=True) / n_valid   # (B, 1, 3)
                centered = gt_pos - tree_centroid
                centered[~valid3] = 0.0
                # Max distance from centroid across ONLY valid tokens (ignore padding)
                # mask_fill with -inf ensures PAD tokens don't contribute to 'amax'
                dist_from_centroid = centered.norm(dim=-1)
                dist_from_centroid = dist_from_centroid.masked_fill(~valid_mask, -1e9)
                tree_radius = dist_from_centroid.amax(dim=1, keepdim=True).unsqueeze(-1).clamp(min=0.1)
                gt_states_norm_local = centered / tree_radius             # (B, T, 3) in ~[-1,1]

            loss_state = F.mse_loss(
                pred_state.float()[valid_mask][..., :3],
                gt_states_norm_local.float()[valid_mask]
            ) if valid_mask.any() else torch.tensor(0.0, device=device)

            # --------------------------------------------------
            # GEOMETRY-AWARE LOSS (The Most Effective Geometric Grounding)
            # --------------------------------------------------
            loss_chamfer = torch.tensor(0.0, device=device)

            if chamfer_loss_fn is not None and batch_idx % chamfer_sample_freq == 0:
                try:
                    # 🔴 SPATIAL SYNCHRONIZATION: Chamfer in Unit Cube Space
                    center = batch["dsm_center"].to(device)
                    scale  = batch["dsm_scale"].to(device)
                    if scale.dim() == 1: scale = scale.unsqueeze(-1)
                    
                    st_tgt = states_tgt
                    st_in  = states_in

                    st_tgt_norm = st_tgt.clone()
                    st_tgt_norm[..., :3] = (st_tgt[..., :3] - center[:, None, :]) / (scale[:, None, :] + 1e-9) * 2.0
                    st_in_norm = st_in.clone()
                    st_in_norm[..., :3] = (st_in[..., :3] - center[:, None, :]) / (scale[:, None, :] + 1e-9) * 2.0

                    s_factor = 2.0 / (scale + 1e-9)
                    
                    # 🔴 BRANCH-AWARE RECURRENT DIFFERENTIABILITY:
                    pred_pos_soft = compute_differentiable_turtle_positions(
                        vlog, st_in_norm, t_tgt, tokenizer.f_bins, tokenizer.theta_bins, tokenizer.phi_bins,
                        scale_factor=s_factor
                    )
                    
                    # Use the advanced chamfer_distance with move_mask and target_mask
                    # 🔴 SCALE FIX: Setting normalize=False ensures the model learns the CORRECT SIZE 
                    # of the branches, not just the relative shape.
                    loss_chamfer, sub_pred_pts, sub_gt_pts = chamfer_loss_fn(
                        pred_pos_soft.float(),
                        st_tgt_norm[..., :3].float(),
                        move_mask=(t_tgt == TokenType.F).float(),
                        target_mask=(t_tgt == TokenType.F).float(),
                        num_points=1024,
                        return_pts=True,
                        normalize=False
                    )

                    with torch.no_grad():
                        # ✅ ALIGNMENT FIX: Only visualize points the loss is actually penalizing (F-tokens)
                        # We un-normalize them back to world-space so they match the main visualization
                        f_mask_pred = (t_tgt[0] == TokenType.F).cpu()
                        
                        # Use world-space coordinates for the plot so they are recognizable
                        # pred_pos_soft[0] is (T, 3) normalized
                        pred_world = (pred_pos_soft[0].cpu() * scale[0].cpu() / 2.0) + center[0].cpu()
                        gt_world   = (st_tgt[0,...,:3].cpu())
                        
                        if f_mask_pred.any():
                            _last_chamfer_pred_np = pred_world[f_mask_pred].float().numpy()
                            _last_chamfer_gt_np   = gt_world[f_mask_pred].float().numpy()
                        else:
                            # Fallback to sparse points if no F-tokens in first batch item
                            _last_chamfer_pred_np = pred_world.float().numpy()
                            _last_chamfer_gt_np   = gt_world.float().numpy()


                except Exception as e:
                    print(f"[WARN] Chamfer calc failed in train5_2: {e}")
                    # import traceback; traceback.print_exc()
                    loss_chamfer = torch.tensor(0.0, device=device)

            # --------------------------------------------------
            # FINAL LOSS
            # --------------------------------------------------
            # species weight 0.2 → 0.02: species CE was dominating the visual encoder gradient
            # and pushing it toward per-species templates.  Structural losses should drive
            # visual encoding of individual tree geometry.
            loss = loss_type + 10.0 * loss_val + 0.02 * loss_sp

            # Rotation smoothness: weight kept near-zero because absolute
            # spherical directions don't require consecutive-segment smoothness
            # (branches legitimately jump to very different angles).
            if use_rotation_smoothness:
                loss = loss + 0.0001 * loss_rot_smooth

            # Geometry-aware loss
            if chamfer_loss_fn is not None:
                loss = loss + 10.0 * chamfer_weight * loss_chamfer

            # Soft angle L1: 2.0 is appropriate (values in [0, bins-1])
            # State MSE weight raised 2.0 → 5.0: this is the ONLY per-tree-discriminative loss;
            # it must dominate enough that visual encoder is forced to encode individual geometry.
            loss = loss + 2.0 * loss_soft_angle + 5.0 * loss_state

            # --------------------------------------------------
            # BACKPROP (with proper AMP scaling)
            # --------------------------------------------------
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            # Accumulate for epoch-level logging (only .item() calls in the file are here — correct)
            epoch_loss            += loss.detach()
            epoch_loss_type       += loss_type.detach()
            epoch_loss_val        += loss_val.detach() if torch.is_tensor(loss_val) else loss_val
            epoch_loss_rot_smooth += loss_rot_smooth.detach()
            epoch_loss_chamfer    += loss_chamfer.detach()
            epoch_loss_soft_angle += loss_soft_angle.detach()
            epoch_loss_state      += loss_state.detach()
            if ep % 5 == 0 and batch_idx == 0:
                with torch.no_grad():
                    # Use bracket-safe generation
                    pred_str = model.generate(
                        dsm[0:1],
                        ortho[0:1],
                        tokenizer,
                        max_len=dataset.window,
                        temperature=1.0,
                        temperature_structural=0.7
                    )

                    tid = batch["tid"][0]
                    scale_0  = batch["dsm_scale"][0].item()
                    s_factor = 2.0 / (scale_0 + 1e-9)
                    
                    # Load ground truth string DIRECTLY from the text file to preserve raw format (B vs S)
                    gt_path = os.path.join(dataset.lstring_dir, f"{tid}.txt")
                    try:
                        with open(gt_path, "r", encoding="utf-8") as f_gt:
                            gt_str = f_gt.read().strip()
                    except:
                        # Fallback to decode
                        gt_types = batch["type_tgt"][0].cpu().tolist() if torch.is_tensor(batch["type_tgt"]) else batch["type_tgt"][0]
                        gt_vals  = batch["val_tgt"][0].cpu().tolist() if torch.is_tensor(batch["val_tgt"]) else batch["val_tgt"][0]
                        gt_str = tokenizer.decode(gt_types, gt_vals)

                    gt_pts = render_lsystem(gt_str, step_scale=s_factor, num_bins_theta=theta_bins, num_bins_phi=phi_bins, num_bins_f=f_bins)
                    pred_pts = render_lsystem(pred_str, step_scale=s_factor, num_bins_theta=theta_bins, num_bins_phi=phi_bins, num_bins_f=f_bins)
                    # states_tgt is tree-local (starts at [0,0,0]), not world-space.
                    # render_lsystem also starts at [0,0,0], so no offset is needed.
                    # (Previously used states_tgt[0,0,:3] as "base_world" which was
                    # always ~[0,0,0], yielding offset ≈ -center/scale — wrong frame.)

                    viz.visualize(
                        ortho=batch["ortho"][0],
                        dsm=batch["dsm"][0],
                        gt_lstring=gt_str,
                        pred_lstring=pred_str,
                        step=batch_idx,
                        gt_pts=gt_pts,
                        pred_pts=pred_pts,
                    )

                    # 🔴 STATE LOSS DEBUG: Visualize the local skeleton grounding
                    try:
                        # Full window visualization to verify length consistency
                        # pred_state predicts centered/radius normalized positions
                        p_state_local = pred_state[0, :, :3].detach().cpu()
                        g_state_local = gt_states_norm_local[0].cpu()
                        
                        viz.show_state_debug(
                            g_state_local.numpy(),
                            p_state_local.numpy(),
                            mask=valid_mask[0].cpu().numpy(),
                            title=f"State Loss Debug (Local Trajectory) ep{ep}"
                        )
                    except Exception as _e_state:
                         print(f"[WARN] State debug viz failed: {_e_state}")

                    # Chamfer INPUT point-clouds
                    if _last_chamfer_pred_np is not None and _last_chamfer_gt_np is not None:
                        try:
                            viz.show_diff_vs_gt(
                                _last_chamfer_gt_np, 
                                _last_chamfer_pred_np,
                                title=f"Chamfer INPUT PCs (GT=blue, Pred=orange) ep{ep}"
                            )
                        except Exception as _e:
                            pass

                            pass

        # .item() is correct here — we are outside the per-batch loop, doing logging
        n_batches = len(loader)
        avg_loss          = epoch_loss.item()            / n_batches
        avg_loss_type     = epoch_loss_type.item()       / n_batches
        avg_loss_val      = (epoch_loss_val.item() if torch.is_tensor(epoch_loss_val) else epoch_loss_val) / n_batches
        avg_loss_rot_smooth = epoch_loss_rot_smooth.item() / n_batches if use_rotation_smoothness else 0.0
        avg_loss_chamfer  = epoch_loss_chamfer.item()    / n_batches
        avg_loss_soft_angle = epoch_loss_soft_angle.item() / n_batches
        avg_loss_state      = epoch_loss_state.item()      / n_batches

        print(f"Epoch {ep} | Total: {avg_loss:.4f} | Type: {avg_loss_type:.4f} | Val: {avg_loss_val:.4f}", end="")
        if use_rotation_smoothness:
            print(f" | RotSmooth: {avg_loss_rot_smooth:.4f}", end="")
        if chamfer_loss_fn is not None:
            print(f" | Chamfer: {avg_loss_chamfer:.4f}", end="")
        print(f" | Angle: {avg_loss_soft_angle:.4f} | State: {avg_loss_state:.4f}")

        # Step LR based on training loss
        scheduler.step(avg_loss)

        # Print current learning rate
        current_lr = opt.param_groups[0]['lr']
        print(f"[LR] Current learning rate: {current_lr:.2e}")

        # ── Visdom loss curves (legend items are clickable to toggle on/off) ──
        losses_to_plot = {
            "Train Total":     avg_loss,
            "Train Type":      avg_loss_type,
            "Train Val":       avg_loss_val,
            "Train SoftAngle": avg_loss_soft_angle,
            "Train State":     avg_loss_state,
            "LR":              current_lr,
        }
        if use_rotation_smoothness:
            losses_to_plot["Train RotSmooth"] = avg_loss_rot_smooth
        if chamfer_loss_fn is not None:
            losses_to_plot["Train Chamfer"] = avg_loss_chamfer
        try:
            viz.plot_losses(ep, losses_to_plot)
        except Exception as _e:
            pass

        # -----------------------------------------------------
        # SAVE CHECKPOINT
        # -----------------------------------------------------
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": ep + 1,
        }, save_path)

        print(f"[OK] Saved checkpoint → {save_path}")


# ============================================================
# OPTIONAL: SIMPLE EVALUATION & GENERATION DEMO
# ============================================================

@torch.no_grad()
def generate_example(model, tokenizer, dsm, ortho, device="cuda"):
    """
    Example generation for sanity check.
    """
    model.eval()
    out = model.generate(dsm.to(device), ortho.to(device), tokenizer)
    print("Generated L-system:")
    print(out)


if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Train L-System model with geometric stability fixes")
    parser.add_argument("--lstrings_path", type=str, default="LSTRINGS_FINAL") # SYMBOLIC_LSTRINGS_d3
    parser.add_argument("--id_cache", type=str, default="1")
    parser.add_argument("--num_trees", type=int, default=450)
    parser.add_argument("--base", type=str, default="/home/grammatikakis1/TREES_DATASET_SIDE")

    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--normalize", action="store_true")
    
    # NEW: Stability enhancement flags
    parser.add_argument("--no_scheduled_sampling", action="store_true", 
                        help="Disable scheduled sampling (use pure teacher forcing)")
    parser.add_argument("--ss_max_prob", type=float, default=0.8,
                        help="Maximum scheduled sampling probability (higher = less exposure bias)")
    
    parser.add_argument("--no_position_weights", action="store_true",
                        help="Disable position-weighted loss (early errors matter more)")
    parser.add_argument("--position_alpha", type=float, default=0.002,
                        help="Position weight decay rate (smaller = slower decay)")
    
    parser.add_argument("--chamfer_loss", action="store_true",
                        help="Enable Chamfer distance loss (geometry-aware, expensive)")
    parser.add_argument("--chamfer_weight", type=float, default=1.0,
                        help="Weight for Chamfer loss (keep ≤ 5× type-CE to avoid collapse)")
    parser.add_argument("--chamfer_freq", type=int, default=1,
                        help="Compute Chamfer loss every N batches")
    parser.add_argument("--max_points_chamfer", type=int, default=1000,
                        help="Number of points for Chamfer loss")
    
    parser.add_argument("--no_rotation_smoothness", action="store_true",
                        help="Disable rotation smoothness regularization")
    parser.add_argument("--rotation_weight", type=float, default=0.01,
                        help="Weight for rotation smoothness loss")
    
    # NEW: Truncated BPTT flags
    parser.add_argument("--no_truncated_bptt", action="store_true",
                        help="Disable truncated BPTT (process full sequences, may OOM on long sequences)")
    # parser.add_argument("--bptt_chunk_size", type=int, default=2048,
    #                     help="Chunk size for truncated BPTT (default: 2048)")
    
    parser.add_argument("--no_species_loss", action="store_true",
                        help="Disable species classification supervision")

    parser.add_argument("--dim", type=int, default=512, help="Dimension of the model")
    parser.add_argument("--heads", type=int, default=16, help="Number of heads")
    parser.add_argument("--layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--student_forcing", action="store_true",
                        help="Enable 100% student forcing (autonomous generation during training)")

    parser.add_argument("--f_bins", type=int, default=12, help="Number of bins for F")
    parser.add_argument("--theta_bins", type=int, default=12, help="Number of bins for Theta")
    parser.add_argument("--phi_bins", type=int, default=15, help="Number of bins for Phi")
    parser.add_argument("--modality", type=str, default="both", choices=["both", "dsm", "ortho"])

    args = parser.parse_args()
    
    bptt_chunk_size = args.window

    save = f"p3_eff{args.id_cache}.pth"

    BASE = args.base

    lstrings_path = args.lstrings_path

    # --------------------------------------------------------
    # Select IDs  (10 % val split by tree ID, not by window)
    # --------------------------------------------------------
    all_ids = sorted([f[:-4] for f in os.listdir(os.path.join(BASE, lstrings_path))
                      if f.endswith(".txt")])
    
    # Shuffle to ensure species diversity (otherwise alphabetically late species like Spruce are excluded)
    random.seed(42)
    random.shuffle(all_ids)

    ids = all_ids[: args.num_trees]
    held_out_ids = all_ids[args.num_trees:]

    # Save memory of held-out IDs for inference
    held_out_file = f"{args.id_cache}_held_out_ids.txt"
    with open(held_out_file, "w", encoding="utf-8") as f:
        for hid in held_out_ids:
            f.write(f"{hid}\n")

    print(f"[INFO] Train trees: {len(ids)}")
    print(f"[INFO] Held-out trees saved to {held_out_file} (Count: {len(held_out_ids)})")
    print(f"[INFO] Stability features:")
    print(f"  - Bracket-safe generation: ENABLED (always on)")
    print(f"  - Scheduled sampling: {'DISABLED' if args.no_scheduled_sampling else 'ENABLED (prefix-consistent)'}")
    print(f"  - Position-weighted loss: {'DISABLED' if args.no_position_weights else 'ENABLED'}")
    print(f"  - Rotation smoothness: {'DISABLED' if args.no_rotation_smoothness else 'ENABLED'}")
    print(f"  - Chamfer loss: {'ENABLED (GT L-system geometry)' if args.chamfer_loss else 'DISABLED'}")
    print(f"  - Strict Unit-Cube Normalization: {'ENABLED' if args.normalize else 'DISABLED'}")


    tokenizer = LSystemTokenizerV2(f_bins=args.f_bins, theta_bins=args.theta_bins, phi_bins=args.phi_bins)
    # (f_bins=10, theta_bins=6, phi_bins=6)

    dataset = LSystemDataset(
        base_path=BASE,
        lstring_dir=lstrings_path,
        normalize=args.normalize,
        tokenizer=tokenizer,
        ids=ids,
        window=args.window,
        overlap=0,
        training=True,
    )

    print(f'[INFO] Dataset loaded. Train samples: {len(dataset)}')

    train_model(
        dataset,
        tokenizer,
        save_path=save,
        batch_size=args.batch,
        epochs=args.epochs,
        env=f"p3_eff{args.id_cache}",
        resume_path=args.resume,
        # Truncated BPTT
        use_truncated_bptt=not args.no_truncated_bptt,
        bptt_chunk_size=bptt_chunk_size,
        # Stability enhancements
        use_scheduled_sampling=not args.no_scheduled_sampling,
        ss_max_prob=args.ss_max_prob,
        use_position_weights=not args.no_position_weights,
        position_weight_alpha=args.position_alpha,
        use_chamfer_loss=args.chamfer_loss,
        chamfer_weight=args.chamfer_weight,
        chamfer_sample_freq=args.chamfer_freq,
        max_points_chamfer=args.max_points_chamfer,
        use_rotation_smoothness=not args.no_rotation_smoothness,
        rotation_smoothness_weight=args.rotation_weight,
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        use_student_forcing=args.student_forcing,
        f_bins=args.f_bins,
        theta_bins=args.theta_bins,
        phi_bins=args.phi_bins,
        modality=args.modality,
        use_species_loss=not args.no_species_loss,
    )

# CUDA_VISIBLE_DEVICES=0 python train.py --lstrings_path "SYMBOLIC_LSTRINGS_d3" --id_cache "symbolic_L" --window 1024 --num_trees 750 --batch 16 --resume p3_effsymbolic_L.pth --epochs 5000 --no_rotation_smoothness --no_position_weights --no_scheduled_sampling --no_truncated_bptt --f_bins 10 --theta_bins 6 --phi_bins 6

# CUDA_VISIBLE_DEVICES=5 python train.py --lstrings_path "SYMBOLIC_LSTRINGS_d2" --id_cache "symbolic_S" --window 256 --num_trees 750 --batch 16 --resume p3_effsymbolic_S.pth --epochs 5000 --no_rotation_smoothness --no_position_weights --no_scheduled_sampling --no_truncated_bptt --f_bins 10 --theta_bins 6 --phi_bins 6

# CUDA_VISIBLE_DEVICES=4 python train.py --lstrings_path "SYMBOL_LSTRINGS_d3" --id_cache "symbolic_L_small" --window 700 --num_trees 750 --batch 16 --resume p3_effsymbolic_L_small.pth --epochs 5000 --no_rotation_smoothness --no_position_weights --no_scheduled_sampling --no_truncated_bptt --f_bins 5 --theta_bins 5 --phi_bins 5

