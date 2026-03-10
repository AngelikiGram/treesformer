import torch
import torch.nn.functional as F
from typing import Tuple

@torch.jit.script
def render_loop_jit_fast(
    p_step: torch.Tensor,
    p_rot: torch.Tensor,
    p_types: torch.Tensor,
    init_pos: torch.Tensor,
    init_H: torch.Tensor,
    init_L: torch.Tensor,
    init_U: torch.Tensor,
    max_depth: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast differentiable renderer with fixed, vectorized stack for max_depth=3.
    Supports up to 3 levels of branching efficiently.
    """
    device = p_types.device
    B, T, _ = p_types.shape
    dtype = p_types.dtype
    out_dtype = torch.float16 if dtype == torch.float16 else torch.float32
    pos_output = torch.zeros(B, T, 3, device=device, dtype=out_dtype)
    orient_output = torch.zeros(B, T, 6, device=device, dtype=out_dtype)
    move_mask_output = torch.zeros(B, T, device=device, dtype=out_dtype)
    cur_pos = init_pos.to(out_dtype)
    H = init_H.to(out_dtype)
    L = init_L.to(out_dtype)
    U = init_U.to(out_dtype)
    TYPE_F, TYPE_R, TYPE_LBR, TYPE_RBR = 0, 1, 2, 3
    cos_rot, sin_rot = torch.cos(p_rot), torch.sin(p_rot)
    eps = 1e-5
    # Fixed-size stack: (B, max_depth, 12)
    stack = torch.zeros(B, max_depth, 12, device=device, dtype=out_dtype)
    stack_ptr = torch.zeros(B, dtype=torch.long, device=device)
    b_idx = torch.arange(B, device=device)
    for t in range(T):
        p = p_types[:, t]
        is_f = p[:, TYPE_F].unsqueeze(-1)
        is_r = p[:, TYPE_R].unsqueeze(-1)
        is_lbr = p[:, TYPE_LBR].unsqueeze(-1)
        is_rbr = p[:, TYPE_RBR].unsqueeze(-1)
        move_mask_output[:, t] = is_f.squeeze(-1)
        cur_pos = cur_pos + H * (p_step[:, t].view(-1, 1) * is_f)
        # Rotations (fused)
        c1, s1 = cos_rot[:, t, 0].view(B, 1), sin_rot[:, t, 0].view(B, 1)
        c2, s2 = cos_rot[:, t, 1].view(B, 1), sin_rot[:, t, 1].view(B, 1)
        c3, s3 = cos_rot[:, t, 2].view(B, 1), sin_rot[:, t, 2].view(B, 1)
        H_y = H * c1 + torch.cross(U, H, dim=-1) * s1 + U * (torch.sum(U * H, dim=-1, keepdim=True) * (1.0 - c1))
        L_y = L * c1 + torch.cross(U, L, dim=-1) * s1 + U * (torch.sum(U * L, dim=-1, keepdim=True) * (1.0 - c1))
        H_x = H_y * c2 + torch.cross(L_y, H_y, dim=-1) * s2 + L_y * (torch.sum(L_y * H_y, dim=-1, keepdim=True) * (1.0 - c2))
        U_x = U * c2 + torch.cross(L_y, U, dim=-1) * s2 + L_y * (torch.sum(L_y * U, dim=-1, keepdim=True) * (1.0 - c2))
        L_z = L_y * c3 + torch.cross(H_x, L_y, dim=-1) * s3 + H_x * (torch.sum(H_x * L_y, dim=-1, keepdim=True) * (1.0 - c3))
        U_z = U_x * c3 + torch.cross(H_x, U_x, dim=-1) * s3 + H_x * (torch.sum(H_x * U_x, dim=-1, keepdim=True) * (1.0 - c3))
        H = H * (1.0 - is_r) + H_x * is_r
        L = L * (1.0 - is_r) + L_z * is_r
        U = U * (1.0 - is_r) + U_z * is_r
        # Always normalize
        H = H / (torch.norm(H, dim=-1, keepdim=True) + eps)
        L_cross = torch.cross(U, H, dim=-1)
        L = L_cross / (torch.norm(L_cross, dim=-1, keepdim=True) + eps)
        U = torch.cross(H, L, dim=-1)
        U = U / (torch.norm(U, dim=-1, keepdim=True) + eps)
        # Stack push (vectorized, max_depth=3)
        lbr_mask = (is_lbr.squeeze(-1) > 0.5) & (stack_ptr < max_depth)
        if lbr_mask.any():
            state = torch.cat([cur_pos, H, L, U], dim=1)
            d = stack_ptr.clamp(0, max_depth-1)
            stack[b_idx, d] = torch.where(lbr_mask.unsqueeze(-1), state, stack[b_idx, d])
            stack_ptr = torch.where(lbr_mask, stack_ptr + 1, stack_ptr)
        # Stack pop (vectorized, max_depth=3)
        rbr_mask = (is_rbr.squeeze(-1) > 0.5) & (stack_ptr > 0)
        if rbr_mask.any():
            d = (stack_ptr - 1).clamp(0, max_depth-1)
            saved = stack[b_idx, d]
            m = rbr_mask.unsqueeze(-1).to(out_dtype)
            cur_pos = cur_pos * (1.0 - m) + saved[:, 0:3] * m
            H = H * (1.0 - m) + saved[:, 3:6] * m
            L = L * (1.0 - m) + saved[:, 6:9] * m
            U = U * (1.0 - m) + saved[:, 9:12] * m
            stack_ptr = torch.where(rbr_mask, stack_ptr - 1, stack_ptr)
        pos_output[:, t] = cur_pos
        orient_output[:, t] = torch.cat([H, U], dim=-1)
    return pos_output, orient_output, move_mask_output


@torch.jit.script
def render_loop_hard(
    p_step: torch.Tensor,      # (B, T)
    p_rot: torch.Tensor,       # (B, T, 3)
    p_types: torch.Tensor,     # (B, T, C) Can be soft probabilities or hard-coded
    init_pos: torch.Tensor,
    init_H: torch.Tensor,
    init_L: torch.Tensor,
    init_U: torch.Tensor,
    max_depth: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, _ = p_types.shape
    device = p_types.device
    eps = 1e-5

    # Lists to collect outputs (better for gradient history than pre-allocated tensor with copy_)
    pos_list = []
    orient_list = []
    move_mask_list = []

    cur_pos = init_pos
    H = init_H
    L = init_L
    U = init_U

    TYPE_F, TYPE_R, TYPE_LBR, TYPE_RBR = 0, 1, 2, 3

    # Stack state (B, D, 12).
    # We initialize with zeros.
    stack = torch.zeros(B, max_depth, 12, device=device)
    
    # Pointer is discrete (Hard logic still needed for addressing)
    stack_ptr = torch.zeros(B, dtype=torch.long, device=device)
    d_indices = torch.arange(max_depth, device=device).view(1, max_depth, 1)

    cos_rot = torch.cos(p_rot)
    sin_rot = torch.sin(p_rot)

    # Pre-allocate batch index so torch.arange is not re-created every step
    b_idx = torch.arange(B, device=device)

    for t in range(T):
        token = p_types[:, t]  # (B, C)
        is_f   = token[:, TYPE_F].view(B, 1)
        is_r   = token[:, TYPE_R].view(B, 1)
        is_lbr = token[:, TYPE_LBR].view(B, 1)
        is_rbr = token[:, TYPE_RBR].view(B, 1)

        move_mask_list.append(is_f.view(-1))

        # ---- FORWARD (Algebraic) ----
        step = p_step[:, t].view(B, 1)
        cur_pos = cur_pos + H * (step * is_f)

        # ---- ROTATION (Algebraic) ----
        c1, s1 = cos_rot[:, t, 0].view(B, 1), sin_rot[:, t, 0].view(B, 1)
        c2, s2 = cos_rot[:, t, 1].view(B, 1), sin_rot[:, t, 1].view(B, 1)
        c3, s3 = cos_rot[:, t, 2].view(B, 1), sin_rot[:, t, 2].view(B, 1)

        H_y = H * c1 + torch.cross(U, H, dim=-1) * s1
        L_y = L * c1 + torch.cross(U, L, dim=-1) * s1

        H_x = H_y * c2 + torch.cross(L_y, H_y, dim=-1) * s2
        U_x = U * c2 + torch.cross(L_y, U, dim=-1) * s2

        L_z = L_y * c3 + torch.cross(H_x, L_y, dim=-1) * s3
        U_z = U_x * c3 + torch.cross(H_x, U_x, dim=-1) * s3

        # Weighted mix for rotation
        H = H * (1.0 - is_r) + H_x * is_r
        L = L * (1.0 - is_r) + L_z * is_r
        U = U * (1.0 - is_r) + U_z * is_r

        # Normalization (Safe)
        H = H / (torch.norm(H, dim=-1, keepdim=True) + eps)
        L = torch.cross(U, H, dim=-1)
        L = L / (torch.norm(L, dim=-1, keepdim=True) + eps)
        U = torch.cross(H, L, dim=-1)

        # ---- Vectorized Stack Update (Compiler Friendly) ----
        state_to_push = torch.cat([cur_pos, H, L, U], dim=-1).unsqueeze(1) # (B, 1, 12)
        # Write IF (is_lbr AND pointer matches slot)
        write_mask = (stack_ptr.view(B, 1, 1) == d_indices) & (is_lbr.view(B, 1, 1) > 0.5)
        stack = torch.where(write_mask, state_to_push, stack)
        
        # Update Pointer (Discrete)
        can_push = (stack_ptr < max_depth)
        stack_ptr = torch.where((is_lbr.view(-1) > 0.5) & can_push, stack_ptr + 1, stack_ptr)

        # ---- POP (Algebraic Restore) ----
        can_pop = (stack_ptr > 0)
        pop_idx = (stack_ptr - 1).clamp(0, max_depth - 1)
        
        # Gather saved from current hard pointer
        saved = stack[b_idx, pop_idx]
        pop_strength = is_rbr * can_pop.view(B, 1).float()
        
        # Blend state
        cur_pos = cur_pos * (1.0 - pop_strength) + saved[:, 0:3] * pop_strength
        H = H * (1.0 - pop_strength) + saved[:, 3:6] * pop_strength
        L = L * (1.0 - pop_strength) + saved[:, 6:9] * pop_strength
        U = U * (1.0 - pop_strength) + saved[:, 9:12] * pop_strength

        # Update Pointer (Discrete)
        stack_ptr = torch.where((is_rbr.view(-1) > 0.5) & can_pop, stack_ptr - 1, stack_ptr)

        pos_list.append(cur_pos)
        orient_list.append(torch.cat([H, U], dim=-1))

    return torch.stack(pos_list, dim=1), torch.stack(orient_list, dim=1), torch.stack(move_mask_list, dim=1)

