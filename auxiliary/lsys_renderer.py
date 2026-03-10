import torch
import numpy as np
import re
from typing import List, Tuple

# ============================================================
# CONSTANTS (must match tokenizer / training)
# ============================================================

# NUM_BINS_THETA = 6
# NUM_BINS_PHI   = 6
# NUM_BINS_F     = 10

# Token layout (must match dataset)
TYPE_F   = 0
TYPE_LBR = 1
TYPE_RBR = 2
TYPE_EOS = 3
TYPE_PAD = 4


# ============================================================
# TORCH JIT RENDERER (ABSOLUTE SPHERICAL)
# ============================================================

@torch.jit.script
def render_loop_jit(
    p_length: torch.Tensor,   # (B, T)  length bins (already converted to real length)
    p_theta: torch.Tensor,    # (B, T)  radians
    p_phi: torch.Tensor,      # (B, T)  radians
    p_types: torch.Tensor,    # (B, T, num_types)
    max_depth: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:

    B, T = p_length.shape
    device = p_length.device
    dtype = p_length.dtype

    pos = torch.zeros((B, 3), device=device, dtype=dtype)

    stack = [torch.zeros((B, 3), device=device, dtype=dtype) for _ in range(max_depth)]
    stack_ptr = torch.zeros((B,), device=device, dtype=torch.long)

    pos_list = torch.jit.annotate(List[torch.Tensor], [])

    for t in range(T):

        types_t = p_types[:, t]

        is_f   = types_t[:, 0].unsqueeze(-1)
        is_lbr = types_t[:, 1].unsqueeze(-1)
        is_rbr = types_t[:, 2].unsqueeze(-1)

        # -------------------------------
        # Absolute spherical direction
        # -------------------------------
        theta = p_theta[:, t]
        phi   = p_phi[:, t]

        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        sin_p = torch.sin(phi)
        cos_p = torch.cos(phi)

        direction = torch.stack([
            sin_t * cos_p,
            sin_t * sin_p,
            cos_t
        ], dim=-1)

        # -------------------------------
        # Move forward
        # -------------------------------
        step = p_length[:, t].unsqueeze(-1)
        pos = pos + direction * (step * is_f)

        # -------------------------------
        # Push stack
        # -------------------------------
        push_mask = (is_lbr.squeeze(-1) > 0.5) & (stack_ptr < max_depth)

        if push_mask.any():
            for d in range(max_depth):
                m = (push_mask & (stack_ptr == d)).unsqueeze(-1).to(dtype)
                stack[d] = stack[d] * (1.0 - m) + pos * m
            stack_ptr = torch.where(push_mask, stack_ptr + 1, stack_ptr)

        # -------------------------------
        # Pop stack
        # -------------------------------
        pop_mask = (is_rbr.squeeze(-1) > 0.5) & (stack_ptr > 0)

        if pop_mask.any():
            pop_idx = torch.clamp(stack_ptr - 1, min=0)
            saved = torch.stack(stack, dim=0)[pop_idx, torch.arange(B, device=device), :]
            m = pop_mask.unsqueeze(-1).to(dtype)
            pos = pos * (1.0 - m) + saved * m
            stack_ptr = torch.where(pop_mask, stack_ptr - 1, stack_ptr)

        pos_list.append(pos)

    return torch.stack(pos_list, dim=1), pos


# ============================================================
# BIN → DIRECTION (NUMPY VERSION)
# ============================================================

def direction_from_bins(theta_bin, phi_bin, num_bins_theta, num_bins_phi):

    theta = (theta_bin + 0.5) / num_bins_theta * np.pi
    phi   = (phi_bin + 0.5) / num_bins_phi * 2.0 * np.pi

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.array([x, y, z], dtype=np.float32)


# ============================================================
# NUMPY RENDERER (FOR VISUALIZATION)
# ============================================================

def render_lsystem(input_data, step_scale=1.0, max_points=200000, num_bins_theta=6, num_bins_phi=6, num_bins_f=10):

    if isinstance(input_data, str):
        seq = parse_lstring(input_data)
    else:
        seq = input_data

    pos = np.array([0., 0., 0.], dtype=np.float32)
    pts = [pos.copy()]
    stack = []

    for sym, param in seq:

        if sym == "SEGMENT":

            length_bin, theta_bin, phi_bin = param

            direction = direction_from_bins(theta_bin, phi_bin, num_bins_theta, num_bins_phi)
            length = (length_bin + 0.5) / num_bins_f * step_scale

            pos = pos + direction * length
            pts.append(pos.copy())

        elif sym == "[":
            stack.append(pos.copy())

        elif sym == "]":
            if stack:
                pos = stack.pop()
                pts.append(pos.copy())

        if len(pts) >= max_points:
            break

    return np.array(pts, dtype=np.float32)


# ============================================================
# PARSER (Supports S and legacy B format)
# ============================================================

def parse_lstring(lstring):

    # Match both S{theta}_{phi}_{len} and B{theta}_{phi}F{len}
    re_seg = re.compile(r"([BSbs])(\d+)_(\d+)[F_]?(\d+)")
    re_token = re.compile(r"(?:[BSbs]\d+_\d+[F_]?\d+|\[|\])")

    seq = []

    for m in re_token.finditer(lstring):

        tok = m.group(0)

        mS = re_seg.match(tok)
        if mS:
            # Groups: 1=type, 2=theta, 3=phi, 4=length
            theta = int(mS.group(2))
            phi   = int(mS.group(3))
            length = int(mS.group(4))

            seq.append(("SEGMENT", (length, theta, phi)))
            continue

        if tok == "[":
            seq.append(("[", None))
        elif tok == "]":
            seq.append(("]", None))

    return seq