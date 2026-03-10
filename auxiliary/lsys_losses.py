import torch
import torch.nn as nn
import torch.nn.functional as F
from auxiliary.lsys_tokenizer import TokenType

def chamfer_distance(source, target, move_mask=None, target_mask=None, num_points=1024, tau=0.1, return_pts=False, normalize=True):
    """
    Compute 'The Most Effective' Chamfer Distance.
    ...
    Args:
        ...
        normalize: If True, internally normalizes both point-clouds to a unit scale (erases absolute size).
                   If False, computes loss in the provided coordinate space (preserves absolute size).
    """
    B, N, _ = source.shape
    B_t, M, _ = target.shape
    device = source.device
    
    # 1. Subsample points if necessary
    def get_indices(mask, n_total, k_num):
        if n_total <= k_num:
            return torch.arange(n_total, device=device)
        if mask is not None:
            # Importance sampling based on mask (favor F tokens)
            w = (mask.float().mean(0) + 0.05) 
            return torch.multinomial(w, k_num, replacement=False)
        return torch.randperm(n_total, device=device)[:k_num]

    idx_s = get_indices(move_mask, N, num_points)
    idx_t = get_indices(target_mask, M, num_points)
        
    s_sub = source[:, idx_s, :].float()
    t_sub = target[:, idx_t, :].float()
    m_s = move_mask[:, idx_s].float() if move_mask is not None else torch.ones(B, len(idx_s), device=device)
    m_t = target_mask[:, idx_t].float() if target_mask is not None else torch.ones(B, len(idx_t), device=device)
    
    # 2. INTERNAL SCALE NORMALIZATION (Optional)
    if normalize:
        with torch.no_grad():
            t_max = t_sub.max(dim=1, keepdim=True)[0] 
            t_min = t_sub.min(dim=1, keepdim=True)[0] 
            t_center = (t_max + t_min) / 2.0
            t_scale = (t_max - t_min).max(dim=2, keepdim=True)[0] + 1e-6 
            
        s_norm = (s_sub - t_center) / t_scale
        t_norm = (t_sub - t_center) / t_scale
    else:
        s_norm = s_sub
        t_norm = t_sub
    
    # 3. Compute Distance Matrix (Squared Euclidean)
    dist_sq = torch.cdist(s_norm, t_norm, p=2)**2 # (B, num_s, num_t)
    
    # 4. Source to Target (Pred -> GT) - Each pred finding its GT match
    # Softmin via softmax over negative distances
    weights_s2t = torch.softmax(-dist_sq / max(tau, 1e-6), dim=2)
    d_soft_s2t = (dist_sq * weights_s2t).sum(dim=2) # (B, num_s)
    
    # Only penalize points that are likely to be geometry in source
    loss_s2t = (d_soft_s2t * m_s).sum() / (m_s.sum() + 1e-6)
    
    # 5. Target to Source (GT -> Pred) - Each GT finding its Pred match
    dist_sq_t2s = dist_sq.transpose(1, 2) # (B, num_t, num_s)
    
    # Penalty for matching to a non-moving token in source:
    penalty = (1.0 - m_s).unsqueeze(1) * 10.0 
    dist_sq_t2s_p = dist_sq_t2s + penalty
    
    weights_t2s = torch.softmax(-dist_sq_t2s_p / max(tau, 1e-6), dim=2)
    d_soft_t2s = (dist_sq_t2s_p * weights_t2s).sum(dim=2) # (B, num_t)
    
    # Only penalize points that are geometry in ground truth
    loss_t2s = (d_soft_t2s * m_t).sum() / (m_t.sum() + 1e-6)
    
    loss = loss_s2t + loss_t2s
    if return_pts:
        return loss, s_norm, t_norm
    return loss





def chamfer_f_tokens(pred_pos, gt_states, gt_types, max_pts=1024, f_mask=None, return_pts=False):
    """
    Trajectory-based geometric loss (absolute position supervision).
    Matches predicted positions to ground truth turtle positions.
    Very efficient and stable for BPTT chunking.
    """
    B, T, _ = pred_pos.shape
    device = pred_pos.device
    
    # Ground truth positions (from pre-computed turtle states)
    gt_pos = gt_states[..., 0:3]
    
    # Mask: only penalize at ground truth 'F' tokens
    # and not at PAD/EOS
    m_gt = (gt_types == TokenType.F).float()
    
    # If soft mask from model is provided, multiply
    if f_mask is not None:
        m_gt = m_gt * f_mask
        
    # MSE loss on positions
    dist = torch.sum((pred_pos - gt_pos)**2, dim=-1) # (B, T)
    
    # Filter for max points if sequence is very long
    if T > max_pts:
        # Subsample high-confidence positions
        m_mean = m_gt.mean(dim=0)
        _, idx = torch.topk(m_mean, k=min(max_pts, T), sorted=False)
        dist = dist[:, idx]
        m_gt = m_gt[:, idx]
        
        sub_pred = pred_pos[:, idx]
        sub_gt = gt_pos[:, idx]
    else:
        sub_pred = pred_pos
        sub_gt = gt_pos
        
    loss = (dist * m_gt).sum() / (m_gt.sum() + 1e-8)
    
    if return_pts:
        return loss, sub_pred, sub_gt
    return loss


# ============================================================================
# ORDERED TRAJECTORY LOSSES (Much Better Than Chamfer!)
# ============================================================================

def trajectory_loss(pred_points, gt_points, move_mask=None):
    """
    Ordered trajectory loss - compares positions step-by-step.
    Much faster and more stable than Chamfer distance.
    
    Args:
        pred_points: (B, T, 3) predicted trajectory
        gt_points: (B, T, 3) ground truth trajectory
        move_mask: (B, T) optional mask for 'F' tokens
    
    Returns:
        MSE loss between trajectories
    """
    # L2 distance at each timestep
    diff = pred_points - gt_points  # (B, T, 3)
    sq_dist = torch.sum(diff ** 2, dim=-1)  # (B, T)
    
    if move_mask is not None:
        # Only penalize positions where movement happened
        loss = (sq_dist * move_mask).sum() / (move_mask.sum() + 1e-8)
    else:
        loss = sq_dist.mean()
    
    return loss


def heading_consistency_loss(pred_points, gt_points, move_mask=None, eps=1e-5):
    """
    Heading/direction loss - compares movement directions between steps.
    Stabilizes rotations and prevents drift.
    
    Args:
        pred_points: (B, T, 3) predicted trajectory
        gt_points: (B, T, 3) ground truth trajectory
        move_mask: (B, T) optional mask for 'F' tokens
    
    Returns:
        Cosine distance loss between heading vectors
    """
    # Compute heading vectors (direction of movement)
    pred_headings = pred_points[:, 1:] - pred_points[:, :-1]  # (B, T-1, 3)
    gt_headings = gt_points[:, 1:] - gt_points[:, :-1]  # (B, T-1, 3)
    
    # Normalize to unit vectors
    pred_headings = F.normalize(pred_headings, dim=-1, eps=eps)
    gt_headings = F.normalize(gt_headings, dim=-1, eps=eps)
    
    # Cosine similarity (dot product of unit vectors)
    cos_sim = torch.sum(pred_headings * gt_headings, dim=-1)  # (B, T-1)
    
    # Loss = 1 - cosine similarity (0 when aligned, 2 when opposite)
    loss = 1.0 - cos_sim
    
    if move_mask is not None:
        # Only penalize headings where movement happened
        mask = move_mask[:, 1:]  # Shift mask by 1
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    else:
        loss = loss.mean()
    
    return loss


def depth_progression_loss(pred_types, gt_types):
    """
    Depth progression loss - ensures bracket depth matches ground truth.
    Prevents bracket structure errors.
    
    Args:
        pred_types: (B, T, 7) soft token probabilities
        gt_types: (B, T) ground truth token indices
    
    Returns:
        MSE loss between depth progressions
    """
    B, T, _ = pred_types.shape
    device = pred_types.device
    
    # Compute predicted depth (soft)
    pred_lbr = pred_types[:, :, TokenType.LBR]  # (B, T)
    pred_rbr = pred_types[:, :, TokenType.RBR]  # (B, T)
    pred_depth = torch.cumsum(pred_lbr - pred_rbr, dim=1)  # (B, T)
    
    # Compute ground truth depth (hard)
    gt_lbr = (gt_types == TokenType.LBR).float()  # (B, T)
    gt_rbr = (gt_types == TokenType.RBR).float()  # (B, T)
    gt_depth = torch.cumsum(gt_lbr - gt_rbr, dim=1)  # (B, T)
    
    # MSE loss
    loss = F.mse_loss(pred_depth, gt_depth)
    
    return loss


def state_consistency_loss(pred_orient, gt_orient, move_mask=None):
    """
    State consistency loss - compares internal turtle state (H, U vectors).
    Cleanest geometric supervision without point-set ambiguity.
    
    Args:
        pred_orient: (B, T, 6) predicted orientations [H, U] concatenated
        gt_orient: (B, T, 6) ground truth orientations [H, U] concatenated
        move_mask: (B, T) optional mask for 'F' tokens
    
    Returns:
        MSE loss between orientation states
    """
    # Split into H and U vectors
    pred_H = pred_orient[:, :, 0:3]  # (B, T, 3)
    pred_U = pred_orient[:, :, 3:6]  # (B, T, 3)
    gt_H = gt_orient[:, :, 0:3]  # (B, T, 3)
    gt_U = gt_orient[:, :, 3:6]  # (B, T, 3)
    
    # MSE on both vectors
    loss_H = F.mse_loss(pred_H, gt_H, reduction='none').sum(dim=-1)  # (B, T)
    loss_U = F.mse_loss(pred_U, gt_U, reduction='none').sum(dim=-1)  # (B, T)
    
    total_loss = loss_H + loss_U  # (B, T)
    
    if move_mask is not None:
        loss = (total_loss * move_mask).sum() / (move_mask.sum() + 1e-8)
    else:
        loss = total_loss.mean()
    
    return loss


def combined_trajectory_loss(pred_points, pred_orient, gt_points, gt_orient, 
                             pred_types, gt_types, move_mask=None,
                             w_traj=1.0, w_heading=0.5, w_depth=0.3, w_state=0.5):
    """
    Combined trajectory-based loss (replacement for Chamfer).
    Much faster, more stable, and more interpretable.
    
    Args:
        pred_points: (B, T, 3) predicted positions
        pred_orient: (B, T, 6) predicted orientations [H, U]
        gt_points: (B, T, 3) ground truth positions
        gt_orient: (B, T, 6) ground truth orientations [H, U]
        pred_types: (B, T, 7) soft token probabilities
        gt_types: (B, T) ground truth token indices
        move_mask: (B, T) optional mask for 'F' tokens
        w_*: weights for each loss component
    
    Returns:
        Weighted combination of trajectory losses
    """
    loss = 0.0
    
    if w_traj > 0:
        loss += w_traj * trajectory_loss(pred_points, gt_points, move_mask)
    
    if w_heading > 0:
        loss += w_heading * heading_consistency_loss(pred_points, gt_points, move_mask)
    
    if w_depth > 0:
        loss += w_depth * depth_progression_loss(pred_types, gt_types)
    
    if w_state > 0:
        loss += w_state * state_consistency_loss(pred_orient, gt_orient, move_mask)
    
    return loss


class SyntaxConsistencyLoss(nn.Module):
    def __init__(self, grammar_matrix, weight=1.0):
        super().__init__()
        self.register_buffer('grammar', grammar_matrix.float())
        self.weight = weight
    
    def forward(self, logits, prev_types):
        B, T, C = logits.shape
        probs = F.softmax(logits, dim=-1)
        
        # Gather valid next tokens for each previous token
        valid_mask = self.grammar[prev_types]
        
        # Mask out invalid transitions
        invalid_prob = probs * (1.0 - valid_mask)
        
        # Mask out PAD tokens from the mean to prevent dilution
        mask = (prev_types != TokenType.PAD).float()
        loss = (invalid_prob.sum(dim=-1) * mask).sum() / (mask.sum() + 1e-6)
        
        return self.weight * loss

class BracketBalanceLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, logits):
        probs = F.softmax(logits, dim=-1)
        
        lbr_probs = probs[:, :, TokenType.LBR]
        rbr_probs = probs[:, :, TokenType.RBR]
        
        bracket_diff = torch.abs(lbr_probs.sum(dim=1) - rbr_probs.sum(dim=1))
        
        updates = lbr_probs - rbr_probs
        depths = torch.cumsum(updates, dim=1)
        neg_depth_penalty = F.relu(-depths).sum(dim=1)
        
        start_penalty = probs[:, 0, TokenType.RBR] * 2.0
        end_penalty = probs[:, -1, TokenType.LBR] * 2.0
        
        total_loss = bracket_diff * 0.5 + neg_depth_penalty + start_penalty + end_penalty
        return self.weight * total_loss.mean()

# ============================================================================
# CHAMFER LOSS WRAPPER
# ============================================================================

class ChamferLoss(nn.Module):
    """
    Module wrapper for chamfer_distance to match the interface expected by trainers.
    """
    def forward(self, source, target, move_mask=None, target_mask=None, num_points=1024, return_pts=False, normalize=True):
        return chamfer_distance(source, target, move_mask, target_mask, num_points, tau=0.1, return_pts=return_pts, normalize=normalize)


def fast_chamfer(pc1: torch.Tensor, pc2: torch.Tensor, max_pts: int = 512) -> torch.Tensor:
    """
    Improved stable fast chamfer.
    Uses hard min but adds stability and scale-normalization locally.
    """
    B, N, _ = pc1.shape
    M = pc2.shape[1]
    k1 = min(max_pts, N)
    k2 = min(max_pts, M)
    
    idx1 = torch.randperm(N, device=pc1.device)[:k1]
    idx2 = torch.randperm(M, device=pc2.device)[:k2]
    
    s = pc1[:, idx1].float()
    t = pc2[:, idx2].float().detach()
    
    with torch.no_grad():
        t_max = t.max(dim=1, keepdim=True)[0]
        t_min = t.min(dim=1, keepdim=True)[0]
        t_center = (t_max + t_min) / 2.0
        t_scale = (t_max - t_min).max(dim=2, keepdim=True)[0] + 1e-6
        
    s_norm = (s - t_center) / t_scale
    t_norm = (t - t_center) / t_scale
    
    d = torch.cdist(s_norm, t_norm)
    # Bidirectional sum of mins
    return (d.min(2)[0].mean(1) + d.min(1)[0].mean(1)).mean()

