import re
import math
import numpy as np
import torch

# ============================================================
#  Token Types
# ============================================================

class TokenType:
    F   = 0
    LBR = 1
    RBR = 2
    EOS = 3
    PAD = 4

NUM_TYPES = 5

# ============================================================
#  Grammar Rules
# ============================================================

def build_grammar_matrix():
    """
    Defines allowed transitions between TokenType IDs.
    F=0, LBR=1, RBR=2, EOS=3, PAD=4
    """
    M = torch.zeros((NUM_TYPES, NUM_TYPES), dtype=torch.float32)

    # From F
    M[TokenType.F, TokenType.F]   = 1
    M[TokenType.F, TokenType.LBR] = 1
    M[TokenType.F, TokenType.RBR] = 1
    M[TokenType.F, TokenType.EOS] = 1

    # From LBR (must follow with segment or nested branch)
    M[TokenType.LBR, TokenType.F]   = 1
    M[TokenType.LBR, TokenType.LBR] = 1

    # From RBR
    M[TokenType.RBR, TokenType.F]   = 1
    M[TokenType.RBR, TokenType.LBR] = 1
    M[TokenType.RBR, TokenType.RBR] = 1
    M[TokenType.RBR, TokenType.EOS] = 1

    # From EOS
    M[TokenType.EOS, TokenType.PAD] = 1

    # From PAD
    M[TokenType.PAD, TokenType.PAD] = 1

    return M

GRAMMAR_MATRIX = build_grammar_matrix()

def compute_grammar_mask(type_sequence):
    """
    type_sequence: list[int] or torch.Tensor
    Returns mask [T, NUM_TYPES] where 1.0 means transition is allowed.
    """
    L = len(type_sequence)
    mask = torch.zeros((L, NUM_TYPES))
    
    # We compute what is allowed at t+1 based on token at t
    for i in range(L - 1):
        prev_type = type_sequence[i]
        if torch.is_tensor(prev_type): prev_type = prev_type.item()
        mask[i] = GRAMMAR_MATRIX[prev_type]
    
    # Final token usually transitions to PAD
    mask[L-1] = GRAMMAR_MATRIX[TokenType.EOS]
    return mask

def apply_grammar_mask(logits, prev_type_id):
    """
    logits: (NUM_TYPES,)
    prev_type_id: int
    """
    grammar_matrix_device = GRAMMAR_MATRIX.to(logits.device)
    mask = grammar_matrix_device[prev_type_id]
    return logits.masked_fill(mask == 0, float("-inf"))

# ============================================================
#  Tokenizer (Type + Value bins)
# ============================================================

class LSystemTokenizerV2:
    """
    Tokenizer for compressed symbolic L-strings:
        S{theta}_{phi}_{length}
        [
        ]

    Produces:
        type_ids:  list[int]
        value_ids: list[[len, theta, phi]]
    """
    def __init__(self, f_bins=10, theta_bins=6, phi_bins=6):
        self.f_bins = f_bins
        self.theta_bins = theta_bins
        self.phi_bins = phi_bins
        # Robust regex: matches S{theta}_{phi}_{len}, B{theta}_{phi}F{len}, and lowercase variants
        self.re_SEGMENT = re.compile(r"([BSbs])(\d+)_(\d+)[F_]?(\d+)")

    def encode(self, s):
        types = []
        values = []
        i = 0
        L = len(s)

        while i < L:
            if s[i] == "[":
                types.append(TokenType.LBR)
                values.append([0, 0, 0])
                i += 1
                continue
            if s[i] == "]":
                types.append(TokenType.RBR)
                values.append([0, 0, 0])
                i += 1
                continue

            m = self.re_SEGMENT.match(s, i)
            if m:
                theta = min(int(m.group(2)), self.theta_bins - 1)
                phi   = min(int(m.group(3)), self.phi_bins - 1)
                f_bin = min(int(m.group(4)), self.f_bins - 1)

                types.append(TokenType.F)
                values.append([f_bin, theta, phi])
                i = m.end()
                continue
            i += 1

        types.append(TokenType.EOS)
        values.append([0, 0, 0])
        return types, values

    def decode(self, types, values):
        out = []
        for t, v in zip(types, values):
            if t == TokenType.F:
                f_bin, theta, phi = v
                out.append(f"B{int(theta)}_{int(phi)}F{int(f_bin)}")
            elif t == TokenType.LBR:
                out.append("[")
            elif t == TokenType.RBR:
                out.append("]")
            elif t == TokenType.EOS:
                break
        return "".join(out)
