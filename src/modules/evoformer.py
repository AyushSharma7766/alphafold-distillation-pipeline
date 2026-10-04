"""
Mini-AlphaFold: Evoformer-Lite Modules
======================================
Stage 2 of the architecture:
Processes single sequence representations and 2D pair representations iteratively:
  1. OuterProductUpdate: Single -> Pair communication
  2. TriangleMultiplicativeUpdate: Pair -> Pair structural consistency
  3. PairTransition: Pair MLP
  4. PairBiasedSelfAttention: Pair-guided attention on Single representation
  5. SingleTransition: Single MLP
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OuterProductUpdate(nn.Module):
    """Communicates information from the 1D single representation to the 2D pair representation.

    Projects single vectors s_i and s_j to a lower dimension c, computes their outer product,
    and projects the result to d_pair:
        delta_z_ij = Linear(Linear(s_i) (x) Linear(s_j))

    Args:
        d_model: Dimensionality of single representation (default: 128)
        d_pair: Dimensionality of pair representation (default: 64)
        c_hidden: Internal projection dimension (default: 16)
    """

    def __init__(self, d_model: int = 128, d_pair: int = 64, c_hidden: int = 16):
        super().__init__()
        self.c_hidden = c_hidden
        self.layer_norm = nn.LayerNorm(d_model)
        self.proj_left = nn.Linear(d_model, c_hidden, bias=False)
        self.proj_right = nn.Linear(d_model, c_hidden, bias=False)
        self.out_proj = nn.Linear(c_hidden * c_hidden, d_pair)

    def forward(self, s: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            s: [B, L, d_model] single representation
            mask: [B, L] bool mask
        Returns:
            delta_z: [B, L, L, d_pair] update to pair representation
        """
        B, L, _ = s.shape
        s_norm = self.layer_norm(s)

        # [B, L, c_hidden]
        a = self.proj_left(s_norm)
        b = self.proj_right(s_norm)

        if mask is not None:
            mask_float = mask.unsqueeze(-1).to(s.dtype)
            a = a * mask_float
            b = b * mask_float

        # Outer product: a_i [B, L, 1, c] * b_j [B, 1, L, c]
        # outer[b, i, j, c1, c2] = a[b, i, c1] * b[b, j, c2]
        # Flattened outer dimension: [B, L, L, c_hidden * c_hidden]
        a_expanded = a.unsqueeze(2).unsqueeze(-1)  # [B, L, 1, c, 1]
        b_expanded = b.unsqueeze(1).unsqueeze(-2)  # [B, 1, L, 1, c]
        outer = (a_expanded * b_expanded).view(B, L, L, self.c_hidden * self.c_hidden)

        delta_z = self.out_proj(outer)
        return delta_z


class TriangleMultiplicativeUpdate(nn.Module):
    """Enforces triangular consistency on the 2D pair representation.

    Approximates the triangle inequality (d_ij <= d_ik + d_kj) in latent space:
        z_ij = LayerNorm(Linear(sum_k (Linear(z_ik) * Linear(z_jk))))

    Args:
        d_pair: Dimensionality of pair representation (default: 64)
        c_hidden: Internal channel dimension (default: 32)
    """

    def __init__(self, d_pair: int = 64, c_hidden: int = 32):
        super().__init__()
        self.c_hidden = c_hidden
        self.layer_norm_in = nn.LayerNorm(d_pair)

        self.proj_a = nn.Linear(d_pair, c_hidden)
        self.proj_b = nn.Linear(d_pair, c_hidden)
        self.gate_a = nn.Linear(d_pair, c_hidden)
        self.gate_b = nn.Linear(d_pair, c_hidden)

        self.layer_norm_out = nn.LayerNorm(c_hidden)
        self.proj_out = nn.Linear(c_hidden, d_pair)
        self.gate_out = nn.Linear(d_pair, d_pair)

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z: [B, L, L, d_pair] pair representation
            mask: [B, L] bool mask
        Returns:
            delta_z: [B, L, L, d_pair] triangular update
        """
        B, L, _, _ = z.shape
        z_norm = self.layer_norm_in(z)

        # Gated projections: [B, L, L, c_hidden]
        a = self.proj_a(z_norm) * torch.sigmoid(self.gate_a(z_norm))
        b = self.proj_b(z_norm) * torch.sigmoid(self.gate_b(z_norm))

        if mask is not None:
            pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1).to(z.dtype)
            a = a * pair_mask
            b = b * pair_mask

        # Sum over intermediary node k: sum_k a_ik * b_jk
        # Einsum: [B, i, k, c] and [B, j, k, c] -> [B, i, j, c]
        x = torch.einsum("bikc,bjkc->bijc", a, b)

        x = self.layer_norm_out(x)
        delta_z = self.proj_out(x) * torch.sigmoid(self.gate_out(z_norm))
        return delta_z


class PairBiasedSelfAttention(nn.Module):
    """Multi-head self-attention on single sequence representation with pair bias.

    Attention logits are augmented with a learned linear projection of the pair representation:
        Attn_ij = (q_i * k_j^T) / sqrt(d_k) + b_ij
    where b_ij is projected from z_ij.

    Args:
        d_model: Dimensionality of single representation (default: 128)
        d_pair: Dimensionality of pair representation (default: 64)
        n_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_model: int = 128, d_pair: int = 64, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.d_pair = d_pair
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.layer_norm_s = nn.LayerNorm(d_model)
        self.layer_norm_z = nn.LayerNorm(d_pair)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Projects pair representation to head-specific bias: [B, L, L, n_heads]
        self.pair_bias_proj = nn.Linear(d_pair, n_heads, bias=False)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            s: [B, L, d_model] single representation
            z: [B, L, L, d_pair] pair representation
            mask: [B, L] bool mask
        Returns:
            s_out: [B, L, d_model] updated single representation
        """
        B, L, _ = s.shape
        s_norm = self.layer_norm_s(s)
        z_norm = self.layer_norm_z(z)

        # Compute Q, K, V: [B, n_heads, L, d_head]
        q = self.q_proj(s_norm).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(s_norm).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(s_norm).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Standard dot-product attention scores: [B, n_heads, L, L]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add pair bias: [B, L, L, n_heads] -> [B, n_heads, L, L]
        pair_bias = self.pair_bias_proj(z_norm).permute(0, 3, 1, 2)
        scores = scores + pair_bias

        # Mask padding tokens
        if mask is not None:
            # key mask: [B, 1, 1, L]
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_expanded, -1e4)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute context: [B, n_heads, L, d_head] -> [B, L, d_model]
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)

        out = self.out_proj(context)
        return out


class Transition(nn.Module):
    """Feed-forward transition layer with GELU activation.

    Args:
        d_in: Input dimensionality
        multiplier: Hidden dimension multiplier (default: 4)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(self, d_in: int, multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        d_hidden = d_in * multiplier
        self.layer_norm = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer_norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return self.dropout(h)


class EvoformerBlock(nn.Module):
    """A single Evoformer-Lite block combining pair updates and pair-biased sequence attention.

    Execution sequence:
      1. z = z + OuterProductUpdate(s)
      2. z = z + TriangleMultiplicativeUpdate(z)
      3. z = z + Transition(z)
      4. s = s + PairBiasedSelfAttention(s, z)
      5. s = s + Transition(s)

    Args:
        d_model: Dimensionality of single representation (default: 128)
        d_pair: Dimensionality of pair representation (default: 64)
        n_heads: Attention heads (default: 8)
        ffn_multiplier: Multiplier for FFN hidden dims (default: 4)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        d_model: int = 128,
        d_pair: int = 64,
        n_heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Pair branch
        self.outer_product = OuterProductUpdate(d_model=d_model, d_pair=d_pair)
        self.triangle_mult = TriangleMultiplicativeUpdate(d_pair=d_pair)
        self.pair_transition = Transition(d_pair, multiplier=2, dropout=dropout)

        # Single branch
        self.self_attn = PairBiasedSelfAttention(
            d_model=d_model, d_pair=d_pair, n_heads=n_heads, dropout=dropout
        )
        self.single_transition = Transition(d_model, multiplier=ffn_multiplier, dropout=dropout)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: [B, L, d_model] single representation
            z: [B, L, L, d_pair] pair representation
            mask: [B, L] bool mask
        Returns:
            s_out: [B, L, d_model]
            z_out: [B, L, L, d_pair]
        """
        # 1. Single -> Pair communication
        z = z + self.outer_product(s, mask=mask)

        # 2. Pair triangular consistency update
        z = z + self.triangle_mult(z, mask=mask)

        # 3. Pair MLP
        z = z + self.pair_transition(z)

        # 4. Pair-biased attention on Single representation
        s = s + self.self_attn(s, z, mask=mask)

        # 5. Single MLP
        s = s + self.single_transition(s)

        return s, z


class EvoformerStack(nn.Module):
    """Stack of N Evoformer-Lite blocks.

    Args:
        n_blocks: Number of blocks (default: 4)
        d_model: Single representation dim (default: 128)
        d_pair: Pair representation dim (default: 64)
        n_heads: Attention heads (default: 8)
        ffn_multiplier: Multiplier for FFN (default: 4)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        n_blocks: int = 4,
        d_model: int = 128,
        d_pair: int = 64,
        n_heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            EvoformerBlock(
                d_model=d_model,
                d_pair=d_pair,
                n_heads=n_heads,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        ])

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            s, z = block(s, z, mask=mask)
        return s, z
