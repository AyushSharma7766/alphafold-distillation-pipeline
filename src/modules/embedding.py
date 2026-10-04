"""
Mini-AlphaFold: Embedding Modules
==================================
Stage 1 of the architecture:
  - SequenceEmbedding: Projects 1D amino acid tokens to single representation [B, L, d_model]
  - RelativePositionEncoding: Encodes relative residue separations |i - j| into pair space [B, L, L, d_pair]
  - PairInitializer: Combines single representation outer-projections with relative position encodings
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_parser import UNK_IDX, NUM_AMINO_ACIDS


class SequenceEmbedding(nn.Module):
    """Embeds amino acid sequence tokens into continuous vectors.

    Args:
        d_model: Dimensionality of the single representation (default: 128)
        vocab_size: Number of amino acid tokens (20 canonical + 1 UNK = 21)
    """

    def __init__(self, d_model: int = 128, vocab_size: int = NUM_AMINO_ACIDS):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=UNK_IDX)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            seq: [B, L] int64 sequence indices
            mask: [B, L] bool mask (True for real residues, False for padding)
        Returns:
            s: [B, L, d_model] float32 single representation
        """
        s = self.embedding(seq) * math.sqrt(self.d_model)
        s = self.layer_norm(s)

        if mask is not None:
            s = s * mask.unsqueeze(-1).to(s.dtype)

        return s


class RelativePositionEncoding(nn.Module):
    """Encodes relative sequence distance |i - j| between residues into pair features.

    AlphaFold relative position encoding clamps distances to [-max_relpos, max_relpos].

    Args:
        d_pair: Dimensionality of the pair representation (default: 64)
        max_relpos: Maximum relative distance before clamping (default: 32)
    """

    def __init__(self, d_pair: int = 64, max_relpos: int = 32):
        super().__init__()
        self.d_pair = d_pair
        self.max_relpos = max_relpos
        self.num_bins = 2 * max_relpos + 1
        self.embedding = nn.Embedding(self.num_bins, d_pair)

    def forward(self, L: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            L: Sequence length
            device: Device to create position tensors on
        Returns:
            rel_pos: [L, L, d_pair] relative position embeddings
        """
        pos = torch.arange(L, device=device)
        # diff[i, j] = i - j
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        clamped = torch.clamp(diff, -self.max_relpos, self.max_relpos)
        # Shift range [-max_relpos, max_relpos] -> [0, 2 * max_relpos]
        bin_indices = clamped + self.max_relpos
        return self.embedding(bin_indices)


class PairInitializer(nn.Module):
    """Initializes the pair representation z_ij from single representations s_i, s_j
    and relative sequence position encodings.

    z_ij = Linear(s_i) + Linear(s_j) + RelPos(i - j)

    Args:
        d_model: Dimensionality of single representation (default: 128)
        d_pair: Dimensionality of pair representation (default: 64)
        max_relpos: Clamping window for relative position (default: 32)
    """

    def __init__(self, d_model: int = 128, d_pair: int = 64, max_relpos: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_pair = d_pair

        self.proj_left = nn.Linear(d_model, d_pair, bias=False)
        self.proj_right = nn.Linear(d_model, d_pair, bias=False)
        self.rel_pos = RelativePositionEncoding(d_pair=d_pair, max_relpos=max_relpos)
        self.layer_norm = nn.LayerNorm(d_pair)

    def forward(self, s: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            s: [B, L, d_model] single representation
            mask: [B, L] bool mask
        Returns:
            z: [B, L, L, d_pair] initial pair representation
        """
        B, L, _ = s.shape

        # Projections: [B, L, d_pair]
        left = self.proj_left(s)
        right = self.proj_right(s)

        # Outer sum: left[i] + right[j] -> [B, L, L, d_pair]
        z = left.unsqueeze(2) + right.unsqueeze(1)

        # Add relative position embedding [L, L, d_pair]
        r_ij = self.rel_pos(L, device=s.device)
        z = z + r_ij.unsqueeze(0)

        z = self.layer_norm(z)

        # Mask padding pairs: pair is valid if both residue i and residue j are valid
        if mask is not None:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, L, L]
            z = z * pair_mask.unsqueeze(-1).to(z.dtype)

        return z
