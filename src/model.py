"""
Mini-AlphaFold: Top-Level Model Architecture
=============================================
Connects all 3 stages:
  Stage 1: Sequence & Relative Position Embeddings -> (s, z)
  Stage 2: Evoformer-Lite Stack                    -> (s_refined, z_refined)
  Stage 3: Structure Module & Distance Heads       -> 3D coordinates & Distances

Supports two structure modules:
  - EGNN (legacy): Equivariant GNN, requires MDS post-processing
  - IPA (default): AlphaFold 2 Invariant Point Attention, direct frame-based coords
"""

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.embedding import SequenceEmbedding, PairInitializer
from src.modules.evoformer import EvoformerStack
from src.modules.ipa import IPAStructureModule


class DistanceHead(nn.Module):
    """Predicts a symmetric pairwise distance matrix from pair features z_ij.

    Symmetrizes pair representation and projects via MLP:
        d_ij = Softplus(Linear(LayerNorm(1/2 * (z_ij + z_ji))))
    """

    def __init__(self, d_pair: int = 64, d_hidden: int = 64):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_pair)
        self.mlp = nn.Sequential(
            nn.Linear(d_pair, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z: [B, L, L, d_pair] pair representation
            mask: [B, L] bool mask
        Returns:
            dist_mat: [B, L, L] symmetric non-negative distance matrix
        """
        # Symmetrize pair features: z_sym[i, j] = 0.5 * (z[i, j] + z[j, i])
        z_sym = 0.5 * (z + z.transpose(1, 2))
        z_norm = self.layer_norm(z_sym)

        dist = self.mlp(z_norm).squeeze(-1)  # [B, L, L]

        # Zero out diagonal (distance from residue to itself is 0)
        B, L, _ = dist.shape
        eye = torch.eye(L, dtype=torch.bool, device=z.device).unsqueeze(0)
        dist = dist.masked_fill(eye, 0.0)

        if mask is not None:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
            dist = dist * pair_mask.to(dist.dtype)

        return dist


class BinnedDistogramHead(nn.Module):
    """Predicts a 64-bin categorical distogram from pair representation z_ij.
    Bins range from min_bin (2.0 A) to max_bin (22.0 A).
    """

    def __init__(
        self,
        d_pair: int = 64,
        d_hidden: int = 64,
        n_bins: int = 64,
        min_bin: float = 2.0,
        max_bin: float = 22.0,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.layer_norm = nn.LayerNorm(d_pair)
        self.mlp = nn.Sequential(
            nn.Linear(d_pair, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_bins),
        )
        bin_edges = torch.linspace(min_bin, max_bin, n_bins + 1)
        self.register_buffer("bin_centers", 0.5 * (bin_edges[:-1] + bin_edges[1:]))

    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        z_sym = 0.5 * (z + z.transpose(1, 2))
        z_norm = self.layer_norm(z_sym)
        logits = self.mlp(z_norm)
        logits = 0.5 * (logits + logits.transpose(1, 2))
        probs = F.softmax(logits, dim=-1)
        expected_dist = torch.sum(probs * self.bin_centers, dim=-1)

        B, L, _ = expected_dist.shape
        eye = torch.eye(L, dtype=torch.bool, device=z.device).unsqueeze(0)
        expected_dist = expected_dist.masked_fill(eye, 0.0)

        if mask is not None:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
            expected_dist = expected_dist * pair_mask.to(expected_dist.dtype)

        return logits, expected_dist


class MiniAlphaFold(nn.Module):
    """Mini-AlphaFold end-to-end structure prediction model.

    Args:
        d_model: Dimension of single representation (default: 128)
        d_pair: Dimension of pair representation (default: 64)
        max_relpos: Maximum relative position offset clamped (default: 32)
        n_evoformer_blocks: Number of Evoformer-Lite blocks (default: 4)
        n_heads: Number of attention heads in Evoformer (default: 8)
        ffn_multiplier: Expansion multiplier for FFN layers (default: 4)
        dropout: Dropout rate (default: 0.1)
        use_distogram: Whether to use 64-bin categorical distogram head (default: False)
        n_ipa_blocks: Number of IPA refinement iterations (default: 4)
    """

    def __init__(
        self,
        d_model: int = 128,
        d_pair: int = 64,
        max_relpos: int = 32,
        n_evoformer_blocks: int = 4,
        n_heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        use_distogram: bool = False,
        n_ipa_blocks: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_pair = d_pair
        self.use_distogram = use_distogram

        # Stage 1: Embeddings
        self.seq_embedding = SequenceEmbedding(d_model=d_model)
        self.pair_initializer = PairInitializer(
            d_model=d_model, d_pair=d_pair, max_relpos=max_relpos
        )

        # Stage 2: Evoformer-Lite Stack
        self.evoformer = EvoformerStack(
            n_blocks=n_evoformer_blocks,
            d_model=d_model,
            d_pair=d_pair,
            n_heads=n_heads,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
        )

        # Stage 3A: Distance Head (always present, for distogram loss + visualization)
        if use_distogram:
            self.distogram_head = BinnedDistogramHead(d_pair=d_pair, d_hidden=d_pair, n_bins=64)
        else:
            self.distance_head = DistanceHead(d_pair=d_pair, d_hidden=d_pair)

        # Stage 3B: Structure Module (IPA)
        self.ipa_structure_module = IPAStructureModule(
            d_single=d_model,
            d_pair=d_pair,
            n_heads=min(n_heads, 4),  # IPA uses fewer heads than Evoformer
            n_qk_points=4,
            n_v_points=4,
            n_blocks=n_ipa_blocks,
            dropout=dropout,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "MiniAlphaFold":
        """Instantiate MiniAlphaFold from a configuration dictionary (e.g. default.yaml)."""
        model_cfg = cfg.get("model", {})
        return cls(
            d_model=model_cfg.get("d_model", 128),
            d_pair=model_cfg.get("d_pair", 64),
            max_relpos=model_cfg.get("max_relpos", 32),
            n_evoformer_blocks=model_cfg.get("n_evoformer_blocks", 4),
            n_heads=model_cfg.get("n_heads", 8),
            ffn_multiplier=model_cfg.get("ffn_multiplier", 4),
            dropout=model_cfg.get("dropout", 0.1),
            use_distogram=model_cfg.get("use_distogram", False),
            n_ipa_blocks=model_cfg.get("n_ipa_blocks", 4),
        )

    def forward(
        self,
        seq: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Stage 1: Embeddings
        s = self.seq_embedding(seq, mask=mask)
        z = self.pair_initializer(s, mask=mask)

        # Stage 2: Evoformer-Lite
        s, z = self.evoformer(s, z, mask=mask)

        # Stage 3A: Distance matrix from pair representation
        dist_logits = None
        if self.use_distogram:
            dist_logits, pred_dist = self.distogram_head(z, mask=mask)
        else:
            pred_dist = self.distance_head(z, mask=mask)

        # Stage 3B: 3D coordinates from Structure Module in FP32
        intermediates = []
        with torch.amp.autocast('cuda', enabled=False):
            ca_coords, backbone_coords, intermediates = self.ipa_structure_module(
                s.float(), z.float(), mask=mask
            )

            coord_diff = ca_coords.unsqueeze(2) - ca_coords.unsqueeze(1)
            coord_dist = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1) + 1e-8)
            if mask is not None:
                pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
                coord_dist = coord_dist * pair_mask.to(coord_dist.dtype)

        return {
            "ca_coords": ca_coords,
            "backbone_coords": backbone_coords,
            "pred_dist": pred_dist,
            "dist_logits": dist_logits,
            "coord_dist": coord_dist,
            "single_repr": s,
            "pair_repr": z,
            "intermediates": intermediates,
        }
