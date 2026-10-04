"""
Mini-AlphaFold: Loss Functions (Numerically Stable)
===================================================
Includes:
  1. FAPELoss: Frame-Aligned Point Error (AlphaFold local frame invariant loss)
  2. DistanceMatrixLoss: Smooth L1 error on pairwise CA distance matrices
  3. ClashLoss: Steric penalty for non-bonded atoms < 3.0 A apart
  4. BondLengthLoss: Regularizes consecutive CA-CA bond lengths to ~3.8 A
  5. CompositeLoss: Weighted combination of all losses, computed in FP32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_local_frames(
    backbone_coords: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constructs local coordinate frames (R_i, t_i) for each residue from backbone atoms.

    Uses Gram-Schmidt orthogonalization from N, CA, C coordinates:
      - Origin: t_i = CA_i
      - e1 = unit(CA_i - N_i)
      - e3 = unit(e1 x (C_i - CA_i))
      - e2 = e3 x e1

    Padded residues (mask == False) are replaced with a canonical dummy frame
    so vectors are never zero and gradients never explode.

    Args:
        backbone_coords: [B, L, 3, 3] (atom 0=N, 1=CA, 2=C)
        mask: [B, L] bool mask
        eps: Numerical stability constant
    Returns:
        R: [B, L, 3, 3] rotation matrices (orthogonal basis vectors as columns)
        t: [B, L, 3] translation origins (CA positions)
    """
    backbone_coords = backbone_coords.float()

    # Handle both 3-atom (N, CA, C) and 4-atom (N, CA, C, O) backbone formats
    if backbone_coords.shape[2] > 3:
        backbone_coords = backbone_coords[:, :, :3]  # Only need N, CA, C for frames

    if mask is not None:
        # Standard ideal peptide geometry for padded positions
        dummy_frame = torch.tensor([
            [-1.46, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.52, 0.0, 0.0],
        ], dtype=torch.float32, device=backbone_coords.device)
        mask_3d = mask.unsqueeze(-1).unsqueeze(-1)
        backbone_coords = torch.where(mask_3d, backbone_coords, dummy_frame)

    n = backbone_coords[:, :, 0]   # [B, L, 3]
    ca = backbone_coords[:, :, 1]  # [B, L, 3]
    c = backbone_coords[:, :, 2]   # [B, L, 3]

    t = ca  # Origin

    # Vector 1: CA - N
    v1 = ca - n
    e1 = F.normalize(v1, dim=-1, eps=eps)

    # Vector 2: C - CA
    v2 = c - ca

    # Vector 3: e1 x v2 (perpendicular to peptide plane)
    v3 = torch.cross(e1, v2, dim=-1)
    e3 = F.normalize(v3, dim=-1, eps=eps)

    # Vector 2: e3 x e1 (in-plane orthogonal vector)
    e2 = torch.cross(e3, e1, dim=-1)

    # Rotation matrix R with columns [e1, e2, e3]
    R = torch.stack([e1, e2, e3], dim=-1)

    return R, t


def project_to_local_frames(
    points: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Projects global Cartesian points into each residue's local frame.

    x_j^(i) = R_i^T * (points_j - t_i)
    """
    diff = points.unsqueeze(1) - t.unsqueeze(2)  # [B, L, L, 3]
    local = torch.einsum("bikd,bijd->bijk", R, diff)
    return local


class FAPELoss(nn.Module):
    """Frame-Aligned Point Error (FAPE) Loss.

    Computes local coordinate error in FP32 with safe clamping against sqrt(0) NaNs.
    """

    def __init__(self, d_clamp: float = 10.0, eps: float = 1e-4):
        super().__init__()
        self.d_clamp = d_clamp
        self.eps = eps

    def forward(
        self,
        pred_backbone: torch.Tensor,
        true_backbone: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_backbone = pred_backbone.float()
        true_backbone = true_backbone.float()

        pred_R, pred_t = construct_local_frames(pred_backbone, mask=mask, eps=self.eps)
        true_R, true_t = construct_local_frames(true_backbone, mask=mask, eps=self.eps)

        pred_ca = pred_backbone[:, :, 1]
        true_ca = true_backbone[:, :, 1]

        pred_local = project_to_local_frames(pred_ca, pred_R, pred_t)
        true_local = project_to_local_frames(true_ca, true_R, true_t)

        diff = pred_local - true_local
        # Safe clamped sqrt prevents infinite gradients at identical coordinates
        dist_sq = torch.sum(diff ** 2, dim=-1)
        dist = torch.sqrt(dist_sq.clamp(min=1e-8))

        clamped_dist = torch.clamp(dist, max=self.d_clamp)

        if mask is not None:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, L, L]
            clamped_dist = clamped_dist * pair_mask.to(clamped_dist.dtype)
            total_pairs = pair_mask.sum().clamp(min=1.0)
            loss = clamped_dist.sum() / total_pairs
        else:
            loss = clamped_dist.mean()

        return loss


class DistanceMatrixLoss(nn.Module):
    """Smooth L1 loss on pairwise CA distance matrices in FP32."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        pred_dist: torch.Tensor,
        true_dist: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_dist = pred_dist.float()
        true_dist = true_dist.float()

        loss_mat = F.smooth_l1_loss(pred_dist, true_dist, beta=self.beta, reduction="none")

        if mask is not None:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
            eye_mask = ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device).unsqueeze(0)
            valid_pairs = pair_mask & eye_mask
            loss = (loss_mat * valid_pairs.to(loss_mat.dtype)).sum() / valid_pairs.sum().clamp(min=1.0)
        else:
            loss = loss_mat.mean()

        return loss


class ClashLoss(nn.Module):
    """Penalizes non-bonded residue pairs with CA-CA distance < clash_dist (default: 3.0 A)."""

    def __init__(self, clash_dist: float = 3.0):
        super().__init__()
        self.clash_dist = clash_dist

    def forward(self, ca_coords: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        ca_coords = ca_coords.float()
        B, L, _ = ca_coords.shape

        diff = ca_coords.unsqueeze(2) - ca_coords.unsqueeze(1)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1).clamp(min=1e-8))

        penalties = F.relu(self.clash_dist - dist) ** 2

        idx = torch.arange(L, device=ca_coords.device)
        non_adjacent = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0)) > 1

        if mask is not None:
            pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)) & non_adjacent.unsqueeze(0)
            loss = (penalties * pair_mask.to(penalties.dtype)).sum() / pair_mask.sum().clamp(min=1.0)
        else:
            loss = (penalties * non_adjacent.unsqueeze(0).to(penalties.dtype)).mean()

        return loss


class BondLengthLoss(nn.Module):
    """Penalizes deviation of consecutive CA-CA distances from canonical 3.8 A."""

    def __init__(self, target_dist: float = 3.8):
        super().__init__()
        self.target_dist = target_dist

    def forward(self, ca_coords: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        ca_coords = ca_coords.float()
        diff = ca_coords[:, 1:] - ca_coords[:, :-1]
        bond_lengths = torch.sqrt(torch.sum(diff ** 2, dim=-1).clamp(min=1e-8))

        bond_loss = (bond_lengths - self.target_dist) ** 2

        if mask is not None:
            bond_mask = mask[:, 1:] & mask[:, :-1]
            loss = (bond_loss * bond_mask.to(bond_loss.dtype)).sum() / bond_mask.sum().clamp(min=1.0)
        else:
            loss = bond_loss.mean()

        return loss


class DistogramCrossEntropyLoss(nn.Module):
    """Categorical cross-entropy loss over 64 pairwise distance bins [2.0, 22.0] A."""

    def __init__(self, min_bin: float = 2.0, max_bin: float = 22.0, n_bins: int = 64):
        super().__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.n_bins = n_bins
        self.bin_width = (max_bin - min_bin) / n_bins

    def forward(self, logits: torch.Tensor, true_dist: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = true_dist.shape
        target_bins = torch.clamp(((true_dist - self.min_bin) / self.bin_width).long(), 0, self.n_bins - 1)
        eye = torch.eye(L, dtype=torch.bool, device=logits.device).unsqueeze(0).expand(B, L, L)
        pair_mask = ~eye
        if mask is not None:
            pair_mask = pair_mask & (mask.unsqueeze(2) & mask.unsqueeze(1))
        flat_logits = logits[pair_mask]
        flat_targets = target_bins[pair_mask]
        return F.cross_entropy(flat_logits, flat_targets)


class PeptideBondLoss(nn.Module):
    """Enforces proper peptide bond distance: C_i → N_{i+1} ≈ 1.33 Å.

    This is critical for IPA models where inter-residue connectivity
    is not directly constrained by the frame representation.
    """

    def __init__(self, target_dist: float = 1.33):
        super().__init__()
        self.target_dist = target_dist

    def forward(
        self, backbone: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        backbone = backbone.float()
        # backbone shape: [B, L, >=3, 3] with 0=N, 1=CA, 2=C
        c_coords = backbone[:, :-1, 2]   # C of residue i
        n_next = backbone[:, 1:, 0]       # N of residue i+1

        bond_len = torch.sqrt(
            torch.sum((c_coords - n_next) ** 2, dim=-1) + 1e-8
        )
        loss = (bond_len - self.target_dist) ** 2

        if mask is not None:
            bond_mask = mask[:, :-1] & mask[:, 1:]
            loss = (loss * bond_mask.float()).sum() / bond_mask.sum().clamp(min=1.0)
        else:
            loss = loss.mean()

        return loss


class CompositeLoss(nn.Module):
    """Combines FAPE, Distance matrix (or Distogram Cross-Entropy), Clash, Bond,
    and Peptide bond losses in FP32. Supports auxiliary FAPE on IPA intermediates."""

    def __init__(
        self,
        lambda_fape: float = 1.0,
        lambda_distmat: float = 0.5,
        lambda_clash: float = 0.1,
        lambda_bond: float = 0.1,
        lambda_peptide: float = 0.2,
        lambda_aux: float = 0.5,
        fape_clamp: float = 10.0,
        clash_dist: float = 3.0,
    ):
        super().__init__()
        self.lambda_fape = lambda_fape
        self.lambda_distmat = lambda_distmat
        self.lambda_clash = lambda_clash
        self.lambda_bond = lambda_bond
        self.lambda_peptide = lambda_peptide
        self.lambda_aux = lambda_aux

        self.fape_loss = FAPELoss(d_clamp=fape_clamp)
        self.dist_loss = DistanceMatrixLoss()
        self.distogram_loss = DistogramCrossEntropyLoss()
        self.clash_loss = ClashLoss(clash_dist=clash_dist)
        self.bond_loss = BondLengthLoss()
        self.peptide_loss = PeptideBondLoss()

    def forward(
        self,
        pred_backbone: torch.Tensor,
        true_backbone: torch.Tensor,
        pred_dist: torch.Tensor,
        true_dist: torch.Tensor,
        mask: torch.Tensor | None = None,
        dist_logits: torch.Tensor | None = None,
        intermediates: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        # Always run geometric losses in FP32 for numerical stability
        pred_backbone = pred_backbone.float()
        true_backbone = true_backbone.float()
        pred_dist = pred_dist.float()
        true_dist = true_dist.float()
        pred_ca = pred_backbone[:, :, 1]

        l_fape = self.fape_loss(pred_backbone, true_backbone, mask=mask)
        if dist_logits is not None:
            l_dist = self.distogram_loss(dist_logits.float(), true_dist, mask=mask)
        else:
            l_dist = self.dist_loss(pred_dist, true_dist, mask=mask)

        l_clash = self.clash_loss(pred_ca, mask=mask)
        l_bond = self.bond_loss(pred_ca, mask=mask)

        # Peptide bond loss (C_i → N_{i+1}) — important for IPA frame-based models
        l_peptide = self.peptide_loss(pred_backbone, mask=mask)

        total_loss = (
            self.lambda_fape * l_fape
            + self.lambda_distmat * l_dist
            + self.lambda_clash * l_clash
            + self.lambda_bond * l_bond
            + self.lambda_peptide * l_peptide
        )

        # Auxiliary FAPE on IPA intermediate structures (AF2-style)
        l_aux = torch.tensor(0.0, device=pred_backbone.device)
        if intermediates and len(intermediates) > 0:
            aux_fape_sum = torch.tensor(0.0, device=pred_backbone.device)
            for inter_bb in intermediates:
                aux_fape_sum = aux_fape_sum + self.fape_loss(
                    inter_bb.float(), true_backbone, mask=mask
                )
            l_aux = aux_fape_sum / len(intermediates)
            total_loss = total_loss + self.lambda_aux * l_aux

        return {
            "loss": total_loss,
            "loss_fape": l_fape,
            "loss_dist": l_dist,
            "loss_clash": l_clash,
            "loss_bond": l_bond,
            "loss_peptide": l_peptide,
            "loss_aux": l_aux,
        }
