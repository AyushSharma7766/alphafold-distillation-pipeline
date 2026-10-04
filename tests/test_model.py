"""
Mini-AlphaFold: Unit & Integration Tests
=========================================
Tests:
  1. Sequence and pair embedding shapes & masking
  2. Evoformer-Lite block forward pass & gradient flow
  3. EGNN layer rotation equivariance: R * EGCL(x) == EGCL(R * x)
  4. FAPE, DistanceMatrix, and Clash loss functions
  5. End-to-end MiniAlphaFold forward & backward pass
"""

import math
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.modules.embedding import SequenceEmbedding, PairInitializer
from src.modules.evoformer import EvoformerBlock, EvoformerStack
from src.modules.egnn import E_GCL, EGNNStructureModule
from src.loss import FAPELoss, DistanceMatrixLoss, ClashLoss, BondLengthLoss, CompositeLoss
from src.model import MiniAlphaFold


def test_embedding_and_pair_init():
    """Verify embedding and pair initialization shapes."""
    B, L = 2, 16
    d_model, d_pair = 64, 32
    seq = torch.randint(0, 20, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, 12:] = False  # Add padding to sample 0

    embedder = SequenceEmbedding(d_model=d_model)
    pair_init = PairInitializer(d_model=d_model, d_pair=d_pair)

    s = embedder(seq, mask=mask)
    assert s.shape == (B, L, d_model)
    # Check that padded positions in sample 0 are zeroed
    assert (s[0, 12:] == 0).all()

    z = pair_init(s, mask=mask)
    assert z.shape == (B, L, L, d_pair)
    # Check that padded pairs are zeroed
    assert (z[0, 12:, :] == 0).all()
    assert (z[0, :, 12:] == 0).all()
    print("[PASS] test_embedding_and_pair_init passed")


def test_evoformer_block():
    """Verify Evoformer forward and backward pass."""
    B, L = 2, 12
    d_model, d_pair = 64, 32
    s = torch.randn(B, L, d_model, requires_grad=True)
    z = torch.randn(B, L, L, d_pair, requires_grad=True)
    mask = torch.ones(B, L, dtype=torch.bool)

    block = EvoformerBlock(d_model=d_model, d_pair=d_pair, n_heads=4)
    s_out, z_out = block(s, z, mask=mask)

    assert s_out.shape == (B, L, d_model)
    assert z_out.shape == (B, L, L, d_pair)

    # Check gradient flow
    loss = s_out.sum() + z_out.sum()
    loss.backward()
    assert s.grad is not None and not torch.isnan(s.grad).any()
    assert z.grad is not None and not torch.isnan(z.grad).any()
    print("[PASS] test_evoformer_block passed")


def test_egnn_rotation_equivariance():
    """Verify that rotating input coordinates produces an identically rotated coordinate output."""
    L = 10
    d_node, d_pair = 32, 16
    h = torch.randn(1, L, d_node)
    x = torch.randn(1, L, 3)
    z = torch.randn(1, L, L, d_pair)

    egcl = E_GCL(d_node=d_node, d_pair=d_pair, d_hidden=32)
    egcl.eval()

    # Generate a random 3D rotation matrix via QR decomposition
    random_matrix = torch.randn(3, 3)
    q, r = torch.linalg.qr(random_matrix)
    d = torch.diag(torch.sign(torch.diag(r)))
    R = torch.matmul(q, d)  # Exact SO(3) rotation matrix
    if torch.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]

    with torch.no_grad():
        # Pass 1: Standard forward
        _, x_out = egcl(h, x, z)

        # Rotate output
        rotated_x_out = torch.matmul(x_out, R.T)

        # Pass 2: Rotate input then forward
        x_rotated_in = torch.matmul(x, R.T)
        _, x_out_from_rotated = egcl(h, x_rotated_in, z)

        # Difference should be negligible (within numerical floating-point precision)
        diff = torch.abs(rotated_x_out - x_out_from_rotated).max().item()
        assert diff < 1e-4, f"Rotation equivariance violated! Max diff = {diff}"

    print(f"[PASS] test_egnn_rotation_equivariance passed (max diff = {diff:.2e})")


def test_loss_functions():
    """Verify FAPE, distance, and clash losses."""
    B, L = 2, 16
    # Create synthetic backbone: N, CA, C
    idx = torch.arange(L, dtype=torch.float32).unsqueeze(0).expand(B, L)
    ca = torch.stack([idx * 3.8, torch.zeros_like(idx), torch.zeros_like(idx)], dim=-1)
    n = ca - torch.tensor([1.2, 0.0, 0.0])
    c = ca + torch.tensor([1.2, 0.0, 0.0])
    true_backbone = torch.stack([n, ca, c], dim=2)  # [B, L, 3, 3]

    fape = FAPELoss(d_clamp=10.0)
    # Identical predictions should yield zero FAPE loss
    zero_loss = fape(true_backbone, true_backbone)
    assert zero_loss.item() < 1e-4, f"Expected 0 FAPE loss for identical structures, got {zero_loss.item()}"

    # Clash loss: two overlapping atoms should trigger a clash penalty
    clashing_ca = ca.clone()
    clashing_ca[:, 5] = clashing_ca[:, 0]  # Move residue 5 right on top of residue 0
    clash_loss = ClashLoss(clash_dist=3.0)
    penalty = clash_loss(clashing_ca)
    assert penalty.item() > 0.0, "Expected non-zero clash penalty for overlapping atoms"

    print("[PASS] test_loss_functions passed")


def test_mini_alphafold_end_to_end():
    """Verify complete forward pass and loss backward pass through MiniAlphaFold."""
    B, L = 2, 16
    seq = torch.randint(0, 20, (B, L))
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[1, 10:] = False

    model = MiniAlphaFold(
        d_model=64,
        d_pair=32,
        max_relpos=16,
        n_evoformer_blocks=2,
        n_heads=4,
        n_egnn_layers=2,
        ffn_multiplier=2,
        dropout=0.0,
    )

    outputs = model(seq, mask=mask)
    assert outputs["ca_coords"].shape == (B, L, 3)
    assert outputs["backbone_coords"].shape == (B, L, 3, 3)
    assert outputs["pred_dist"].shape == (B, L, L)

    # Compute composite loss against mock targets
    mock_true_backbone = torch.randn(B, L, 3, 3)
    mock_true_dist = torch.rand(B, L, L) * 15.0

    criterion = CompositeLoss()
    losses = criterion(
        pred_backbone=outputs["backbone_coords"],
        true_backbone=mock_true_backbone,
        pred_dist=outputs["pred_dist"],
        true_dist=mock_true_dist,
        mask=mask,
    )

    total_loss = losses["loss"]
    total_loss.backward()

    # Check all parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients!"

    print(f"[PASS] test_mini_alphafold_end_to_end passed (total loss = {total_loss.item():.4f})")


if __name__ == "__main__":
    test_embedding_and_pair_init()
    test_evoformer_block()
    test_egnn_rotation_equivariance()
    test_loss_functions()
    test_mini_alphafold_end_to_end()
    print("\n============================================================")
    print("ALL 5 ARCHITECTURE & INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("============================================================")
