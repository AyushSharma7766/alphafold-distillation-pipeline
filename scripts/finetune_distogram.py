"""
Mini-AlphaFold: Distogram Cross-Entropy Fine-Tuning Script
Transfers pretrained Evoformer weights from checkpoints/best_model.pt and
trains the 64-bin BinnedDistogramHead with Cross-Entropy and FAPE loss.
"""

import sys
import time
import math
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np

from src.model import MiniAlphaFold
from src.loss import CompositeLoss
from src.dataset import ProteinDataset, collate_proteins
from predict import kabsch_rmsd, compute_contact_precision
from scripts.benchmark_accuracy import evaluate_model, print_results_table, BENCHMARK_TARGETS

def get_subset_dataloaders(processed_dir="data/processed", n_train=400, n_val=60, batch_size=4, max_seq_len=128):
    train_ids_path = Path(processed_dir) / "train_domains.txt"
    val_ids_path = Path(processed_dir) / "val_domains.txt"

    with open(train_ids_path) as f:
        train_ids = [l.strip() for l in f if l.strip()][:n_train]
    with open(val_ids_path) as f:
        val_ids = [l.strip() for l in f if l.strip()][:n_val]

    train_ds = ProteinDataset(processed_dir, domain_ids=train_ids, max_seq_len=max_seq_len, preload=False)
    val_ds = ProteinDataset(processed_dir, domain_ids=val_ids, max_seq_len=max_seq_len, preload=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_proteins, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_proteins, num_workers=0)

    return train_dl, val_dl

def run_distogram_finetune(epochs=5, batch_size=4, lr=1.5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INIT] Starting Distogram Fine-Tuning on {device}...")

    # 1. Initialize model with Distogram Head
    model = MiniAlphaFold(
        d_model=128,
        d_pair=64,
        max_relpos=32,
        n_evoformer_blocks=4,
        n_heads=8,
        n_egnn_layers=3,
        dropout=0.1,
        use_distogram=True,
    ).to(device)

    # 2. Transfer pretrained weights from best_model.pt (skip old distance_head)
    ckpt_path = Path("checkpoints/best_model.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in ckpt["model_state_dict"].items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"[PRETRAIN] Transferred {len(pretrained_dict)} pretrained weight tensors from {ckpt_path}.")

    # 3. Criterion with Cross-Entropy Distogram Loss
    criterion = CompositeLoss(
        lambda_fape=1.0,
        lambda_distmat=1.0,  # Cross-entropy weight
        lambda_clash=0.1,
        lambda_bond=0.1,
        fape_clamp=10.0,
        clash_dist=3.0,
    ).to(device)

    # 4. Data
    train_dl, val_dl = get_subset_dataloaders(n_train=300, n_val=40, batch_size=batch_size)
    print(f"[DATA] Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

    # 5. Optimizer & Scaler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    save_path = Path("checkpoints/distogram_best.pt")

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, total_ce, total_fape = 0.0, 0.0, 0.0
        t0 = time.time()

        for batch in train_dl:
            seq = batch["seq"].to(device)
            backbone = batch["backbone"].to(device)
            dist_mat = batch["dist_mat"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(seq, mask=mask)

            losses = criterion(
                pred_backbone=outputs["backbone_coords"].float(),
                true_backbone=backbone.float(),
                pred_dist=outputs["pred_dist"].float(),
                true_dist=dist_mat.float(),
                mask=mask,
                dist_logits=outputs.get("dist_logits", None),
            )
            loss = losses["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_ce += losses["loss_dist"].item()
            total_fape += losses["loss_fape"].item()

        n_b = len(train_dl)
        dur = time.time() - t0
        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {total_loss/n_b:.4f} (CE Distogram: {total_ce/n_b:.3f}, FAPE: {total_fape/n_b:.3f}) | {dur:.1f}s")

        # Validation
        model.eval()
        val_loss, val_ce = 0.0, 0.0
        with torch.no_grad():
            for batch in val_dl:
                seq = batch["seq"].to(device)
                backbone = batch["backbone"].to(device)
                dist_mat = batch["dist_mat"].to(device)
                mask = batch["mask"].to(device)

                outputs = model(seq, mask=mask)
                losses = criterion(
                    pred_backbone=outputs["backbone_coords"].float(),
                    true_backbone=backbone.float(),
                    pred_dist=outputs["pred_dist"].float(),
                    true_dist=dist_mat.float(),
                    mask=mask,
                    dist_logits=outputs.get("dist_logits", None),
                )
                val_loss += losses["loss"].item()
                val_ce += losses["loss_dist"].item()

        n_v = max(len(val_dl), 1)
        mean_val = val_loss / n_v
        print(f"         --> Val Loss: {mean_val:.4f} (Val CE: {val_ce/n_v:.3f})")

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "use_distogram": True,
            }, save_path)
            print(f"         --> Saved new best checkpoint to {save_path}!")

    print(f"\n[DONE] Fine-tuning finished. Best checkpoint saved to {save_path}.")

if __name__ == "__main__":
    run_distogram_finetune(epochs=5, batch_size=4, lr=1.5e-4)
