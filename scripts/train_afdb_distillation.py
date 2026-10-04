"""
Mini-AlphaFold: AFDB Multi-Organism Distillation Training Script
==============================================================
Trains MiniAlphaFold with the 64-bin BinnedDistogramHead on high-confidence
structures distilled from AlphaFold Reference Proteomes (data/afdb_processed).
Starts from checkpoints/distogram_best.pt to continue refining accuracy.
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import ProteinDataset, collate_proteins
from src.loss import CompositeLoss
from src.model import MiniAlphaFold
from scripts.benchmark_accuracy import evaluate_model, print_results_table, BENCHMARK_TARGETS

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("afdb_train")


def get_afdb_dataloaders(
    afdb_dir: Path | str = "data/afdb_processed",
    batch_size: int = 4,
    max_seq_len: int | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = 1000,
):
    afdb_dir = Path(afdb_dir)
    splits_file = afdb_dir / "splits.json"

    if not splits_file.exists():
        # Discover all available .pt files directly
        all_ids = sorted([p.stem for p in afdb_dir.glob("*.pt")])
        if len(all_ids) == 0:
            raise RuntimeError(f"No .pt files found in {afdb_dir}. Run stream_afdb_distillation.py first.")
        n_train = int(len(all_ids) * 0.85)
        train_ids = all_ids[:n_train]
        val_ids = all_ids[n_train:]
    else:
        with open(splits_file) as f:
            splits = json.load(f)
        train_ids = splits.get("train", [])
        val_ids = splits.get("val", [])

    if max_train_samples:
        train_ids = train_ids[:max_train_samples]
    if max_val_samples:
        val_ids = val_ids[:max_val_samples]

    log.info(f"Loading AFDB dataset from {afdb_dir}...")
    log.info(f"  Train samples: {len(train_ids)} | Val samples: {len(val_ids)}")

    train_ds = ProteinDataset(afdb_dir, domain_ids=train_ids, max_seq_len=max_seq_len, preload=False)
    val_ds = ProteinDataset(afdb_dir, domain_ids=val_ids, max_seq_len=max_seq_len, preload=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_proteins, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_proteins, num_workers=0)

    return train_dl, val_dl


def train_afdb_distillation(
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1.0e-4,
    max_train_samples: int | None = None,
    max_val_samples: int | None = 1000,
    checkpoint_in: str | None = None,
    checkpoint_out: str = "checkpoints/distogram_afdb_best.pt",
    afdb_dir: str = "data/afdb_processed",
):
    if checkpoint_in is None:
        checkpoint_in = "checkpoints/distogram_afdb_best.pt" if Path("checkpoints/distogram_afdb_best.pt").exists() else "checkpoints/distogram_best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Starting AFDB Distillation Training on {device}...")

    # 1. Initialize model with 64-bin distogram head + IPA structure module
    model = MiniAlphaFold(
        d_model=128,
        d_pair=64,
        max_relpos=32,
        n_evoformer_blocks=4,
        n_heads=8,
        dropout=0.1,
        use_distogram=True,
        n_ipa_blocks=4,
    ).to(device)

    # 2. Transfer pretrained weights
    in_path = Path(checkpoint_in)
    if in_path.exists():
        ckpt = torch.load(in_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in ckpt["model_state_dict"].items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        log.info(f"Loaded {len(pretrained_dict)} pretrained weight tensors from {in_path}.")

    # 3. Composite loss (Cross-Entropy + FAPE + Clash + Bond + Peptide + Auxiliary)
    criterion = CompositeLoss(
        lambda_fape=5.0,        # Increased: strong signal for IPA coordinate learning
        lambda_distmat=0.5,
        lambda_clash=0.1,
        lambda_bond=0.1,
        lambda_peptide=0.2,
        lambda_aux=0.5,
        fape_clamp=100.0,       # Greatly increased: allows gradients to flow globally over long chains
        clash_dist=3.0,
    ).to(device)

    # 4. Data loaders
    train_dl, val_dl = get_afdb_dataloaders(
        afdb_dir=afdb_dir,
        batch_size=batch_size,
        max_seq_len=None,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )

    # 5. Optimizer & Cosine LR Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    out_path = Path(checkpoint_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_ce, total_fape = 0.0, 0.0, 0.0
        n_batches = 0
        t0 = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=25),
            TaskProgressColumn(),
            TextColumn("[yellow]{task.completed}/{task.total} batches"),
            TextColumn("•"),
            TextColumn("[bold green]{task.fields[loss_str]}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=4,
        ) as progress:
            train_task = progress.add_task(
                f"Epoch {epoch}/{epochs}",
                total=len(train_dl),
                loss_str="Loss: ...",
            )
            for batch in train_dl:
                seq = batch["seq"].to(device)
                mask = batch["mask"].to(device)
                target_ca = batch["ca_coords"].to(device)
                target_backbone = batch["backbone"].to(device)
                target_dist = batch["dist_mat"].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    out = model(seq, mask=mask)
                    loss_dict = criterion(
                        pred_backbone=out["backbone_coords"],
                        true_backbone=target_backbone,
                        pred_dist=out["pred_dist"],
                        true_dist=target_dist,
                        mask=mask,
                        dist_logits=out.get("dist_logits"),
                        intermediates=out.get("intermediates"),
                    )
                    loss = loss_dict["loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                b_loss = float(loss.item())
                b_ce = float(loss_dict.get("loss_dist", 0.0))
                b_fape = float(loss_dict.get("loss_fape", 0.0))

                total_loss += b_loss
                total_ce += b_ce
                total_fape += b_fape
                n_batches += 1

                progress.update(
                    train_task,
                    advance=1,
                    loss_str=f"Loss: {b_loss:.2f} (CE: {b_ce:.2f}, FAPE: {b_fape:.2f})",
                )

        scheduler.step()
        train_loss = total_loss / max(1, n_batches)
        train_ce = total_ce / max(1, n_batches)
        train_fape = total_fape / max(1, n_batches)

        # Validation with live progress
        model.eval()
        val_total, val_ce_total = 0.0, 0.0
        v_batches = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]Validating"),
            BarColumn(bar_width=20),
            TaskProgressColumn(),
            TextColumn("[yellow]{task.completed}/{task.total} batches"),
            TextColumn("•"),
            TextColumn("[bold green]{task.fields[loss_str]}"),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=4,
        ) as v_progress:
            val_task = v_progress.add_task("Validating", total=len(val_dl), loss_str="Val CE: ...")
            with torch.no_grad():
                for v_batch in val_dl:
                    seq = v_batch["seq"].to(device)
                    mask = v_batch["mask"].to(device)
                    target_backbone = v_batch["backbone"].to(device)
                    target_dist = v_batch["dist_mat"].to(device)

                    out = model(seq, mask=mask)
                    v_loss_dict = criterion(
                        pred_backbone=out["backbone_coords"],
                        true_backbone=target_backbone,
                        pred_dist=out["pred_dist"],
                        true_dist=target_dist,
                        mask=mask,
                        dist_logits=out.get("dist_logits"),
                        intermediates=out.get("intermediates"),
                    )
                    v_loss = float(v_loss_dict["loss"].item())
                    v_ce = float(v_loss_dict.get("loss_dist", 0.0))
                    val_total += v_loss
                    val_ce_total += v_ce
                    v_batches += 1

                    v_progress.update(val_task, advance=1, loss_str=f"Val CE: {v_ce:.2f}")

        val_loss = val_total / max(1, v_batches)
        val_ce = val_ce_total / max(1, v_batches)
        elapsed = time.time() - t0

        console.print(
            f"[bold white]Epoch {epoch:2d}/{epochs:2d}[/bold white] ({elapsed:4.1f}s) | "
            f"Train Loss: [bold yellow]{train_loss:.4f}[/bold yellow] (CE: {train_ce:.3f}, FAPE: {train_fape:.3f}) | "
            f"Val Loss: [bold green]{val_loss:.4f}[/bold green] (Val CE: {val_ce:.3f})"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_ce": val_ce,
                },
                out_path,
            )
            console.print(f"  [bold green]--> Saved new best AFDB checkpoint to: {out_path} (Val Loss: {val_loss:.4f})[/bold green]")

    console.print(f"\n[bold green]Distillation training finished! Best checkpoint: [cyan]{out_path}[/cyan][/bold green]")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--afdb_dir", type=str, default="data/afdb_processed")
    args = parser.parse_args()

    train_afdb_distillation(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, afdb_dir=args.afdb_dir)
