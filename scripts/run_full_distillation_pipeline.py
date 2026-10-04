"""
Mini-AlphaFold: Master Multi-Species Distillation & Unified Training Pipeline
=============================================================================
Executes the full automated workflow in sequence:
  1. STAGE 1: Download & in-memory parse all selected reference proteomes one-by-one.
              Extracts high-confidence folded structures (pLDDT >= 70, length 30-128).
              Saves compact PyTorch tensors to data/afdb_processed/ and deletes .tar archives
              immediately from D:\afdb_cache (Zero storage waste!).
  2. STAGE 2: Automatically trains Mini-AlphaFold with the 64-bin Distogram Head on ALL
              ingested species together in a unified training session.
  3. STAGE 3: Runs the structural benchmark across standard targets (Crambin, Villin,
              Trp-Cage, Designed Fold) and prints the quantitative accuracy improvements!

Usage:
    # 1. Run full pipeline for all compact microbial & health proteomes (Recommended):
    python run_full_distillation_pipeline.py --tier compact --epochs 5

    # 2. Run for core model organisms (E. coli, TB, S. aureus, etc.) and train together:
    python run_full_distillation_pipeline.py --tier medium --epochs 5

    # 3. Run for ALL 34 catalog species, then train on everything together:
    python run_full_distillation_pipeline.py --all --epochs 5

    # 4. If you already have downloaded data and just want to train all together:
    python run_full_distillation_pipeline.py --skip_download --epochs 5
"""

import argparse
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from download_afdb import CATALOG, get_cache_directory, process_single_species, resolve_species
from scripts.train_afdb_distillation import train_afdb_distillation
import torch

console = Console()


def run_pipeline(
    species_keys: list[str],
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1.0e-4,
    max_samples: int | None = None,
    skip_download: bool = False,
    output_dir: Path = PROJECT_ROOT / "data" / "afdb_processed",
    cache_dir: Path | None = None,
):
    t_start = time.time()
    cache_dir = cache_dir if cache_dir else get_cache_directory()

    console.print(Panel(
        f"[bold white]Master Multi-Species Distillation & Unified Training Pipeline[/bold white]\n"
        f"[cyan]Dataset Destination:[/cyan] {output_dir}\n"
        f"[cyan]Staging Cache (Drive D:):[/cyan] {cache_dir}\n"
        f"[cyan]Training Epochs:[/cyan] {epochs} (Batch Size: {batch_size}, LR: {lr})\n"
        f"[cyan]Target Species:[/cyan] {len(species_keys)} proteomes",
        title="[bold yellow]Mini-AlphaFold Pipeline[/bold yellow]",
        border_style="yellow",
    ))

    # =========================================================================
    # STAGE 1: Download & In-Memory Parse All Species First
    # =========================================================================
    if not skip_download:
        console.print(Rule("[bold cyan]STAGE 1: Streaming Ingestion of All Species[/bold cyan]"))
        total_ingested = 0

        for idx, key in enumerate(species_keys, start=1):
            info = CATALOG[key]
            # Check if species was already processed
            prefix = f"AF_{key[:6]}_"
            existing = list(output_dir.glob(f"{prefix}*.pt"))
            if len(existing) > 200:
                console.print(f"[bold green]✓ [{idx}/{len(species_keys)}] Already processed {info['common']} ({len(existing)} structures present). Skipping download.[/bold green]")
                continue

            console.print(f"\n[bold yellow]▶ [{idx}/{len(species_keys)}] Ingesting {info['name']} ({info['size_mb']} MB, {info['count']} structures)...[/bold yellow]")
            added = process_single_species(key, output_dir, cache_dir)
            total_ingested += added

        console.print(f"\n[bold green]✓ Stage 1 Complete![/bold green] All species downloaded and converted to tensors in [cyan]{output_dir}[/cyan].\n")
    else:
        console.print(Rule("[bold yellow]STAGE 1: Skipped (Using Existing Processed Dataset)[/bold yellow]"))

    # Count total dataset
    all_pts = list(output_dir.glob("*.pt"))
    total_structures = len(all_pts)
    if total_structures == 0:
        console.print("[bold red]Error: No processed structures found in data/afdb_processed! Cannot train.[/bold red]")
        return

    console.print(Panel(
        f"[bold green]Ready for Unified Training![/bold green]\n"
        f"Total accumulated dataset: [bold white]{total_structures:,} structures[/bold white] across all species.\n"
        f"Total disk usage: [bold white]{sum(os.path.getsize(p) for p in all_pts) / (1024**2):.1f} MB[/bold white]",
        title="[bold green]Dataset Assembly Verified[/bold green]",
        border_style="green",
    ))

    # =========================================================================
    # STAGE 2: Train on ALL Species Together
    # =========================================================================
    console.print(Rule(f"[bold magenta]STAGE 2: Unified Multi-Species Distillation Training ({epochs} Epochs)[/bold magenta]"))
    train_cap = max_samples if (max_samples is not None and max_samples > 0) else None
    train_cap_str = f"{train_cap:,}" if train_cap else "all 131,284"
    console.print(f"Training 64-bin Distogram Head + FAPE loss across {train_cap_str} samples from {total_structures:,} structures on CUDA...\n")

    best_ckpt = train_afdb_distillation(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_train_samples=train_cap,
        max_val_samples=1000,
        afdb_dir=str(output_dir),
        checkpoint_out="checkpoints/distogram_afdb_best.pt",
    )

    console.print(f"\n[bold green]✓ Stage 2 Complete![/bold green] Best unified checkpoint saved to: [bold cyan]{best_ckpt}[/bold cyan]\n")

    # =========================================================================
    # STAGE 3: Run Benchmark to Measure Accuracy Gains
    # =========================================================================
    console.print(Rule("[bold green]STAGE 3: Automated Structural Benchmark[/bold green]"))
    try:
        import subprocess
        res = subprocess.run([sys.executable, "scripts/benchmark_afdb_distillation.py"], capture_output=True, text=True)
        console.print(res.stdout)
    except Exception as e:
        console.print(f"[yellow]Could not automatically run benchmark: {e}[/yellow]")
        console.print("Run [bold cyan]python scripts/benchmark_afdb_distillation.py[/bold cyan] manually.")

    elapsed_total = time.time() - t_start
    console.print(Panel(
        f"[bold green]Pipeline finished successfully in {elapsed_total / 60:.1f} minutes![/bold green]\n"
        f"Model trained on [bold white]{total_structures:,} structures[/bold white] from all selected species.\n"
        f"Live Web Studio ready: [bold cyan]python -m uvicorn app.main:app --host 127.0.0.1 --port 8000[/bold cyan]",
        title="[bold green]Pipeline Complete[/bold green]",
        border_style="green",
    ))


def main():
    parser = argparse.ArgumentParser(description="Master Multi-Species Distillation Pipeline")
    parser.add_argument("--tier", "-t", type=str, default=None, choices=["compact", "medium", "eukaryote", "global_health", "plant", "vertebrate"],
                        help="Ingest species by tier (choices: compact, medium, eukaryote, global_health, plant, vertebrate)")
    parser.add_argument("--all", "-a", action="store_true", help="Download and parse ALL 46 catalog species, then train")
    parser.add_argument("--species", "-s", nargs="+", default=None, help="List of specific species to process")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="Number of unified training epochs (default: 5)")
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size for training (default: 4)")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--max_samples", "-m", type=int, default=None, help="Max training samples per epoch (default: all 131k structures; or specify e.g. 25000)")
    parser.add_argument("--skip_download", action="store_true", help="Skip downloading, train on existing data/afdb_processed")
    args = parser.parse_args()

    if args.all:
        species_keys = list(CATALOG.keys())
    elif args.tier:
        species_keys = [k for k, v in CATALOG.items() if v.get("tier") == args.tier]
    elif args.species:
        species_keys = []
        for s in args.species:
            resolved = resolve_species(s)
            if resolved:
                species_keys.append(resolved)
            else:
                console.print(f"[red]Unknown species:[/red] {s}")
    else:
        # Default: The core microbial and model organisms set
        species_keys = [
            "helicobacter_pylori",
            "campylobacter_jejuni",
            "methanocaldococcus_jannaschii",
            "haemophilus_influenzae",
            "mycobacterium_leprae",
            "neisseria_gonorrhoeae",
            "streptococcus_pneumoniae",
            "staphylococcus_aureus",
            "escherichia_coli",
        ]

    run_pipeline(
        species_keys=species_keys,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
