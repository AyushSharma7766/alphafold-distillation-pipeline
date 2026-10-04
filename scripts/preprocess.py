"""
Mini-AlphaFold: Batch Preprocessing Script
============================================
Processes all downloaded CATH PDB files into PyTorch-ready tensors.

For each valid PDB file, produces a .pt file containing:
  - domain_id:        str
  - sequence:         str (1-letter codes)
  - sequence_indices: [L] int64 tensor
  - backbone_coords:  [L, 3, 3] float32 tensor (N, CA, C backbone atoms)
  - ca_coords:        [L, 3] float32 tensor
  - distance_matrix:  [L, L] float32 tensor
  - length:           int

Also generates a metadata.json summary file with dataset statistics.

Usage:
    python preprocess.py --input data/raw --output data/processed
    python preprocess.py --input data/raw --output data/processed --workers 8
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

# Add project root to path so we can import src modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_parser import ProteinParser, ProteinData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# Worker Function (for multiprocessing)
# ============================================================

# Global parser instance (initialized per worker process)
_parser: Optional[ProteinParser] = None


def _init_worker(min_len: int, max_len: int):
    """Initialize a ProteinParser in each worker process."""
    global _parser
    _parser = ProteinParser(quiet=True, validate_geometry=True)


def _process_single_file(args: tuple) -> Optional[dict]:
    """Process a single PDB file → save as .pt and return metadata.

    This function runs in a worker process.

    Args:
        args: (pdb_path, output_dir, min_length, max_length)

    Returns:
        Metadata dict if successful, None otherwise.
    """
    pdb_path, output_dir, min_length, max_length = args
    global _parser

    if _parser is None:
        _parser = ProteinParser(quiet=True, validate_geometry=True)

    pdb_path = Path(pdb_path)
    domain_id = pdb_path.stem

    # Parse PDB
    protein = _parser.parse(
        pdb_path,
        domain_id=domain_id,
        min_length=min_length,
        max_length=max_length,
    )

    if protein is None:
        return None

    # Save as .pt file
    out_path = Path(output_dir) / f"{domain_id}.pt"
    torch.save(protein.to_dict(), out_path)

    # Return metadata for statistics
    return {
        "domain_id": domain_id,
        "length": protein.length,
        "sequence": protein.sequence,
        "dist_min": float(protein.distance_matrix.min()),
        "dist_max": float(protein.distance_matrix.max()),
        "dist_mean": float(protein.distance_matrix.mean()),
    }


# ============================================================
# Main Preprocessing Pipeline
# ============================================================

def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    min_length: int = 30,
    max_length: int = 200,
    num_workers: int = 1,
) -> dict:
    """Process all PDB files in input_dir and save as .pt tensors.

    Args:
        input_dir:   Directory containing .pdb files
        output_dir:  Directory to write .pt files and metadata.json
        min_length:  Minimum sequence length to include
        max_length:  Maximum sequence length to include
        num_workers: Number of parallel worker processes

    Returns:
        Metadata dictionary with dataset statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all PDB files
    pdb_files = sorted(input_dir.glob("*.pdb"))
    if not pdb_files:
        log.error(f"No .pdb files found in {input_dir}")
        return {}

    log.info(f"Found {len(pdb_files)} PDB files in {input_dir}")
    log.info(f"Length filter: [{min_length}, {max_length}] residues")
    log.info(f"Output: {output_dir}")
    log.info(f"Workers: {num_workers}")

    # Prepare arguments for worker function
    work_args = [
        (str(pdb_path), str(output_dir), min_length, max_length)
        for pdb_path in pdb_files
    ]

    # Process files (parallel or sequential)
    results: list[Optional[dict]] = []
    start_time = time.time()

    if num_workers > 1:
        with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(min_length, max_length),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_single_file, work_args),
                total=len(work_args),
                desc="Processing PDBs",
                unit="file",
            ):
                results.append(result)
    else:
        _init_worker(min_length, max_length)
        for args in tqdm(work_args, desc="Processing PDBs", unit="file"):
            results.append(_process_single_file(args))

    elapsed = time.time() - start_time

    # Collect statistics
    successful = [r for r in results if r is not None]
    failed_count = len(results) - len(successful)

    log.info(f"\nProcessing complete in {elapsed:.1f}s")
    log.info(f"  Successful: {len(successful)}")
    log.info(f"  Skipped:    {failed_count} (invalid, wrong length, or bad geometry)")

    if not successful:
        log.error("No proteins were successfully processed!")
        return {}

    # Compute statistics
    lengths = [r["length"] for r in successful]
    dist_mins = [r["dist_min"] for r in successful]
    dist_maxs = [r["dist_max"] for r in successful]

    # Amino acid frequency across the dataset
    all_sequences = "".join(r["sequence"] for r in successful)
    aa_counts = Counter(all_sequences)
    total_residues = sum(aa_counts.values())
    aa_freq = {
        aa: round(count / total_residues, 4)
        for aa, count in sorted(aa_counts.items())
    }

    # Length distribution buckets
    length_buckets = Counter()
    for l in lengths:
        bucket = f"{(l // 25) * 25}-{(l // 25) * 25 + 24}"
        length_buckets[bucket] += 1

    metadata = {
        "dataset": "CATH S40 Non-Redundant",
        "total_domains": len(successful),
        "skipped_domains": failed_count,
        "length_filter": {"min": min_length, "max": max_length},
        "length_stats": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 1),
            "median": sorted(lengths)[len(lengths) // 2],
        },
        "length_distribution": dict(sorted(length_buckets.items())),
        "distance_stats_angstrom": {
            "min": round(min(dist_mins), 2),
            "max": round(max(dist_maxs), 2),
        },
        "amino_acid_frequency": aa_freq,
        "total_residues": total_residues,
        "backbone_atoms": ["N", "CA", "C"],
        "processing_time_seconds": round(elapsed, 1),
        "domain_ids": [r["domain_id"] for r in successful],
    }

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"  Metadata saved to: {meta_path}")

    # Print summary
    log.info(f"\n{'='*50}")
    log.info(f"Dataset Summary")
    log.info(f"{'='*50}")
    log.info(f"  Domains:     {metadata['total_domains']}")
    log.info(f"  Lengths:     {metadata['length_stats']}")
    log.info(f"  Total AAs:   {total_residues:,}")
    log.info(f"  Dist range:  [{metadata['distance_stats_angstrom']['min']}, "
             f"{metadata['distance_stats_angstrom']['max']}] Å")

    return metadata


# ============================================================
# Train / Val / Test Split
# ============================================================

def generate_splits(
    output_dir: Path,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> None:
    """Generate train/val/test split files from processed data.

    Creates three text files in output_dir:
      - train_domains.txt
      - val_domains.txt
      - test_domains.txt

    Each file contains one domain ID per line.
    """
    import random

    meta_path = output_dir / "metadata.json"
    if not meta_path.exists():
        log.error(f"metadata.json not found in {output_dir}. Run preprocessing first.")
        return

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    domain_ids = metadata["domain_ids"]
    random.seed(seed)
    random.shuffle(domain_ids)

    n = len(domain_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = domain_ids[:n_train]
    val_ids = domain_ids[n_train : n_train + n_val]
    test_ids = domain_ids[n_train + n_val :]

    splits = {
        "train_domains.txt": train_ids,
        "val_domains.txt": val_ids,
        "test_domains.txt": test_ids,
    }

    for filename, ids in splits.items():
        path = output_dir / filename
        path.write_text("\n".join(ids), encoding="utf-8")
        log.info(f"  {filename}: {len(ids)} domains")

    log.info(f"  Split ratio: {train_frac}/{val_frac}/{test_frac}")
    log.info(f"  Random seed: {seed}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CATH PDB files into PyTorch tensors for Mini-AlphaFold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python preprocess.py --input data/raw --output data/processed

  # With length filtering and parallelism
  python preprocess.py --input data/raw --output data/processed \\
      --min-length 40 --max-length 128 --workers 8

  # Quick test with a small subset
  python preprocess.py --input data/raw --output data/processed \\
      --max-length 80 --workers 1
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/raw",
        help="Input directory with .pdb files (default: data/raw)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/processed",
        help="Output directory for .pt files (default: data/processed)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=30,
        help="Minimum sequence length (default: 30)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Maximum sequence length (default: 200)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=max(1, cpu_count() - 1),
        help=f"Number of worker processes (default: {max(1, cpu_count() - 1)})",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip generating train/val/test splits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits (default: 42)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        log.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Run preprocessing
    metadata = preprocess_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        num_workers=args.workers,
    )

    if not metadata:
        sys.exit(1)

    # Generate splits
    if not args.no_split:
        log.info("\nGenerating train/val/test splits...")
        generate_splits(output_dir, seed=args.seed)

    log.info(f"\n✓ Preprocessing complete! Next: build the PyTorch Dataset.")


if __name__ == "__main__":
    main()
