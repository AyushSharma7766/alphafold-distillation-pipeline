"""
Mini-AlphaFold: PyTorch Dataset & DataLoader
==============================================
Loads preprocessed .pt files (from scripts/preprocess.py) and provides
batched, padded tensors for training.

Each batch yields:
  - seq           [B, L_max]      int64   — amino acid indices (0-19, 20=UNK/pad)
  - backbone      [B, L_max, 3, 3] float32 — N/CA/C coords in Å
  - ca_coords     [B, L_max, 3]   float32 — CA atom positions in Å
  - dist_mat      [B, L_max, L_max] float32 — pairwise CA distance matrix
  - mask          [B, L_max]      bool    — True for real residues, False for padding
  - lengths       [B]             int64   — original sequence lengths

Variable-length proteins are padded to the longest sequence in each batch
(not to the global max_seq_len). This saves significant memory and compute
on batches of shorter proteins.

Usage:
    from src.dataset import get_dataloaders

    train_dl, val_dl, test_dl = get_dataloaders(
        processed_dir="data/processed",
        batch_size=2,
        num_workers=2,
        max_seq_len=128,
    )

    for batch in train_dl:
        seq, backbone, ca, distmat, mask, lengths = (
            batch["seq"], batch["backbone"], batch["ca_coords"],
            batch["dist_mat"], batch["mask"], batch["lengths"],
        )
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path when script is executed directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import Dataset, DataLoader

from src.data_parser import UNK_IDX

log = logging.getLogger(__name__)


# ============================================================
# Dataset
# ============================================================

class ProteinDataset(Dataset):
    """PyTorch Dataset for preprocessed Mini-AlphaFold protein tensors.

    Loads .pt files lazily on first access and caches them in memory.
    For datasets that fit in RAM (typical for CATH S40 with max_seq_len ≤ 128),
    this gives fast epoch iteration after the first pass.

    Args:
        processed_dir: Path to directory containing .pt files
        domain_ids:    List of domain IDs to include (from split files)
        max_seq_len:   Maximum sequence length filter (drop longer proteins)
        preload:       If True, load all tensors into memory at init
    """

    def __init__(
        self,
        processed_dir: str | Path,
        domain_ids: Optional[list[str]] = None,
        max_seq_len: int = 128,
        preload: bool = False,
    ):
        self.processed_dir = Path(processed_dir)
        self.max_seq_len = max_seq_len

        # Discover available .pt files
        if domain_ids is not None:
            # Use the provided split list
            self.domain_ids = [
                did for did in domain_ids
                if (self.processed_dir / f"{did}.pt").exists()
            ]
        else:
            # Use all .pt files in the directory
            self.domain_ids = [
                p.stem for p in sorted(self.processed_dir.glob("*.pt"))
            ]

        # In-memory cache: domain_id → dict of tensors
        self._cache: dict[str, dict] = {}

        # Filter by max_seq_len using a quick pre-scan
        if max_seq_len is not None:
            self.domain_ids = self._filter_by_length(self.domain_ids, max_seq_len)

        if preload:
            log.info(f"Preloading {len(self.domain_ids)} proteins into memory...")
            for did in self.domain_ids:
                self._load(did)
            log.info(f"  Preloaded {len(self._cache)} proteins.")

    def _filter_by_length(
        self, domain_ids: list[str], max_len: int
    ) -> list[str]:
        """Filter domain IDs by sequence length without fully loading tensors."""
        kept = []
        for did in domain_ids:
            path = self.processed_dir / f"{did}.pt"
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                if data["length"] <= max_len:
                    # Cache it since we already loaded it
                    self._cache[did] = data
                    kept.append(did)
            except Exception:
                continue  # Skip corrupted files silently
        return kept

    def _load(self, domain_id: str) -> Optional[dict]:
        """Load a single .pt file, using cache if available."""
        if domain_id in self._cache:
            return self._cache[domain_id]

        path = self.processed_dir / f"{domain_id}.pt"
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            self._cache[domain_id] = data
            return data
        except Exception as e:
            log.warning(f"Skipping corrupted sample {domain_id}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.domain_ids)

    def __getitem__(self, idx: int) -> dict:
        """Return a single protein sample as a dict of tensors."""
        for attempt in range(10):
            curr_idx = (idx + attempt) % len(self.domain_ids)
            domain_id = self.domain_ids[curr_idx]
            data = self._load(domain_id)
            if data is not None and "sequence_indices" in data and "backbone_coords" in data:
                return {
                    "seq": data["sequence_indices"],       # [L]
                    "backbone": data["backbone_coords"],   # [L, 3, 3]
                    "ca_coords": data["ca_coords"],        # [L, 3]
                    "dist_mat": data["distance_matrix"],    # [L, L]
                    "length": data["length"],
                    "domain_id": data["domain_id"],
                }
        raise RuntimeError(f"Failed to load valid protein data after 10 attempts near index {idx}")


# ============================================================
# Collation (Variable-Length → Padded Batch)
# ============================================================

def collate_proteins(samples: list[dict]) -> dict:
    """Collate variable-length protein samples into a padded batch.

    Pads to the maximum length *within the batch* (not globally),
    which saves memory for batches of short proteins.

    Padding values:
      - seq:       UNK_IDX (20) for padding positions
      - backbone:  0.0 for padding coordinates
      - ca_coords: 0.0 for padding coordinates
      - dist_mat:  0.0 for padding positions
      - mask:      False for padding positions

    Args:
        samples: List of dicts from ProteinDataset.__getitem__

    Returns:
        Batched dict with:
          - seq:        [B, L_max] int64
          - backbone:   [B, L_max, 3, 3] float32
          - ca_coords:  [B, L_max, 3] float32
          - dist_mat:   [B, L_max, L_max] float32
          - mask:       [B, L_max] bool
          - lengths:    [B] int64
          - domain_ids: list[str] of length B
    """
    batch_size = len(samples)
    lengths = [s["length"] for s in samples]
    max_len = max(lengths)

    # Pre-allocate padded tensors
    seq = torch.full((batch_size, max_len), fill_value=UNK_IDX, dtype=torch.long)
    backbone = torch.zeros(batch_size, max_len, 3, 3, dtype=torch.float32)
    ca_coords = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    dist_mat = torch.zeros(batch_size, max_len, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    domain_ids = []

    for i, sample in enumerate(samples):
        L = sample["length"]

        seq[i, :L] = sample["seq"]
        backbone[i, :L] = sample["backbone"]
        ca_coords[i, :L] = sample["ca_coords"]
        dist_mat[i, :L, :L] = sample["dist_mat"]
        mask[i, :L] = True

        domain_ids.append(sample["domain_id"])

    return {
        "seq": seq,                                      # [B, L_max]
        "backbone": backbone,                            # [B, L_max, 3, 3]
        "ca_coords": ca_coords,                          # [B, L_max, 3]
        "dist_mat": dist_mat,                            # [B, L_max, L_max]
        "mask": mask,                                    # [B, L_max]
        "lengths": torch.tensor(lengths, dtype=torch.long),  # [B]
        "domain_ids": domain_ids,                        # list[str]
    }


# ============================================================
# Split File Loading
# ============================================================

def load_split_ids(processed_dir: Path, split: str) -> list[str]:
    """Load domain IDs from a split file (train/val/test).

    Args:
        processed_dir: Path to the processed data directory
        split: One of "train", "val", "test"

    Returns:
        List of domain ID strings
    """
    split_file = processed_dir / f"{split}_domains.txt"
    if not split_file.exists():
        log.warning(f"Split file not found: {split_file}")
        return []

    ids = [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return ids


# ============================================================
# DataLoader Factory
# ============================================================

def get_dataloaders(
    processed_dir: str | Path = "data/processed",
    batch_size: int = 2,
    num_workers: int = 2,
    max_seq_len: int = 128,
    preload: bool = True,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders.

    Reads the train/val/test split files created by preprocess.py
    and returns one DataLoader per split.

    Args:
        processed_dir: Path to directory with .pt files and split files
        batch_size:    Batch size for training (val/test use same)
        num_workers:   DataLoader worker processes
        max_seq_len:   Drop proteins longer than this
        preload:       Pre-load all tensors into RAM at init
        pin_memory:    Pin memory for GPU transfer (set True if using CUDA)

    Returns:
        (train_loader, val_loader, test_loader) tuple
    """
    processed_dir = Path(processed_dir)

    # Load split domain IDs
    train_ids = load_split_ids(processed_dir, "train")
    val_ids = load_split_ids(processed_dir, "val")
    test_ids = load_split_ids(processed_dir, "test")

    # If no splits exist or train is empty, fall back to all data
    if not train_ids:
        if val_ids or test_ids:
            log.warning("Train split is empty. Using all data as training set.")
        else:
            log.warning("No split files found. Using all data as training set.")
        train_ids = None  # Will discover all .pt files

    log.info(f"Dataset splits — train: {len(train_ids) if train_ids else 'all'}, "
             f"val: {len(val_ids)}, test: {len(test_ids)}")

    # Create datasets
    train_ds = ProteinDataset(
        processed_dir, domain_ids=train_ids,
        max_seq_len=max_seq_len, preload=preload,
    )
    val_ds = ProteinDataset(
        processed_dir, domain_ids=val_ids,
        max_seq_len=max_seq_len, preload=preload,
    )
    test_ds = ProteinDataset(
        processed_dir, domain_ids=test_ids,
        max_seq_len=max_seq_len, preload=preload,
    )

    log.info(f"Dataset sizes — train: {len(train_ds)}, "
             f"val: {len(val_ds)}, test: {len(test_ds)}")

    # Shared DataLoader kwargs
    use_persistent = num_workers > 0
    common_kwargs = dict(
        collate_fn=collate_proteins,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )

    def _make_loader(
        ds: ProteinDataset, shuffle: bool, drop_last: bool
    ) -> DataLoader:
        """Create a DataLoader, handling empty datasets gracefully."""
        workers = num_workers if len(ds) > 0 else 0
        return DataLoader(
            ds,
            batch_size=min(batch_size, max(len(ds), 1)),
            shuffle=shuffle and len(ds) > 0,
            num_workers=workers,
            drop_last=drop_last and len(ds) > batch_size,
            persistent_workers=use_persistent and workers > 0,
            **common_kwargs,
        )

    train_loader = _make_loader(train_ds, shuffle=True, drop_last=True)
    val_loader = _make_loader(val_ds, shuffle=False, drop_last=False)
    test_loader = _make_loader(test_ds, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader


# ============================================================
# CLI — Quick Inspection / Smoke Test
# ============================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    processed_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    max_seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    print(f"\n{'='*60}")
    print(f"Mini-AlphaFold Dataset Smoke Test")
    print(f"{'='*60}")
    print(f"  Processed dir:  {processed_dir}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Max seq length: {max_seq_len}")
    print()

    train_dl, val_dl, test_dl = get_dataloaders(
        processed_dir=processed_dir,
        batch_size=batch_size,
        num_workers=0,  # Use 0 workers for debugging
        max_seq_len=max_seq_len,
        preload=True,
        pin_memory=False,
    )

    print(f"\nDataLoader sizes:")
    print(f"  Train: {len(train_dl)} batches ({len(train_dl.dataset)} proteins)")
    print(f"  Val:   {len(val_dl)} batches ({len(val_dl.dataset)} proteins)")
    print(f"  Test:  {len(test_dl)} batches ({len(test_dl.dataset)} proteins)")

    # Iterate through one batch from each split
    for name, dl in [("Train", train_dl), ("Val", val_dl), ("Test", test_dl)]:
        if len(dl) == 0:
            print(f"\n  [{name}] Empty — no batches to show")
            continue

        batch = next(iter(dl))
        B = batch["seq"].shape[0]
        L = batch["seq"].shape[1]

        print(f"\n  [{name}] First batch:")
        print(f"    Batch size:        {B}")
        print(f"    Padded length:     {L}")
        print(f"    seq:               {batch['seq'].shape}  {batch['seq'].dtype}")
        print(f"    backbone:          {batch['backbone'].shape}  {batch['backbone'].dtype}")
        print(f"    ca_coords:         {batch['ca_coords'].shape}  {batch['ca_coords'].dtype}")
        print(f"    dist_mat:          {batch['dist_mat'].shape}  {batch['dist_mat'].dtype}")
        print(f"    mask:              {batch['mask'].shape}  {batch['mask'].dtype}")
        print(f"    lengths:           {batch['lengths'].tolist()}")
        print(f"    domain_ids:        {batch['domain_ids']}")

        # Verify mask consistency
        for i in range(B):
            real_len = batch["lengths"][i].item()
            mask_len = batch["mask"][i].sum().item()
            assert real_len == mask_len, (
                f"Mask mismatch: lengths[{i}]={real_len}, mask sum={mask_len}"
            )

            # Check that padding positions are zeroed
            if real_len < L:
                pad_ca = batch["ca_coords"][i, real_len:]
                assert (pad_ca == 0).all(), "CA coords not zeroed in padding!"

                pad_seq = batch["seq"][i, real_len:]
                assert (pad_seq == UNK_IDX).all(), "Seq not padded with UNK_IDX!"

        print(f"    [OK] Padding & mask consistency verified")

    print(f"\n{'='*60}")
    print(f"[OK] All checks passed!")
    print(f"{'='*60}\n")
