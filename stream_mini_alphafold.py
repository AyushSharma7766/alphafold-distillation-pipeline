"""
Milestone 4: Continuous Streaming Distillation on AFDB
======================================================
This script orchestrates infinite training by streaming entire proteomes from the
AlphaFold Database (AFDB).

Architecture: Master/Worker pattern
  - MASTER: Downloads, extracts, preprocesses .cif.gz → .pt tensors (single process)
  - WORKER: Spawned as a subprocess for each species, runs PyTorch Lightning DDP
            training across all GPUs, then exits cleanly.

This prevents DDP zombie processes from corrupting the download/extract loop.

Usage:
    python stream_mini_alphafold.py --epochs_per_species 2 --devices auto
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import glob
import gzip
import time
import shutil
import tarfile
import argparse
import concurrent.futures
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# OpenFold imports
from openfold.data.data_pipeline import DataPipeline
from openfold.data.feature_pipeline import FeaturePipeline
from openfold.config import model_config

# Import from existing scripts
from download_afdb import CATALOG, multi_thread_download, get_cache_directory
from train_mini_alphafold import MiniAlphaFoldModule, OpenFoldFeatureDataset, collate_fn

# Resolve absolute path of this script (needed for subprocess spawning)
SCRIPT_PATH = os.path.abspath(__file__)


def process_cif_gz(cif_gz_path: str, output_dir: str, data_pipeline, feature_pipeline):
    """Parses a single AFDB .cif.gz file into OpenFold feature tensors using BioPython."""
    domain_id = Path(cif_gz_path).stem.replace(".cif", "")
    output_path = os.path.join(output_dir, f"{domain_id}.pt")
    
    if os.path.exists(output_path):
        return True, "already exists"
        
    try:
        with gzip.open(cif_gz_path, 'rt') as f:
            cif_str = f.read()
            
        import io
        import numpy as np
        from Bio.PDB.MMCIFParser import MMCIFParser
        from openfold.np import protein, residue_constants
        from openfold.data.data_pipeline import make_pdb_features
        
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("none", io.StringIO(cif_str))
        models = list(structure.get_models())
        if not models:
            return False, "No models found in CIF"
            
        atom_positions, aatype, atom_mask, residue_index, b_factors = [], [], [], [], []
        
        for chain in models[0]:
            for res in chain:
                if res.id[0] != ' ':
                    continue
                res_name = res.resname
                # Safely map 3-letter to 1-letter, default to 'X' if unknown
                one_letter = residue_constants.restype_3to1.get(res_name, 'X')
                
                # 'X' (UNK) is index 20 in OpenFold
                res_idx = residue_constants.restype_order.get(one_letter, 20)
                
                pos = np.zeros((37, 3), dtype=np.float32)
                mask = np.zeros((37,), dtype=np.float32)
                bfs = np.zeros((37,), dtype=np.float32)
                
                for atom in res:
                    if atom.name in residue_constants.atom_order:
                        idx = residue_constants.atom_order[atom.name]
                        pos[idx] = atom.coord
                        mask[idx] = 1.0
                        bfs[idx] = atom.bfactor
                        
                atom_positions.append(pos)
                aatype.append(res_idx)
                atom_mask.append(mask)
                residue_index.append(res.id[1])
                b_factors.append(bfs)
                
        if len(aatype) < 30 or len(aatype) > 400:
            return False, f"Length out of bounds: {len(aatype)}"
            
        prot = protein.Protein(
            atom_positions=np.array(atom_positions),
            aatype=np.array(aatype),
            atom_mask=np.array(atom_mask),
            residue_index=np.array(residue_index),
            b_factors=np.array(b_factors)
        )
        
        features = make_pdb_features(prot, description=domain_id, is_distillation=True)
        
        # Add dummy MSA features (just the target sequence itself)
        # We must use prot.aatype (integer, 1D) instead of features["aatype"] (one-hot, 2D)
        features["msa"] = np.array([prot.aatype], dtype=np.int64)
        features["deletion_matrix_int"] = np.zeros((1, len(prot.aatype)), dtype=np.int64)
        # num_alignments must be repeated num_res times to survive squeeze_features
        features["num_alignments"] = np.array([1] * len(prot.aatype), dtype=np.int64)
        features["msa_species_identifiers"] = np.array([b''], dtype=object)
        
        processed = feature_pipeline.process_features(features, mode='train')
        
        tensor_dict = {}
        for k, v in processed.items():
            if hasattr(v, 'numpy'):
                tensor_dict[k] = v
            else:
                try:
                    tensor_dict[k] = torch.tensor(v)
                except Exception:
                    pass
                    
        torch.save(tensor_dict, output_path)
        return True, "OK"
    except Exception:
        import traceback
        return False, traceback.format_exc()


# ═══════════════════════════════════════════════════════════════════
#  MASTER: Downloads, extracts, preprocesses, spawns DDP workers
# ═══════════════════════════════════════════════════════════════════

def master_loop(args, cache_dir, tmp_pdb_dir, features_dir, data_pipeline, feature_pipeline):
    print("\n🚀 Starting Continuous Streaming Distillation (MASTER MODE)...")
    
    skip_mode = False
    if args.start_from is not None:
        skip_mode = True
        print(f"⏩ Fast-forwarding catalog to: {args.start_from}")
    
    for species_key, info in CATALOG.items():
        if skip_mode:
            if species_key == args.start_from:
                skip_mode = False
            else:
                continue
                
        print(f"\n" + "="*60)
        print(f"🧬 STAGE: {info['name']} ({info['size_mb']} MB)")
        print("="*60)
        
        # 1. Download
        tar_dest = cache_dir / info["tar"]
        expected_bytes = info["size_mb"] * 1000000 * 0.95  # 95% threshold for safety
        
        if tar_dest.exists() and tar_dest.stat().st_size < expected_bytes:
            print(f"  ⚠️  Found corrupted/partial download ({tar_dest.stat().st_size / 1e6:.1f} MB). Deleting...")
            tar_dest.unlink()
            
        if not tar_dest.exists():
            try:
                multi_thread_download(info["url"], tar_dest)
            except Exception as e:
                print(f"  ⚠️  Download failed: {e}")
                if tar_dest.exists():
                    tar_dest.unlink()
                continue
            
        # 2. Extract & Preprocess
        print(f"Extracting and Preprocessing structures in parallel...")
        os.makedirs(tmp_pdb_dir, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)
        
        extracted_files = []
        try:
            with tarfile.open(tar_dest, "r") as tar:
                members = [m for m in tar if m.name.endswith(".cif.gz")]
                print(f"  Found {len(members)} structures in archive. Extracting...")
                for m in members: 
                    tar.extract(m, tmp_pdb_dir)
                    extracted_files.append(os.path.join(tmp_pdb_dir, m.name))
        except Exception as e:
            print(f"  ⚠️  Failed to extract tar: {e}")
            continue
                
        t0 = time.time()
        success = 0
        failed = 0
        target = 1500
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            # Process in chunks of 500 to avoid submitting 20,000 tasks at once
            for i in range(0, len(extracted_files), 500):
                if success >= target:
                    break
                    
                chunk = extracted_files[i : i + 500]
                futures = [ex.submit(process_cif_gz, f, str(features_dir), data_pipeline, feature_pipeline) for f in chunk]
                
                for fut in concurrent.futures.as_completed(futures):
                    ok, msg = fut.result()
                    if ok: 
                        success += 1
                        if success == 1 or success % 100 == 0:
                            print(f"  [{success}/{target}] valid tensors generated...")
                    else:
                        failed += 1
                
        print(f"✅ Generated {success} training tensors ({failed} skipped) in {time.time()-t0:.1f}s")
        
        # 3. Spawn DDP Training Worker as a clean subprocess
        if success > 0:
            print("🚀 Spawning DDP Training Worker...")
            import subprocess
            cmd = [
                sys.executable, SCRIPT_PATH, 
                "--worker_mode",
                "--features_path", str(features_dir),
                "--devices", str(args.devices),
                "--epochs_per_species", str(args.epochs_per_species),
                "--batch_size", str(args.batch_size),
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  ⚠️  Training worker exited with code {result.returncode}, continuing to next species...")
        else:
            print("  ⚠️  No valid tensors generated, skipping training.")
            
        # 4. Cleanup to save disk space
        print("🧹 Cleaning up staging files...")
        shutil.rmtree(tmp_pdb_dir, ignore_errors=True)
        shutil.rmtree(features_dir, ignore_errors=True)
        try:
            tar_dest.unlink() 
        except Exception:
            pass
        
    print("\n🎉 All species in the catalog have been processed!")


# ═══════════════════════════════════════════════════════════════════
#  WORKER: Runs PyTorch Lightning DDP training, then exits
# ═══════════════════════════════════════════════════════════════════

def worker_train(args):
    """This function runs inside a clean subprocess spawned by the master."""
    model = MiniAlphaFoldModule()
    
    devices = args.devices
    if devices.isdigit():
        devices = int(devices)
    
    # Check for existing checkpoint to dynamically adjust max_epochs
    last_ckpt_path = None
    ckpts = glob.glob("lightning_logs/lightning_logs/version_*/checkpoints/*.ckpt")
    completed_epochs = 0
    if ckpts:
        last_ckpt_path = max(ckpts, key=os.path.getctime)
        try:
            ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
            # PyTorch Lightning's ckpt['epoch'] is the 0-indexed epoch that just finished.
            completed_epochs = ckpt.get("epoch", 0) + 1
            del ckpt
        except Exception:
            pass
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"📦 Resuming from {last_ckpt_path} (Completed {completed_epochs} epochs)")
            
    trainer = pl.Trainer(
        max_epochs=completed_epochs + args.epochs_per_species,
        accelerator="gpu",
        devices=devices,
        strategy="auto",
        precision="bf16-mixed",
        enable_progress_bar=True,
        log_every_n_steps=10,
        default_root_dir="lightning_logs",
    )
    
    dataset = OpenFoldFeatureDataset(args.features_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=4,
    )
    
    trainer.fit(model, dataloader, ckpt_path=last_ckpt_path)


# ═══════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Mini-AlphaFold Streaming Distillation")
    parser.add_argument("--epochs_per_species", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--devices", type=str, default="auto",
                        help="Number of GPUs: '1', '4', or 'auto' for all available")
    parser.add_argument("--start_from", type=str, default=None,
                        help="Species key to resume from (e.g., 'pseudomonas_aeruginosa')")
    
    # Internal flags for the Master/Worker subprocess pattern
    parser.add_argument("--worker_mode", action="store_true",
                        help="(Internal) Run as a DDP training worker")
    parser.add_argument("--features_path", type=str, default=None,
                        help="(Internal) Path to the features directory for worker mode")
    args = parser.parse_args()
    
    if args.worker_mode:
        # We are a DDP Worker subprocess
        if args.features_path is None:
            print("ERROR: --features_path is required in worker mode")
            sys.exit(1)
        worker_train(args)
    else:
        # We are the Master process
        cache_dir = get_cache_directory()
        tmp_pdb_dir = cache_dir / "tmp_pdb"
        features_dir = cache_dir / "features"
        
        config = model_config("initial_training", train=True)
        config.data.common.use_templates = False
        config.data.train.max_msa = 1
        config.data.train.max_extra_msa = 1
        data_pipeline = DataPipeline(template_featurizer=None)
        feature_pipeline = FeaturePipeline(config.data)
        
        master_loop(args, cache_dir, tmp_pdb_dir, features_dir, data_pipeline, feature_pipeline)


if __name__ == "__main__":
    main()
