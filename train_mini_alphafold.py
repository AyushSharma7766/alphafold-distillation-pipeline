"""
Milestone 2/3: Train Mini-AlphaFold on preprocessed OpenProteinSet features.
Uses PyTorch Lightning for mixed-precision training.

Usage:
    python train_mini_alphafold.py --feature_dir data/features --batch_size 1 --epochs 100
"""
import os
import glob
import torch
import argparse
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# OpenFold imports
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.loss import AlphaFoldLoss
from openfold.utils.tensor_utils import tensor_tree_map


class OpenFoldFeatureDataset(Dataset):
    """Loads preprocessed OpenFold feature dictionaries (.pt files)."""
    def __init__(self, feature_dir, max_length=400):
        self.files = glob.glob(os.path.join(feature_dir, "*.pt"))
        self.max_length = max_length
        print(f"Found {len(self.files)} training examples in {feature_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pt_path = self.files[idx]
        try:
            feat = torch.load(pt_path, map_location="cpu", weights_only=False)
            
            # The features must have the recycling dimension at the end.
            # Our preprocessor ran `feature_pipeline.process_features` which adds it.
            # But we must ensure it's in the exact shape the model expects.
            return feat
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
            # Fallback to another file if one is corrupted
            return self.__getitem__((idx + 1) % len(self.files))


def collate_fn(batch):
    """OpenFold expects batch_size=1 and doesn't batch across sequence length."""
    # Since sequences have variable lengths, we just use batch_size=1
    # and return the dictionary with a batch dimension of 1.
    feat = batch[0]
    return tensor_tree_map(lambda t: t.unsqueeze(0), feat)


class MiniAlphaFoldModule(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize configuration
        self.config = model_config("initial_training", train=True)
        # Mini-AlphaFold settings (24 blocks, reduced channels)
        self.config.model.evoformer_stack.no_blocks = 24
        self.config.model.evoformer_stack.c_m = 192
        self.config.model.evoformer_stack.c_z = 96
        self.config.model.extra_msa.enabled = False
        self.config.model.template.enabled = False
        
        # Turn on activation checkpointing to save memory on H200
        self.config.model.evoformer_stack.blocks_per_ckpt = 1
        
        # Initialize model and loss
        self.model = AlphaFold(self.config)
        self.loss_fn = AlphaFoldLoss(self.config.loss)
        
    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        
        # Remove recycling dimension for the loss function
        # (The model expects it, but the loss function does not)
        batch_no_recycle = tensor_tree_map(lambda t: t[..., 0], batch)
        
        loss, loss_breakdown = self.loss_fn(out, batch_no_recycle, _return_breakdown=True)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, batch_size=1)
        if "fape" in loss_breakdown:
            fape_val = loss_breakdown["fape"]
            if isinstance(fape_val, torch.Tensor): fape_val = fape_val.item()
            self.log("train_fape", fape_val, prog_bar=True, batch_size=1)
            
        if "distogram" in loss_breakdown:
            dist_val = loss_breakdown["distogram"]
            if isinstance(dist_val, torch.Tensor): dist_val = dist_val.item()
            self.log("train_distogram", dist_val, prog_bar=True, batch_size=1)
            
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        # Optional learning rate scheduler could go here
        return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, default="data/features")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # 1. Dataset & DataLoader
    dataset = OpenFoldFeatureDataset(args.feature_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn
    )

    # 2. Model
    model = MiniAlphaFoldModule(lr=args.lr)
    
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Initialized Mini-AlphaFold with {n_params:.1f}M parameters")

    # 3. Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",  # Best for Hopper H200
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # 4. Train
    print("Starting training loop...")
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
