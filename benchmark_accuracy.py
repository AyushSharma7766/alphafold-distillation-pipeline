import os
import glob
import torch
import numpy as np
import time
from pathlib import Path
from scipy.spatial.transform import Rotation

# OpenFold imports
from openfold.data import feature_pipeline
from openfold.np import protein
from openfold.config import model_config
from openfold.utils.tensor_utils import tensor_tree_map
# Local imports
from train_mini_alphafold import MiniAlphaFoldModule

def get_latest_checkpoint(log_dir="lightning_logs"):
    """Finds the most recently modified .ckpt file in the lightning_logs directory."""
    ckpts = glob.glob(f"{log_dir}/**/checkpoints/*.ckpt", recursive=True)
    if not ckpts:
        raise FileNotFoundError("No checkpoints found in lightning_logs!")
    return max(ckpts, key=os.path.getctime)

def calculate_rmsd(pred_coords, true_coords):
    """Calculates C-alpha RMSD using the Kabsch algorithm."""
    # Ensure numpy
    if torch.is_tensor(pred_coords):
        pred_coords = pred_coords.detach().cpu().numpy()
    if torch.is_tensor(true_coords):
        true_coords = true_coords.detach().cpu().numpy()
        
    # Center the coordinates
    p_center = pred_coords.mean(axis=0)
    t_center = true_coords.mean(axis=0)
    p = pred_coords - p_center
    t = true_coords - t_center
    
    # Calculate optimal rotation (Kabsch)
    rot, _ = Rotation.align_vectors(t, p)
    p_aligned = rot.apply(p)
    
    # Calculate RMSD
    rmsd = np.sqrt(np.mean(np.sum((p_aligned - t)**2, axis=1)))
    return float(rmsd)

def save_to_pdb(prot: protein.Protein, filename: str):
    """Saves an OpenFold Protein object to a standard PDB file."""
    pdb_str = protein.to_pdb(prot)
    with open(filename, "w") as f:
        f.write(pdb_str)
    print(f"💾 Saved prediction to {filename}")

def evaluate_pdb(model, pdb_path: str, output_dir: str):
    """Runs a single PDB through the model and calculates accuracy metrics."""
    pdb_path = Path(pdb_path)
    print(f"\n🧪 Evaluating {pdb_path.name}...")
    
    # 1. Parse ground truth
    with open(pdb_path, 'r') as f:
        pdb_str = f.read()
    
    # Extract native coordinates for baseline
    prot_true = protein.from_pdb_string(pdb_str)
    
    # 2. Build Features using OpenFold's make_pdb_features
    from openfold.data.data_pipeline import make_pdb_features
    
    config = model_config("model_1_ptm")
    config.data.common.use_templates = False
    fp = feature_pipeline.FeaturePipeline(config.data)
    
    L = len(prot_true.aatype)
    features = make_pdb_features(prot_true, description=pdb_path.stem, is_distillation=False)
    
    # Add dummy MSA features (just the target sequence itself)
    features["msa"] = np.array([prot_true.aatype], dtype=np.int64)
    features["deletion_matrix_int"] = np.zeros((1, L), dtype=np.int64)
    features["num_alignments"] = np.array([1] * L, dtype=np.int64)
    features["msa_species_identifiers"] = np.array([b''], dtype=object)
    
    processed = fp.process_features(features, mode='predict')
    
    # Convert to batch-dimension tensors on correct device
    batch = {k: torch.tensor(v).unsqueeze(0).to(model.device) for k, v in processed.items()}
    
    # 3. Inference
    t0 = time.time()
    with torch.no_grad():
        out = model.model(batch)
    t1 = time.time()
    print(f"⚡ Inference completed in {t1-t0:.2f}s")
    
    # 4. Extract Predicted Coordinates
    # OpenFold outputs 'final_atom_positions' of shape [B, N, 37, 3]
    # We grab the CA atoms (index 1 in residue_constants.atom_order)
    pred_positions = out["final_atom_positions"][0].cpu().numpy()
    
    # CA RMSD
    ca_pred = pred_positions[:, 1, :]
    ca_true = prot_true.atom_positions[:, 1, :]
    
    # Filter out missing CA atoms in ground truth using the atom_mask
    ca_mask = prot_true.atom_mask[:, 1].astype(bool)
    ca_pred_masked = ca_pred[ca_mask]
    ca_true_masked = ca_true[ca_mask]
    
    rmsd = calculate_rmsd(ca_pred_masked, ca_true_masked)
    print(f"📊 C-alpha RMSD: {rmsd:.2f} Å (Angstroms)")
    
    # 5. Save Predicted Structure to PDB
    # Construct a new Protein object using the predicted coordinates
    prot_pred = protein.Protein(
        atom_positions=pred_positions,
        aatype=prot_true.aatype,
        atom_mask=prot_true.atom_mask,
        residue_index=prot_true.residue_index,
        chain_index=getattr(prot_true, 'chain_index', np.zeros_like(prot_true.aatype)),
        b_factors=np.zeros_like(prot_true.b_factors) # No real b-factors for prediction
    )
    
    os.makedirs(output_dir, exist_ok=True)
    out_pdb = os.path.join(output_dir, f"{pdb_path.stem}_pred.pdb")
    save_to_pdb(prot_pred, out_pdb)
    
    return rmsd

def main():
    print("="*60)
    print("🔬 Mini-AlphaFold Benchmarking Suite")
    print("="*60)
    
    # 1. Load Model
    ckpt_path = get_latest_checkpoint()
    print(f"📦 Loading weights from: {ckpt_path}")
    
    # Load onto GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("⚙️  Initializing model architecture...")
    model = MiniAlphaFoldModule.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    print("✅ Model loaded and set to evaluation mode.")
    
    # 2. Run Benchmarks
    # We'll use a small X-ray crystal test PDB (1crn.pdb)
    test_pdbs = ["1crn.pdb"]
    
    # Download them if they are missing
    import urllib.request
    for pdb in test_pdbs:
        if not os.path.exists(pdb):
            pdb_id = pdb.split('.')[0]
            print(f"📥 Downloading test file {pdb}...")
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
            try:
                urllib.request.urlretrieve(url, pdb)
            except Exception as e:
                print(f"Failed to download {pdb}: {e}")
    
    results = {}
    for pdb in test_pdbs:
        if os.path.exists(pdb):
            rmsd = evaluate_pdb(model, pdb, "benchmark_results")
            results[pdb] = rmsd
        else:
            print(f"⚠️ Test file {pdb} not found in workspace. Skipping.")
            
    print("\n" + "="*60)
    print("🏆 FINAL RESULTS SUMMARY")
    print("="*60)
    for pdb, rmsd in results.items():
        print(f" - {pdb}: {rmsd:.2f} Å RMSD")
        
    print("\n(Note: An RMSD < 3.0 Å means the backbone folded correctly! An RMSD < 1.5 Å is atomic accuracy).")

if __name__ == "__main__":
    main()
