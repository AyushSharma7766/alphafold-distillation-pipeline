import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from scripts.benchmark_accuracy import extract_ca_and_seq_from_pdb, BENCHMARK_TARGETS
from train import kabsch_rmsd

def test_helix_geometry():
    # True Villin HP-36 (1vii.pdb)
    true_ca, seq = extract_ca_and_seq_from_pdb("data/raw/1vii.pdb")
    print(f"Loaded 1vii.pdb: Length {len(seq)}")
    
    # Let's inspect the dihedral angles of true Villin:
    def calc_dihedrals(coords):
        b1 = coords[1:-2] - coords[:-3]
        b2 = coords[2:-1] - coords[1:-2]
        b3 = coords[3:] - coords[2:-1]
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1 /= np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-8
        n2 /= np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-8
        m1 = np.cross(n1, b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8))
        x = np.sum(n1 * n2, axis=-1)
        y = np.sum(m1 * n2, axis=-1)
        return np.degrees(np.arctan2(y, x))

    true_tau = calc_dihedrals(true_ca)
    print("True 1VII dihedrals (residues 2-10, Helix 1):", np.round(true_tau[:8], 1))
    
    # Calculate bond lengths of true Villin:
    bonds = np.linalg.norm(true_ca[1:] - true_ca[:-1], axis=-1)
    print(f"True 1VII CA-CA bond lengths: mean={np.mean(bonds):.2f}, min={np.min(bonds):.2f}, max={np.max(bonds):.2f}")

test_helix_geometry()
