"""
Mini-AlphaFold: Systematic Accuracy Benchmarking Tool
Evaluates:
  - CA-RMSD (Kabsch aligned, Angstroms)
  - TM-Score (structural similarity 0 to 1, >0.5 indicates same global fold)
  - Contact Precision P@L and P@L/5 (medium/long-range contacts |i-j| >= 6, cutoff <= 8.0 A)
  - Q3 Secondary Structure Accuracy (% identical SS assignment)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any
import numpy as np
import torch
from predict import Predictor, kabsch_rmsd, compute_contact_precision, assign_secondary_structure

BENCHMARK_TARGETS = [
    {"name": "Crambin (1CRN)", "pdb": "data/raw/1CRN.pdb", "length": 46},
    {"name": "Villin HP-36 (1VII)", "pdb": "data/raw/1vii.pdb", "length": 36},
    {"name": "Trp-Cage (1L2Y)", "pdb": "data/raw/1l2y.pdb", "length": 20},
    {"name": "Designed Fold (2DK4)", "pdb": "data/raw/2dk4A00.pdb", "length": 76},
]


def compute_tm_score(coords_pred: np.ndarray, coords_true: np.ndarray) -> float:
    """Computes TM-score after Kabsch alignment.
    TM-score = 1/L * sum(1 / (1 + (d_i / d_0)^2))
    d0 = 1.24 * (L - 15)^(1/3) - 1.8
    """
    L = len(coords_pred)
    if L < 16:
        d0 = 0.5
    else:
        d0 = max(0.5, 1.24 * ((L - 15) ** (1.0 / 3.0)) - 1.8)

    # Kabsch optimal superposition
    p = coords_pred - np.mean(coords_pred, axis=0)
    q = coords_true - np.mean(coords_true, axis=0)
    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    p_rot = p @ R.T

    d = np.linalg.norm(p_rot - q, axis=-1)
    tm = float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))
    return round(tm, 4)


def compute_q3_score(ss_pred: str, ss_true: str) -> float:
    """Computes 3-state secondary structure accuracy (Q3: Helix, Sheet, Coil)."""
    if len(ss_pred) != len(ss_true) or len(ss_pred) == 0:
        return 0.0
    matches = sum(1 for p, t in zip(ss_pred, ss_true) if p == t)
    return round(matches / len(ss_pred), 4)


def extract_ca_and_seq_from_pdb(pdb_path: str | Path) -> tuple[np.ndarray, str]:
    from predict import ONE_TO_THREE
    three_to_one = {v: k for k, v in ONE_TO_THREE.items()}

    ca_list = []
    seq_list = []
    seen_residues = set()

    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resi = int(line[22:26].strip())
                if resi in seen_residues:
                    continue
                seen_residues.add(resi)
                resn = line[17:20].strip()
                seq_list.append(three_to_one.get(resn, "A"))
                ca_list.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            elif line.startswith("ENDMDL"):
                break  # take Model 1 for NMR
    return np.array(ca_list), "".join(seq_list)


def evaluate_model(predictor: Predictor, relax: bool = True) -> list[dict[str, Any]]:
    results = []

    for target in BENCHMARK_TARGETS:
        pdb_p = Path(target["pdb"])
        if not pdb_p.exists():
            continue

        true_ca, seq = extract_ca_and_seq_from_pdb(pdb_p)
        L = len(true_ca)
        if L == 0:
            continue

        # Predict
        pred = predictor.predict_sequence(seq, relax=relax)
        pred_ca = pred["ca_coords"][:L]

        # Ground truth distance matrix & secondary structure
        true_diff = true_ca[:, None, :] - true_ca[None, :, :]
        true_dist = np.sqrt(np.sum(true_diff ** 2, axis=-1))
        _, _, ss_true = assign_secondary_structure(true_ca, true_dist)

        # Metrics
        rmsd = kabsch_rmsd(torch.from_numpy(pred_ca), torch.from_numpy(true_ca))
        tm = compute_tm_score(pred_ca, true_ca)
        q3 = compute_q3_score(pred["ss_string"][:L], ss_true)

        contacts = compute_contact_precision(
            torch.from_numpy(pred["coord_dist"][:L, :L]).unsqueeze(0),
            torch.from_numpy(true_dist).unsqueeze(0),
        )

        res = {
            "target": target["name"],
            "length": L,
            "rmsd": round(rmsd, 2),
            "tm_score": tm,
            "p_at_l": round(contacts["p_at_l"] * 100, 1),
            "p_at_l5": round(contacts["p_at_l5"] * 100, 1),
            "q3_ss": round(q3 * 100, 1),
        }
        results.append(res)

    return results


def print_results_table(title: str, results: list[dict[str, Any]]):
    print(f"\n{'='*75}")
    print(f"{title}")
    print(f"{'='*75}")
    print(f"{'Target':<24} | {'Length':<6} | {'RMSD (A)':<8} | {'TM-Score':<8} | {'P@L (%)':<8} | {'Q3 SS (%)'}")
    print(f"{'-'*75}")
    mean_rmsd = np.mean([r['rmsd'] for r in results])
    mean_tm = np.mean([r['tm_score'] for r in results])
    mean_pl = np.mean([r['p_at_l'] for r in results])
    mean_q3 = np.mean([r['q3_ss'] for r in results])

    for r in results:
        print(f"{r['target']:<24} | {r['length']:<6} | {r['rmsd']:<8.2f} | {r['tm_score']:<8.4f} | {r['p_at_l']:<8.1f} | {r['q3_ss']:.1f}%")
    print(f"{'-'*75}")
    print(f"{'AVERAGE':<24} | {'--':<6} | {mean_rmsd:<8.2f} | {mean_tm:<8.4f} | {mean_pl:<8.1f} | {mean_q3:.1f}%")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    predictor = Predictor()
    res = evaluate_model(predictor, relax=True)
    print_results_table("BASELINE BENCHMARK: Current Mini-AlphaFold Model", res)
