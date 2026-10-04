"""
Mini-AlphaFold: In-Silico Variant & Mutation Sensitivity Analyzer
================================================================
Evaluates the biophysical and structural impact of single- and multi-point
mutations on predicted 3D protein folds.

Features:
  - Parses standard mutation syntax (e.g. 'I7P', 'V8A', 'C3A')
  - Predicts both Wild-Type (WT) and Mutant 3D structures and distance maps
  - Aligns backbones via Kabsch algorithm to measure per-residue displacement
  - Generates differential distance heatmaps (|D_mut - D_wt|) and contact disruption
  - Exports WT and Mutant .pdb files with full N, CA, C, O geometry

Usage:
    python src/mutation_analyzer.py --domain-id 1crnA00 --mutation I7P
    python src/mutation_analyzer.py --sequence TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN --mutation I7P
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from predict import Predictor, save_pdb
from train import kabsch_rmsd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_mutation(mut_str: str) -> tuple[str, int, str]:
    """Parses standard single-letter mutation string, e.g. 'I7P' or 'V8A'.

    Returns:
        (from_aa, 0_indexed_pos, to_aa)
    """
    mut_str = mut_str.strip().upper()
    from_aa = mut_str[0]
    to_aa = mut_str[-1]
    pos_1based = int(mut_str[1:-1])
    return from_aa, pos_1based - 1, to_aa


def apply_mutations(sequence: str, mutations: list[str]) -> tuple[str, list[dict[str, Any]]]:
    """Applies a list of mutations to a sequence and validates them."""
    seq_chars = list(sequence)
    parsed_muts = []

    for mut in mutations:
        from_aa, pos, to_aa = parse_mutation(mut)
        if pos < 0 or pos >= len(sequence):
            raise ValueError(f"Mutation position {pos + 1} is out of bounds for sequence of length {len(sequence)}")
        actual_aa = sequence[pos]
        if actual_aa != from_aa:
            log.warning(
                f"Mutation {mut} expects '{from_aa}' at position {pos + 1}, but found '{actual_aa}'. Proceeding with '{actual_aa}' -> '{to_aa}'."
            )
        seq_chars[pos] = to_aa
        parsed_muts.append({
            "from_aa": actual_aa,
            "pos": pos,
            "to_aa": to_aa,
            "str": f"{actual_aa}{pos + 1}{to_aa}",
        })

    mutated_seq = "".join(seq_chars)
    return mutated_seq, parsed_muts


def align_coordinates(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """Superimposes coords_b onto coords_a using Kabsch SVD alignment.

    Args:
        coords_a: [L, 3] reference (WT)
        coords_b: [L, 3] mobile (Mutant)

    Returns:
        aligned_b: [L, 3] mobile coordinates aligned to reference
    """
    ca = coords_a - coords_a.mean(axis=0)
    cb = coords_b - coords_b.mean(axis=0)

    h = cb.T @ ca
    u, s, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    diag = np.eye(3)
    diag[2, 2] = 1.0 if d >= 0 else -1.0
    r = vt.T @ diag @ u.T

    aligned_b = (cb @ r.T) + coords_a.mean(axis=0)
    return aligned_b


def analyze_mutation(
    wt_sequence: str,
    mutations: list[str],
    predictor: Optional[Predictor] = None,
    name: str = "protein",
    out_dir: Path | str = "predictions/mutations",
) -> dict[str, Any]:
    """Performs end-to-end mutation analysis."""
    if predictor is None:
        predictor = Predictor()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mut_sequence, parsed_muts = apply_mutations(wt_sequence, mutations)
    mut_label = "_".join(m["str"] for m in parsed_muts)

    log.info(f"Predicting Wild-Type ({len(wt_sequence)} aa)...")
    wt_res = predictor.predict_sequence(wt_sequence)

    log.info(f"Predicting Mutant [{mut_label}] ({len(mut_sequence)} aa)...")
    mut_res = predictor.predict_sequence(mut_sequence)

    # 1. Differential Distance Matrix: Delta D = |D_mut - D_wt|
    delta_dist = np.abs(mut_res["pred_dist"] - wt_res["pred_dist"])

    # 2. Contact Disruption (< 8.0 A threshold)
    wt_contacts = wt_res["coord_dist"] <= 8.0
    mut_contacts = mut_res["coord_dist"] <= 8.0
    lost_contacts = int(np.sum(wt_contacts & ~mut_contacts)) // 2
    gained_contacts = int(np.sum(~wt_contacts & mut_contacts)) // 2

    # 3. Superposition & Per-Residue Displacement
    aligned_mut_ca = align_coordinates(wt_res["ca_coords"], mut_res["ca_coords"])
    per_res_disp = np.linalg.norm(aligned_mut_ca - wt_res["ca_coords"], axis=-1)  # [L]
    global_rmsd = float(np.sqrt(np.mean(per_res_disp ** 2)))

    # Save PDBs
    wt_pdb = out_dir / f"{name}_WT.pdb"
    mut_pdb = out_dir / f"{name}_MUT_{mut_label}.pdb"
    save_pdb(wt_res["backbone_coords"], wt_sequence, wt_pdb, helices=wt_res["helices"], sheets=wt_res["sheets"])
    save_pdb(mut_res["backbone_coords"], mut_sequence, mut_pdb, helices=mut_res["helices"], sheets=mut_res["sheets"])

    # Generate 4-panel diagnostic plot
    plot_file = out_dir / f"{name}_mutation_{mut_label}.png"
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)

    # Panel 1: WT Distance Matrix
    im1 = axes[0].imshow(wt_res["pred_dist"], cmap="viridis_r", vmin=0, vmax=25)
    axes[0].set_title("Wild-Type Distances", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Residue Index")
    axes[0].set_ylabel("Residue Index")
    plt.colorbar(im1, ax=axes[0], label="Distance (Å)")

    # Panel 2: Mutant Distance Matrix
    im2 = axes[1].imshow(mut_res["pred_dist"], cmap="viridis_r", vmin=0, vmax=25)
    axes[1].set_title(f"Mutant ({mut_label}) Distances", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Residue Index")
    plt.colorbar(im2, ax=axes[1], label="Distance (Å)")

    # Panel 3: Delta Distance Matrix
    vmax_delta = max(5.0, float(np.percentile(delta_dist, 98)))
    im3 = axes[2].imshow(delta_dist, cmap="magma", vmin=0, vmax=vmax_delta)
    axes[2].set_title(f"|Δ Distance| (Max: {delta_dist.max():.2f} Å)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Residue Index")
    plt.colorbar(im3, ax=axes[2], label="|Δ Distance| (Å)")

    # Highlight mutation positions with crosshairs
    for m in parsed_muts:
        p = m["pos"]
        axes[2].axvline(p, color="cyan", linestyle="--", alpha=0.7, linewidth=1.2)
        axes[2].axhline(p, color="cyan", linestyle="--", alpha=0.7, linewidth=1.2)

    # Panel 4: Per-Residue Displacement Profile
    res_indices = np.arange(1, len(wt_sequence) + 1)
    axes[3].plot(res_indices, per_res_disp, color="#e63946", linewidth=2.0, label="Displacement")
    axes[3].fill_between(res_indices, per_res_disp, color="#e63946", alpha=0.2)
    for m in parsed_muts:
        p = m["pos"] + 1
        disp_val = per_res_disp[m["pos"]]
        axes[3].scatter([p], [disp_val], color="#1d3557", s=70, zorder=5)
        axes[3].annotate(
            m["str"],
            (p, disp_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            color="#1d3557",
            fontsize=9,
        )
    axes[3].set_title(f"Displacement Profile (RMSD: {global_rmsd:.2f} Å)", fontsize=11, fontweight="bold")
    axes[3].set_xlabel("Residue Position")
    axes[3].set_ylabel("Displacement (Å)")
    axes[3].grid(True, linestyle=":", alpha=0.5)

    title_str = f"In-Silico Mutation Analysis: {name} [{mut_label}]"
    fig.suptitle(title_str, fontsize=13, fontweight="bold")
    plt.savefig(plot_file, dpi=200)
    plt.close()
    log.info(f"  [PLOT] Saved mutation analysis to: {plot_file}")

    result = {
        "wt_sequence": wt_sequence,
        "mut_sequence": mut_sequence,
        "mut_label": mut_label,
        "mutations": parsed_muts,
        "global_rmsd": global_rmsd,
        "mean_displacement": float(np.mean(per_res_disp)),
        "max_displacement": float(np.max(per_res_disp)),
        "lost_contacts": lost_contacts,
        "gained_contacts": gained_contacts,
        "wt_pdb": str(wt_pdb),
        "mut_pdb": str(mut_pdb),
        "plot_file": str(plot_file),
        "per_res_disp": per_res_disp.tolist(),
        "delta_dist": delta_dist,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="In-Silico Variant & Mutation Sensitivity Analyzer")
    parser.add_argument("--domain-id", type=str, default=None, help="CATH domain ID from data/processed")
    parser.add_argument("--sequence", type=str, default=None, help="Raw 1D amino acid sequence")
    parser.add_argument("--mutation", type=str, required=True, nargs="+", help="One or more mutations (e.g., I7P V8A)")
    parser.add_argument("--name", type=str, default=None, help="Name for outputs")
    parser.add_argument("--output-dir", type=str, default="predictions/mutations", help="Output directory")
    args = parser.parse_args()

    if args.domain_id:
        pt_path = Path("data/processed") / f"{args.domain_id}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Domain not found: {pt_path}")
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        wt_seq = data["sequence"]
        name = args.name or args.domain_id
    elif args.sequence:
        wt_seq = args.sequence
        name = args.name or "mutant_protein"
    else:
        # Default: Crambin 1crnA00
        pt_path = Path("data/processed/1crnA00.pt")
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        wt_seq = data["sequence"]
        name = "1crnA00"

    predictor = Predictor()
    res = analyze_mutation(
        wt_sequence=wt_seq,
        mutations=args.mutation,
        predictor=predictor,
        name=name,
        out_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print(f"MUTATION ANALYSIS COMPLETE FOR {name} [{res['mut_label']}]")
    print("=" * 60)
    print(f"  Wild-Type Sequence: {res['wt_sequence']}")
    print(f"  Mutant Sequence:    {res['mut_sequence']}")
    print(f"  Global RMSD:        {res['global_rmsd']:.2f} Å")
    print(f"  Mean Displacement:  {res['mean_displacement']:.2f} Å")
    print(f"  Max Displacement:   {res['max_displacement']:.2f} Å")
    print(f"  Contacts Lost (<8Å):{res['lost_contacts']}")
    print(f"  Contacts Gained:    {res['gained_contacts']}")
    print(f"  Wild-Type PDB:      {res['wt_pdb']}")
    print(f"  Mutant PDB:         {res['mut_pdb']}")
    print(f"  Diagnostic Plot:    {res['plot_file']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
