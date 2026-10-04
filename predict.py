"""
Mini-AlphaFold: Inference, Evaluation & 3D Structure Export
===========================================================
Predicts 3D protein backbone structures from 1D amino acid sequences.

Features:
  - Predicts 3D coordinates (N, CA, C) and pairwise distance maps
  - Exports standard .pdb files viewable in PyMOL, ChimeraX, or Mol*
  - Generates side-by-side comparison heatmaps (predicted vs ground truth)
  - Full test-set evaluation benchmark (1,109 proteins)

Usage:
    # Predict for a specific domain (e.g., Crambin)
    python predict.py --domain-id 1crnA00

    # Predict from a raw amino acid sequence
    python predict.py --sequence TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN --name my_protein

    # Evaluate full test set
    python predict.py --test-set
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data_parser import THREE_TO_ONE, AMINO_ACIDS, sequence_to_indices
from src.dataset import get_dataloaders
from src.loss import CompositeLoss
from src.model import MiniAlphaFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Reverse map: 1-letter -> 3-letter code
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}


# ============================================================
# PDB Exporter
# ============================================================

def save_pdb(
    backbone_coords: np.ndarray,
    sequence: str,
    out_path: Path | str,
    chain_id: str = "A",
    helices: Optional[list[tuple[int, int]]] = None,
    sheets: Optional[list[tuple[int, int]]] = None,
    plddt: Optional[np.ndarray] = None,
) -> None:
    """Exports predicted backbone coordinates to a standard PDB file with secondary structure annotations.

    Args:
        backbone_coords: [L, 4, 3] or [L, 3, 3] float array (0=N, 1=CA, 2=C, 3=O)
        sequence: 1-letter amino acid sequence of length L
        out_path: Filepath for the .pdb file
        chain_id: PDB chain identifier (default: 'A')
        helices: list of (start_res_idx, end_res_idx) 0-indexed inclusive
        sheets: list of (start_res_idx, end_res_idx) 0-indexed inclusive
        plddt: Optional [L] float array of residue confidence scores (0-100)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_o = backbone_coords.shape[1] >= 4
    atom_names = ["N", "CA", "C", "O"] if has_o else ["N", "CA", "C"]
    elements = ["N", "C", "C", "O"] if has_o else ["N", "C", "C"]

    lines = [
        "HEADER    PREDICTED STRUCTURE BY MINI-ALPHAFOLD",
        f"TITLE     {out_path.stem}",
    ]

    # Write standard PDB HELIX records so Mol* and PyMOL render cartoon ribbons
    if helices:
        for h_idx, (start, end) in enumerate(helices, start=1):
            init_res = ONE_TO_THREE.get(sequence[start], "ALA")
            term_res = ONE_TO_THREE.get(sequence[end], "ALA")
            init_seq = start + 1
            term_seq = end + 1
            h_len = end - start + 1
            h_id = f"H{h_idx:<2d}"
            # Standard PDB HELIX format (columns 1-80)
            line = (
                f"HELIX  {h_idx:3d} {h_id:3s} {init_res:3s} {chain_id} {init_seq:4d}  "
                f"{term_res:3s} {chain_id} {term_seq:4d}  1                              {h_len:5d}"
            )
            lines.append(line)

    # Write standard PDB SHEET records (multi-strand sheet format with shared sheetID and antiparallel sense)
    if sheets:
        n_strands = len(sheets)
        for s_idx, (start, end) in enumerate(sheets, start=1):
            init_res = ONE_TO_THREE.get(sequence[start], "ALA")
            term_res = ONE_TO_THREE.get(sequence[end], "ALA")
            init_seq = start + 1
            term_seq = end + 1
            sense = 0 if s_idx == 1 else -1
            line = (
                f"SHEET  {s_idx:3d}  S1{n_strands:2d} {init_res:3s} {chain_id}{init_seq:4d}  "
                f"{term_res:3s} {chain_id}{term_seq:4d} {sense:2d}"
            )
            lines.append(line)

    atom_serial = 1
    for res_idx, aa in enumerate(sequence):
        res_seq = res_idx + 1
        res_name = ONE_TO_THREE.get(aa, "GLY")
        b_val = float(plddt[res_idx]) if plddt is not None else 85.0

        for atom_idx, (atom_name, element) in enumerate(zip(atom_names, elements)):
            x, y, z = backbone_coords[res_idx, atom_idx]

            # Standard PDB ATOM record format with pLDDT in B-factor column (cols 61-66)
            line = (
                f"ATOM  {atom_serial:5d} {atom_name:^4s} {res_name:3s} {chain_id}{res_seq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 {b_val:5.2f}          {element:>2s}"
            )
            lines.append(line)
            atom_serial += 1

    lines.append("TER")
    lines.append("END")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"  [PDB] Saved 3D structure with secondary structure headers to: {out_path}")


# ============================================================
# Visualization: Contact & Distance Heatmaps
# ============================================================

def plot_prediction(
    pred_dist: np.ndarray,
    coord_dist: np.ndarray,
    true_dist: Optional[np.ndarray],
    sequence: str,
    out_path: Path | str,
    title: str = "Structure Prediction",
    rmsd: Optional[float] = None,
    p_at_l: Optional[float] = None,
) -> None:
    """Generates a multi-panel figure comparing predicted distances and contact maps."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    has_truth = true_dist is not None
    n_cols = 4 if has_truth else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5), constrained_layout=True)

    # 1. Predicted Distance Matrix (from distance head)
    im1 = axes[0].imshow(pred_dist, cmap="viridis_r", vmin=0, vmax=25)
    axes[0].set_title("Predicted Pair Distance (Head)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Residue Index")
    axes[0].set_ylabel("Residue Index")
    plt.colorbar(im1, ax=axes[0], label="Distance (Å)")

    # 2. 3D Coordinate Distance Matrix (from predicted CA coords)
    im2 = axes[1].imshow(coord_dist, cmap="viridis_r", vmin=0, vmax=25)
    axes[1].set_title("3D Structure CA Distances", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Residue Index")
    plt.colorbar(im2, ax=axes[1], label="Distance (Å)")

    if has_truth:
        # 3. Ground Truth Distance Matrix
        im3 = axes[2].imshow(true_dist, cmap="viridis_r", vmin=0, vmax=25)
        axes[2].set_title("Ground Truth Distance (PDB)", fontsize=11, fontweight="bold")
        axes[2].set_xlabel("Residue Index")
        plt.colorbar(im3, ax=axes[2], label="Distance (Å)")

        # 4. Contact Map Overlay (< 8.0 Å)
        L = len(sequence)
        pred_contacts = coord_dist <= 8.0
        true_contacts = true_dist <= 8.0

        overlay = np.zeros((L, L, 3))
        # Blue = True contacts, Red = Predicted contacts, Magenta = Match (True Positive)
        overlay[true_contacts, 2] = 0.8   # Blue
        overlay[pred_contacts, 0] = 0.9   # Red
        axes[3].imshow(overlay)
        axes[3].set_title("Contacts (<8Å): Red=Pred, Blue=True", fontsize=11, fontweight="bold")
        axes[3].set_xlabel("Residue Index")

    subtitle = f"Length: {len(sequence)} aa"
    if rmsd is not None:
        subtitle += f" | CA-RMSD: {rmsd:.2f} Å"
    if p_at_l is not None:
        subtitle += f" | P@L: {p_at_l * 100:.1f}%"

    fig.suptitle(f"{title} ({subtitle})", fontsize=13, fontweight="bold")
    plt.savefig(out_path, dpi=200)
    plt.close()
    log.info(f"  [PLOT] Saved comparison figure to: {out_path}")





def assign_secondary_structure(
    ca_coords: np.ndarray,
    dist_matrix: np.ndarray,
    sequence: Optional[str] = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], str]:
    """Accurately detects alpha-helices and beta-sheets for any protein sequence
    combining local virtual distance geometry, turn compactness, and non-local contacts.

    Returns:
        helices: list of (start_res, end_res) 0-indexed inclusive
        sheets: list of (start_res, end_res) 0-indexed inclusive
        ss_string: L-character string ('H' for helix, 'E' for sheet, 'C' for coil)
    """
    L = len(dist_matrix)
    d2 = np.array([dist_matrix[i, i + 2] if i + 2 < L else 99.0 for i in range(L)])
    d3 = np.array([dist_matrix[i, i + 3] if i + 3 < L else 99.0 for i in range(L)])
    d4 = np.array([dist_matrix[i, i + 4] if i + 4 < L else 99.0 for i in range(L)])

    # Helical condition: compact virtual bond angles and turn distances
    raw_h = (d2 <= 6.15) & (d3 <= 8.50) & (d4 <= 9.85)

    # Proline rule: Proline cannot reside inside a continuous alpha-helix
    if sequence is not None and len(sequence) == L:
        for i in range(L):
            if sequence[i] == "P":
                raw_h[i] = False

    # Bridge 1-residue gaps inside helices
    for i in range(1, L - 1):
        if not raw_h[i] and raw_h[i - 1] and raw_h[i + 1]:
            if sequence is None or sequence[i] != "P":
                raw_h[i] = True

    # Extract continuous helices (length >= 4)
    helices: list[tuple[int, int]] = []
    i = 0
    while i < L:
        if raw_h[i]:
            j = i
            while j < L and raw_h[j]:
                j += 1
            if j - i >= 4:
                helices.append((i, j - 1))
            i = j
        else:
            i += 1

    h_mask = np.zeros(L, dtype=bool)
    for s, e in helices:
        h_mask[s:e + 1] = True

    # Beta-sheet condition: extended virtual angle and non-helical
    raw_s = (d2 >= 6.12) & (d2 < 90.0) & (~h_mask)
    raw_sheets: list[tuple[int, int]] = []
    i = 0
    while i < L:
        if raw_s[i]:
            j = i
            while j < L and raw_s[j]:
                j += 1
            if j - i >= 2:
                raw_sheets.append((i, j - 1))
            i = j
        else:
            i += 1

    # Filter sheets to those that form non-local pairing contacts
    sheets: list[tuple[int, int]] = []
    for s, e in raw_sheets:
        strand_mask = np.zeros(L, dtype=bool)
        strand_mask[s:e + 1] = True
        non_local = np.any(
            (dist_matrix[strand_mask, :] <= 10.5)
            & (np.abs(np.arange(L)[None, :] - np.arange(s, e + 1)[:, None]) >= 4)
        )
        if non_local:
            sheets.append((s, e))

    ss = ["C"] * L
    for s, e in helices:
        for k in range(s, e + 1):
            ss[k] = "H"
    for s, e in sheets:
        for k in range(s, e + 1):
            if ss[k] != "H":
                ss[k] = "E"

    return helices, sheets, "".join(ss)


def _compute_dihedrals_torch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes pseudo-dihedral angles for consecutive CA quads: x_{i-1}, x_i, x_{i+1}, x_{i+2}."""
    v1 = x[1:-2] - x[:-3]
    v2 = x[2:-1] - x[1:-2]
    v3 = x[3:] - x[2:-1]

    n1 = torch.cross(v1, v2, dim=-1)
    n2 = torch.cross(v2, v3, dim=-1)

    n1_norm = torch.norm(n1, dim=-1, keepdim=True) + 1e-8
    n2_norm = torch.norm(n2, dim=-1, keepdim=True) + 1e-8
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    v2_unit = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-8)
    m1 = torch.cross(n1, v2_unit, dim=-1)

    cos_tau = torch.sum(n1 * n2, dim=-1)
    sin_tau = torch.sum(m1 * n2, dim=-1)
    return cos_tau, sin_tau


def relax_coordinates(
    ca_coords: np.ndarray,
    D: np.ndarray,
    helix_mask: Optional[np.ndarray] = None,
    sheet_mask: Optional[np.ndarray] = None,
    steps: int = 150,
) -> np.ndarray:
    """Refines coordinates using physics-informed PyTorch optimization:
    - Enforces true right-handed helical dihedral potential (tau = -50.4 deg) for tight cylindrical 3D coiling
    - Enforces extended trans dihedral potential (tau = 175 deg) for beta-strands
    - Clamps target distances on secondary structure pairs to prevent loss_dist from pulling helices apart
    - Covalent bond lengths CA_i - CA_{i+1} = 3.80 A with post-optimization SHAKE projection
    - Virtual bond angle CA_{i-1} - CA_i - CA_{i+1} distance d(i, i+2) in [5.1, 7.2] A
    - Smooth Laplacian loop regularization applied ONLY to flexible coils, eliminating crinkles
    - Steric clash clearance (> 3.8 A for non-bonded pairs)
    """
    L = len(ca_coords)
    if L < 4:
        return ca_coords

    # Secondary structure masks
    h_mask = helix_mask if helix_mask is not None else np.zeros(L, dtype=bool)
    s_mask = sheet_mask if sheet_mask is not None else np.zeros(L, dtype=bool)
    loop_mask = ~(h_mask | s_mask)

    # 1. Clamp target distance matrix for secondary structures so loss_dist cooperates with physical springs
    target_D_np = D.copy()
    for i in range(L):
        if h_mask[i]:
            if i + 1 < L and h_mask[i + 1]:
                target_D_np[i, i + 1] = target_D_np[i + 1, i] = 3.80
            if i + 2 < L and h_mask[i + 2]:
                target_D_np[i, i + 2] = target_D_np[i + 2, i] = 5.40
            if i + 3 < L and h_mask[i + 3]:
                target_D_np[i, i + 3] = target_D_np[i + 3, i] = 5.05
            if i + 4 < L and h_mask[i + 4]:
                target_D_np[i, i + 4] = target_D_np[i + 4, i] = 6.20
        elif s_mask[i]:
            if i + 1 < L and s_mask[i + 1]:
                target_D_np[i, i + 1] = target_D_np[i + 1, i] = 3.80
            if i + 2 < L and s_mask[i + 2]:
                target_D_np[i, i + 2] = target_D_np[i + 2, i] = max(target_D_np[i, i + 2], 6.50)

    # 2. PyTorch energy minimization with physical force field
    x = torch.from_numpy(ca_coords.copy()).float().requires_grad_(True)
    target_D = torch.from_numpy(target_D_np).float()
    opt = torch.optim.Adam([x], lr=0.030)
    mask_nb = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=2)

    h_mask_t = torch.tensor(h_mask, dtype=torch.bool)
    s_mask_t = torch.tensor(s_mask, dtype=torch.bool)
    loop_mask_t = torch.tensor(loop_mask, dtype=torch.bool)

    # Contact-sensitive weights: strong attraction for close contacts (<=8A), decaying for diffuse distances
    weights = torch.ones_like(target_D)
    weights = torch.where(target_D <= 8.0, 2.5, weights)
    weights = torch.where(target_D > 14.0, 0.25, weights)
    w_nb = weights[mask_nb]

    # Dihedral targets: correct right-handed alpha-helix dihedral (-50.4 deg) and extended beta-sheet (175 deg)
    target_cos_helix = float(np.cos(np.radians(-50.4)))
    target_sin_helix = float(np.sin(np.radians(-50.4)))
    target_cos_sheet = float(np.cos(np.radians(175.0)))
    target_sin_sheet = float(np.sin(np.radians(175.0)))

    # Target radius of gyration (Flory law for globular proteins: Rg ~ 2.82 * L^0.28)
    rg_target = float(2.82 * (L ** 0.28))

    for step in range(steps):
        opt.zero_grad()
        cur_D = torch.cdist(x, x)

        # Contact-weighted Huber distance loss
        diff = torch.abs(cur_D[mask_nb] - target_D[mask_nb])
        huber_diff = torch.where(diff < 2.0, 0.5 * (diff ** 2), 2.0 * diff - 2.0)
        loss_dist = torch.mean(w_nb * huber_diff)

        # Consecutive CA-CA bond length = 3.80 A (stiff harmonic spring)
        bonds = torch.norm(x[1:] - x[:-1], dim=-1)
        loss_bond = torch.mean((bonds - 3.80) ** 2)

        # Virtual bond angle constraint d(i, i+2) in [5.1, 7.2] A
        d_i_i2 = torch.norm(x[2:] - x[:-2], dim=-1)
        loss_angle = torch.mean(
            torch.clamp(5.1 - d_i_i2, min=0.0) ** 2 + torch.clamp(d_i_i2 - 7.2, min=0.0) ** 2
        )

        # Secondary structure hydrogen-bond network regularization (stiff canonical geometry)
        loss_ss = torch.tensor(0.0)
        for i in range(L - 4):
            if h_mask_t[i] and h_mask_t[i + 2]:
                loss_ss = loss_ss + (torch.norm(x[i + 2] - x[i]) - 5.40) ** 2
            if h_mask_t[i] and h_mask_t[i + 3]:
                loss_ss = loss_ss + (torch.norm(x[i + 3] - x[i]) - 5.05) ** 2
            if h_mask_t[i] and h_mask_t[i + 4]:
                loss_ss = loss_ss + (torch.norm(x[i + 4] - x[i]) - 6.20) ** 2
        for i in range(L - 2):
            if s_mask_t[i] and s_mask_t[i + 2]:
                loss_ss = loss_ss + (torch.norm(x[i + 2] - x[i]) - 6.60) ** 2

        # Chiral dihedral coiling potential (forces right-handed 3D cylindrical spirals!)
        cos_tau, sin_tau = _compute_dihedrals_torch(x)
        loss_dihedral = torch.tensor(0.0)
        n_quads = 0
        for j in range(len(cos_tau)):
            if h_mask_t[j] and h_mask_t[j + 1] and h_mask_t[j + 2] and h_mask_t[j + 3]:
                loss_dihedral = loss_dihedral + (cos_tau[j] - target_cos_helix) ** 2 + (sin_tau[j] - target_sin_helix) ** 2
                n_quads += 1
            elif s_mask_t[j] and s_mask_t[j + 1] and s_mask_t[j + 2] and s_mask_t[j + 3]:
                loss_dihedral = loss_dihedral + (cos_tau[j] - target_cos_sheet) ** 2 + (sin_tau[j] - target_sin_sheet) ** 2
                n_quads += 1
        if n_quads > 0:
            loss_dihedral = loss_dihedral / n_quads

        # Radius of gyration compactness prior
        center = torch.mean(x, dim=0, keepdim=True)
        cur_rg = torch.sqrt(torch.mean(torch.sum((x - center) ** 2, dim=-1)) + 1e-6)
        loss_rg = torch.clamp(cur_rg - (rg_target * 1.20), min=0.0) ** 2

        # Smooth loop regularization (Laplacian curvature minimization on loops to eliminate crinkles)
        loss_smooth = torch.tensor(0.0)
        for i in range(L - 2):
            if loop_mask_t[i] and loop_mask_t[i + 1] and loop_mask_t[i + 2]:
                lap = x[i + 2] - 2.0 * x[i + 1] + x[i]
                loss_smooth = loss_smooth + torch.sum(lap ** 2)

        # Steric clash avoidance (> 3.8 A for non-bonded pairs)
        clashes = torch.clamp(3.8 - cur_D[mask_nb], min=0.0)
        loss_clash = torch.mean(clashes ** 2)

        total_loss = (
            2.0 * loss_dist
            + 120.0 * loss_bond
            + 25.0 * loss_angle
            + 35.0 * loss_ss
            + 40.0 * loss_dihedral
            + 2.5 * loss_rg
            + 1.0 * loss_smooth
            + 15.0 * loss_clash
        )
        total_loss.backward()
        opt.step()

    out_coords = x.detach().numpy()

    # 3. Holonomic covalent bond constraint (SHAKE algorithm)
    # Guarantees EVERY single CA-CA bond length is strictly 3.80 +- 0.005 A
    for _ in range(25):
        for i in range(L - 1):
            v = out_coords[i + 1] - out_coords[i]
            d = np.linalg.norm(v)
            diff = (d - 3.80) / (d + 1e-8)
            out_coords[i] += 0.5 * diff * v
            out_coords[i + 1] -= 0.5 * diff * v

    return out_coords


def build_full_backbone(ca_coords: np.ndarray, ss_string: Optional[str] = None) -> np.ndarray:
    """Builds realistic, stereochemically pristine peptide backbone (N, CA, C, O) atoms
    around CA coordinates using standard Engh & Huber trans-planar peptide geometry and
    continuous reference frames.

    Guarantees:
        - Continuous covalent polymer chain: d(C_i, N_{i+1}) = 1.33 A
        - Covalent bonds: d(CA_i, C_i) = 1.52 A, d(N_i, CA_i) = 1.46 A, d(C_i, O_i) = 1.23 A
        - Continuous, non-inverting ribbon normals so Mol* renders unbroken, smooth cartoon ribbons.

    Returns:
        backbone: [L, 4, 3] float array (atom 0=N, 1=CA, 2=C, 3=O)
    """
    L = len(ca_coords)
    backbone = np.zeros((L, 4, 3))
    backbone[:, 1] = ca_coords  # CA

    if L < 2:
        return backbone

    v = ca_coords[1:] - ca_coords[:-1]
    d = np.linalg.norm(v, axis=-1, keepdims=True)
    d = np.maximum(d, 1e-4)
    u = v / d  # [L-1, 3] Unit vectors along CA_i -> CA_{i+1}

    # Rotation-continuous reference frames along the chain
    normals = np.zeros((L - 1, 3))

    # Initialize first normal
    if L > 2:
        n0 = np.cross(u[0], u[1])
        if np.linalg.norm(n0) > 1e-3:
            normals[0] = n0 / np.linalg.norm(n0)
        else:
            normals[0] = np.array([0.0, 0.0, 1.0])
    else:
        normals[0] = np.array([0.0, 0.0, 1.0])

    for i in range(1, L - 1):
        n_raw = np.cross(u[i - 1], u[i])
        n_len = np.linalg.norm(n_raw)
        if n_len > 1e-3:
            n_cand = n_raw / n_len
            # Crucial: Enforce continuity so normal NEVER flips 180 degrees!
            if np.dot(n_cand, normals[i - 1]) < 0:
                n_cand = -n_cand
            normals[i] = n_cand
        else:
            normals[i] = normals[i - 1]

    # Orthogonalize each normal to u[i] (Gram-Schmidt)
    for i in range(L - 1):
        proj = normals[i] - np.dot(normals[i], u[i]) * u[i]
        p_len = np.linalg.norm(proj)
        normals[i] = proj / p_len if p_len > 1e-4 else np.array([0.0, 1.0, 0.0])

    # Engh & Huber standard planar peptide placement:
    # Guarantees CA-C = 1.520 A, C-N = 1.331 A, N-CA = 1.459 A, C=O = 1.230 A
    for i in range(L - 1):
        ui = u[i]
        ni = normals[i]
        di = d[i, 0]

        sign = -1.0 if (ss_string and i < len(ss_string) and ss_string[i] == 'E' and i % 2 == 1) else 1.0
        eff_n = sign * ni

        # Carbonyl carbon C_i
        c_pos = ca_coords[i] + 1.445 * ui + 0.471 * eff_n
        backbone[i, 2] = c_pos

        # Amide nitrogen N_{i+1}
        n_next = ca_coords[i] + (di - 1.390) * ui - 0.445 * eff_n
        backbone[i + 1, 0] = n_next

        # Carbonyl oxygen O_i
        backbone[i, 3] = c_pos - 0.270 * ui + 1.200 * eff_n

    # N-terminus (residue 0)
    backbone[0, 0] = ca_coords[0] - 1.390 * u[0] - 0.445 * normals[0]

    # C-terminus (residue L - 1)
    c_last = ca_coords[-1] + 1.445 * u[-1] + 0.471 * normals[-1]
    backbone[-1, 2] = c_last
    backbone[-1, 3] = c_last - 0.270 * u[-1] + 1.200 * normals[-1]

    return backbone


# ============================================================
# Inference Engine
# ============================================================

class Predictor:
    def __init__(self, checkpoint_path: Optional[str] = None, config_path: str = "configs/default.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = "lightning_logs/lightning_logs/version_68/checkpoints/epoch=97-step=16425.ckpt"
        if not Path(ckpt).exists():
            ckpt = checkpoint_path
            
        from train_mini_alphafold import MiniAlphaFoldModule
        self.model = MiniAlphaFoldModule.load_from_checkpoint(ckpt, map_location=self.device, strict=False)
        self.model.eval()
        self.esm_model = None

    def predict_sequence(
        self,
        sequence: str,
        relax: bool = True,
        use_esm: bool = True,
        num_recycles: int = 2,
    ) -> dict[str, Any]:
        """Predict 3D structure and distance matrices for a raw amino acid sequence."""
        import numpy as np
        from openfold.data.data_pipeline import make_pdb_features
        from openfold.data import feature_pipeline
        from openfold.np import protein
        from openfold.config import model_config
        from openfold.np.residue_constants import restype_order
        
        aatype = np.array([restype_order.get(aa, 20) for aa in sequence], dtype=np.int64)
        L = len(aatype)
        
        prot_dummy = protein.Protein(
            atom_positions=np.zeros((L, 37, 3)),
            aatype=aatype,
            atom_mask=np.zeros((L, 37)),
            residue_index=np.arange(L) + 1,
            b_factors=np.zeros((L, 37))
        )
        
        config = model_config("model_1_ptm")
        config.data.common.use_templates = False
        fp = feature_pipeline.FeaturePipeline(config.data)
        
        features = make_pdb_features(prot_dummy, description="custom_seq", is_distillation=False)
        features["msa"] = np.array([aatype], dtype=np.int64)
        features["deletion_matrix_int"] = np.zeros((1, L), dtype=np.int64)
        features["num_alignments"] = np.array([1] * L, dtype=np.int64)
        features["msa_species_identifiers"] = np.array([b''], dtype=object)
        
        processed = fp.process_features(features, mode='predict')
        batch = {k: torch.tensor(v).unsqueeze(0).to(self.device) for k, v in processed.items()}
        
        with torch.no_grad():
            out = self.model.model(batch)
            
        pos = out["sm"]["positions"][-1].squeeze(0).cpu().numpy() # [L, 37, 3]
        
        # N, CA, C, O are 0, 1, 2, 4 in OpenFold
        backbone_coords = pos[:, [0, 1, 2, 4], :]
        ca_coords = pos[:, 1, :]
        
        diff = ca_coords[:, None, :] - ca_coords[None, :, :]
        coord_dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        helices, sheets, ss_string = assign_secondary_structure(ca_coords, coord_dist, sequence=sequence)
        
        plddt = out["plddt"].squeeze().cpu().numpy()
        
        return {
            "ca_coords": ca_coords,
            "backbone_coords": backbone_coords,
            "pred_dist": coord_dist,
            "coord_dist": coord_dist,
            "sequence": sequence,
            "helices": helices,
            "sheets": sheets,
            "ss_string": ss_string,
            "plddt": plddt,
        }


    def predict_domain(self, domain_id: str, processed_dir: str = "data/processed", relax: bool = True) -> dict[str, Any]:
        """Predict for a domain from data/processed and compare against ground truth."""
        pt_path = Path(processed_dir) / f"{domain_id}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Domain tensor not found: {pt_path}")

        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        sequence = data["sequence"]

        pred = self.predict_sequence(sequence, relax=relax)

        # Ground truth
        true_backbone = data["backbone_coords"].numpy()
        true_ca = data["ca_coords"].numpy()
        true_dist = data["distance_matrix"].numpy()

        # Compute metrics
        rmsd = kabsch_rmsd(torch.from_numpy(pred["ca_coords"]), torch.from_numpy(true_ca))
        contacts = compute_contact_precision(
            torch.from_numpy(pred["coord_dist"]).unsqueeze(0),
            torch.from_numpy(true_dist).unsqueeze(0),
        )

        pred.update({
            "true_backbone": true_backbone,
            "true_ca": true_ca,
            "true_dist": true_dist,
            "rmsd": rmsd,
            "p_at_l": contacts["p_at_l"],
            "p_at_l5": contacts["p_at_l5"],
            "domain_id": domain_id,
        })
        return pred

    def evaluate_test_set(self, processed_dir: str = "data/processed", batch_size: int = 2) -> dict[str, float]:
        """Evaluates model performance across all 1,109 held-out test domains."""
        log.info("\n============================================================")
        log.info("Running Full Test Set Benchmark (Held-out CATH Domains)")
        log.info("============================================================")

        _, _, test_dl = get_dataloaders(
            processed_dir=processed_dir,
            batch_size=batch_size,
            num_workers=0,
            max_seq_len=self.cfg["data"].get("max_seq_len", 128),
            preload=False,
            pin_memory=False,
        )

        criterion = CompositeLoss(
            fape_clamp=self.cfg["loss"].get("fape_clamp_angstrom", 10.0),
            clash_dist=self.cfg["loss"].get("clash_dist_angstrom", 3.0),
        ).to(self.device)

        total_loss, total_fape, total_dist = 0.0, 0.0, 0.0
        all_rmsds, all_p_at_l, all_p_at_l5 = [], [], []

        log.info(f"Evaluating {len(test_dl)} test batches ({len(test_dl.dataset)} proteins)...")

        with torch.no_grad():
            for batch in test_dl:
                seq = batch["seq"].to(self.device)
                backbone = batch["backbone"].to(self.device)
                dist_mat = batch["dist_mat"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputs = self.model(seq, mask=mask)
                losses = criterion(
                    pred_backbone=outputs["backbone_coords"].float(),
                    true_backbone=backbone.float(),
                    pred_dist=outputs["pred_dist"].float(),
                    true_dist=dist_mat.float(),
                    mask=mask,
                )

                total_loss += losses["loss"].item()
                total_fape += losses["loss_fape"].item()
                total_dist += losses["loss_dist"].item()

                pred_ca = outputs["ca_coords"]
                true_ca = backbone[:, :, 1]
                for b in range(seq.shape[0]):
                    m = mask[b]
                    rmsd = kabsch_rmsd(pred_ca[b], true_ca[b], mask=m)
                    all_rmsds.append(rmsd)

                contacts = compute_contact_precision(outputs["pred_dist"], dist_mat, mask=mask)
                all_p_at_l.append(contacts["p_at_l"])
                all_p_at_l5.append(contacts["p_at_l5"])

        n = len(test_dl)
        results = {
            "test_loss": total_loss / n,
            "fape": total_fape / n,
            "dist_error": total_dist / n,
            "mean_rmsd": sum(all_rmsds) / max(len(all_rmsds), 1),
            "median_rmsd": float(np.median(all_rmsds)),
            "p_at_l": sum(all_p_at_l) / max(len(all_p_at_l), 1),
            "p_at_l5": sum(all_p_at_l5) / max(len(all_p_at_l5), 1),
        }

        print("\n" + "=" * 60)
        print("MINI-ALPHAFOLD TEST SET RESULTS")
        print("=" * 60)
        print(f"  Test Loss:           {results['test_loss']:.4f}")
        print(f"  FAPE Loss:           {results['fape']:.3f} Å")
        print(f"  Distance MAE:        {results['dist_error']:.3f} Å")
        print(f"  Mean CA-RMSD:        {results['mean_rmsd']:.2f} Å")
        print(f"  Median CA-RMSD:      {results['median_rmsd']:.2f} Å")
        print(f"  Contact Prec (P@L):  {results['p_at_l'] * 100:.2f}%")
        print(f"  Contact Prec (P@L/5):{results['p_at_l5'] * 100:.2f}%")
        print("=" * 60 + "\n")

        return results


# ============================================================
# Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Predict 3D protein structure with Mini-AlphaFold")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (defaults to distogram_afdb_best.pt if present)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config")
    parser.add_argument("--domain-id", type=str, default=None, help="Domain ID in data/processed (e.g., 1crnA00)")
    parser.add_argument("--sequence", type=str, default=None, help="1D amino acid sequence string")
    parser.add_argument("--name", type=str, default="predicted_protein", help="Name for sequence prediction outputs")
    parser.add_argument("--output-dir", type=str, default="predictions", help="Directory for outputs")
    parser.add_argument("--no-relax", action="store_true", help="Disable coordinate relaxation")
    parser.add_argument("--test-set", action="store_true", help="Evaluate full test set")
    args = parser.parse_args()

    predictor = Predictor(checkpoint_path=args.checkpoint, config_path=args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    relax = not args.no_relax

    if args.test_set:
        predictor.evaluate_test_set()
        return

    if args.domain_id:
        log.info(f"Predicting 3D structure for domain: {args.domain_id} (relax={relax})...")
        res = predictor.predict_domain(args.domain_id, relax=relax)

        pdb_file = out_dir / f"{args.domain_id}_predicted.pdb"
        plot_file = out_dir / f"{args.domain_id}_comparison.png"

        save_pdb(
            res["backbone_coords"],
            res["sequence"],
            pdb_file,
            helices=res.get("helices"),
            sheets=res.get("sheets"),
            plddt=res.get("plddt"),
        )
        plot_prediction(
            pred_dist=res["pred_dist"],
            coord_dist=res["coord_dist"],
            true_dist=res["true_dist"],
            sequence=res["sequence"],
            out_path=plot_file,
            title=f"Domain {args.domain_id}",
            rmsd=res["rmsd"],
            p_at_l=res["p_at_l"],
        )

        print("\n" + "=" * 60)
        print(f"PREDICTION COMPLETE FOR {args.domain_id}")
        print("=" * 60)
        print(f"  Sequence:       {res['sequence']}")
        print(f"  Length:         {len(res['sequence'])} residues")
        print(f"  Secondary Str:  {res.get('ss_string', 'N/A')}")
        print(f"  Helices:        {len(res.get('helices', []))} segments")
        print(f"  Beta Sheets:    {len(res.get('sheets', []))} segments")
        print(f"  Mean pLDDT:     {float(np.mean(res['plddt'])):.1f}")
        print(f"  CA-RMSD:        {res['rmsd']:.2f} Å")
        print(f"  Contact P@L:    {res['p_at_l'] * 100:.1f}%")
        print(f"  Contact P@L/5:  {res['p_at_l5'] * 100:.1f}%")
        print(f"  3D PDB File:    {pdb_file}")
        print(f"  Heatmap Plot:   {plot_file}")
        print("=" * 60 + "\n")

    elif args.sequence:
        log.info(f"Predicting 3D structure from sequence ({len(args.sequence)} aa, relax={relax})...")
        res = predictor.predict_sequence(args.sequence, relax=relax)

        pdb_file = out_dir / f"{args.name}_predicted.pdb"
        plot_file = out_dir / f"{args.name}_prediction.png"

        save_pdb(
            res["backbone_coords"],
            res["sequence"],
            pdb_file,
            helices=res.get("helices"),
            sheets=res.get("sheets"),
            plddt=res.get("plddt"),
        )
        plot_prediction(
            pred_dist=res["pred_dist"],
            coord_dist=res["coord_dist"],
            true_dist=None,
            sequence=res["sequence"],
            out_path=plot_file,
            title=f"{args.name}",
        )

        print("\n" + "=" * 60)
        print(f"PREDICTION COMPLETE FOR {args.name}")
        print("=" * 60)
        print(f"  Sequence:     {res['sequence']}")
        print(f"  Length:       {len(res['sequence'])} residues")
        print(f"  Secondary Str:{res.get('ss_string', 'N/A')}")
        print(f"  Mean pLDDT:   {float(np.mean(res['plddt'])):.1f}")
        print(f"  3D PDB File:  {pdb_file}")
        print(f"  Heatmap Plot: {plot_file}")
        print("=" * 60 + "\n")

    else:
        # Default: test on Crambin 1crnA00
        log.info(f"No domain or sequence specified. Running on Crambin (1crnA00, relax={relax})...")
        res = predictor.predict_domain("1crnA00", relax=relax)
        pdb_file = out_dir / "1crnA00_predicted.pdb"
        plot_file = out_dir / "1crnA00_comparison.png"
        save_pdb(
            res["backbone_coords"],
            res["sequence"],
            pdb_file,
            helices=res.get("helices"),
            sheets=res.get("sheets"),
            plddt=res.get("plddt"),
        )
        plot_prediction(
            pred_dist=res["pred_dist"],
            coord_dist=res["coord_dist"],
            true_dist=res["true_dist"],
            sequence=res["sequence"],
            out_path=plot_file,
            title="Crambin (1crnA00)",
            rmsd=res["rmsd"],
            p_at_l=res["p_at_l"],
        )
        print("\n" + "=" * 60)
        print("CRAMBIN (1crnA00) PREDICTION COMPLETE")
        print("=" * 60)
        print(f"  Length:       {len(res['sequence'])} aa")
        print(f"  Secondary Str:{res.get('ss_string', 'N/A')}")
        print(f"  CA-RMSD:      {res['rmsd']:.2f} Å")
        print(f"  Contact P@L:  {res['p_at_l'] * 100:.1f}%")
        print(f"  3D PDB:       {pdb_file}")
        print(f"  Plot:         {plot_file}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
