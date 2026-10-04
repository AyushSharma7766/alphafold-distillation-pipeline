"""
Mini-AlphaFold: PDB Data Parser
================================
Parses PDB files to extract:
  - 1D amino acid sequence (as integer indices)
  - 3D backbone atom coordinates (N, CA, C)
  - Pairwise CA distance matrix [L × L]

Designed for CATH S40 domain PDB files. Handles edge cases:
  - Non-standard residues (MSE → M, etc.)
  - Disordered atoms (takes first altloc)
  - Multi-model files (takes first model, e.g. NMR)
  - Missing backbone atoms (skips incomplete residues)

Usage:
    parser = ProteinParser()
    protein = parser.parse("domain.pdb", domain_id="1a0aA00")
    if protein is not None:
        torch.save(protein.to_dict(), "output.pt")
"""

import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from Bio.PDB import PDBParser as BioPDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Suppress Biopython warnings about discontinuous chains, etc.
warnings.filterwarnings("ignore", category=PDBConstructionWarning)


# ============================================================
# Amino Acid Mappings
# ============================================================

# Standard 3-letter → 1-letter amino acid codes
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

# Non-standard residues → closest standard 1-letter code
NONSTANDARD_MAP: dict[str, str] = {
    "MSE": "M",   # Selenomethionine → Methionine
    "SEC": "C",   # Selenocysteine → Cysteine
    "CSE": "C",   # Selenocysteine variant
    "HSD": "H",   # CHARMM histidine protonation states
    "HSE": "H",
    "HSP": "H",
    "HIE": "H",   # Amber histidine protonation states
    "HID": "H",
    "HIP": "H",
    "CSD": "C",   # 3-sulfinoalanine
    "SEP": "S",   # Phosphoserine
    "TPO": "T",   # Phosphothreonine
    "PTR": "Y",   # Phosphotyrosine
    "MLY": "K",   # N-dimethyl-lysine
    "CSO": "C",   # S-hydroxycysteine
    "CME": "C",   # S,S-(2-hydroxyethyl)thiocysteine
}

# Canonical 1-letter codes → integer index (alphabetical order)
# 20 standard amino acids + unknown/padding token = 21 total
AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: dict[str, int] = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
UNK_IDX: int = len(AMINO_ACIDS)  # Index 20 = unknown/padding
NUM_AMINO_ACIDS: int = len(AMINO_ACIDS) + 1  # 21 (including UNK)

# Backbone atom names in canonical order
BACKBONE_ATOMS: list[str] = ["N", "CA", "C"]
N_BACKBONE_ATOMS: int = len(BACKBONE_ATOMS)  # 3


# ============================================================
# Data Container
# ============================================================

@dataclass
class ProteinData:
    """Container for parsed protein structure data.

    Attributes:
        domain_id:        Unique identifier (e.g., CATH domain ID)
        sequence:         1-letter amino acid sequence string
        sequence_indices: [L] int64 tensor, amino acid indices (0-19, 20=UNK)
        backbone_coords:  [L, 3, 3] float32 tensor, (N/CA/C) × (x, y, z) in Å
        ca_coords:        [L, 3] float32 tensor, CA atom coordinates in Å
        distance_matrix:  [L, L] float32 tensor, pairwise CA distances in Å
        length:           Number of residues L
    """

    domain_id: str
    sequence: str
    sequence_indices: torch.Tensor
    backbone_coords: torch.Tensor
    ca_coords: torch.Tensor
    distance_matrix: torch.Tensor
    length: int

    def to_dict(self) -> dict:
        """Serialize to dictionary for torch.save()."""
        return {
            "domain_id": self.domain_id,
            "sequence": self.sequence,
            "sequence_indices": self.sequence_indices,
            "backbone_coords": self.backbone_coords,
            "ca_coords": self.ca_coords,
            "distance_matrix": self.distance_matrix,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProteinData":
        """Deserialize from a saved dictionary."""
        return cls(**d)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Domain:           {self.domain_id}",
            f"Sequence:         {self.sequence[:50]}{'...' if self.length > 50 else ''}",
            f"Length:           {self.length} residues",
            f"Backbone coords:  {list(self.backbone_coords.shape)}  (L × 3_atoms × 3_xyz)",
            f"CA coords:        {list(self.ca_coords.shape)}",
            f"Distance matrix:  {list(self.distance_matrix.shape)}",
            f"Dist range:       [{self.distance_matrix.min():.2f}, {self.distance_matrix.max():.2f}] Å",
        ]
        return "\n".join(lines)


# ============================================================
# Helper Functions
# ============================================================

def residue_to_one_letter(resname: str) -> Optional[str]:
    """Convert a 3-letter residue name to 1-letter amino acid code.

    Handles standard amino acids and common non-standard variants.
    Returns None for unrecognized residues (water, ligands, ions, etc.)
    """
    resname = resname.strip().upper()
    if resname in THREE_TO_ONE:
        return THREE_TO_ONE[resname]
    if resname in NONSTANDARD_MAP:
        return NONSTANDARD_MAP[resname]
    return None


def sequence_to_indices(sequence: str) -> torch.Tensor:
    """Convert a 1-letter amino acid sequence to an integer index tensor.

    Standard amino acids are mapped to indices 0-19 (alphabetical).
    Unknown / non-standard residues are mapped to UNK_IDX (20).

    Args:
        sequence: String of 1-letter amino acid codes

    Returns:
        [L] int64 tensor of amino acid indices
    """
    indices = [AA_TO_IDX.get(aa, UNK_IDX) for aa in sequence]
    return torch.tensor(indices, dtype=torch.long)


def compute_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distance matrix.

    Args:
        coords: [L, 3] tensor of 3D coordinates (Å)

    Returns:
        [L, L] symmetric distance matrix (Å).
        Diagonal is 0. dist[i][j] = ||coords[i] - coords[j]||
    """
    # cdist expects [batch, N, D] input
    return torch.cdist(
        coords.unsqueeze(0), coords.unsqueeze(0), p=2.0
    ).squeeze(0)


def validate_backbone_geometry(backbone_coords: torch.Tensor) -> bool:
    """Sanity-check backbone coordinates for obvious problems.

    Checks:
      1. No NaN/Inf values
      2. CA-CA distances between consecutive residues are reasonable
         (typical: 3.8 Å ± tolerance; we allow 2.0-5.0 Å)

    Args:
        backbone_coords: [L, 3, 3] backbone atom coordinates

    Returns:
        True if geometry looks reasonable, False otherwise
    """
    if torch.isnan(backbone_coords).any() or torch.isinf(backbone_coords).any():
        return False

    # Check consecutive CA-CA distances
    ca_coords = backbone_coords[:, 1, :]  # [L, 3]
    if ca_coords.shape[0] < 2:
        return False

    ca_dists = torch.norm(ca_coords[1:] - ca_coords[:-1], dim=-1)

    # Typical CA-CA distance is ~3.8 Å
    # Allow 2.0–6.0 Å to account for some flexibility and missing residues
    if (ca_dists < 2.0).any() or (ca_dists > 6.0).any():
        return False

    return True


# ============================================================
# Main Parser Class
# ============================================================

class ProteinParser:
    """Parse PDB files and extract backbone structure data.

    Extracts N, CA, C backbone atoms for each residue, computes
    pairwise CA distance matrices, and encodes sequences as integer tensors.

    Args:
        quiet: Suppress Biopython parser warnings (default: True)
        validate_geometry: Run geometry validation checks (default: True)

    Example:
        >>> parser = ProteinParser()
        >>> protein = parser.parse("1a0aA00.pdb", domain_id="1a0aA00")
        >>> if protein is not None:
        ...     print(protein.summary())
        ...     torch.save(protein.to_dict(), "1a0aA00.pt")
    """

    def __init__(self, quiet: bool = True, validate_geometry: bool = True):
        self._parser = BioPDBParser(QUIET=quiet)
        self._validate = validate_geometry

    def parse(
        self,
        pdb_path: Union[str, Path],
        domain_id: str = "",
        min_length: int = 30,
        max_length: int = 200,
    ) -> Optional[ProteinData]:
        """Parse a PDB file and extract backbone data.

        Args:
            pdb_path:   Path to the .pdb file
            domain_id:  Identifier for this domain (default: filename stem)
            min_length: Reject proteins shorter than this (default: 30)
            max_length: Reject proteins longer than this (default: 200)

        Returns:
            ProteinData if parsing succeeds and length is within bounds.
            None if the file is invalid, cannot be parsed, or is outside
            the length range.
        """
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            return None

        if not domain_id:
            domain_id = pdb_path.stem

        # --- Parse PDB structure ---
        try:
            structure = self._parser.get_structure(domain_id, str(pdb_path))
        except Exception:
            return None

        # Take the first model (handles NMR multi-model files)
        try:
            model = structure[0]
        except (KeyError, IndexError):
            return None

        # --- Extract residues with complete backbone ---
        sequence_letters: list[str] = []
        backbone_coords_list: list[list[np.ndarray]] = []

        for chain in model:
            for residue in chain:
                # Identify residue type from het flag
                het_flag = residue.get_id()[0]

                # Standard residue: het_flag == ' '
                # Modified residue: het_flag == 'H_XXX' (e.g., H_MSE)
                # Water: het_flag == 'W'
                if het_flag == "W":
                    continue

                if het_flag != " ":
                    # It's a HETATM. Check if it's a known modified residue.
                    if het_flag.startswith("H_"):
                        modified_name = het_flag[2:]
                        if modified_name not in NONSTANDARD_MAP:
                            continue  # Unknown HETATM, skip
                    else:
                        continue  # Non-residue HETATM (ligand, ion, etc.)

                # Convert residue name to 1-letter code
                resname = residue.get_resname()
                one_letter = residue_to_one_letter(resname)
                if one_letter is None:
                    continue  # Unrecognized residue type

                # Extract backbone atom coordinates: N, CA, C
                atom_coords: list[np.ndarray] = []
                complete = True

                for atom_name in BACKBONE_ATOMS:
                    if atom_name not in residue:
                        complete = False
                        break

                    atom = residue[atom_name]

                    # Handle disordered atoms: take first alternate location
                    if atom.is_disordered():
                        atom = atom.disordered_get_list()[0]

                    atom_coords.append(atom.get_vector().get_array().copy())

                if not complete:
                    continue  # Skip residues with missing backbone atoms

                sequence_letters.append(one_letter)
                backbone_coords_list.append(atom_coords)

        # --- Validate length ---
        length = len(sequence_letters)
        if length < min_length or length > max_length:
            return None

        # --- Build tensors ---
        sequence = "".join(sequence_letters)
        seq_indices = sequence_to_indices(sequence)

        # backbone_coords: [L, 3, 3] — (residue, atom_type[N/CA/C], xyz)
        backbone_coords = torch.tensor(
            np.array(backbone_coords_list, dtype=np.float32),
            dtype=torch.float32,
        )

        # --- Geometry validation ---
        if self._validate and not validate_backbone_geometry(backbone_coords):
            return None

        # CA coordinates: [L, 3] — index 1 in atom dimension = CA
        ca_coords = backbone_coords[:, 1, :].contiguous()

        # Pairwise CA distance matrix: [L, L]
        dist_matrix = compute_distance_matrix(ca_coords)

        return ProteinData(
            domain_id=domain_id,
            sequence=sequence,
            sequence_indices=seq_indices,
            backbone_coords=backbone_coords,
            ca_coords=ca_coords,
            distance_matrix=dist_matrix,
            length=length,
        )


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_parser.py <pdb_file> [domain_id]")
        print("Example: python data_parser.py data/raw/1a0aA00.pdb 1a0aA00")
        sys.exit(1)

    pdb_file = sys.argv[1]
    did = sys.argv[2] if len(sys.argv) > 2 else ""

    parser = ProteinParser()
    result = parser.parse(pdb_file, domain_id=did)

    if result is None:
        print(f"[FAIL] Failed to parse: {pdb_file}")
        sys.exit(1)

    print(f"[OK] Successfully parsed: {pdb_file}\n")
    print(result.summary())

    print(f"\nFirst 5 CA positions (Å):")
    n_show = min(5, result.length)
    for i in range(n_show):
        x, y, z = result.ca_coords[i].tolist()
        print(f"  {result.sequence[i]}{i+1:>4d}:  ({x:8.3f}, {y:8.3f}, {z:8.3f})")

    print(f"\nDistance matrix sample (top-left 5×5):")
    n_show = min(5, result.length)
    dm = result.distance_matrix[:n_show, :n_show]
    for i in range(n_show):
        row = "  ".join(f"{dm[i, j]:6.2f}" for j in range(n_show))
        print(f"  {row}")
