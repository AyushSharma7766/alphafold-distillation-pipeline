import json
from pathlib import Path

pdb_text = Path("data/AF-A0A1B0GTW7-F1.pdb").read_text(encoding="utf-8")

# Find first occurrence of each amino acid
residues = {}
for line in pdb_text.splitlines():
    if line.startswith("ATOM"):
        resn = line[17:20].strip()
        resi = int(line[22:26].strip())
        if resn not in residues:
            residues[resn] = resi

print(f"Found {len(residues)} amino acids in CIROP:", sorted(residues.keys()))
