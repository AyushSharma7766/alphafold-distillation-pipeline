import json
from pathlib import Path
import numpy as np

pdb_lines = Path("data/AF-A0A1B0GTW7-F1.pdb").read_text(encoding="utf-8").splitlines()

# 20 Canonical Amino Acids Metadata
METADATA = {
    "ALA": {
        "code1": "A", "code3": "ALA", "name": "Alanine", "category": "Hydrophobic",
        "formula": "C3H7NO2", "mw": 89.09, "pI": 6.00, "hydropathy": 1.8, "charge": "Neutral",
        "codons": ["GCU", "GCC", "GCA", "GCG"],
        "sidechain": "Methyl group (-CH3)",
        "role": "Small, non-polar and hydrophobic. Highly versatile, commonly found in alpha-helices and beta-sheets. Used in alanine-scanning mutagenesis to evaluate functional contributions of sidechains."
    },
    "ARG": {
        "code1": "R", "code3": "ARG", "name": "Arginine", "category": "Positively Charged (Basic)",
        "formula": "C6H14N4O2", "mw": 174.20, "pI": 10.76, "hydropathy": -4.5, "charge": "Positive (+1)",
        "codons": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],
        "sidechain": "Guanidinium group (-CH2-CH2-CH2-NH-C(=NH2+)-NH2)",
        "role": "Strongly basic with a positively charged guanidinium group across physiological pH. Forms critical salt bridges with Asp/Glu and binds negatively charged phosphate backbones in DNA/RNA."
    },
    "ASN": {
        "code1": "N", "code3": "ASN", "name": "Asparagine", "category": "Polar / Uncharged",
        "formula": "C4H8N2O3", "mw": 132.12, "pI": 5.41, "hydropathy": -3.5, "charge": "Neutral",
        "codons": ["AAU", "AAC"],
        "sidechain": "Carboxamide (-CH2-CO-NH2)",
        "role": "Amide derivative of aspartate. Major site for N-linked glycosylation (Asn-X-Ser/Thr consensus motif). Frequent in turns and loops forming stabilizing sidechain-backbone hydrogen bonds."
    },
    "ASP": {
        "code1": "D", "code3": "ASP", "name": "Aspartate", "category": "Negatively Charged (Acidic)",
        "formula": "C4H7NO4", "mw": 133.10, "pI": 2.77, "hydropathy": -3.5, "charge": "Negative (-1)",
        "codons": ["GAU", "GAC"],
        "sidechain": "Carboxylate (-CH2-COO-)",
        "role": "Acidic and negatively charged at pH 7. Frequently coordinates catalytic divalent metal ions (Mg2+, Zn2+, Ca2+) and forms electrostatic salt bridges stabilizing tertiary folds."
    },
    "CYS": {
        "code1": "C", "code3": "CYS", "name": "Cysteine", "category": "Polar / Uncharged",
        "formula": "C3H7NO2S", "mw": 121.16, "pI": 5.07, "hydropathy": 2.5, "charge": "Neutral",
        "codons": ["UGU", "UGC"],
        "sidechain": "Thiol / Sulfhydryl (-CH2-SH)",
        "role": "Forms covalent disulfide bridges (-S-S-) with other cysteines, locking together protein tertiary and quaternary architectures. Coordinates zinc and other metal centers in catalytic domains."
    },
    "GLN": {
        "code1": "Q", "code3": "GLN", "name": "Glutamine", "category": "Polar / Uncharged",
        "formula": "C5H10N2O3", "mw": 146.15, "pI": 5.65, "hydropathy": -3.5, "charge": "Neutral",
        "codons": ["CAA", "CAG"],
        "sidechain": "Carboxamide (-CH2-CH2-CO-NH2)",
        "role": "Amide derivative of glutamate. Acts as an excellent hydrogen-bond donor and acceptor. Often exposed on protein surfaces mediating protein-protein interactions."
    },
    "GLU": {
        "code1": "E", "code3": "GLU", "name": "Glutamate", "category": "Negatively Charged (Acidic)",
        "formula": "C5H9NO4", "mw": 147.13, "pI": 3.22, "hydropathy": -3.5, "charge": "Negative (-1)",
        "codons": ["GAA", "GAG"],
        "sidechain": "Carboxylate (-CH2-CH2-COO-)",
        "role": "Catalytic nucleophile in HEXXH zinc metallopeptidases (like CIROP Glu 306). Forms salt bridges and coordinates catalytic metal ions."
    },
    "GLY": {
        "code1": "G", "code3": "GLY", "name": "Glycine", "category": "Special / Flexible",
        "formula": "C2H5NO2", "mw": 75.07, "pI": 5.97, "hydropathy": -0.4, "charge": "Neutral",
        "codons": ["GGU", "GGC", "GGA", "GGG"],
        "sidechain": "Single hydrogen atom (-H)",
        "role": "Achiral and the smallest amino acid. Imparts maximum conformational flexibility to polypeptide chains. Abundant in tight turns, hinges, and fibrous collagen helices."
    },
    "HIS": {
        "code1": "H", "code3": "HIS", "name": "Histidine", "category": "Positively Charged (Basic)",
        "formula": "C6H9N3O2", "mw": 155.16, "pI": 7.59, "hydropathy": -3.2, "charge": "Partial Positive",
        "codons": ["CAU", "CAC"],
        "sidechain": "Imidazole ring (-CH2-C3H3N2)",
        "role": "Near-neutral pKa (~6.0) allows it to switch between protonated and deprotonated states at physiological pH. Critical for enzymatic acid-base catalysis and coordination of Zn2+ (CIROP His 305/309) and Fe2+."
    },
    "ILE": {
        "code1": "I", "code3": "ILE", "name": "Isoleucine", "category": "Hydrophobic",
        "formula": "C6H13NO2", "mw": 131.17, "pI": 6.02, "hydropathy": 4.5, "charge": "Neutral",
        "codons": ["AUU", "AUC", "AUA"],
        "sidechain": "sec-Butyl group (-CH(CH3)-CH2-CH3)",
        "role": "Branched-chain hydrophobic amino acid with a chiral beta-carbon. Strongly stabilizes beta-sheets and hydrophobic interior cores of globular proteins."
    },
    "LEU": {
        "code1": "L", "code3": "LEU", "name": "Leucine", "category": "Hydrophobic",
        "formula": "C6H13NO2", "mw": 131.17, "pI": 5.98, "hydropathy": 3.8, "charge": "Neutral",
        "codons": ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"],
        "sidechain": "Isobutyl group (-CH2-CH(CH3)2)",
        "role": "Most abundant amino acid in proteins. High alpha-helix propensity (such as CIROP residue Leu 222). Drives hydrophobic collapse during protein folding and forms leucine zippers in transcription factors."
    },
    "LYS": {
        "code1": "K", "code3": "LYS", "name": "Lysine", "category": "Positively Charged (Basic)",
        "formula": "C6H14N2O2", "mw": 146.19, "pI": 9.74, "hydropathy": -3.9, "charge": "Positive (+1)",
        "codons": ["AAA", "AAG"],
        "sidechain": "Butylamine (-CH2-CH2-CH2-CH2-NH3+)",
        "role": "Flexible positively charged basic sidechain. Target of widespread post-translational modifications including acetylation, methylation, and ubiquitination. Forms key surface salt bridges."
    },
    "MET": {
        "code1": "M", "code3": "MET", "name": "Methionine", "category": "Hydrophobic",
        "formula": "C5H11NO2S", "mw": 149.21, "pI": 5.74, "hydropathy": 1.9, "charge": "Neutral",
        "codons": ["AUG"],
        "sidechain": "Thioether (-CH2-CH2-S-CH3)",
        "role": "The universal initiator amino acid encoded by the AUG start codon. Hydrophobic and non-reactive under physiological conditions; can act as an antioxidant buffer through methionine sulfoxide formation."
    },
    "PHE": {
        "code1": "F", "code3": "PHE", "name": "Phenylalanine", "category": "Hydrophobic (Aromatic)",
        "formula": "C9H11NO2", "mw": 165.19, "pI": 5.48, "hydropathy": 2.8, "charge": "Neutral",
        "codons": ["UUU", "UUC"],
        "sidechain": "Benzyl / Phenyl ring (-CH2-C6H5)",
        "role": "Bulky aromatic amino acid. Strongly hydrophobic, packing tightly into the central cores of globular domains via aromatic pi-pi and pi-cation interactions."
    },
    "PRO": {
        "code1": "P", "code3": "PRO", "name": "Proline", "category": "Special / Cyclic",
        "formula": "C5H9NO2", "mw": 115.13, "pI": 6.30, "hydropathy": -1.6, "charge": "Neutral",
        "codons": ["CCU", "CCC", "CCA", "CCG"],
        "sidechain": "Pyrrolidine ring (-CH2-CH2-CH2- bonded to backbone N)",
        "role": "Only cyclic imino acid. Constrained backbone dihedral angles (phi ~ -65 deg). Serves as a classic alpha-helix and beta-sheet breaker, introducing rigid kinks and turns into polypeptide folds."
    },
    "SER": {
        "code1": "S", "code3": "SER", "name": "Serine", "category": "Polar / Uncharged",
        "formula": "C3H7NO3", "mw": 105.09, "pI": 5.68, "hydropathy": -0.8, "charge": "Neutral",
        "codons": ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"],
        "sidechain": "Hydroxymethyl (-CH2-OH)",
        "role": "Polar uncharged residue with a nucleophilic hydroxyl group. Major site of regulatory phosphorylation by Ser/Thr kinases. Catalytic nucleophile in serine proteases (chymotrypsin, trypsin)."
    },
    "THR": {
        "code1": "T", "code3": "THR", "name": "Threonine", "category": "Polar / Uncharged",
        "formula": "C4H9NO3", "mw": 119.12, "pI": 5.60, "hydropathy": -0.7, "charge": "Neutral",
        "codons": ["ACU", "ACC", "ACA", "ACG"],
        "sidechain": "Hydroxyethyl (-CH(OH)-CH3)",
        "role": "Branched beta-carbon containing a secondary hydroxyl group. Prominent site for O-linked glycosylation and reversible phosphorylation in cellular signaling cascades."
    },
    "TRP": {
        "code1": "W", "code3": "TRP", "name": "Tryptophan", "category": "Hydrophobic (Aromatic)",
        "formula": "C11H12N2O2", "mw": 204.23, "pI": 5.89, "hydropathy": -0.9, "charge": "Neutral",
        "codons": ["UGG"],
        "sidechain": "Indole ring (-CH2-C8H6N)",
        "role": "Largest amino acid with a bicyclic indole ring. Dominates intrinsic protein UV absorbance (280 nm) and fluorescence. Anchors transmembrane proteins at water-lipid bilayer interfaces."
    },
    "TYR": {
        "code1": "Y", "code3": "TYR", "name": "Tyrosine", "category": "Polar / Aromatic",
        "formula": "C9H11NO3", "mw": 181.19, "pI": 5.66, "hydropathy": -1.3, "charge": "Neutral",
        "codons": ["UAU", "UAC"],
        "sidechain": "Phenol ring (-CH2-C6H4-OH)",
        "role": "Aromatic amino acid with an ionizable phenolic hydroxyl group (pKa ~ 10.1). Primary target of receptor tyrosine kinase (RTK) phosphorylation. Facilitates electron transfer in enzymes."
    },
    "VAL": {
        "code1": "V", "code3": "VAL", "name": "Valine", "category": "Hydrophobic",
        "formula": "C5H11NO2", "mw": 117.15, "pI": 5.96, "hydropathy": 4.2, "charge": "Neutral",
        "codons": ["GUU", "GUC", "GUA", "GUG"],
        "sidechain": "Isopropyl group (-CH(CH3)2)",
        "role": "Branched-chain aliphatic hydrophobic residue. High beta-sheet propensity. Critical component of hydrophobic packing in internal protein cores."
    },
}

# Find first occurrence of each residue in CIROP and extract all its atoms
found_resis = {}
for line in pdb_lines:
    if line.startswith("ATOM"):
        resn = line[17:20].strip()
        resi = int(line[22:26].strip())
        if resn not in found_resis:
            found_resis[resn] = resi

amino_acids_db = []

for code3, meta in METADATA.items():
    target_resi = found_resis[code3]
    atoms = []
    coords = []
    for line in pdb_lines:
        if line.startswith("ATOM") and int(line[22:26].strip()) == target_resi:
            atom_name = line[12:16].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            elem = line[76:78].strip() if len(line) >= 78 and line[76:78].strip() else atom_name[0]
            atoms.append((atom_name, elem))
            coords.append([x, y, z])
    
    # Center coordinates at (0, 0, 0)
    arr = np.array(coords)
    center = np.mean(arr, axis=0)
    centered = arr - center
    
    # Build clean standalone PDB string
    pdb_out = []
    for idx, ((atom_name, elem), (cx, cy, cz)) in enumerate(zip(atoms, centered), start=1):
        line = f"ATOM  {idx:5d} {atom_name:^4s} {code3:3s} A   1    {cx:8.3f}{cy:8.3f}{cz:8.3f}  1.00 20.00          {elem:>2s}"
        pdb_out.append(line)
    pdb_out.append("END")
    pdb_str = "\n".join(pdb_out)
    
    entry = dict(meta)
    entry["pdb"] = pdb_str
    amino_acids_db.append(entry)

out_py = Path("src/amino_acids_db.py")
out_py.write_text(
    f'"""\n20 Canonical Amino Acids Database\nIncludes chemical properties, formulas, pI, hydropathy, roles, and 3D PDB coordinates centered at origin.\n"""\n\nAMINO_ACIDS = {json.dumps(amino_acids_db, indent=2)}\n',
    encoding="utf-8"
)
print(f"Successfully generated {out_py} with {len(amino_acids_db)} amino acids!")
