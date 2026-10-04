"""
20 Canonical Amino Acids Database
Includes chemical properties, formulas, pI, hydropathy, roles, and 3D PDB coordinates centered at origin.
"""

AMINO_ACIDS = [
  {
    "code1": "A",
    "code3": "ALA",
    "name": "Alanine",
    "category": "Hydrophobic",
    "formula": "C3H7NO2",
    "mw": 89.09,
    "pI": 6.0,
    "hydropathy": 1.8,
    "charge": "Neutral",
    "codons": [
      "GCU",
      "GCC",
      "GCA",
      "GCG"
    ],
    "sidechain": "Methyl group (-CH3)",
    "role": "Small, non-polar and hydrophobic. Highly versatile, commonly found in alpha-helices and beta-sheets. Used in alanine-scanning mutagenesis to evaluate functional contributions of sidechains.",
    "pdb": "ATOM      1  N   ALA A   1       0.682   0.436  -1.456  1.00 20.00           N\nATOM      2  CA  ALA A   1      -0.408   0.173  -0.541  1.00 20.00           C\nATOM      3  C   ALA A   1      -0.024   0.733   0.832  1.00 20.00           C\nATOM      4  CB  ALA A   1      -0.679  -1.334  -0.537  1.00 20.00           C\nATOM      5  O   ALA A   1       0.429  -0.006   1.704  1.00 20.00           O\nEND"
  },
  {
    "code1": "R",
    "code3": "ARG",
    "name": "Arginine",
    "category": "Positively Charged (Basic)",
    "formula": "C6H14N4O2",
    "mw": 174.2,
    "pI": 10.76,
    "hydropathy": -4.5,
    "charge": "Positive (+1)",
    "codons": [
      "CGU",
      "CGC",
      "CGA",
      "CGG",
      "AGA",
      "AGG"
    ],
    "sidechain": "Guanidinium group (-CH2-CH2-CH2-NH-C(=NH2+)-NH2)",
    "role": "Strongly basic with a positively charged guanidinium group across physiological pH. Forms critical salt bridges with Asp/Glu and binds negatively charged phosphate backbones in DNA/RNA.",
    "pdb": "ATOM      1  N   ARG A   1       2.776  -2.059  -0.675  1.00 20.00           N\nATOM      2  CA  ARG A   1       2.517  -1.059   0.359  1.00 20.00           C\nATOM      3  C   ARG A   1       3.176  -1.589   1.620  1.00 20.00           C\nATOM      4  CB  ARG A   1       1.011  -0.824   0.589  1.00 20.00           C\nATOM      5  O   ARG A   1       2.932  -2.734   1.999  1.00 20.00           O\nATOM      6  CG  ARG A   1       0.358   0.075  -0.473  1.00 20.00           C\nATOM      7  CD  ARG A   1      -1.091   0.404  -0.075  1.00 20.00           C\nATOM      8  NE  ARG A   1      -1.712   1.398  -0.978  1.00 20.00           N\nATOM      9 NH1  ARG A   1      -3.720   1.592   0.130  1.00 20.00           N\nATOM     10 NH2  ARG A   1      -3.332   2.857  -1.660  1.00 20.00           N\nATOM     11  CZ  ARG A   1      -2.912   1.940  -0.833  1.00 20.00           C\nEND"
  },
  {
    "code1": "N",
    "code3": "ASN",
    "name": "Asparagine",
    "category": "Polar / Uncharged",
    "formula": "C4H8N2O3",
    "mw": 132.12,
    "pI": 5.41,
    "hydropathy": -3.5,
    "charge": "Neutral",
    "codons": [
      "AAU",
      "AAC"
    ],
    "sidechain": "Carboxamide (-CH2-CO-NH2)",
    "role": "Amide derivative of aspartate. Major site for N-linked glycosylation (Asn-X-Ser/Thr consensus motif). Frequent in turns and loops forming stabilizing sidechain-backbone hydrogen bonds.",
    "pdb": "ATOM      1  N   ASN A   1      -0.441   1.729   0.746  1.00 20.00           N\nATOM      2  CA  ASN A   1       0.167   0.909  -0.293  1.00 20.00           C\nATOM      3  C   ASN A   1       0.083   1.521  -1.703  1.00 20.00           C\nATOM      4  CB  ASN A   1      -0.430  -0.507  -0.214  1.00 20.00           C\nATOM      5  O   ASN A   1       0.461   0.835  -2.648  1.00 20.00           O\nATOM      6  CG  ASN A   1      -0.031  -1.256   1.039  1.00 20.00           C\nATOM      7 ND2  ASN A   1      -0.855  -2.173   1.483  1.00 20.00           N\nATOM      8 OD1  ASN A   1       1.046  -1.060   1.593  1.00 20.00           O\nEND"
  },
  {
    "code1": "D",
    "code3": "ASP",
    "name": "Aspartate",
    "category": "Negatively Charged (Acidic)",
    "formula": "C4H7NO4",
    "mw": 133.1,
    "pI": 2.77,
    "hydropathy": -3.5,
    "charge": "Negative (-1)",
    "codons": [
      "GAU",
      "GAC"
    ],
    "sidechain": "Carboxylate (-CH2-COO-)",
    "role": "Acidic and negatively charged at pH 7. Frequently coordinates catalytic divalent metal ions (Mg2+, Zn2+, Ca2+) and forms electrostatic salt bridges stabilizing tertiary folds.",
    "pdb": "ATOM      1  N   ASP A   1       1.151  -1.479   0.082  1.00 20.00           N\nATOM      2  CA  ASP A   1      -0.149  -0.800   0.134  1.00 20.00           C\nATOM      3  C   ASP A   1      -1.326  -1.759  -0.129  1.00 20.00           C\nATOM      4  CB  ASP A   1      -0.133   0.416  -0.807  1.00 20.00           C\nATOM      5  O   ASP A   1      -2.261  -1.828   0.675  1.00 20.00           O\nATOM      6  CG  ASP A   1       0.681   1.553  -0.183  1.00 20.00           C\nATOM      7 OD1  ASP A   1       0.252   2.015   0.901  1.00 20.00           O\nATOM      8 OD2  ASP A   1       1.783   1.882  -0.672  1.00 20.00           O\nEND"
  },
  {
    "code1": "C",
    "code3": "CYS",
    "name": "Cysteine",
    "category": "Polar / Uncharged",
    "formula": "C3H7NO2S",
    "mw": 121.16,
    "pI": 5.07,
    "hydropathy": 2.5,
    "charge": "Neutral",
    "codons": [
      "UGU",
      "UGC"
    ],
    "sidechain": "Thiol / Sulfhydryl (-CH2-SH)",
    "role": "Forms covalent disulfide bridges (-S-S-) with other cysteines, locking together protein tertiary and quaternary architectures. Coordinates zinc and other metal centers in catalytic domains.",
    "pdb": "ATOM      1  N   CYS A   1       1.181  -0.102  -1.421  1.00 20.00           N\nATOM      2  CA  CYS A   1       0.529  -0.356  -0.141  1.00 20.00           C\nATOM      3  C   CYS A   1      -0.961  -0.709  -0.327  1.00 20.00           C\nATOM      4  CB  CYS A   1       0.738   0.873   0.743  1.00 20.00           C\nATOM      5  O   CYS A   1      -1.609  -0.235  -1.260  1.00 20.00           O\nATOM      6  SG  CYS A   1       0.121   0.527   2.408  1.00 20.00           S\nEND"
  },
  {
    "code1": "Q",
    "code3": "GLN",
    "name": "Glutamine",
    "category": "Polar / Uncharged",
    "formula": "C5H10N2O3",
    "mw": 146.15,
    "pI": 5.65,
    "hydropathy": -3.5,
    "charge": "Neutral",
    "codons": [
      "CAA",
      "CAG"
    ],
    "sidechain": "Carboxamide (-CH2-CH2-CO-NH2)",
    "role": "Amide derivative of glutamate. Acts as an excellent hydrogen-bond donor and acceptor. Often exposed on protein surfaces mediating protein-protein interactions.",
    "pdb": "ATOM      1  N   GLN A   1       0.372  -1.233  -2.339  1.00 20.00           N\nATOM      2  CA  GLN A   1      -0.160  -0.563  -1.166  1.00 20.00           C\nATOM      3  C   GLN A   1      -1.670  -0.324  -1.303  1.00 20.00           C\nATOM      4  CB  GLN A   1       0.635   0.735  -0.952  1.00 20.00           C\nATOM      5  O   GLN A   1      -2.387  -0.456  -0.315  1.00 20.00           O\nATOM      6  CG  GLN A   1       0.426   1.366   0.432  1.00 20.00           C\nATOM      7  CD  GLN A   1       0.848   0.440   1.565  1.00 20.00           C\nATOM      8 NE2  GLN A   1      -0.082  -0.213   2.230  1.00 20.00           N\nATOM      9 OE1  GLN A   1       2.019   0.252   1.851  1.00 20.00           O\nEND"
  },
  {
    "code1": "E",
    "code3": "GLU",
    "name": "Glutamate",
    "category": "Negatively Charged (Acidic)",
    "formula": "C5H9NO4",
    "mw": 147.13,
    "pI": 3.22,
    "hydropathy": -3.5,
    "charge": "Negative (-1)",
    "codons": [
      "GAA",
      "GAG"
    ],
    "sidechain": "Carboxylate (-CH2-CH2-COO-)",
    "role": "Catalytic nucleophile in HEXXH zinc metallopeptidases (like CIROP Glu 306). Forms salt bridges and coordinates catalytic metal ions.",
    "pdb": "ATOM      1  N   GLU A   1       0.568   1.854   1.665  1.00 20.00           N\nATOM      2  CA  GLU A   1      -0.381   0.749   1.426  1.00 20.00           C\nATOM      3  C   GLU A   1      -0.461  -0.240   2.605  1.00 20.00           C\nATOM      4  CB  GLU A   1       0.062  -0.088   0.218  1.00 20.00           C\nATOM      5  O   GLU A   1      -1.510  -0.819   2.902  1.00 20.00           O\nATOM      6  CG  GLU A   1       0.080   0.594  -1.159  1.00 20.00           C\nATOM      7  CD  GLU A   1       0.458  -0.415  -2.269  1.00 20.00           C\nATOM      8 OE1  GLU A   1       0.481  -0.031  -3.459  1.00 20.00           O\nATOM      9 OE2  GLU A   1       0.699  -1.603  -1.930  1.00 20.00           O\nEND"
  },
  {
    "code1": "G",
    "code3": "GLY",
    "name": "Glycine",
    "category": "Special / Flexible",
    "formula": "C2H5NO2",
    "mw": 75.07,
    "pI": 5.97,
    "hydropathy": -0.4,
    "charge": "Neutral",
    "codons": [
      "GGU",
      "GGC",
      "GGA",
      "GGG"
    ],
    "sidechain": "Single hydrogen atom (-H)",
    "role": "Achiral and the smallest amino acid. Imparts maximum conformational flexibility to polypeptide chains. Abundant in tight turns, hinges, and fibrous collagen helices.",
    "pdb": "ATOM      1  N   GLY A   1       0.809   1.159   0.590  1.00 20.00           N\nATOM      2  CA  GLY A   1       0.856   0.145  -0.452  1.00 20.00           C\nATOM      3  C   GLY A   1      -0.533  -0.469  -0.577  1.00 20.00           C\nATOM      4  O   GLY A   1      -1.133  -0.834   0.438  1.00 20.00           O\nEND"
  },
  {
    "code1": "H",
    "code3": "HIS",
    "name": "Histidine",
    "category": "Positively Charged (Basic)",
    "formula": "C6H9N3O2",
    "mw": 155.16,
    "pI": 7.59,
    "hydropathy": -3.2,
    "charge": "Partial Positive",
    "codons": [
      "CAU",
      "CAC"
    ],
    "sidechain": "Imidazole ring (-CH2-C3H3N2)",
    "role": "Near-neutral pKa (~6.0) allows it to switch between protonated and deprotonated states at physiological pH. Critical for enzymatic acid-base catalysis and coordination of Zn2+ (CIROP His 305/309) and Fe2+.",
    "pdb": "ATOM      1  N   HIS A   1       0.644  -1.344  -2.880  1.00 20.00           N\nATOM      2  CA  HIS A   1       0.227  -0.978  -1.527  1.00 20.00           C\nATOM      3  C   HIS A   1      -1.188  -0.393  -1.494  1.00 20.00           C\nATOM      4  CB  HIS A   1       1.223  -0.019  -0.867  1.00 20.00           C\nATOM      5  O   HIS A   1      -2.008  -0.886  -0.724  1.00 20.00           O\nATOM      6  CG  HIS A   1       0.723   0.425   0.483  1.00 20.00           C\nATOM      7 CD2  HIS A   1       0.168   1.637   0.782  1.00 20.00           C\nATOM      8 ND1  HIS A   1       0.567  -0.375   1.589  1.00 20.00           N\nATOM      9 CE1  HIS A   1      -0.037   0.349   2.544  1.00 20.00           C\nATOM     10 NE2  HIS A   1      -0.320   1.582   2.095  1.00 20.00           N\nEND"
  },
  {
    "code1": "I",
    "code3": "ILE",
    "name": "Isoleucine",
    "category": "Hydrophobic",
    "formula": "C6H13NO2",
    "mw": 131.17,
    "pI": 6.02,
    "hydropathy": 4.5,
    "charge": "Neutral",
    "codons": [
      "AUU",
      "AUC",
      "AUA"
    ],
    "sidechain": "sec-Butyl group (-CH(CH3)-CH2-CH3)",
    "role": "Branched-chain hydrophobic amino acid with a chiral beta-carbon. Strongly stabilizes beta-sheets and hydrophobic interior cores of globular proteins.",
    "pdb": "ATOM      1  N   ILE A   1       0.482   0.505   1.917  1.00 20.00           N\nATOM      2  CA  ILE A   1      -0.313  -0.236   0.932  1.00 20.00           C\nATOM      3  C   ILE A   1      -0.054  -1.733   1.111  1.00 20.00           C\nATOM      4  CB  ILE A   1      -0.024   0.228  -0.515  1.00 20.00           C\nATOM      5  O   ILE A   1       1.010  -2.230   0.733  1.00 20.00           O\nATOM      6 CG1  ILE A   1      -0.279   1.746  -0.659  1.00 20.00           C\nATOM      7 CG2  ILE A   1      -0.896  -0.599  -1.485  1.00 20.00           C\nATOM      8 CD1  ILE A   1       0.076   2.322  -2.032  1.00 20.00           C\nEND"
  },
  {
    "code1": "L",
    "code3": "LEU",
    "name": "Leucine",
    "category": "Hydrophobic",
    "formula": "C6H13NO2",
    "mw": 131.17,
    "pI": 5.98,
    "hydropathy": 3.8,
    "charge": "Neutral",
    "codons": [
      "UUA",
      "UUG",
      "CUU",
      "CUC",
      "CUA",
      "CUG"
    ],
    "sidechain": "Isobutyl group (-CH2-CH(CH3)2)",
    "role": "Most abundant amino acid in proteins. High alpha-helix propensity (such as CIROP residue Leu 222). Drives hydrophobic collapse during protein folding and forms leucine zippers in transcription factors.",
    "pdb": "ATOM      1  N   LEU A   1       1.289   1.832  -1.170  1.00 20.00           N\nATOM      2  CA  LEU A   1       0.730   0.882  -0.183  1.00 20.00           C\nATOM      3  C   LEU A   1       1.792   0.181   0.697  1.00 20.00           C\nATOM      4  CB  LEU A   1      -0.160  -0.143  -0.916  1.00 20.00           C\nATOM      5  O   LEU A   1       1.623   0.109   1.907  1.00 20.00           O\nATOM      6  CG  LEU A   1      -1.180  -0.842   0.009  1.00 20.00           C\nATOM      7 CD1  LEU A   1      -2.363   0.078   0.324  1.00 20.00           C\nATOM      8 CD2  LEU A   1      -1.729  -2.096  -0.672  1.00 20.00           C\nEND"
  },
  {
    "code1": "K",
    "code3": "LYS",
    "name": "Lysine",
    "category": "Positively Charged (Basic)",
    "formula": "C6H14N2O2",
    "mw": 146.19,
    "pI": 9.74,
    "hydropathy": -3.9,
    "charge": "Positive (+1)",
    "codons": [
      "AAA",
      "AAG"
    ],
    "sidechain": "Butylamine (-CH2-CH2-CH2-CH2-NH3+)",
    "role": "Flexible positively charged basic sidechain. Target of widespread post-translational modifications including acetylation, methylation, and ubiquitination. Forms key surface salt bridges.",
    "pdb": "ATOM      1  N   LYS A   1       1.262  -1.113   1.822  1.00 20.00           N\nATOM      2  CA  LYS A   1      -0.188  -1.064   1.517  1.00 20.00           C\nATOM      3  C   LYS A   1      -0.884  -2.417   1.793  1.00 20.00           C\nATOM      4  CB  LYS A   1      -0.399  -0.589   0.051  1.00 20.00           C\nATOM      5  O   LYS A   1      -2.062  -2.434   2.133  1.00 20.00           O\nATOM      6  CG  LYS A   1      -0.079   0.909  -0.230  1.00 20.00           C\nATOM      7  CD  LYS A   1       0.140   1.250  -1.736  1.00 20.00           C\nATOM      8  CE  LYS A   1       0.834   2.623  -1.981  1.00 20.00           C\nATOM      9  NZ  LYS A   1       1.373   2.836  -3.373  1.00 20.00           N\nEND"
  },
  {
    "code1": "M",
    "code3": "MET",
    "name": "Methionine",
    "category": "Hydrophobic",
    "formula": "C5H11NO2S",
    "mw": 149.21,
    "pI": 5.74,
    "hydropathy": 1.9,
    "charge": "Neutral",
    "codons": [
      "AUG"
    ],
    "sidechain": "Thioether (-CH2-CH2-S-CH3)",
    "role": "The universal initiator amino acid encoded by the AUG start codon. Hydrophobic and non-reactive under physiological conditions; can act as an antioxidant buffer through methionine sulfoxide formation.",
    "pdb": "ATOM      1  N   MET A   1       1.456  -1.484  -1.091  1.00 20.00           N\nATOM      2  CA  MET A   1       0.515  -0.956  -0.076  1.00 20.00           C\nATOM      3  C   MET A   1       0.058  -2.017   0.917  1.00 20.00           C\nATOM      4  CB  MET A   1      -0.649  -0.170  -0.682  1.00 20.00           C\nATOM      5  O   MET A   1       0.507  -1.926   2.044  1.00 20.00           O\nATOM      6  CG  MET A   1      -0.133   1.153  -1.265  1.00 20.00           C\nATOM      7  SD  MET A   1      -1.105   2.600  -0.796  1.00 20.00           S\nATOM      8  CE  MET A   1      -0.650   2.798   0.953  1.00 20.00           C\nEND"
  },
  {
    "code1": "F",
    "code3": "PHE",
    "name": "Phenylalanine",
    "category": "Hydrophobic (Aromatic)",
    "formula": "C9H11NO2",
    "mw": 165.19,
    "pI": 5.48,
    "hydropathy": 2.8,
    "charge": "Neutral",
    "codons": [
      "UUU",
      "UUC"
    ],
    "sidechain": "Benzyl / Phenyl ring (-CH2-C6H5)",
    "role": "Bulky aromatic amino acid. Strongly hydrophobic, packing tightly into the central cores of globular domains via aromatic pi-pi and pi-cation interactions.",
    "pdb": "ATOM      1  N   PHE A   1      -0.223  -2.552   0.040  1.00 20.00           N\nATOM      2  CA  PHE A   1      -0.141  -1.582   1.125  1.00 20.00           C\nATOM      3  C   PHE A   1      -0.811  -2.195   2.353  1.00 20.00           C\nATOM      4  CB  PHE A   1      -0.807  -0.250   0.741  1.00 20.00           C\nATOM      5  O   PHE A   1      -2.021  -2.080   2.539  1.00 20.00           O\nATOM      6  CG  PHE A   1      -0.025   0.607  -0.234  1.00 20.00           C\nATOM      7 CD1  PHE A   1       0.970   1.485   0.239  1.00 20.00           C\nATOM      8 CD2  PHE A   1      -0.324   0.566  -1.609  1.00 20.00           C\nATOM      9 CE1  PHE A   1       1.661   2.319  -0.659  1.00 20.00           C\nATOM     10 CE2  PHE A   1       0.367   1.401  -2.506  1.00 20.00           C\nATOM     11  CZ  PHE A   1       1.357   2.279  -2.031  1.00 20.00           C\nEND"
  },
  {
    "code1": "P",
    "code3": "PRO",
    "name": "Proline",
    "category": "Special / Cyclic",
    "formula": "C5H9NO2",
    "mw": 115.13,
    "pI": 6.3,
    "hydropathy": -1.6,
    "charge": "Neutral",
    "codons": [
      "CCU",
      "CCC",
      "CCA",
      "CCG"
    ],
    "sidechain": "Pyrrolidine ring (-CH2-CH2-CH2- bonded to backbone N)",
    "role": "Only cyclic imino acid. Constrained backbone dihedral angles (phi ~ -65 deg). Serves as a classic alpha-helix and beta-sheet breaker, introducing rigid kinks and turns into polypeptide folds.",
    "pdb": "ATOM      1  N   PRO A   1      -0.305  -0.151  -1.239  1.00 20.00           N\nATOM      2  CA  PRO A   1      -0.618   0.751  -0.139  1.00 20.00           C\nATOM      3  C   PRO A   1       0.669   1.545   0.121  1.00 20.00           C\nATOM      4  CB  PRO A   1      -1.019  -0.131   1.046  1.00 20.00           C\nATOM      5  O   PRO A   1       1.746   0.934   0.158  1.00 20.00           O\nATOM      6  CG  PRO A   1      -0.261  -1.432   0.790  1.00 20.00           C\nATOM      7  CD  PRO A   1      -0.212  -1.519  -0.737  1.00 20.00           C\nEND"
  },
  {
    "code1": "S",
    "code3": "SER",
    "name": "Serine",
    "category": "Polar / Uncharged",
    "formula": "C3H7NO3",
    "mw": 105.09,
    "pI": 5.68,
    "hydropathy": -0.8,
    "charge": "Neutral",
    "codons": [
      "UCU",
      "UCC",
      "UCA",
      "UCG",
      "AGU",
      "AGC"
    ],
    "sidechain": "Hydroxymethyl (-CH2-OH)",
    "role": "Polar uncharged residue with a nucleophilic hydroxyl group. Major site of regulatory phosphorylation by Ser/Thr kinases. Catalytic nucleophile in serine proteases (chymotrypsin, trypsin).",
    "pdb": "ATOM      1  N   SER A   1       0.725   0.143  -1.454  1.00 20.00           N\nATOM      2  CA  SER A   1      -0.310  -0.488  -0.628  1.00 20.00           C\nATOM      3  C   SER A   1      -1.250   0.573  -0.069  1.00 20.00           C\nATOM      4  CB  SER A   1       0.324  -1.287   0.509  1.00 20.00           C\nATOM      5  O   SER A   1      -0.760   1.579   0.452  1.00 20.00           O\nATOM      6  OG  SER A   1       1.273  -0.520   1.191  1.00 20.00           O\nEND"
  },
  {
    "code1": "T",
    "code3": "THR",
    "name": "Threonine",
    "category": "Polar / Uncharged",
    "formula": "C4H9NO3",
    "mw": 119.12,
    "pI": 5.6,
    "hydropathy": -0.7,
    "charge": "Neutral",
    "codons": [
      "ACU",
      "ACC",
      "ACA",
      "ACG"
    ],
    "sidechain": "Hydroxyethyl (-CH(OH)-CH3)",
    "role": "Branched beta-carbon containing a secondary hydroxyl group. Prominent site for O-linked glycosylation and reversible phosphorylation in cellular signaling cascades.",
    "pdb": "ATOM      1  N   THR A   1      -0.495   1.229  -1.441  1.00 20.00           N\nATOM      2  CA  THR A   1      -0.397   0.258  -0.346  1.00 20.00           C\nATOM      3  C   THR A   1      -0.969   0.806   0.948  1.00 20.00           C\nATOM      4  CB  THR A   1       1.044  -0.243  -0.197  1.00 20.00           C\nATOM      5  O   THR A   1      -1.779   0.126   1.579  1.00 20.00           O\nATOM      6 CG2  THR A   1       1.202  -1.317   0.876  1.00 20.00           C\nATOM      7 OG1  THR A   1       1.391  -0.858  -1.422  1.00 20.00           O\nEND"
  },
  {
    "code1": "W",
    "code3": "TRP",
    "name": "Tryptophan",
    "category": "Hydrophobic (Aromatic)",
    "formula": "C11H12N2O2",
    "mw": 204.23,
    "pI": 5.89,
    "hydropathy": -0.9,
    "charge": "Neutral",
    "codons": [
      "UGG"
    ],
    "sidechain": "Indole ring (-CH2-C8H6N)",
    "role": "Largest amino acid with a bicyclic indole ring. Dominates intrinsic protein UV absorbance (280 nm) and fluorescence. Anchors transmembrane proteins at water-lipid bilayer interfaces.",
    "pdb": "ATOM      1  N   TRP A   1      -1.100  -3.285  -0.764  1.00 20.00           N\nATOM      2  CA  TRP A   1       0.144  -2.493  -0.880  1.00 20.00           C\nATOM      3  C   TRP A   1       1.326  -3.389  -1.234  1.00 20.00           C\nATOM      4  CB  TRP A   1       0.426  -1.650   0.378  1.00 20.00           C\nATOM      5  O   TRP A   1       1.499  -4.448  -0.630  1.00 20.00           O\nATOM      6  CG  TRP A   1      -0.329  -0.354   0.423  1.00 20.00           C\nATOM      7 CD1  TRP A   1      -1.675  -0.254   0.462  1.00 20.00           C\nATOM      8 CD2  TRP A   1       0.161   1.029   0.361  1.00 20.00           C\nATOM      9 CE2  TRP A   1      -0.967   1.904   0.344  1.00 20.00           C\nATOM     10 CE3  TRP A   1       1.433   1.644   0.309  1.00 20.00           C\nATOM     11 NE1  TRP A   1      -2.049   1.062   0.317  1.00 20.00           N\nATOM     12 CH2  TRP A   1       0.425   3.878   0.305  1.00 20.00           C\nATOM     13 CZ2  TRP A   1      -0.855   3.301   0.333  1.00 20.00           C\nATOM     14 CZ3  TRP A   1       1.562   3.049   0.280  1.00 20.00           C\nEND"
  },
  {
    "code1": "Y",
    "code3": "TYR",
    "name": "Tyrosine",
    "category": "Polar / Aromatic",
    "formula": "C9H11NO3",
    "mw": 181.19,
    "pI": 5.66,
    "hydropathy": -1.3,
    "charge": "Neutral",
    "codons": [
      "UAU",
      "UAC"
    ],
    "sidechain": "Phenol ring (-CH2-C6H4-OH)",
    "role": "Aromatic amino acid with an ionizable phenolic hydroxyl group (pKa ~ 10.1). Primary target of receptor tyrosine kinase (RTK) phosphorylation. Facilitates electron transfer in enzymes.",
    "pdb": "ATOM      1  N   TYR A   1      -0.476   2.333   2.958  1.00 20.00           N\nATOM      2  CA  TYR A   1      -0.616   1.511   1.760  1.00 20.00           C\nATOM      3  C   TYR A   1      -1.776   0.537   1.924  1.00 20.00           C\nATOM      4  CB  TYR A   1       0.669   0.738   1.451  1.00 20.00           C\nATOM      5  O   TYR A   1      -1.760  -0.308   2.819  1.00 20.00           O\nATOM      6  CG  TYR A   1       0.644   0.086   0.076  1.00 20.00           C\nATOM      7 CD1  TYR A   1       0.699  -1.317  -0.047  1.00 20.00           C\nATOM      8 CD2  TYR A   1       0.525   0.880  -1.083  1.00 20.00           C\nATOM      9 CE1  TYR A   1       0.642  -1.923  -1.318  1.00 20.00           C\nATOM     10 CE2  TYR A   1       0.465   0.279  -2.354  1.00 20.00           C\nATOM     11  OH  TYR A   1       0.465  -1.692  -3.709  1.00 20.00           O\nATOM     12  CZ  TYR A   1       0.521  -1.123  -2.476  1.00 20.00           C\nEND"
  },
  {
    "code1": "V",
    "code3": "VAL",
    "name": "Valine",
    "category": "Hydrophobic",
    "formula": "C5H11NO2",
    "mw": 117.15,
    "pI": 5.96,
    "hydropathy": 4.2,
    "charge": "Neutral",
    "codons": [
      "GUU",
      "GUC",
      "GUA",
      "GUG"
    ],
    "sidechain": "Isopropyl group (-CH(CH3)2)",
    "role": "Branched-chain aliphatic hydrophobic residue. High beta-sheet propensity. Critical component of hydrophobic packing in internal protein cores.",
    "pdb": "ATOM      1  N   VAL A   1       0.452  -1.141  -1.445  1.00 20.00           N\nATOM      2  CA  VAL A   1       0.603  -0.364  -0.230  1.00 20.00           C\nATOM      3  C   VAL A   1       0.327  -1.341   0.904  1.00 20.00           C\nATOM      4  CB  VAL A   1      -0.347   0.851  -0.202  1.00 20.00           C\nATOM      5  O   VAL A   1      -0.797  -1.467   1.392  1.00 20.00           O\nATOM      6 CG1  VAL A   1      -0.063   1.724   1.028  1.00 20.00           C\nATOM      7 CG2  VAL A   1      -0.174   1.735  -1.444  1.00 20.00           C\nEND"
  }
]
