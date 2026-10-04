"""
Regenerate All Preset PDB Structures with Universal Engine
==========================================================
Recomputes and saves all preset PDB structures using the updated Mini-AlphaFold
universal secondary structure detection, tight helical coiling, beta-sheet arrow
formatting, and smooth loop engine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from predict import Predictor, save_pdb
from app.main import PRESET_PROTEINS

def main():
    print("Loading Predictor...")
    predictor = Predictor()

    out_dir = Path("predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in PRESET_PROTEINS:
        pid = p["id"]
        seq = p["sequence"]
        name = p["name"]
        print(f"\nFolding {pid} ({name}, {len(seq)} aa)...")
        res = predictor.predict_sequence(seq, relax=True)
        pdb_path = out_dir / f"{pid}_predicted.pdb"
        save_pdb(
            res["backbone_coords"],
            seq,
            pdb_path,
            helices=res["helices"],
            sheets=res["sheets"],
            plddt=res["plddt"],
        )
        print(f"  SS: {res['ss_string']}")
        print(f"  Helices: {[(s+1, e+1) for s, e in res['helices']]}")
        print(f"  Sheets:  {[(s+1, e+1) for s, e in res['sheets']]}")
        print(f"  Saved: {pdb_path}")

    # Also update latest_predicted.pdb with Crambin
    latest_path = out_dir / "latest_predicted.pdb"
    crambin_p = out_dir / "crambin_predicted.pdb"
    if crambin_p.exists():
        latest_path.write_text(crambin_p.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\nUpdated {latest_path} with crambin prediction.")

    print("\nAll presets regenerated successfully!")

if __name__ == "__main__":
    main()
