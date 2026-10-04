import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import esm
import torch
import numpy as np
from scripts.benchmark_accuracy import BENCHMARK_TARGETS, extract_ca_and_seq_from_pdb, compute_tm_score, compute_q3_score
from predict import Predictor, kabsch_rmsd, compute_contact_precision, distance_matrix_to_3d_coords, relax_coordinates, assign_secondary_structure

# Load ESM-2 8M
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

predictor = Predictor()
results = []

print("\n" + "="*75)
print("PHASE 2: 3D FOLDING WITH ESM-2 CONTACT-GUIDED DISTANCE GEOMETRY")
print("="*75)
print(f"{'Target':<24} | {'Length':<6} | {'Baseline RMSD':<14} | {'Phase 2 RMSD':<14} | {'TM-Score'}")
print("-"*75)

b_rmsds = {"Crambin (1CRN)": 7.66, "Villin HP-36 (1VII)": 6.95, "Trp-Cage (1L2Y)": 5.03, "Designed Fold (2DK4)": 9.87}

for target in BENCHMARK_TARGETS:
    true_ca, seq = extract_ca_and_seq_from_pdb(target["pdb"])
    L = len(true_ca)
    
    # 1. ESM-2 contacts
    data = [(target["name"], seq)]
    _, _, tokens = batch_converter(data)
    with torch.no_grad():
        esm_out = esm_model(tokens, repr_layers=[6], return_contacts=True)
    contacts = esm_out["contacts"][0, :L, :L].numpy()

    # 2. Mini-AlphaFold predicted distance matrix
    from src.data_parser import sequence_to_indices
    indices = sequence_to_indices(seq)
    seq_tensor = indices.unsqueeze(0).to(predictor.device)
    mask_tensor = torch.ones(1, L, dtype=torch.bool, device=predictor.device)
    with torch.no_grad():
        out = predictor.model(seq_tensor, mask=mask_tensor)
        pred_dist = out["pred_dist"][0].cpu().numpy()

    # 3. Blend predicted distances with evolutionary contact constraints:
    # High-probability contacts (p > 0.4) are tightened to physical contact distance (~6.0 A)
    blended_D = pred_dist.copy()
    mask_contact = (contacts > 0.35) & (np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= 5)
    blended_D[mask_contact] = np.minimum(blended_D[mask_contact], 5.8 + 2.5 * (1.0 - contacts[mask_contact]))
    # Symmetrize
    blended_D = 0.5 * (blended_D + blended_D.T)
    np.fill_diagonal(blended_D, 0.0)

    # 4. Fold into 3D
    pred_ca = distance_matrix_to_3d_coords(blended_D)
    helices, sheets, ss_string = assign_secondary_structure(pred_ca, blended_D)
    h_mask = np.array([ch == "H" for ch in ss_string], dtype=bool)
    s_mask = np.array([ch == "E" for ch in ss_string], dtype=bool)
    pred_ca = relax_coordinates(pred_ca, blended_D, helix_mask=h_mask, sheet_mask=s_mask)[:L]

    # Metrics
    rmsd = kabsch_rmsd(torch.from_numpy(pred_ca), torch.from_numpy(true_ca))
    tm = compute_tm_score(pred_ca, true_ca)
    base_r = b_rmsds.get(target["name"], 7.38)
    diff = rmsd - base_r
    sign = "+" if diff > 0 else ""
    print(f"{target['name']:<24} | {L:<6} | {base_r:<14.2f} | {rmsd:<14.2f} | {tm:.4f} ({sign}{diff:.2f} A)")
    results.append({"name": target["name"], "rmsd": rmsd, "tm": tm})

mean_rmsd = np.mean([r["rmsd"] for r in results])
mean_tm = np.mean([r["tm"] for r in results])
print("-"*75)
print(f"{'AVERAGE':<24} | {'--':<6} | {7.38:<14.2f} | {mean_rmsd:<14.2f} | {mean_tm:.4f}")
print("="*75 + "\n")
