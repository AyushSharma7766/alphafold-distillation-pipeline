import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import esm
import torch
import numpy as np
from scripts.benchmark_accuracy import BENCHMARK_TARGETS, extract_ca_and_seq_from_pdb, compute_tm_score, compute_q3_score
from predict import kabsch_rmsd, compute_contact_precision, distance_matrix_to_3d_coords, relax_coordinates, assign_secondary_structure

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

print("\n" + "="*75)
print("PHASE 2 EVALUATION: ESM-2 (8M) Evolutionary Representations")
print("="*75)
print(f"{'Target':<24} | {'Length':<6} | {'Baseline P@L':<13} | {'ESM-2 P@L (%)':<14} | {'Gain'}")
print("-"*75)

p_l_baseline = {"Crambin (1CRN)": 23.9, "Villin HP-36 (1VII)": 11.1, "Trp-Cage (1L2Y)": 5.0, "Designed Fold (2DK4)": 3.9}
gains = []

for target in BENCHMARK_TARGETS:
    true_ca, seq = extract_ca_and_seq_from_pdb(target["pdb"])
    L = len(true_ca)
    data = [(target["name"], seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)
    contacts = results["contacts"][0, :L, :L].numpy()

    true_diff = true_ca[:, None, :] - true_ca[None, :, :]
    true_dist = np.sqrt(np.sum(true_diff ** 2, axis=-1))

    # Evaluate contact precision P@L (|i-j| >= 6, cutoff <= 8.0 A)
    mask_diag = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= 6
    pred_c = contacts.copy()
    pred_c[~mask_diag] = -1e9
    true_bin = (true_dist <= 8.0) & mask_diag

    top_l = np.argsort(pred_c.ravel())[::-1][:L]
    prec = float(np.mean(true_bin.ravel()[top_l])) * 100.0

    b_val = p_l_baseline.get(target["name"], 11.0)
    diff = prec - b_val
    gains.append(diff)
    print(f"{target['name']:<24} | {L:<6} | {b_val:<13.1f}% | {prec:<14.1f}% | +{diff:.1f}%")

print("-"*75)
print(f"{'AVERAGE':<24} | {'--':<6} | {11.0:<13.1f}% | {np.mean([p_l_baseline[t['name']]+g for t, g in zip(BENCHMARK_TARGETS, gains)]):<14.1f}% | +{np.mean(gains):.1f}%")
print("="*75 + "\n")
