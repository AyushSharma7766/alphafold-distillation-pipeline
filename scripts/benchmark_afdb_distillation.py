import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from src.model import MiniAlphaFold
from predict import (
    distance_matrix_to_3d_coords,
    relax_coordinates,
    assign_secondary_structure,
    kabsch_rmsd,
    compute_contact_precision,
    sequence_to_indices,
)
from scripts.benchmark_accuracy import (
    BENCHMARK_TARGETS,
    extract_ca_and_seq_from_pdb,
    compute_tm_score,
    compute_q3_score,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniAlphaFold(
    d_model=128,
    d_pair=64,
    max_relpos=32,
    n_evoformer_blocks=4,
    n_heads=8,
    n_egnn_layers=3,
    dropout=0.1,
    use_distogram=True,
).to(device)

ckpt_path = "checkpoints/distogram_afdb_best.pt"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

results = []

print("\n" + "="*75)
print(f"AFDB DISTILLATION BENCHMARK: {ckpt_path}")
print("="*75)
print(f"{'Target':<24} | {'Length':<6} | {'RMSD (A)':<8} | {'TM-Score':<8} | {'P@L (%)':<8} | {'Q3 SS (%)'}")
print("-"*75)

for target in BENCHMARK_TARGETS:
    true_ca, seq = extract_ca_and_seq_from_pdb(target["pdb"])
    L = len(true_ca)
    indices = sequence_to_indices(seq)
    seq_tensor = indices.unsqueeze(0).to(device)
    mask_tensor = torch.ones(1, L, dtype=torch.bool, device=device)

    with torch.no_grad():
        out = model(seq_tensor, mask=mask_tensor)
        pred_dist = out["pred_dist"][0].cpu().numpy()

    true_diff = true_ca[:, None, :] - true_ca[None, :, :]
    true_dist = np.sqrt(np.sum(true_diff ** 2, axis=-1))
    _, _, ss_true = assign_secondary_structure(true_ca, true_dist)

    pred_ca = distance_matrix_to_3d_coords(pred_dist)
    helices, sheets, ss_string = assign_secondary_structure(pred_ca, pred_dist)
    h_mask = np.array([ch == "H" for ch in ss_string], dtype=bool)
    s_mask = np.array([ch == "E" for ch in ss_string], dtype=bool)
    pred_ca = relax_coordinates(pred_ca, pred_dist, helix_mask=h_mask, sheet_mask=s_mask)[:L]

    rmsd = kabsch_rmsd(torch.from_numpy(pred_ca), torch.from_numpy(true_ca))
    tm = compute_tm_score(pred_ca, true_ca)
    q3 = compute_q3_score(ss_string[:L], ss_true)
    contacts = compute_contact_precision(
        torch.from_numpy(out["coord_dist"][0, :L, :L].cpu().numpy()).unsqueeze(0),
        torch.from_numpy(true_dist).unsqueeze(0),
    )

    print(f"{target['name']:<24} | {L:<6} | {rmsd:<8.2f} | {tm:<8.4f} | {contacts['p_at_l']*100:<8.1f} | {q3*100:.1f}%")
    results.append({"name": target["name"], "rmsd": rmsd, "tm": tm, "pl": contacts["p_at_l"]*100, "q3": q3*100})

mean_rmsd = np.mean([r["rmsd"] for r in results])
mean_tm = np.mean([r["tm"] for r in results])
mean_pl = np.mean([r["pl"] for r in results])
mean_q3 = np.mean([r["q3"] for r in results])
print("-"*75)
print(f"{'AVERAGE':<24} | {'--':<6} | {mean_rmsd:<8.2f} | {mean_tm:<8.4f} | {mean_pl:<8.1f} | {mean_q3:.1f}%")
print("="*75 + "\n")
