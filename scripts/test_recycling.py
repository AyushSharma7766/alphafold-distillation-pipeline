import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from predict import Predictor, kabsch_rmsd, compute_contact_precision, assign_secondary_structure
from scripts.benchmark_accuracy import BENCHMARK_TARGETS, extract_ca_and_seq_from_pdb, compute_tm_score, compute_q3_score

def eval_with_recycles(predictor: Predictor, num_recycles: int = 1):
    results = []
    model = predictor.model

    for target in BENCHMARK_TARGETS:
        pdb_p = Path(target["pdb"])
        if not pdb_p.exists(): continue
        true_ca, seq = extract_ca_and_seq_from_pdb(pdb_p)
        L = len(true_ca)
        if L == 0: continue

        from src.data_parser import sequence_to_indices
        indices = sequence_to_indices(seq)
        seq_tensor = indices.unsqueeze(0).to(predictor.device)
        mask_tensor = torch.ones(1, L, dtype=torch.bool, device=predictor.device)

        with torch.no_grad():
            s_init = model.seq_embedding(seq_tensor, mask=mask_tensor)
            z_init = model.pair_initializer(s_init, mask=mask_tensor)
            s, z = s_init, z_init
            for r in range(num_recycles):
                s, z = model.evoformer(s, z, mask=mask_tensor)
                if r < num_recycles - 1:
                    s = s + 0.5 * s_init
                    z = z + 0.5 * z_init

            pred_dist = model.distance_head(z, mask=mask_tensor)[0].cpu().numpy()

        from predict import distance_matrix_to_3d_coords, relax_coordinates
        pred_ca = distance_matrix_to_3d_coords(pred_dist)
        helices, sheets, ss_string = assign_secondary_structure(pred_ca, pred_dist)
        h_mask = np.array([ch == "H" for ch in ss_string], dtype=bool)
        s_mask = np.array([ch == "E" for ch in ss_string], dtype=bool)
        pred_ca = relax_coordinates(pred_ca, pred_dist, helix_mask=h_mask, sheet_mask=s_mask)[:L]

        # Metrics
        rmsd = kabsch_rmsd(torch.from_numpy(pred_ca), torch.from_numpy(true_ca))
        tm = compute_tm_score(pred_ca, true_ca)
        results.append({"target": target["name"], "rmsd": rmsd, "tm": tm})

    mean_rmsd = np.mean([r["rmsd"] for r in results])
    mean_tm = np.mean([r["tm"] for r in results])
    print(f"Recycles = {num_recycles}: Mean RMSD = {mean_rmsd:.3f} A, Mean TM = {mean_tm:.4f}")
    for r in results:
        print(f"   {r['target']}: RMSD = {r['rmsd']:.2f} A, TM = {r['tm']:.4f}")

if __name__ == "__main__":
    p = Predictor()
    for rec in [1, 2, 3]:
        eval_with_recycles(p, num_recycles=rec)
