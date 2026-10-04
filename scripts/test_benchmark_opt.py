import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from src.model import MiniAlphaFold
from src.data_parser import sequence_to_indices
from predict import distance_matrix_to_3d_coords, _compute_dihedrals_torch, kabsch_rmsd
from scripts.benchmark_accuracy import extract_ca_and_seq_from_pdb, BENCHMARK_TARGETS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('checkpoints/distogram_afdb_best.pt', map_location=device)
model = MiniAlphaFold(d_model=128, d_pair=64, max_relpos=32, n_evoformer_blocks=4, n_heads=8, n_egnn_layers=3, dropout=0.1, use_distogram=True).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

def test_opt(target):
    true_ca, seq = extract_ca_and_seq_from_pdb(target['pdb'])
    L = len(true_ca)
    indices = sequence_to_indices(seq).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(indices)
        D = out['pred_dist'][0].cpu().numpy()
    pred_ca = distance_matrix_to_3d_coords(D)
    
    # Helix detection
    raw_h = np.zeros(L, dtype=bool)
    for i in range(L - 4):
        if D[i, i+3] <= 7.4 and D[i, i+4] <= 8.6 and D[i, i+2] <= 6.2:
            raw_h[i:i+4] = True
            
    x = torch.from_numpy(pred_ca.copy()).float().requires_grad_(True)
    target_D = torch.from_numpy(D).float()
    opt = torch.optim.Adam([x], lr=0.030)
    mask_nb = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=2)
    h_mask_t = torch.tensor(raw_h, dtype=torch.bool)
    
    weights = torch.ones_like(target_D)
    weights = torch.where(target_D <= 8.0, 2.5, weights)
    weights = torch.where(target_D > 14.0, 0.25, weights)
    w_nb = weights[mask_nb]
    
    target_cos = float(np.cos(np.radians(50.0)))
    target_sin = float(np.sin(np.radians(50.0)))
    
    for step in range(120):
        opt.zero_grad()
        cur_D = torch.cdist(x, x)
        diff = torch.abs(cur_D[mask_nb] - target_D[mask_nb])
        loss_dist = torch.mean(w_nb * torch.where(diff < 2.0, 0.5 * (diff**2), 2.0 * diff - 2.0))
        bonds = torch.norm(x[1:] - x[:-1], dim=-1)
        loss_bond = torch.mean((bonds - 3.80) ** 2)
        d_i_i2 = torch.norm(x[2:] - x[:-2], dim=-1)
        loss_angle = torch.mean(torch.clamp(5.1 - d_i_i2, min=0.0)**2 + torch.clamp(d_i_i2 - 7.2, min=0.0)**2)
        
        # Helical dihedrals
        cos_t, sin_t = _compute_dihedrals_torch(x)
        loss_dih = torch.tensor(0.0)
        n_q = 0
        for j in range(len(cos_t)):
            if h_mask_t[j] and h_mask_t[j+1] and h_mask_t[j+2] and h_mask_t[j+3]:
                loss_dih = loss_dih + (cos_t[j] - target_cos)**2 + (sin_t[j] - target_sin)**2
                n_q += 1
        if n_q > 0:
            loss_dih = loss_dih / n_q
            
        loss = 1.5 * loss_dist + 85.0 * loss_bond + 25.0 * loss_angle + 10.0 * loss_dih
        loss.backward()
        opt.step()
        
    ca_opt = x.detach().numpy()
    
    # SHAKE
    for _ in range(15):
        for i in range(L - 1):
            v = ca_opt[i+1] - ca_opt[i]
            d = np.linalg.norm(v)
            diff = (d - 3.80) / (d + 1e-8)
            ca_opt[i] += 0.5 * diff * v
            ca_opt[i+1] -= 0.5 * diff * v
            
    rmsd = kabsch_rmsd(torch.from_numpy(ca_opt), torch.from_numpy(true_ca))
    print(f"{target['name']}: RMSD = {rmsd:.2f} A, bond mean = {np.mean(np.linalg.norm(ca_opt[1:] - ca_opt[:-1], axis=-1)):.2f}")

for t in BENCHMARK_TARGETS:
    test_opt(t)
