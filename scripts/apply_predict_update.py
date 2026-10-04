import re
from pathlib import Path

predict_path = Path("predict.py")
content = predict_path.read_text(encoding="utf-8")

# Replace relax_coordinates implementation
pattern = r"def relax_coordinates\([\s\S]*?return out_coords\n"
new_relax = '''def relax_coordinates(
    ca_coords: np.ndarray,
    D: np.ndarray,
    helix_mask: Optional[np.ndarray] = None,
    sheet_mask: Optional[np.ndarray] = None,
    steps: int = 150,
) -> np.ndarray:
    """Refines coordinates using physics-informed PyTorch optimization:
    - Enforces true right-handed helical dihedral potential (tau = -50.4 deg) for tight cylindrical 3D coiling
    - Enforces extended trans dihedral potential (tau = 175 deg) for beta-strands
    - Clamps target distances on secondary structure pairs to prevent loss_dist from pulling helices apart
    - Covalent bond lengths CA_i - CA_{i+1} = 3.80 A with post-optimization SHAKE projection
    - Virtual bond angle CA_{i-1} - CA_i - CA_{i+1} distance d(i, i+2) in [5.1, 7.2] A
    - Smooth Laplacian loop regularization applied ONLY to flexible coils, eliminating crinkles
    - Steric clash clearance (> 3.8 A for non-bonded pairs)
    """
    L = len(ca_coords)
    if L < 4:
        return ca_coords

    # Secondary structure masks
    h_mask = helix_mask if helix_mask is not None else np.zeros(L, dtype=bool)
    s_mask = sheet_mask if sheet_mask is not None else np.zeros(L, dtype=bool)
    loop_mask = ~(h_mask | s_mask)

    # 1. Clamp target distance matrix for secondary structures so loss_dist cooperates with physical springs
    target_D_np = D.copy()
    for i in range(L):
        if h_mask[i]:
            if i + 1 < L and h_mask[i + 1]:
                target_D_np[i, i + 1] = target_D_np[i + 1, i] = 3.80
            if i + 2 < L and h_mask[i + 2]:
                target_D_np[i, i + 2] = target_D_np[i + 2, i] = 5.40
            if i + 3 < L and h_mask[i + 3]:
                target_D_np[i, i + 3] = target_D_np[i + 3, i] = 5.05
            if i + 4 < L and h_mask[i + 4]:
                target_D_np[i, i + 4] = target_D_np[i + 4, i] = 6.20
        elif s_mask[i]:
            if i + 1 < L and s_mask[i + 1]:
                target_D_np[i, i + 1] = target_D_np[i + 1, i] = 3.80
            if i + 2 < L and s_mask[i + 2]:
                target_D_np[i, i + 2] = target_D_np[i + 2, i] = max(target_D_np[i, i + 2], 6.50)

    # 2. PyTorch energy minimization with physical force field
    x = torch.from_numpy(ca_coords.copy()).float().requires_grad_(True)
    target_D = torch.from_numpy(target_D_np).float()
    opt = torch.optim.Adam([x], lr=0.030)
    mask_nb = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=2)

    h_mask_t = torch.tensor(h_mask, dtype=torch.bool)
    s_mask_t = torch.tensor(s_mask, dtype=torch.bool)
    loop_mask_t = torch.tensor(loop_mask, dtype=torch.bool)

    # Contact-sensitive weights: strong attraction for close contacts (<=8A), decaying for diffuse distances
    weights = torch.ones_like(target_D)
    weights = torch.where(target_D <= 8.0, 2.5, weights)
    weights = torch.where(target_D > 14.0, 0.25, weights)
    w_nb = weights[mask_nb]

    # Dihedral targets: correct right-handed alpha-helix dihedral (-50.4 deg) and extended beta-sheet (175 deg)
    target_cos_helix = float(np.cos(np.radians(-50.4)))
    target_sin_helix = float(np.sin(np.radians(-50.4)))
    target_cos_sheet = float(np.cos(np.radians(175.0)))
    target_sin_sheet = float(np.sin(np.radians(175.0)))

    # Target radius of gyration (Flory law for globular proteins: Rg ~ 2.82 * L^0.28)
    rg_target = float(2.82 * (L ** 0.28))

    for step in range(steps):
        opt.zero_grad()
        cur_D = torch.cdist(x, x)

        # Contact-weighted Huber distance loss
        diff = torch.abs(cur_D[mask_nb] - target_D[mask_nb])
        huber_diff = torch.where(diff < 2.0, 0.5 * (diff ** 2), 2.0 * diff - 2.0)
        loss_dist = torch.mean(w_nb * huber_diff)

        # Consecutive CA-CA bond length = 3.80 A (stiff harmonic spring)
        bonds = torch.norm(x[1:] - x[:-1], dim=-1)
        loss_bond = torch.mean((bonds - 3.80) ** 2)

        # Virtual bond angle constraint d(i, i+2) in [5.1, 7.2] A
        d_i_i2 = torch.norm(x[2:] - x[:-2], dim=-1)
        loss_angle = torch.mean(
            torch.clamp(5.1 - d_i_i2, min=0.0) ** 2 + torch.clamp(d_i_i2 - 7.2, min=0.0) ** 2
        )

        # Secondary structure hydrogen-bond network regularization (stiff canonical geometry)
        loss_ss = torch.tensor(0.0)
        for i in range(L - 4):
            if h_mask_t[i] and h_mask_t[i + 2]:
                loss_ss = loss_ss + (torch.norm(x[i + 2] - x[i]) - 5.40) ** 2
            if h_mask_t[i] and h_mask_t[i + 3]:
                loss_ss = loss_ss + (torch.norm(x[i + 3] - x[i]) - 5.05) ** 2
            if h_mask_t[i] and h_mask_t[i + 4]:
                loss_ss = loss_ss + (torch.norm(x[i + 4] - x[i]) - 6.20) ** 2
        for i in range(L - 2):
            if s_mask_t[i] and s_mask_t[i + 2]:
                loss_ss = loss_ss + (torch.norm(x[i + 2] - x[i]) - 6.60) ** 2

        # Chiral dihedral coiling potential (forces right-handed 3D cylindrical spirals!)
        cos_tau, sin_tau = _compute_dihedrals_torch(x)
        loss_dihedral = torch.tensor(0.0)
        n_quads = 0
        for j in range(len(cos_tau)):
            if h_mask_t[j] and h_mask_t[j + 1] and h_mask_t[j + 2] and h_mask_t[j + 3]:
                loss_dihedral = loss_dihedral + (cos_tau[j] - target_cos_helix) ** 2 + (sin_tau[j] - target_sin_helix) ** 2
                n_quads += 1
            elif s_mask_t[j] and s_mask_t[j + 1] and s_mask_t[j + 2] and s_mask_t[j + 3]:
                loss_dihedral = loss_dihedral + (cos_tau[j] - target_cos_sheet) ** 2 + (sin_tau[j] - target_sin_sheet) ** 2
                n_quads += 1
        if n_quads > 0:
            loss_dihedral = loss_dihedral / n_quads

        # Radius of gyration compactness prior
        center = torch.mean(x, dim=0, keepdim=True)
        cur_rg = torch.sqrt(torch.mean(torch.sum((x - center) ** 2, dim=-1)) + 1e-6)
        loss_rg = torch.clamp(cur_rg - (rg_target * 1.20), min=0.0) ** 2

        # Smooth loop regularization (Laplacian curvature minimization on loops to eliminate crinkles)
        loss_smooth = torch.tensor(0.0)
        for i in range(L - 2):
            if loop_mask_t[i] and loop_mask_t[i + 1] and loop_mask_t[i + 2]:
                lap = x[i + 2] - 2.0 * x[i + 1] + x[i]
                loss_smooth = loss_smooth + torch.sum(lap ** 2)

        # Steric clash avoidance (> 3.8 A for non-bonded pairs)
        clashes = torch.clamp(3.8 - cur_D[mask_nb], min=0.0)
        loss_clash = torch.mean(clashes ** 2)

        total_loss = (
            2.0 * loss_dist
            + 120.0 * loss_bond
            + 25.0 * loss_angle
            + 35.0 * loss_ss
            + 40.0 * loss_dihedral
            + 2.5 * loss_rg
            + 1.0 * loss_smooth
            + 15.0 * loss_clash
        )
        total_loss.backward()
        opt.step()

    out_coords = x.detach().numpy()

    # 3. Holonomic covalent bond constraint (SHAKE algorithm)
    # Guarantees EVERY single CA-CA bond length is strictly 3.80 +- 0.005 A
    for _ in range(25):
        for i in range(L - 1):
            v = out_coords[i + 1] - out_coords[i]
            d = np.linalg.norm(v)
            diff = (d - 3.80) / (d + 1e-8)
            out_coords[i] += 0.5 * diff * v
            out_coords[i + 1] -= 0.5 * diff * v

    return out_coords
'''

assert re.search(pattern, content) is not None, "relax_coordinates pattern not found!"
content = re.sub(pattern, new_relax, content, count=1)

# In predict_sequence, pass sequence to assign_secondary_structure
old_assign_call = "helices, sheets, ss_string = assign_secondary_structure(ca_coords, pred_dist)"
new_assign_call = "helices, sheets, ss_string = assign_secondary_structure(ca_coords, pred_dist, sequence=sequence)"
assert old_assign_call in content, "assign_secondary_structure call not found in predict_sequence!"
content = content.replace(old_assign_call, new_assign_call, 1)

predict_path.write_text(content, encoding="utf-8")
print("predict.py successfully updated with universal coiling and smoothing engine!")
