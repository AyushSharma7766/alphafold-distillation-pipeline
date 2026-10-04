"""
Mini-AlphaFold: Invariant Point Attention (IPA) Structure Module
================================================================
Implements AlphaFold 2's Structure Module with Invariant Point Attention.

Key innovations over EGNN:
  - Rigid body frames (R ∈ SO(3), t ∈ R³) per residue instead of raw coords
  - SE(3)-invariant attention via local frame point projections
  - Weight-shared iterative refinement (all blocks share parameters, like AF2)
  - Ideal backbone geometry from refined frames

Reference: Jumper et al., Nature 596, 583-589 (2021), Algorithm 22.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Rigid Body Frame Utilities
# ============================================================

def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix.

    Args:
        q: [..., 4] quaternions (w, x, y, z format)
    Returns:
        R: [..., 3, 3] rotation matrices
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),      1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)

    return R


def identity_frames(
    batch_size: int, n_residues: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create identity rigid body frames.

    Returns:
        R: [B, L, 3, 3] identity rotation matrices
        t: [B, L, 3] zero translations
    """
    R = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
    R = R.expand(batch_size, n_residues, 3, 3).contiguous()
    t = torch.zeros(batch_size, n_residues, 3, device=device)
    return R, t


def compose_frames(
    R1: torch.Tensor, t1: torch.Tensor,
    R2: torch.Tensor, t2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose rigid transformations: T1 ∘ T2.

    T_out(x) = R1(R2·x + t2) + t1 = (R1·R2)·x + (R1·t2 + t1)
    """
    R = torch.matmul(R1, R2)
    t = torch.matmul(R1, t2.unsqueeze(-1)).squeeze(-1) + t1
    return R, t


# ============================================================
# Invariant Point Attention (IPA)
# ============================================================

class InvariantPointAttention(nn.Module):
    """AlphaFold 2's IPA mechanism (Algorithm 22).

    Attention uses three complementary signals:
      1. Feature-space Q·K dot-product similarity
      2. Pair representation bias z_ij (structural context)
      3. 3D point distances in local residue frames (SE(3)-invariant geometry)

    The 3D point distances allow the module to reason about spatial
    relationships while remaining invariant to global rotations/translations.

    Args:
        d_single: Single representation dimension
        d_pair: Pair representation dimension
        n_heads: Number of attention heads
        n_qk_points: Query/key 3D points per head (for geometric reasoning)
        n_v_points: Value 3D points per head (for geometric output)
    """

    def __init__(
        self,
        d_single: int = 128,
        d_pair: int = 64,
        n_heads: int = 4,
        n_qk_points: int = 4,
        n_v_points: int = 4,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_single // n_heads
        self.n_qk_points = n_qk_points
        self.n_v_points = n_v_points

        # Feature-space QKV projections
        self.q_proj = nn.Linear(d_single, d_single, bias=False)
        self.k_proj = nn.Linear(d_single, d_single, bias=False)
        self.v_proj = nn.Linear(d_single, d_single, bias=False)

        # 3D point QKV projections (points predicted in local frame)
        self.q_pt_proj = nn.Linear(d_single, n_heads * n_qk_points * 3, bias=False)
        self.k_pt_proj = nn.Linear(d_single, n_heads * n_qk_points * 3, bias=False)
        self.v_pt_proj = nn.Linear(d_single, n_heads * n_v_points * 3, bias=False)

        # Pair representation bias on attention logits
        self.pair_proj = nn.Linear(d_pair, n_heads, bias=False)

        # Learnable per-head weights for point attention
        # Initialized to softplus^{-1}(1) ≈ 0.541 so initial gamma = 1.0
        self.head_weights = nn.Parameter(
            torch.full((n_heads,), math.log(math.exp(1.0) - 1.0))
        )

        # Gated output projection
        feat_out = n_heads * self.d_head
        pt_out = n_heads * n_v_points * 3
        pt_norm_out = n_heads * n_v_points
        pair_out = n_heads
        total_out = feat_out + pt_out + pt_norm_out + pair_out

        self.gate_proj = nn.Linear(d_single, total_out)
        self.out_proj = nn.Linear(total_out, d_single)
        nn.init.zeros_(self.out_proj.bias)

        # Layer norms
        self.ln_s = nn.LayerNorm(d_single)
        self.ln_z = nn.LayerNorm(d_pair)

        # Attention scaling constants (AF2 normalization)
        self.feat_scale = 1.0 / math.sqrt(self.d_head)
        self.pt_scale = math.sqrt(2.0 / (9.0 * n_qk_points))

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            s: [B, L, d_single] single representation
            z: [B, L, L, d_pair] pair representation
            R: [B, L, 3, 3] current residue rotation matrices
            t: [B, L, 3] current residue translations (CA positions)
            mask: [B, L] bool mask
        Returns:
            out: [B, L, d_single] updated single representation
        """
        B, L, _ = s.shape
        H, D = self.n_heads, self.d_head
        Qp, Vp = self.n_qk_points, self.n_v_points

        s_n = self.ln_s(s)
        z_n = self.ln_z(z)

        # ---- 1. Feature-space attention ----
        q = self.q_proj(s_n).view(B, L, H, D)
        k = self.k_proj(s_n).view(B, L, H, D)
        v = self.v_proj(s_n).view(B, L, H, D)

        attn_feat = torch.einsum("bihd,bjhd->bijh", q, k) * self.feat_scale

        # ---- 2. Pair bias ----
        pair_bias = self.pair_proj(z_n)  # [B, L, L, H]

        # ---- 3. 3D Point attention (the key innovation) ----
        # Predict points in local residue frame
        q_pts = self.q_pt_proj(s_n).view(B, L, H, Qp, 3)
        k_pts = self.k_pt_proj(s_n).view(B, L, H, Qp, 3)
        v_pts = self.v_pt_proj(s_n).view(B, L, H, Vp, 3)

        # Transform local points to global frame: x_global = R @ x_local + t
        q_g = torch.einsum("blij,blhpj->blhpi", R, q_pts) + t[:, :, None, None, :]
        k_g = torch.einsum("blij,blhpj->blhpi", R, k_pts) + t[:, :, None, None, :]
        v_g = torch.einsum("blij,blhpj->blhpi", R, v_pts) + t[:, :, None, None, :]

        # Point attention: -γ/2 * w_C * Σ_p ‖q_global_p - k_global_p‖²
        pt_diff = q_g[:, :, None] - k_g[:, None, :]  # [B, L, L, H, Qp, 3]
        pt_dist_sq = torch.sum(pt_diff ** 2, dim=(-1, -2))  # [B, L, L, H]

        gamma = F.softplus(self.head_weights)  # [H], positive per-head weight
        attn_pts = -0.5 * gamma * pt_dist_sq * self.pt_scale

        # ---- Combined attention logits ----
        attn_logits = attn_feat + pair_bias + attn_pts

        if mask is not None:
            key_mask = mask[:, None, :, None].expand_as(attn_logits)
            attn_logits = attn_logits.masked_fill(~key_mask, -1e4)

        attn_weights = F.softmax(attn_logits, dim=2)  # [B, L, L, H] softmax over j

        # ---- Aggregate outputs ----
        # Feature values
        o_feat = torch.einsum("bijh,bjhd->bihd", attn_weights, v)

        # 3D point values: aggregate in global frame, transform back to local
        o_pts_g = torch.einsum("bijh,bjhpc->bihpc", attn_weights, v_g)
        R_inv = R.transpose(-1, -2)
        o_pts_centered = o_pts_g - t[:, :, None, None, :]
        o_pts_local = torch.einsum("blij,blhpj->blhpi", R_inv, o_pts_centered)

        # Point norms: SE(3)-invariant scalar features from aggregated points
        o_pts_norm = torch.sqrt(
            torch.sum(o_pts_local ** 2, dim=-1) + 1e-8
        )  # [B, L, H, Vp]

        # Pair aggregation (reuse pair_bias projection)
        o_pair = torch.einsum("bijh,bijh->bih", attn_weights, pair_bias)

        # ---- Flatten and gate ----
        o_feat_flat = o_feat.reshape(B, L, -1)
        o_pts_flat = o_pts_local.reshape(B, L, -1)
        o_pts_norm_flat = o_pts_norm.reshape(B, L, -1)
        o_pair_flat = o_pair.reshape(B, L, -1)

        raw = torch.cat([o_feat_flat, o_pts_flat, o_pts_norm_flat, o_pair_flat], dim=-1)
        gate = torch.sigmoid(self.gate_proj(s_n))
        out = self.out_proj(gate * raw)

        return out


# ============================================================
# Backbone Frame Update
# ============================================================

class BackboneUpdate(nn.Module):
    """Predicts a small rigid body update (rotation + translation) from single repr.

    Rotation is parameterized as quaternion (1, x, y, z) which gives
    near-identity rotation when x, y, z ≈ 0 (ensured by zero initialization).
    """

    def __init__(self, d_single: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_single, 6)
        # Small Xavier init for weights (breaks zero attractor for translations)
        # Zero init for rotation (quat part stays near-identity), small init for translation
        nn.init.zeros_(self.proj.weight[:3])  # rotation part: zero-init (safe)
        nn.init.xavier_normal_(self.proj.weight[3:], gain=0.1)  # translation: small signal
        nn.init.zeros_(self.proj.bias)

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: [B, L, d_single]
        Returns:
            quat: [B, L, 4] quaternion (w=1, x, y, z) → near-identity
            trans: [B, L, 3] translation update
        """
        update = self.proj(s)

        # Quaternion: (1, x, y, z) → identity when zero-initialized
        ones = torch.ones(
            *update.shape[:-1], 1,
            device=update.device, dtype=update.dtype,
        )
        quat = torch.cat([ones, update[..., :3]], dim=-1)
        trans = update[..., 3:6]
        return quat, trans


# ============================================================
# Ideal Backbone Geometry (Engh & Huber)
# ============================================================

# Ideal backbone atom positions in local residue frame (Ångströms)
# Convention: CA at origin, N along negative x-axis, C in +xy quadrant
# N-CA bond: 1.458 Å, CA-C bond: 1.525 Å, N-CA-C angle: 111.2°
_IDEAL_N = torch.tensor([-1.458, 0.000, 0.000])
_IDEAL_C = torch.tensor([0.550, 1.422, 0.000])
_IDEAL_O = torch.tensor([-0.213, 2.388, 0.000])


# ============================================================
# IPA Structure Module (replaces EGNN)
# ============================================================

class IPAStructureModule(nn.Module):
    """AlphaFold 2-style Structure Module with Invariant Point Attention.

    Architecture (per iteration, all weights shared across iterations):
      1. s ← s + Dropout(IPA(s, z, T))
      2. s ← LayerNorm(s)
      3. s ← s + Transition(s)
      4. T ← compose(T, BackboneUpdate(s))   [no stop_grad for gradient flow]

    After all iterations, backbone atoms (N, CA, C, O) are placed
    using Engh & Huber ideal geometry within each residue's final frame.

    Additionally, a direct coordinate prediction head bypasses the frame
    composition chain to provide robust 3D coordinate output.

    Args:
        d_single: Single representation dimension (default: 128)
        d_pair: Pair representation dimension (default: 64)
        n_heads: IPA attention heads (default: 4)
        n_qk_points: Query/key 3D points per head (default: 4)
        n_v_points: Value 3D points per head (default: 4)
        n_blocks: Number of refinement iterations (default: 4)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        d_single: int = 128,
        d_pair: int = 64,
        n_heads: int = 4,
        n_qk_points: int = 4,
        n_v_points: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_blocks = n_blocks

        # Input projection
        self.ln_input = nn.LayerNorm(d_single)

        # Shared IPA block (weight-tied across all iterations, like AF2)
        self.ipa = InvariantPointAttention(
            d_single=d_single,
            d_pair=d_pair,
            n_heads=n_heads,
            n_qk_points=n_qk_points,
            n_v_points=n_v_points,
        )
        self.ipa_dropout = nn.Dropout(dropout)
        self.ipa_ln = nn.LayerNorm(d_single)

        # Transition MLP (shared across iterations)
        self.transition = nn.Sequential(
            nn.LayerNorm(d_single),
            nn.Linear(d_single, d_single * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_single * 2, d_single),
            nn.Dropout(dropout),
        )

        # Backbone frame update (shared, zero-initialized)
        self.bb_update = BackboneUpdate(d_single)

        # Direct coordinate prediction head (bypass for gradient flow)
        # This directly predicts XYZ from the single representation
        self.coord_head = nn.Sequential(
            nn.LayerNorm(d_single),
            nn.Linear(d_single, d_single),
            nn.ReLU(),
            nn.Linear(d_single, 3),
        )
        # Initialize with small weights so initial prediction is near zero
        nn.init.xavier_normal_(self.coord_head[3].weight, gain=0.01)
        nn.init.zeros_(self.coord_head[3].bias)

        # Register ideal atom positions as non-trainable buffers
        self.register_buffer("ideal_N", _IDEAL_N)
        self.register_buffer("ideal_C", _IDEAL_C)
        self.register_buffer("ideal_O", _IDEAL_O)

    def _frames_to_backbone(
        self, R: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Convert rigid body frames to backbone atom coordinates.

        Places N, CA, C, O atoms at ideal Engh & Huber positions
        within each residue's local coordinate frame.

        Args:
            R: [B, L, 3, 3] rotation matrices
            t: [B, L, 3] translations (= CA positions)
        Returns:
            backbone: [B, L, 4, 3] (atom order: N=0, CA=1, C=2, O=3)
        """
        n = torch.einsum("blij,j->bli", R, self.ideal_N) + t
        c = torch.einsum("blij,j->bli", R, self.ideal_C) + t
        o = torch.einsum("blij,j->bli", R, self.ideal_O) + t
        return torch.stack([n, t, c, o], dim=2)

    def _ca_to_backbone(self, ca: torch.Tensor) -> torch.Tensor:
        """Build backbone atoms (N, CA, C, O) from CA coordinates using
        differentiable chain geometry.

        Uses the CA-to-CA direction vectors to place N and C atoms at
        ideal bond lengths along the chain direction. This is fully
        differentiable so FAPE gradients flow through to CA predictions.

        Args:
            ca: [B, L, 3] CA coordinates
        Returns:
            backbone: [B, L, 4, 3] (atom order: N=0, CA=1, C=2, O=3)
        """
        B, L, _ = ca.shape
        backbone = torch.zeros(B, L, 4, 3, device=ca.device, dtype=ca.dtype)
        backbone[:, :, 1] = ca  # CA positions

        if L < 2:
            return backbone

        # Direction vectors along the chain
        # d[i] = ca[i+1] - ca[i], normalized
        d = ca[:, 1:] - ca[:, :-1]  # [B, L-1, 3]
        d_norm = torch.sqrt((d ** 2).sum(-1, keepdim=True).clamp(min=1e-8))
        u = d / d_norm  # [B, L-1, 3] unit vectors

        # N atom: 1.46 Å before CA along chain direction
        # For residue i, N_i is placed opposite to the CA_i -> CA_{i+1} direction
        backbone[:, 1:, 0] = ca[:, 1:] - 1.46 * u  # N_{i+1} from direction i
        backbone[:, 0, 0] = ca[:, 0] - 1.46 * u[:, 0]  # N_0 from first direction

        # C atom: 1.52 Å after CA along chain direction
        backbone[:, :-1, 2] = ca[:, :-1] + 1.52 * u  # C_i along direction i
        backbone[:, -1, 2] = ca[:, -1] + 1.52 * u[:, -1]  # C_last from last direction

        # O atom: 1.23 Å from C, perpendicular to chain
        # Create a perpendicular vector by crossing consecutive directions
        perp = torch.zeros_like(ca)
        if L > 2:
            cross = torch.cross(u[:, :-1], u[:, 1:], dim=-1)  # [B, L-2, 3]
            cross_norm = torch.sqrt((cross ** 2).sum(-1, keepdim=True).clamp(min=1e-8))
            cross = cross / cross_norm
            perp[:, 1:-1] = cross
        # For ends, use a default perpendicular
        perp[:, 0] = perp[:, 1] if L > 2 else torch.tensor([0., 1., 0.], device=ca.device)
        perp[:, -1] = perp[:, -2] if L > 2 else torch.tensor([0., 1., 0.], device=ca.device)
        # Where perp is zero, use fallback
        perp_mag = torch.sqrt((perp ** 2).sum(-1, keepdim=True).clamp(min=1e-8))
        fallback = torch.tensor([0., 1., 0.], device=ca.device).expand_as(perp)
        perp = torch.where(perp_mag > 0.1, perp / perp_mag, fallback)

        backbone[:, :, 3] = backbone[:, :, 2] + 1.23 * perp  # O from C + perpendicular

        return backbone

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            s: [B, L, d_single] refined single representation from Evoformer
            z: [B, L, L, d_pair] refined pair representation from Evoformer
            mask: [B, L] bool mask
        Returns:
            ca_coords: [B, L, 3] predicted CA positions
            backbone_coords: [B, L, 4, 3] predicted backbone (N, CA, C, O)
            intermediates: list of [B, L, 4, 3] backbone at each iteration
                           (for auxiliary FAPE losses during training)
        """
        B, L, _ = s.shape

        s = self.ln_input(s)

        # Initialize all residue frames to identity rotation
        R, t = identity_frames(B, L, s.device)
        # Initialize translations along a linear chain (3.8 Å spacing)
        chain_pos = torch.arange(L, device=s.device, dtype=s.dtype).unsqueeze(0).expand(B, -1)
        t = t.clone()
        t[..., 0] = chain_pos * 3.8

        intermediates = []

        for block_idx in range(self.n_blocks):
            # IPA attention + residual connection
            s = s + self.ipa_dropout(self.ipa(s, z, R, t, mask))
            s = self.ipa_ln(s)

            # Transition MLP + residual
            s = s + self.transition(s)

            # Frame update: compose current frame with predicted delta
            # CRITICAL FIX: Only detach on early iterations, NOT the last one.
            # This allows FAPE gradients to flow through to BackboneUpdate
            # on the final iteration while keeping training stable.
            quat, delta_t = self.bb_update(s)
            delta_R = quat_to_rot(quat)
            if block_idx < self.n_blocks - 1:
                R, t = compose_frames(R.detach(), t.detach(), delta_R, delta_t)
            else:
                # Last iteration: let gradients flow through!
                R, t = compose_frames(R, t, delta_R, delta_t)

            # Mask: reset padded residues to identity frame
            if mask is not None:
                mask_f = mask.unsqueeze(-1).float()
                t = t * mask_f
                mask_4d = mask[:, :, None, None].float()
                eye = torch.eye(3, device=R.device)
                R = R * mask_4d + eye * (1.0 - mask_4d)

            # Store intermediate backbone for auxiliary losses
            intermediates.append(self._frames_to_backbone(R, t))

        # Direct coordinate prediction (bypasses frame composition entirely)
        # This head has a clean gradient path: s -> coord_head -> ca_direct
        ca_direct = self.coord_head(s)  # [B, L, 3]

        # Add chain position bias so the direct head starts from a sensible place
        chain_bias = torch.zeros(B, L, 3, device=s.device, dtype=s.dtype)
        chain_bias[..., 0] = chain_pos * 3.8
        ca_direct = ca_direct + chain_bias

        # Use direct CA coordinates as the primary output
        ca_coords = ca_direct

        # Build geometrically consistent backbone from direct CA predictions
        # This ensures FAPE's construct_local_frames gets proper N, CA, C atoms
        backbone_coords = self._ca_to_backbone(ca_coords)

        if mask is not None:
            ca_coords = ca_coords * mask.unsqueeze(-1).float()
            backbone_coords = backbone_coords * mask[:, :, None, None].float()

        return ca_coords, backbone_coords, intermediates

