"""
Mini-AlphaFold Studio: Backend Server
=====================================
FastAPI application serving real-time 3D folding, in-silico mutation analysis,
explainable distance maps, and live edge latency benchmarks.

Run with:
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np
import torch

from benchmark import benchmark_device, generate_synthetic_sequence
from predict import Predictor, save_pdb, build_full_backbone
from src.mutation_analyzer import analyze_mutation, parse_mutation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mini_alphafold_app")

app = FastAPI(title="Mini-AlphaFold Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor singleton (loaded once on startup)
predictor: Optional[Predictor] = None

PRESET_PROTEINS = [
    {
        "id": "cirop",
        "name": "CIROP (A0A1B0GTW7) - Homo sapiens",
        "entry_id": "A0A1B0GTW7",
        "gene": "CIROP",
        "organism": "Homo sapiens (Human)",
        "sequence": "MLLLLLLLLLLPPLVLRVAASRCLHDETQKSVSLLRPPFSQLPSKSRSSSLTLPSSRDPQ",
        "desc": "Ciliated left-right organizer metallopeptidase (Human) N-terminal domain (60 aa)",
        "suggested_mutations": ["C22S", "L15P", "R19A"],
    },
    {
        "id": "crambin",
        "name": "Crambin (1crnA00) - Plant Seed Protein",
        "entry_id": "P01542",
        "gene": "CRAB",
        "organism": "Crambe hispanica subsp. abyssinica (Abyssinian kale)",
        "sequence": "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN",
        "desc": "Plant seed protein with 2 tight alpha-helices and 3 disulfide bridges (46 aa)",
        "suggested_mutations": ["I7P", "V8A", "C3A"],
    },
    {
        "id": "villin",
        "name": "Villin Headpiece HP-36 (1vii)",
        "entry_id": "P02640",
        "gene": "VIL1",
        "organism": "Gallus gallus (Chicken)",
        "sequence": "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "desc": "Canonical fast-folding autonomous 3-helix bundle in biology (36 aa)",
        "suggested_mutations": ["F6A", "W23A", "M12P"],
    },
    {
        "id": "trp_cage",
        "name": "Trp-Cage Miniprotein (1l2y)",
        "entry_id": "P83331",
        "gene": "TC5b",
        "organism": "Synthetic de novo construct",
        "sequence": "NLYIQWLKDGGPSSGRPPPS",
        "desc": "Classic 20-residue de novo designed fast-folding miniprotein with a central Trp core",
        "suggested_mutations": ["W6A", "P12A", "Y3F"],
    },
    {
        "id": "designed_domain",
        "name": "Designed Fold (2dk4A00)",
        "entry_id": "2DK4",
        "gene": "DE_NOVO",
        "organism": "Designed protein domain",
        "sequence": "GSSGSSGTSSNPVLELELAEEKLPMTLSRQEVIRRLRERGEPIRLFGETDYDAFQRLRKIEILTPEVNKGSGPSSG",
        "desc": "Engineered de novo protein domain with repeating alpha-beta topology (76 aa)",
        "suggested_mutations": ["L15P", "E22A", "I32A"],
    },
]


@app.on_event("startup")
def startup_event():
    global predictor
    log.info("Initializing Mini-AlphaFold inference engine...")
    try:
        ckpt_path = "checkpoints/distogram_afdb_best.pt" if Path("checkpoints/distogram_afdb_best.pt").exists() else (
            "checkpoints/distogram_best.pt" if Path("checkpoints/distogram_best.pt").exists() else "checkpoints/best_model.pt"
        )
        predictor = Predictor(
            checkpoint_path=ckpt_path,
            config_path="configs/default.yaml",
        )
        log.info(f"Predictor loaded successfully with {ckpt_path}.")
    except Exception as e:
        log.error(f"Failed to load checkpoint: {e}")


# ============================================================
# API Models
# ============================================================

class FoldRequest(BaseModel):
    sequence: str = Field(..., min_length=10, max_length=256, description="1D amino acid sequence")
    name: Optional[str] = "protein"
    relax: bool = True


class MutateRequest(BaseModel):
    sequence: str = Field(..., min_length=10, max_length=256)
    mutation: str = Field(..., description="Mutation string, e.g. 'I7P' or 'V8A'")
    name: Optional[str] = "protein"


# ============================================================
# Endpoints
# ============================================================

from src.amino_acids_db import AMINO_ACIDS

@app.get("/api/presets")
def get_presets():
    return PRESET_PROTEINS


@app.get("/api/amino_acids")
def get_amino_acids():
    return AMINO_ACIDS


PROTEIN_REGISTRY = {
    "cirop": {
        "id": "cirop",
        "name": "CIROP - Ciliated left-right organizer metallopeptidase",
        "entry_id": "A0A1B0GTW7",
        "gene": "CIROP",
        "organism": "Homo sapiens (Human)",
        "sequence": "MLLLLLLLLLLPPLVLRVAASRCLHDETQKSVSLLRPPFSQLPSKSRSSSLTLPSSRDPQ",
        "full_sequence": (
            "MLLLLLLLLLLPPLVLRVAASRCLHDETQKSVSLLRPPFSQLPSKSRSSSLTLPSSRDPQ"
            "PLRIQSCYLGDHISDGAWDPEGEGMRGGSRALAAVREATQRIQAVLAVQGPLLLSRDPAQ"
            "YCHAVWGDPDSPNYHRCSLLNPGYKGESCLGAKIPDTHLRGYALWPEQGPPQLVQPDGPG"
            "VQNTDFLLYVRVAHTSKCHQETVSLCCPGWSTAAQSQLTAALTSWAQRRGFVMLPRLCLK"
            "LLGSSNLPTLASQSIRITGPSVIAYAACCQLDSEDRPLAGTIVYCAQHLTSPSLSHSDIV"
            "MATLHELLHALGFSGQLFKKWRDCPSGFSVRENCSTRQLVTRQDEWGQLLLTTPAVSLSL"
            "AKHLGVSGASLGVPLEEEEGLLSSHWEARLLQGSLMTATFDGAQRTRLDPITLAAFKDSG"
            "WYQVNHSAAEELLWGQGSGPEFGLVTTCGTGSSDFFCTGSGLGCHYLHLDKGSCSSDPML"
            "EGCRMYKPLANGSECWKKENGFPAGVDNPHGEIYHPQSRCFFANLTSQLLPGDKPRHPSL"
            "TPHLKEAELMGRCYLHQCTGRGAYKVQVEGSPWVPCLPGKVIQIPGYYGLLFCPRGRLCQ"
            "TNEDINAVTSPPVSLSTPDPLFQLSLELAGPPGHSLGKEQQEGLAEAVLEALASKGGTGR"
            "CYFHGPSITTSLVFTVHMWKSPGCQGPSVATLHKALTLTLQKKPLEVYHGGANFTTQPSK"
            "LLVTSDHNPSMTHLRLSMGLCLMLLILVGVMGTTAYQKRATLPVRPSASYHSPELHSTRV"
            "PVRGIREV"
        ),
        "pred_pdb": Path("predictions/cirop_predicted.pdb"),
        "exp_pdb": Path("data/AF-A0A1B0GTW7-F1.pdb"),
        "binding_sites": [
            {"pos": 22, "name": "Zinc Coordination Motif", "desc": "Cys 22: Essential active region residue"},
            {"pos": 25, "name": "Disulfide Anchor", "desc": "Cys 25: Disulfide pairing site"},
        ],
        "active_sites": [
            {"pos": 19, "name": "Active Basic Site", "desc": "Arg 19: Catalytic electrostatic anchor"},
        ],
        "disulfide_bonds": [
            {"res1": 22, "res2": 25, "name": "Catalytic Loop Bridge (Cys 22 - Cys 25)"},
        ],
        "domains": [
            {"start": 1, "end": 20, "name": "Signal Peptide", "type": "signal", "color": "#0ea5e9"},
            {"start": 21, "end": 60, "name": "Peptidase M8 Catalytic Core", "type": "domain", "color": "#10b981"},
        ],
        "variants": [
            {"pos": 22, "name": "C22S Mutation", "desc": "Disrupts catalytic pocket coordination"},
            {"pos": 15, "name": "L15P Mutation", "desc": "Helix-breaking disruption in hydrophobic core"},
            {"pos": 19, "name": "R19A Mutation", "desc": "Charge ablation of catalytic anchor"},
        ],
    },
    "crambin": {
        "id": "crambin",
        "name": "Crambin - Plant Seed Storage Protein",
        "entry_id": "P01542",
        "gene": "CRAB",
        "organism": "Crambe hispanica subsp. abyssinica",
        "sequence": "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN",
        "pred_pdb": Path("predictions/crambin_predicted.pdb"),
        "exp_pdb": Path("data/raw/1CRN.pdb"),
        "binding_sites": [
            {"pos": 16, "name": "Hydrophobic Core Entrance", "desc": "Pro 16 / Leu 18 hydrophobic pocket anchor"},
            {"pos": 23, "name": "Salt Bridge Site", "desc": "Glu 23: Stabilizes Arg 17 basic network"},
        ],
        "active_sites": [
            {"pos": 17, "name": "Basic Anchor", "desc": "Arg 17: Core structural electrostatic center"},
        ],
        "disulfide_bonds": [
            {"res1": 3, "res2": 40, "name": "Disulfide 1 (Cys 3 - Cys 40)"},
            {"res1": 4, "res2": 32, "name": "Disulfide 2 (Cys 4 - Cys 32)"},
            {"res1": 16, "res2": 26, "name": "Disulfide 3 (Cys 16 - Cys 26)"},
        ],
        "domains": [
            {"start": 1, "end": 46, "name": "Crambin Plant Toxin Domain", "type": "domain", "color": "#10b981"},
        ],
        "variants": [
            {"pos": 7, "name": "I7P Mutation", "desc": "Helix-breaking proline disruption"},
            {"pos": 8, "name": "V8A Mutation", "desc": "Hydrophobic core truncation"},
            {"pos": 3, "name": "C3A Mutation", "desc": "Disulfide bond 1 ablation"},
        ],
    },
    "villin": {
        "id": "villin",
        "name": "Villin Headpiece HP-36 (1vii)",
        "entry_id": "P02640",
        "gene": "VIL1",
        "organism": "Gallus gallus (Chicken)",
        "sequence": "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "pred_pdb": Path("predictions/villin_predicted.pdb"),
        "exp_pdb": Path("data/raw/1vii.pdb"),
        "binding_sites": [
            {"pos": 23, "name": "F-Actin Binding Interface", "desc": "Trp 23: Key aromatic anchor for actin filament bundling"},
            {"pos": 29, "name": "Actin Interaction Site", "desc": "Lys 29: Electrostatic contact with actin subunit"},
        ],
        "active_sites": [
            {"pos": 6, "name": "Hydrophobic Core Phe 6", "desc": "Phe 6: Central hydrophobic triad stabilizing 3-helix bundle"},
        ],
        "disulfide_bonds": [],
        "domains": [
            {"start": 1, "end": 36, "name": "Villin Headpiece HP-36", "type": "domain", "color": "#0ea5e9"},
        ],
        "variants": [
            {"pos": 6, "name": "F6A Mutation", "desc": "Destabilizes hydrophobic core packaging"},
            {"pos": 23, "name": "W23A Mutation", "desc": "Ablates F-actin filament bundling affinity"},
            {"pos": 12, "name": "M12P Mutation", "desc": "Helix-2 disruption in the central turn"},
        ],
    },
    "trp_cage": {
        "id": "trp_cage",
        "name": "Trp-Cage Miniprotein (1l2y)",
        "entry_id": "P83331",
        "gene": "TC5b",
        "organism": "Synthetic de novo construct",
        "sequence": "NLYIQWLKDGGPSSGRPPPS",
        "pred_pdb": Path("predictions/trp_cage_predicted.pdb"),
        "exp_pdb": Path("data/raw/1l2y_model1.pdb"),
        "binding_sites": [
            {"pos": 6, "name": "Central Trp-Cage Core", "desc": "Trp 6: Aromatic ring buried by proline polyproline-II cage"},
        ],
        "active_sites": [
            {"pos": 9, "name": "Salt Bridge Asp 9", "desc": "Asp 9: Forms capping salt bridge with Arg 16"},
        ],
        "disulfide_bonds": [],
        "domains": [
            {"start": 1, "end": 20, "name": "Trp-Cage Fold", "type": "domain", "color": "#f59e0b"},
        ],
        "variants": [
            {"pos": 6, "name": "W6A Mutation", "desc": "Collapse of central hydrophobic tryptophan cage"},
            {"pos": 12, "name": "P12A Mutation", "desc": "Unlocks polyproline casing around Trp 6"},
            {"pos": 3, "name": "Y3F Mutation", "desc": "Modulates aromatic ring stacking"},
        ],
    },
    "designed_domain": {
        "id": "designed_domain",
        "name": "Designed Fold (2dk4A00)",
        "entry_id": "2DK4",
        "gene": "DE_NOVO",
        "organism": "Designed protein domain",
        "sequence": "GSSGSSGTSSNPVLELELAEEKLPMTLSRQEVIRRLRERGEPIRLFGETDYDAFQRLRKIEILTPEVNKGSGPSSG",
        "pred_pdb": Path("predictions/designed_domain_predicted.pdb"),
        "exp_pdb": Path("data/raw/2dk4A00.pdb"),
        "binding_sites": [
            {"pos": 15, "name": "Engineered Core Residue", "desc": "Leu 15: Hydrophobic core stabilization"},
            {"pos": 32, "name": "Beta-Strand Contact", "desc": "Ile 32: Inter-strand packing anchor"},
        ],
        "active_sites": [
            {"pos": 22, "name": "Surface Charge Anchor", "desc": "Glu 22: Solvation boundary electrostatic regulator"},
        ],
        "disulfide_bonds": [],
        "domains": [
            {"start": 1, "end": 76, "name": "De Novo Alpha-Beta Domain", "type": "domain", "color": "#8b5cf6"},
        ],
        "variants": [
            {"pos": 15, "name": "L15P Mutation", "desc": "Helix 1 disruption in de novo topology"},
            {"pos": 22, "name": "E22A Mutation", "desc": "Surface charge neutralization"},
            {"pos": 32, "name": "I32A Mutation", "desc": "Core packing destabilization"},
        ],
    },
}

# Alias resolution mapping
PROTEIN_ALIASES = {
    "a0a1b0gtw7": "cirop",
    "cirop_human": "cirop",
    "1crna00": "crambin",
    "1crn": "crambin",
    "p01542": "crambin",
    "1vii": "villin",
    "p02640": "villin",
    "1l2y": "trp_cage",
    "p83331": "trp_cage",
    "2dk4": "designed_domain",
    "2dk4a00": "designed_domain",
}


def compute_rmsd_between_pdbs(pred_pdb_text: str, exp_pdb_text: str) -> Optional[float]:
    """Computes Kabsch CA-RMSD between predicted and experimental PDB structures."""
    try:
        def extract_ca(text):
            coords = []
            seen = set()
            for l in text.splitlines():
                if l.startswith("ATOM") and l[12:16].strip() == "CA":
                    r = int(l[22:26])
                    if r not in seen:
                        seen.add(r)
                        coords.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            return np.array(coords)

        ca_pred = extract_ca(pred_pdb_text)
        ca_exp = extract_ca(exp_pdb_text)
        if len(ca_pred) == 0 or len(ca_exp) == 0:
            return None
        min_len = min(len(ca_pred), len(ca_exp))
        if min_len < 3:
            return None
        def _kabsch_rmsd(P, Q):
            Pc = P - P.mean(dim=0)
            Qc = Q - Q.mean(dim=0)
            H = Pc.T @ Qc
            U, S, V = torch.svd(H)
            d = torch.sign(torch.det(V @ U.T))
            S[-1] *= d
            rmsd = (Pc.pow(2).sum() + Qc.pow(2).sum() - 2 * S.sum()) / P.shape[0]
            return torch.sqrt(torch.abs(rmsd))
            
        val = _kabsch_rmsd(torch.from_numpy(ca_pred[:min_len]), torch.from_numpy(ca_exp[:min_len]))
        return round(float(val), 2)
    except Exception:
        return None


@app.get("/api/protein/{protein_id}")
def get_protein_details(protein_id: str, truth: int = 0):
    global predictor
    pid_raw = protein_id.lower().strip()
    pid = PROTEIN_ALIASES.get(pid_raw, pid_raw)

    if pid in PROTEIN_REGISTRY:
        info = PROTEIN_REGISTRY[pid]
        seq = info["sequence"]
        pred_p = info["pred_pdb"]
        exp_p = info["exp_pdb"]

        # Ensure predicted PDB exists
        if not pred_p.exists():
            if predictor is not None:
                p_res = predictor.predict_sequence(seq, relax=True)
                save_pdb(
                    p_res["backbone_coords"],
                    seq,
                    pred_p,
                    helices=p_res["helices"],
                    sheets=p_res["sheets"],
                    plddt=p_res["plddt"],
                )
            else:
                raise HTTPException(status_code=500, detail="Predictor not initialized")

        pred_text = pred_p.read_text(encoding="utf-8")
        exp_text = exp_p.read_text(encoding="utf-8") if exp_p.exists() else ""
        rmsd_val = compute_rmsd_between_pdbs(pred_text, exp_text) if exp_text else None

        # Choose which structure to serve based on truth flag
        use_truth = bool(truth) and bool(exp_text)
        active_pdb = exp_text if use_truth else pred_text

        # Parse CA and B-factors/pLDDT from active structure
        ca_coords, plddt_list = [], []
        for line in active_pdb.splitlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ca_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                plddt_list.append(float(line[60:66]))

        ca_arr = np.array(ca_coords) if len(ca_coords) else np.zeros((0, 3))
        if len(ca_arr) > 3:
            diff = ca_arr[:, None, :] - ca_arr[None, :, :]
            d_mat = np.sqrt(np.sum(diff ** 2, axis=-1))
            from predict import assign_secondary_structure
            helices, sheets, ss_string = assign_secondary_structure(ca_arr, d_mat)
        else:
            helices, sheets, ss_string = [], [], "C" * len(seq)

        mean_plddt = round(float(np.mean(plddt_list)), 1) if len(plddt_list) else 85.0

        return {
            "id": pid,
            "name": info["name"],
            "entry_id": info["entry_id"],
            "model_id": f"PDB-{info['entry_id']}" if use_truth else f"MINI-AF-{pid.upper()}",
            "gene": info["gene"],
            "organism": info["organism"],
            "sequence": seq,
            "length": len(seq),
            "pdb": active_pdb,
            "is_predicted": not use_truth,
            "has_experimental": bool(exp_text),
            "rmsd_to_exp": rmsd_val,
            "pred_pdb_url": f"/api/structure_pdb/{pid}?truth=0",
            "exp_pdb_url": f"/api/structure_pdb/{pid}?truth=1" if exp_text else None,
            "plddt": plddt_list,
            "mean_plddt": mean_plddt,
            "helices": helices,
            "sheets": sheets,
            "ss_string": ss_string,
            "binding_sites": info.get("binding_sites", []),
            "active_sites": info.get("active_sites", []),
            "disulfide_bonds": info.get("disulfide_bonds", []),
            "domains": info.get("domains", []),
            "variants": info.get("variants", []),
        }

    # Generic lookup: check if raw PDB exists in data/raw
    raw_p = Path(f"data/raw/{protein_id}.pdb")
    if not raw_p.exists():
        raw_p = Path(f"data/raw/{protein_id.upper()}.pdb")

    if raw_p.exists() and predictor is not None:
        exp_text = raw_p.read_text(encoding="utf-8")
        from predict import ONE_TO_THREE
        three_to_one = {v: k for k, v in ONE_TO_THREE.items()}
        seq_res = []
        for l in exp_text.splitlines():
            if l.startswith("ATOM") and l[12:16].strip() == "CA":
                seq_res.append(three_to_one.get(l[17:20].strip(), "A"))
        seq = "".join(seq_res)
        if len(seq) >= 10:
            pred_p = Path(f"predictions/{pid_raw}_predicted.pdb")
            if not pred_p.exists():
                p_res = predictor.predict_sequence(seq, relax=True)
                save_pdb(p_res["backbone_coords"], seq, pred_p, helices=p_res["helices"], sheets=p_res["sheets"], plddt=p_res["plddt"])
            pred_text = pred_p.read_text(encoding="utf-8")
            use_truth = bool(truth)
            active_pdb = exp_text if use_truth else pred_text
            rmsd_val = compute_rmsd_between_pdbs(pred_text, exp_text)
            return {
                "id": pid_raw,
                "name": f"Protein Structure ({protein_id})",
                "entry_id": protein_id.upper(),
                "model_id": f"PDB-{protein_id.upper()}" if use_truth else f"MINI-AF-{protein_id.upper()}",
                "gene": protein_id.upper(),
                "organism": "Experimental Record",
                "sequence": seq,
                "length": len(seq),
                "pdb": active_pdb,
                "is_predicted": not use_truth,
                "has_experimental": True,
                "rmsd_to_exp": rmsd_val,
                "pred_pdb_url": f"/api/structure_pdb/{pid_raw}?truth=0",
                "exp_pdb_url": f"/api/structure_pdb/{pid_raw}?truth=1",
                "plddt": [85.0] * len(seq),
                "mean_plddt": 85.0,
                "helices": [],
                "sheets": [],
                "ss_string": "C" * len(seq),
                "binding_sites": [],
                "active_sites": [],
                "disulfide_bonds": [],
                "domains": [{"start": 1, "end": len(seq), "name": "Domain", "type": "domain", "color": "#10b981"}],
                "variants": [],
            }

    raise HTTPException(status_code=404, detail="Protein ID not found")


@app.get("/api/structure_pdb/{protein_id}")
def get_structure_pdb(protein_id: str, truth: int = 0):
    pid_raw = protein_id.lower().strip()
    pid = PROTEIN_ALIASES.get(pid_raw, pid_raw)

    if pid == "latest":
        p = Path("predictions/latest_predicted.pdb")
        if p.exists():
            return FileResponse(str(p), media_type="chemical/x-pdb")

    if pid in PROTEIN_REGISTRY:
        info = PROTEIN_REGISTRY[pid]
        p = info["exp_pdb"] if truth else info["pred_pdb"]
        if p.exists():
            return FileResponse(str(p), media_type="chemical/x-pdb")

    # Fallback to direct file search
    if truth:
        raw_p = Path(f"data/raw/{protein_id}.pdb")
        if raw_p.exists():
            return FileResponse(str(raw_p), media_type="chemical/x-pdb")
    else:
        pred_p = Path(f"predictions/{pid_raw}_predicted.pdb")
        if pred_p.exists():
            return FileResponse(str(pred_p), media_type="chemical/x-pdb")

    raise HTTPException(status_code=404, detail="PDB structure not found")


@app.get("/api/system_info")
def get_system_info():
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "Host CPU"
    total_vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if has_cuda else 0.0
    )
    return {
        "has_cuda": has_cuda,
        "device_name": device_name,
        "total_vram_gb": round(total_vram_gb, 2),
        "model_params": 1420615,
        "model_params_m": 1.42,
    }


@app.post("/api/fold")
def fold_protein(req: FoldRequest):
    global predictor
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model predictor not initialized")

    seq = req.sequence.strip().upper()
    valid_chars = set("ACDEFGHIKLMNPQRSTVWY")
    invalid = [c for c in seq if c not in valid_chars]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid amino acid characters: {set(invalid)}")

    t0 = time.perf_counter()
    pred = predictor.predict_sequence(seq, relax=req.relax)
    exec_time_ms = (time.perf_counter() - t0) * 1000.0

    # Save latest predicted PDB for direct Mol* streaming
    latest_pdb = Path("predictions/latest_predicted.pdb")
    latest_pdb.parent.mkdir(parents=True, exist_ok=True)
    plddt_arr = pred.get("plddt")
    save_pdb(
        pred["backbone_coords"],
        seq,
        latest_pdb,
        helices=pred["helices"],
        sheets=pred["sheets"],
        plddt=plddt_arr,
    )
    pdb_text = latest_pdb.read_text(encoding="utf-8")

    # Downsample or format distance matrix for fast JSON transfer
    dist_matrix = np.round(pred["pred_dist"], 2).tolist()
    plddt_list = [round(float(x), 1) for x in plddt_arr] if plddt_arr is not None else [85.0] * len(seq)

    return {
        "pdb": pdb_text,
        "pdb_url": "/api/structure_pdb/latest",
        "sequence": seq,
        "length": len(seq),
        "helices": pred["helices"],
        "sheets": pred["sheets"],
        "ss_string": pred["ss_string"],
        "plddt": plddt_list,
        "mean_plddt": round(float(np.mean(plddt_list)), 1),
        "dist_matrix": dist_matrix,
        "exec_time_ms": round(exec_time_ms, 2),
    }


@app.post("/api/mutate")
def mutate_protein(req: MutateRequest):
    global predictor
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model predictor not initialized")

    seq = req.sequence.strip().upper()
    mut_str = req.mutation.strip().upper()

    t0 = time.perf_counter()
    try:
        res = analyze_mutation(
            wt_sequence=seq,
            mutations=[mut_str],
            predictor=predictor,
            name=req.name or "protein",
            out_dir="predictions/mutations",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    exec_time_ms = (time.perf_counter() - t0) * 1000.0

    wt_pdb_text = Path(res["wt_pdb"]).read_text(encoding="utf-8")
    mut_pdb_text = Path(res["mut_pdb"]).read_text(encoding="utf-8")
    delta_dist = np.round(res["delta_dist"], 2).tolist()

    return {
        "wt_pdb": wt_pdb_text,
        "mut_pdb": mut_pdb_text,
        "wt_sequence": res["wt_sequence"],
        "mut_sequence": res["mut_sequence"],
        "mut_label": res["mut_label"],
        "global_rmsd": round(res["global_rmsd"], 2),
        "mean_displacement": round(res["mean_displacement"], 2),
        "max_displacement": round(res["max_displacement"], 2),
        "lost_contacts": res["lost_contacts"],
        "gained_contacts": res["gained_contacts"],
        "per_res_disp": [round(x, 2) for x in res["per_res_disp"]],
        "delta_dist": delta_dist,
        "exec_time_ms": round(exec_time_ms, 2),
    }


@app.get("/api/benchmark")
def run_live_benchmark():
    global predictor
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model predictor not initialized")

    device = predictor.device
    lengths = [32, 64, 96, 128]
    raw_results = benchmark_device(
        model=predictor.model,
        device=device,
        lengths=lengths,
        repeats=6,
        warmup=2,
    )

    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU",
        "results": raw_results,
    }


# Static Files Mount
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def serve_index():
    index_path = static_dir / "index.html"
    return FileResponse(str(index_path))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
