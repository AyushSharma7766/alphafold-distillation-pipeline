/**
 * Mini-AlphaFold Studio: UniProtKB ProtVista & Mol* 3D Studio
 * Features:
 *  - Screen Recording 2 Replica: ProtVista 100% Width Responsive Sequence & Feature Viewer
 *    (Draggable overview brush [ < 300 --- 329 > ], detail track, UniProt Features Table)
 *  - Screen Recording 1 Replica: Mol* 3D Viewer with Ribbons, Tubes, and Sheets
 *    (Ball-and-stick atom spheres + sticks, neon yellow halo, dashed H-bonds, Mol* info box)
 *  - 20 Canonical Amino Acids 3D Explorer with dedicated ball-and-stick 3D viewer & biochemical encyclopedia
 */

const THREE_TO_FULL = {
  ALA: "Alanine", CYS: "Cysteine", ASP: "Aspartate", GLU: "Glutamate",
  PHE: "Phenylalanine", GLY: "Glycine", HIS: "Histidine", ILE: "Isoleucine",
  LYS: "Lysine", LEU: "Leucine", MET: "Methionine", ASN: "Asparagine",
  PRO: "Proline", GLN: "Glutamine", ARG: "Arginine", SER: "Serine",
  THR: "Threonine", VAL: "Valine", TRP: "Tryptophan", TYR: "Tyrosine",
};

let viewer3D = null;
let viewerWT = null;
let viewerMut = null;
let viewerAA = null;
let pdbeViewer = null;
let molstarInitialized = false;
let currentPDB = "";
let currentProteinId = "cirop";
let currentTruthMode = 0; // 0 = Mini-AlphaFold Predicted 3D, 1 = Experimental Ground Truth
let currentDistMatrix = null;
let currentData = null;
let currentColorScheme = "plddt"; // 'plddt', 'ss', 'spectrum'
let isSpinning = false;
let isAASpinning = true;
let dispChart = null;
let selectedResidue = 306; // Catalytic active site Glu 306 (or 222)
let parsedPDBAtoms = [];

// ProtVista active window (default 300 to 330, matching Video 2!)
let pvWindowStart = 300;
let pvWindowEnd = 330;
let isDraggingBrush = false;
let isDraggingLeftHandle = false;
let isDraggingRightHandle = false;
let dragStartX = 0;
let dragStartWindowStart = 0;
let dragStartWindowEnd = 0;

// 20 Amino Acids Database
let allAminoAcids = [];
let selectedAminoAcid = null;
let currentAAFilter = "all";

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initThemeToggle();
  initSystemInfo();
  initPresets();
  init3DViewer();
  initProtVistaBrush();
  initAminoAcidsExplorer();
  initEventListeners();
});

// ============================================================
// 1. Navigation & Theme
// ============================================================
function initTabs() {
  const tabBtns = document.querySelectorAll(".uniprot-tab-btn");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
        if (targetId === "tab-structure") {
          if (viewer3D) viewer3D.render();
        } else if (targetId === "tab-explain" && currentDistMatrix) {
          renderDistanceHeatmap(currentDistMatrix);
        } else if (targetId === "tab-aminoacids") {
          onShowAminoAcidsTab();
        }
      }
    });
  });
}

function initThemeToggle() {
  const btn = document.getElementById("btn-theme-toggle");
  btn.addEventListener("click", () => {
    document.body.classList.toggle("theme-dark");
    const isDark = document.body.classList.contains("theme-dark");
    btn.textContent = isDark ? "Theme: Dark Studio" : "Theme: UniProt Light";
    const bgCol = isDark ? "#0d121f" : "#ffffff";
    if (pdbeViewer && pdbeViewer.canvas) {
      pdbeViewer.canvas.setBgColor(isDark ? { r: 13, g: 18, b: 31 } : { r: 255, g: 255, b: 255 });
    }
    if (viewer3D) {
      viewer3D.setBackgroundColor(bgCol);
      viewer3D.render();
    }
    if (viewerAA) {
      viewerAA.setBackgroundColor(bgCol);
      viewerAA.render();
    }
  });
}

async function initSystemInfo() {
  try {
    const res = await fetch("/api/system_info");
    const data = await res.json();
    const pill = document.getElementById("hw-info");
    if (data.has_cuda) {
      pill.textContent = `${data.device_name} (${data.total_vram_gb} GB) | 1.42M Params`;
    } else {
      pill.textContent = "CPU Mode | 1.42M Params";
    }
  } catch (e) {
    console.error("System info error:", e);
  }
}

// ============================================================
// 2. Presets Management (CIROP, Crambin, Villin, etc.)
// ============================================================
async function initPresets() {
  try {
    const res = await fetch("/api/presets");
    const presets = await res.json();
    const container = document.getElementById("presets-container");
    container.innerHTML = "";

    presets.forEach((p, idx) => {
      const card = document.createElement("div");
      card.className = `preset-card-item ${idx === 0 ? "active" : ""}`;
      card.innerHTML = `
        <div class="preset-item-title">${p.name}</div>
        <div class="preset-item-desc">${p.desc}</div>
      `;
      card.addEventListener("click", () => {
        document.querySelectorAll(".preset-card-item").forEach((c) => c.classList.remove("active"));
        card.classList.add("active");
        loadPreset(p);
      });
      container.appendChild(card);

      if (idx === 0) {
        loadPreset(p);
      }
    });
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

async function loadPreset(preset) {
  // Update Header Banner
  document.getElementById("entry-acc-breadcrumb").textContent = preset.entry_id || preset.id;
  document.getElementById("entry-id-breadcrumb").textContent = preset.gene ? `${preset.gene}_HUMAN` : preset.name;
  document.getElementById("entry-acc").textContent = preset.entry_id || preset.id;
  document.getElementById("entry-name").textContent = preset.name;
  document.getElementById("entry-organism").textContent = preset.organism || "Homo sapiens (Human)";
  document.getElementById("entry-gene").textContent = preset.gene || "CIROP";
  document.getElementById("seq-custom-input").value = preset.sequence;
  document.getElementById("mut-base-seq").value = preset.sequence;

  if (preset.suggested_mutations && preset.suggested_mutations.length > 0) {
    document.getElementById("mut-code").value = preset.suggested_mutations[0];
  }

  const pid = (preset.id || "").toLowerCase();
  await loadPreindexedProtein(pid, 0);
}

function applySmoothTrackball() {
  if (pdbeViewer && pdbeViewer.plugin && pdbeViewer.plugin.canvas3d) {
    try {
      pdbeViewer.plugin.canvas3d.setProps({
        trackball: {
          staticMoving: false,
          dynamicDampingFactor: 0.12,
          zoomSpeed: 2.2,
          rotateSpeed: 3.5,
          panSpeed: 1.0,
        },
      });
    } catch (e) {
      console.warn("Trackball smooth setProps error:", e);
    }
  }
}

async function loadPreindexedProtein(proteinId, truth = 0) {
  const btn = document.getElementById("btn-fold-custom");
  const label = document.getElementById("btn-fold-label");
  btn.disabled = true;
  label.innerHTML = `<span class="uniprot-spinner"></span> Loading Structure...`;

  currentProteinId = proteinId;
  currentTruthMode = truth;

  try {
    const res = await fetch(`/api/protein/${proteinId}?truth=${truth}`);
    if (!res.ok) throw new Error("Failed to load protein data");
    const data = await res.json();

    // Toggle active styling on predicted vs experimental buttons
    const btnPred = document.getElementById("btn-show-predicted");
    const btnExp = document.getElementById("btn-show-experimental");
    if (btnPred) btnPred.classList.toggle("active", truth === 0);
    if (btnExp) {
      btnExp.classList.toggle("active", truth === 1);
      btnExp.style.display = data.has_experimental ? "inline-flex" : "none";
    }

    currentData = data;
    currentPDB = data.pdb;
    parsePDB(data.pdb);

    // Update length & name in banner
    document.getElementById("entry-length").textContent = data.length;
    document.getElementById("entry-acc").textContent = data.entry_id || "A0A1B0GTW7";
    document.getElementById("entry-name").textContent = data.name;
    document.getElementById("pv-overview-end").textContent = data.length;

    // Model indicator banner tag
    const modelTag = document.getElementById("entry-model-tag");
    if (modelTag) {
      modelTag.textContent = truth === 1 ? "PDB Experimental Record" : "Mini-AlphaFold 3D Prediction";
    }

    // RMSD badge
    const rmsdBadge = document.getElementById("rmsd-badge");
    const rmsdVal = document.getElementById("rmsd-val");
    if (rmsdBadge && rmsdVal) {
      if (data.has_experimental && data.rmsd_to_exp != null) {
        rmsdVal.textContent = data.rmsd_to_exp;
        rmsdBadge.style.display = "inline-flex";
      } else {
        rmsdBadge.style.display = "none";
      }
    }

    // Load PDB into Mol* (or 3Dmol fallback)
    const el = document.getElementById("viewport-3d");
    const pdbUrl = `/api/structure_pdb/${proteinId}?truth=${truth}&t=${Date.now()}`;
    let molstarLoaded = false;
    if (pdbeViewer) {
      try {
        const opts = {
          customData: { url: pdbUrl, format: "pdb" },
          alphafoldView: true,
          bgColor: { r: 255, g: 255, b: 255 },
          hideControls: true,
          subscribeEvents: true,
        };
        if (!molstarInitialized) {
          molstarInitialized = true;
          await pdbeViewer.render(el, opts);
        } else {
          await pdbeViewer.visual.update(opts);
        }
        molstarLoaded = true;
        setTimeout(applySmoothTrackball, 150);
        setTimeout(applySmoothTrackball, 600);
      } catch (mErr) {
        console.warn("Mol* render failed, falling back to 3Dmol:", mErr);
      }
    }

    if (!molstarLoaded && typeof $3Dmol !== "undefined" && el) {
      try {
        if (!viewer3D) viewer3D = $3Dmol.createViewer(el, { backgroundColor: "#ffffff" });
        viewer3D.clear();
        viewer3D.addModel(data.pdb, "pdb");
        applyStyling(viewer3D, currentColorScheme);
        viewer3D.zoomTo();
        viewer3D.render();
      } catch (tdErr) {
        console.warn("3Dmol render notice:", tdErr);
      }
    }

    // Stats
    document.getElementById("stat-mean-plddt").textContent = `${data.mean_plddt}`;
    document.getElementById("stat-time").textContent = truth === 1 ? "PDB Experimental" : "AI Inferred (~25 ms)";

    // Initialize ProtVista window
    pvWindowStart = 1;
    pvWindowEnd = Math.min(35, data.length);
    selectedResidue = (data.binding_sites && data.binding_sites.length > 0)
      ? data.binding_sites[0].pos
      : (data.active_sites && data.active_sites.length > 0)
      ? data.active_sites[0].pos
      : 1;

    renderProtVistaOverview();
    renderProtVistaDetail();
    renderUniProtFeaturesTable();

    // Update FASTA
    renderFastaBox(data.sequence, data.ss_string);

    // Initial 3D selection
    try {
      selectResidueIn3D(selectedResidue, null, false);
    } catch (selErr) {
      console.warn("Initial selectResidueIn3D notice:", selErr);
    }
  } catch (err) {
    console.error("Error loading preindexed protein:", err);
  } finally {
    btn.disabled = false;
    label.textContent = "Fold 3D Structure";
  }
}

async function foldSequence(sequence, relax = true) {
  const seq = (sequence || "").trim().toUpperCase();
  if (!seq || seq.length < 10) {
    alert("Sequence must be at least 10 amino acids long.");
    return;
  }
  const btn = document.getElementById("btn-fold-custom");
  const label = document.getElementById("btn-fold-label");
  btn.disabled = true;
  label.innerHTML = `<span class="uniprot-spinner"></span> Predicting 3D Structure...`;

  try {
    const res = await fetch("/api/fold", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence: seq, relax: relax, name: "custom_fold" })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to fold protein sequence");
    }
    const data = await res.json();

    currentData = {
      id: "custom",
      name: "Custom Predicted Protein",
      entry_id: "AF-PRED",
      model_id: "AF-PREDICTED-F1",
      gene: "CUSTOM",
      organism: "Synthetic Construct",
      sequence: data.sequence,
      length: data.length,
      pdb: data.pdb,
      plddt: data.plddt,
      mean_plddt: data.mean_plddt,
      helices: data.helices,
      sheets: data.sheets,
      ss_string: data.ss_string,
      binding_sites: [],
      active_sites: [],
      disulfide_bonds: [],
      domains: [{ start: 1, end: data.length, name: "Predicted Core", type: "domain", color: "#10b981" }],
      variants: []
    };
    currentPDB = data.pdb;
    currentDistMatrix = data.dist_matrix;
    parsePDB(data.pdb);

    // Update Banner
    document.getElementById("entry-acc-breadcrumb").textContent = "AF-PRED";
    document.getElementById("entry-id-breadcrumb").textContent = "CUSTOM_PREDICT";
    document.getElementById("entry-acc").textContent = "AF-PRED";
    document.getElementById("entry-name").textContent = `Predicted Structure (${data.length} aa)`;
    document.getElementById("entry-organism").textContent = "Synthetic Construct";
    document.getElementById("entry-gene").textContent = "CUSTOM";
    document.getElementById("entry-length").textContent = data.length;
    document.getElementById("pv-overview-end").textContent = data.length;

    // Stats
    document.getElementById("stat-mean-plddt").textContent = `${data.mean_plddt}`;
    document.getElementById("stat-time").textContent = `${data.exec_time_ms} ms`;

    // Render ProtVista
    pvWindowStart = 1;
    pvWindowEnd = Math.min(35, data.length);
    selectedResidue = 1;
    renderProtVistaOverview();
    renderProtVistaDetail();
    renderUniProtFeaturesTable();
    renderFastaBox(data.sequence, data.ss_string);

    // Update Distance Map if in explain tab
    renderDistanceHeatmap(data.dist_matrix);

    // Stream into Mol*!
    const el = document.getElementById("viewport-3d");
    const latestUrl = `/api/structure_pdb/latest?t=${Date.now()}`;
    if (pdbeViewer) {
      const opts = {
        customData: { url: latestUrl, format: "pdb" },
        alphafoldView: true,
        bgColor: { r: 255, g: 255, b: 255 },
        hideControls: true,
        subscribeEvents: true
      };
      if (!molstarInitialized) {
        molstarInitialized = true;
        await pdbeViewer.render(el, opts);
      } else {
        await pdbeViewer.visual.update(opts);
      }
      setTimeout(applySmoothTrackball, 100);
      setTimeout(applySmoothTrackball, 500);

      // Mode switch updates
      currentProteinId = "latest";
      const btnPred = document.getElementById("btn-show-predicted");
      const btnExp = document.getElementById("btn-show-experimental");
      if (btnPred) {
        btnPred.classList.add("active");
        btnPred.style.display = "inline-flex";
      }
      if (btnExp) {
        btnExp.classList.remove("active");
        btnExp.style.display = "none";
      }
      const rmsdBadge = document.getElementById("rmsd-badge");
      if (rmsdBadge) rmsdBadge.style.display = "none";
    } else if (typeof $3Dmol !== "undefined") {
      if (!viewer3D) viewer3D = $3Dmol.createViewer(el, { backgroundColor: "#ffffff" });
      viewer3D.clear();
      viewer3D.addModel(data.pdb, "pdb");
      applyStyling(viewer3D, currentColorScheme);
      viewer3D.zoomTo();
      viewer3D.render();
    }

    selectResidueIn3D(1, null, false);
  } catch (err) {
    console.error("Fold error:", err);
    alert(`Prediction Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    label.textContent = "Fold 3D Structure";
  }
}

function parsePDB(pdbText) {
  parsedPDBAtoms = [];
  if (!pdbText) return;
  const lines = pdbText.split("\n");
  for (const line of lines) {
    if (line.startsWith("ATOM")) {
      const atomName = line.substring(12, 16).trim();
      const resn = line.substring(17, 20).trim();
      const chain = line.substring(21, 22).trim();
      const resi = parseInt(line.substring(22, 26).trim(), 10);
      const x = parseFloat(line.substring(30, 38).trim());
      const y = parseFloat(line.substring(38, 46).trim());
      const z = parseFloat(line.substring(46, 54).trim());
      const b = parseFloat(line.substring(60, 66).trim());
      const elem = line.length >= 78 ? line.substring(76, 78).trim() : atomName[0];
      parsedPDBAtoms.push({ atomName, resn, chain, resi, x, y, z, b, elem });
    }
  }
}

// ============================================================
// 3. 3D Structure Viewer & Ball-and-Stick Rendering (Video 1)
// ============================================================
function init3DViewer() {
  const el = document.getElementById("viewport-3d");
  if (!el) return;

  if (typeof PDBeMolstarPlugin !== "undefined") {
    pdbeViewer = new PDBeMolstarPlugin();
  } else if (typeof $3Dmol !== "undefined") {
    viewer3D = $3Dmol.createViewer(el, { backgroundColor: "#ffffff" });
  }

  // Enable smooth trackball zoom dampening
  el.addEventListener("wheel", () => {
    applySmoothTrackball();
  }, { passive: true });

  // Subscribe to PDBe Molstar custom events (Screen Recording 1 & 2)
  document.addEventListener("PDB.molstar.mouseover", (e) => {
    if (e && e.eventData) {
      handleMolstarHover(e.eventData);
    }
  });

  document.addEventListener("PDB.molstar.mouseout", () => {
    handleAtomUnhover();
  });

  document.addEventListener("PDB.molstar.click", (e) => {
    if (e && e.eventData) {
      const resNum = e.eventData.residueNumber || e.eventData.seq_id || e.eventData.auth_seq_id;
      if (resNum && currentData) {
        selectResidueIn3D(resNum, null, false);
      }
    }
  });

  // Track mouse position over 3D viewport wrapper for floating tooltip
  const wrapper = document.querySelector(".canvas-3d-wrapper");
  if (wrapper) {
    wrapper.addEventListener("mousemove", (e) => {
      const rect = wrapper.getBoundingClientRect();
      const tooltip = document.getElementById("hover-tooltip");
      if (tooltip && tooltip.style.display === "block") {
        const x = e.clientX - rect.left + 16;
        const y = e.clientY - rect.top + 16;
        tooltip.style.left = `${Math.min(x, rect.width - 240)}px`;
        tooltip.style.top = `${Math.min(y, rect.height - 130)}px`;
      }
    });
  }
}

function applyStyling(viewer, scheme) {
  if (!viewer) return;
  viewer.setStyle({}, {});

  if (scheme === "plddt") {
    viewer.setStyle(
      {},
      {
        cartoon: {
          colorfunc: function (atom) {
            const b = atom.b;
            if (b >= 90) return "#0053d6"; // Very high (Dark Blue)
            if (b >= 70) return "#65cbf3"; // Confident (Light Blue)
            if (b >= 50) return "#ffdb13"; // Low (Yellow)
            return "#ff7d45";              // Very low (Orange)
          },
          thickness: 0.75,
          arrows: true,
          tubes: true,
        },
      }
    );
  } else if (scheme === "ss") {
    viewer.setStyle({ ss: "h" }, { cartoon: { color: "#ff7f0e", thickness: 0.85 } });
    viewer.setStyle({ ss: "s" }, { cartoon: { color: "#1f77b4", thickness: 0.85, arrows: true } });
    viewer.setStyle({ ss: "c" }, { cartoon: { color: "#94a3b8", thickness: 0.35, style: "tube" } });
  } else {
    viewer.setStyle({}, { cartoon: { color: "spectrum", thickness: 0.7, arrows: true } });
  }
  viewer.render();
}

function selectResidueIn3D(resNum, extraNote = null, zoomToRes = false) {
  if (!currentData) return;
  selectedResidue = resNum;

  // 1. Native Mol* ball-and-stick selection & camera focus
  if (pdbeViewer && pdbeViewer.visual) {
    try {
      pdbeViewer.visual.select({
        data: [{
          residue_number: resNum,
          representation: "ball-and-stick",
          color: { r: 250, g: 204, b: 21 },
          focus: zoomToRes
        }]
      });
    } catch (e) {
      console.warn("pdbeViewer select notice:", e);
    }
  } else if (viewer3D) {
    applyStyling(viewer3D, currentColorScheme);
    viewer3D.addStyle(
      { resi: resNum },
      {
        sphere: { radius: 0.28, colorscheme: "Jmol" },
        stick: { radius: 0.16, colorscheme: "Jmol" },
        cartoon: { color: "#facc15", thickness: 1.15 },
      }
    );
    viewer3D.addStyle(
      { resi: [resNum - 2, resNum - 1, resNum + 1, resNum + 2] },
      { stick: { radius: 0.14, colorscheme: "Jmol" } }
    );
    renderLocalHydrogenBonds(resNum);
    if (zoomToRes) viewer3D.zoomTo({ resi: resNum }, 600);
    viewer3D.render();
  }

  // 2. Update bottom-right Mol* Info Card
  updateMolstarBox(resNum, extraNote);

  // 3. Update Inspector Panel
  const seq = currentData.sequence;
  const aa1 = seq[resNum - 1] || "A";
  updateResidueInspector(resNum, aa1, currentData);

  // 4. Synchronize ProtVista detail track
  if (resNum < pvWindowStart || resNum > pvWindowEnd) {
    const span = pvWindowEnd - pvWindowStart;
    pvWindowStart = Math.max(1, Math.floor(resNum - span / 2));
    pvWindowEnd = Math.min(currentData.length, pvWindowStart + span);
    renderProtVistaOverview();
  }
  renderProtVistaDetail();
  highlightUniProtTableRow(resNum);
}

function renderLocalHydrogenBonds(resNum) {
  if (!parsedPDBAtoms || parsedPDBAtoms.length === 0) return;

  const nearAtoms = parsedPDBAtoms.filter((a) => a.resi >= resNum - 4 && a.resi <= resNum + 4);
  const nitrogens = nearAtoms.filter((a) => a.atomName === "N");
  const oxygens = nearAtoms.filter((a) => a.atomName === "O");

  for (const n of nitrogens) {
    for (const o of oxygens) {
      if (Math.abs(n.resi - o.resi) >= 2) {
        const dx = n.x - o.x;
        const dy = n.y - o.y;
        const dz = n.z - o.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist >= 2.6 && dist <= 3.5) {
          viewer3D.addLine({
            start: { x: n.x, y: n.y, z: n.z },
            end: { x: o.x, y: o.y, z: o.z },
            color: "#38bdf8",
            dashed: true,
            dashLength: 0.3,
            gapLength: 0.2,
            lineWidth: 3,
          });
        }
      }
    }
  }
}

function updateMolstarBox(resNum, extraNote = null) {
  const box = document.getElementById("molstar-info-box");
  if (!box || !currentData) return;

  const modelId = currentData.model_id || "AF-A0A1B0GTW7-F1";
  const entryId = currentData.entry_id || "A0A1B0GTW7";
  const seq = currentData.sequence;
  const aa1 = seq[resNum - 1] || "A";
  const aa3 = get3Letter(aa1);
  const plddt = currentData.plddt ? currentData.plddt[resNum - 1] : 85.0;
  const ssCode = currentData.ss_string ? currentData.ss_string[resNum - 1] : "C";
  const ssName = ssCode === "H" ? "Alpha Helix" : ssCode === "E" ? "Beta Strand" : "Coil";

  let siteLabel = extraNote;
  if (!siteLabel && currentData.binding_sites) {
    const bs = currentData.binding_sites.find((s) => s.pos === resNum);
    if (bs) siteLabel = bs.name;
  }
  if (!siteLabel && currentData.active_sites) {
    const as = currentData.active_sites.find((s) => s.pos === resNum);
    if (as) siteLabel = as.name;
  }

  document.getElementById("ms-model").innerHTML = `${modelId}: Chain<sup style="font-style: italic; font-size: 10px;">i</sup> A`;
  document.getElementById("ms-res").innerHTML = `${entryId}: ${resNum} <span style="font-weight: 500; font-size: 11px; color: #71717a;">(${aa3})</span>`;
  document.getElementById("ms-detail").textContent = siteLabel
    ? `${siteLabel} · ${ssName} · pLDDT ${plddt.toFixed(1)}`
    : `${ssName} · pLDDT: ${plddt.toFixed(1)} (${getPlddtLabel(plddt)})`;

  box.style.display = "block";
}

function handleMolstarHover(eventData) {
  const resNum = eventData.residueNumber || eventData.seq_id || eventData.auth_seq_id;
  if (!resNum || !currentData) return;

  const seq = currentData.sequence;
  const aa1 = seq[resNum - 1] || "A";
  const aa3 = get3Letter(aa1);
  const resFull = THREE_TO_FULL[aa3] || aa3;
  const plddt = currentData.plddt ? currentData.plddt[resNum - 1] : 85.0;
  const ssCode = currentData.ss_string ? currentData.ss_string[resNum - 1] : "C";
  const ssName = ssCode === "H" ? "Alpha Helix" : ssCode === "E" ? "Beta Strand" : "Coil / Loop";

  let tag = "";
  if (currentData.binding_sites) {
    const bs = currentData.binding_sites.find((s) => s.pos === resNum);
    if (bs) tag = ` [${bs.name}]`;
  }
  if (currentData.active_sites) {
    const as = currentData.active_sites.find((s) => s.pos === resNum);
    if (as) tag = ` [${as.name}]`;
  }

  // Floating Tooltip
  const tooltip = document.getElementById("hover-tooltip");
  if (tooltip) {
    document.getElementById("tt-residue").textContent = `${resFull} ${resNum} (${aa3})${tag}`;
    document.getElementById("tt-pos").textContent = `Residue ${resNum} of ${currentData.length}`;
    document.getElementById("tt-ss").textContent = ssName;
    document.getElementById("tt-plddt").textContent = `${Number(plddt).toFixed(1)} (${getPlddtLabel(plddt)})`;
    document.getElementById("tt-coords").textContent = `Chain A · Position ${resNum}`;
    tooltip.style.display = "block";
  }

  updateMolstarBox(resNum);
  highlightProtVistaCell(resNum);
}

function handleAtomHover(atom, event) {
  const tooltip = document.getElementById("hover-tooltip");
  if (!tooltip || !currentData) return;

  const resIdx = atom.resi;
  const res3 = atom.resn || "ALA";
  const resFull = THREE_TO_FULL[res3] || res3;
  const plddt = atom.b ? atom.b.toFixed(1) : "85.0";
  const ssCode = currentData.ss_string ? currentData.ss_string[resIdx - 1] : "C";
  const ssName = ssCode === "H" ? "Alpha Helix" : ssCode === "E" ? "Beta Strand" : "Coil / Loop";

  let tag = "";
  if (currentData.binding_sites) {
    const bs = currentData.binding_sites.find((s) => s.pos === resIdx);
    if (bs) tag = ` [${bs.name}]`;
  }
  if (currentData.active_sites) {
    const as = currentData.active_sites.find((s) => s.pos === resIdx);
    if (as) tag = ` [${as.name}]`;
  }

  document.getElementById("tt-residue").textContent = `${resFull} ${resIdx} (${res3})${tag}`;
  document.getElementById("tt-pos").textContent = `Residue ${resIdx} of ${currentData.length}`;
  document.getElementById("tt-ss").textContent = ssName;
  document.getElementById("tt-plddt").textContent = `${plddt} (${getPlddtLabel(atom.b)})`;
  document.getElementById("tt-coords").textContent = `(${atom.x.toFixed(2)}, ${atom.y.toFixed(2)}, ${atom.z.toFixed(2)}) Å`;

  if (event) {
    const wrapper = document.querySelector(".canvas-3d-wrapper");
    const rect = wrapper.getBoundingClientRect();
    const x = event.clientX - rect.left + 16;
    const y = event.clientY - rect.top + 16;
    tooltip.style.left = `${Math.min(x, rect.width - 240)}px`;
    tooltip.style.top = `${Math.min(y, rect.height - 130)}px`;
    tooltip.style.display = "block";
  }

  updateMolstarBox(resIdx);
  highlightProtVistaCell(resIdx);
}

function handleAtomUnhover() {
  const tooltip = document.getElementById("hover-tooltip");
  if (tooltip) tooltip.style.display = "none";
  if (selectedResidue) {
    updateMolstarBox(selectedResidue);
    highlightProtVistaCell(selectedResidue);
  }
}

function highlightProtVistaCell(resIdx) {
  document.querySelectorAll(".pv-seq-char").forEach((el) => el.classList.remove("highlighted"));
  const cell = document.getElementById(`pv-cell-${resIdx}`);
  if (cell) {
    cell.classList.add("highlighted");
  }
}

function getPlddtLabel(score) {
  if (score >= 90) return "Very High";
  if (score >= 70) return "Confident";
  if (score >= 50) return "Low";
  return "Very Low";
}

function updateResidueInspector(resIdx, aa1, data) {
  const res3 = get3Letter(aa1);
  const resFull = THREE_TO_FULL[res3] || res3;
  const plddt = data.plddt ? data.plddt[resIdx - 1] : 85.0;
  const ssCode = data.ss_string ? data.ss_string[resIdx - 1] : "C";

  let ssDesc = "Coil / Loop";
  if (ssCode === "H") ssDesc = "Alpha Helix";
  else if (ssCode === "E") ssDesc = "Beta Strand";

  let extraSiteDesc = "";
  if (data.binding_sites) {
    const bs = data.binding_sites.find((b) => b.pos === resIdx);
    if (bs) extraSiteDesc = ` · ${bs.desc}`;
  }
  if (data.active_sites) {
    const as = data.active_sites.find((a) => a.pos === resIdx);
    if (as) extraSiteDesc = ` · ${as.desc}`;
  }

  document.getElementById("insp-res").textContent = `${resFull} (${res3} · ${aa1})`;
  document.getElementById("insp-pos").textContent = `Position ${resIdx} of ${data.length}`;
  document.getElementById("insp-ss").textContent = `${ssDesc}${extraSiteDesc}`;
  document.getElementById("insp-plddt").textContent = `${plddt.toFixed(1)} (${getPlddtLabel(plddt)})`;
}

function get3Letter(c) {
  const map = {
    A: "ALA", C: "CYS", D: "ASP", E: "GLU", F: "PHE", G: "GLY", H: "HIS",
    I: "ILE", K: "LYS", L: "LEU", MET: "MET", ASN: "ASN", P: "PRO", Q: "GLN",
    R: "ARG", S: "SER", T: "THR", V: "VAL", W: "TRP", Y: "TYR",
  };
  return map[c] || "GLY";
}

// ============================================================
// 4. ProtVista 100% Width Responsive Feature Viewer (Video 2)
// ============================================================
function initProtVistaBrush() {
  const brush = document.getElementById("pv-brush-window");
  const handleLeft = document.getElementById("pv-handle-left");
  const handleRight = document.getElementById("pv-handle-right");
  const bar = document.getElementById("pv-overview-bar");

  // Draggable window
  brush.addEventListener("mousedown", (e) => {
    if (e.target === handleLeft || e.target === handleRight) return;
    isDraggingBrush = true;
    dragStartX = e.clientX;
    dragStartWindowStart = pvWindowStart;
    dragStartWindowEnd = pvWindowEnd;
    brush.style.cursor = "grabbing";
    e.preventDefault();
  });

  // Left handle
  handleLeft.addEventListener("mousedown", (e) => {
    isDraggingLeftHandle = true;
    dragStartX = e.clientX;
    dragStartWindowStart = pvWindowStart;
    e.stopPropagation();
    e.preventDefault();
  });

  // Right handle
  handleRight.addEventListener("mousedown", (e) => {
    isDraggingRightHandle = true;
    dragStartX = e.clientX;
    dragStartWindowEnd = pvWindowEnd;
    e.stopPropagation();
    e.preventDefault();
  });

  window.addEventListener("mousemove", (e) => {
    if (!currentData) return;
    const barWidth = bar.clientWidth;
    const totalL = currentData.length;
    const deltaX = e.clientX - dragStartX;
    const deltaRes = Math.round((deltaX / barWidth) * totalL);

    if (isDraggingBrush) {
      const span = dragStartWindowEnd - dragStartWindowStart;
      let newStart = dragStartWindowStart + deltaRes;
      let newEnd = newStart + span;
      if (newStart < 1) {
        newStart = 1;
        newEnd = span + 1;
      }
      if (newEnd > totalL) {
        newEnd = totalL;
        newStart = totalL - span;
      }
      pvWindowStart = newStart;
      pvWindowEnd = newEnd;
      renderProtVistaOverview();
      renderProtVistaDetail();
    } else if (isDraggingLeftHandle) {
      let newStart = Math.min(pvWindowEnd - 5, Math.max(1, dragStartWindowStart + deltaRes));
      pvWindowStart = newStart;
      renderProtVistaOverview();
      renderProtVistaDetail();
    } else if (isDraggingRightHandle) {
      let newEnd = Math.max(pvWindowStart + 5, Math.min(totalL, dragStartWindowEnd + deltaRes));
      pvWindowEnd = newEnd;
      renderProtVistaOverview();
      renderProtVistaDetail();
    }
  });

  window.addEventListener("mouseup", () => {
    isDraggingBrush = false;
    isDraggingLeftHandle = false;
    isDraggingRightHandle = false;
    brush.style.cursor = "grab";
  });

  // Zoom In / Out / Reset buttons
  document.getElementById("btn-pv-zoom-in").addEventListener("click", () => {
    zoomProtVista(0.7);
  });
  document.getElementById("btn-pv-zoom-out").addEventListener("click", () => {
    zoomProtVista(1.4);
  });
  document.getElementById("btn-pv-zoom-fit").addEventListener("click", () => {
    if (!currentData) return;
    pvWindowStart = 300;
    pvWindowEnd = 330;
    renderProtVistaOverview();
    renderProtVistaDetail();
  });
}

function zoomProtVista(factor) {
  if (!currentData) return;
  const totalL = currentData.length;
  const center = (pvWindowStart + pvWindowEnd) / 2;
  const currentSpan = pvWindowEnd - pvWindowStart;
  const newSpan = Math.max(10, Math.min(totalL, Math.round(currentSpan * factor)));
  pvWindowStart = Math.max(1, Math.round(center - newSpan / 2));
  pvWindowEnd = Math.min(totalL, pvWindowStart + newSpan);
  if (pvWindowEnd - pvWindowStart < newSpan) {
    pvWindowStart = Math.max(1, pvWindowEnd - newSpan);
  }
  renderProtVistaOverview();
  renderProtVistaDetail();
}

function renderProtVistaOverview() {
  if (!currentData) return;
  const totalL = currentData.length;
  const leftPct = ((pvWindowStart - 1) / totalL) * 100;
  const widthPct = ((pvWindowEnd - pvWindowStart + 1) / totalL) * 100;

  const brush = document.getElementById("pv-brush-window");
  brush.style.left = `${leftPct}%`;
  brush.style.width = `${Math.max(3.5, widthPct)}%`;
  document.getElementById("pv-brush-badge").textContent = `${pvWindowStart} – ${pvWindowEnd}`;

  // Mini markers across the bar
  const mini = document.getElementById("pv-mini-features");
  mini.innerHTML = "";
  if (currentData.binding_sites) {
    currentData.binding_sites.forEach((site) => {
      const dot = document.createElement("div");
      dot.style.position = "absolute";
      dot.style.left = `${((site.pos - 1) / totalL) * 100}%`;
      dot.style.top = "6px";
      dot.style.width = "4px";
      dot.style.height = "16px";
      dot.style.background = "#f59e0b";
      dot.style.borderRadius = "1px";
      mini.appendChild(dot);
    });
  }
  if (currentData.active_sites) {
    currentData.active_sites.forEach((site) => {
      const dot = document.createElement("div");
      dot.style.position = "absolute";
      dot.style.left = `${((site.pos - 1) / totalL) * 100}%`;
      dot.style.top = "4px";
      dot.style.width = "5px";
      dot.style.height = "20px";
      dot.style.background = "#ef4444";
      dot.style.borderRadius = "1px";
      mini.appendChild(dot);
    });
  }
}

function renderProtVistaDetail() {
  if (!currentData) return;

  const featuresTrack = document.getElementById("pv-features-track");
  const ssTrack = document.getElementById("pv-ss-track");
  const seqTrack = document.getElementById("pv-seq-track");

  featuresTrack.innerHTML = "";
  ssTrack.innerHTML = "";
  seqTrack.innerHTML = "";

  const windowLen = pvWindowEnd - pvWindowStart + 1;
  const seq = currentData.sequence;
  const ss = currentData.ss_string || "C".repeat(seq.length);
  const plddt = currentData.plddt || Array(seq.length).fill(85);

  // 1. Feature markers within active window
  // Active sites
  if (currentData.active_sites) {
    currentData.active_sites.forEach((s) => {
      if (s.pos >= pvWindowStart && s.pos <= pvWindowEnd) {
        const pct = ((s.pos - pvWindowStart + 0.5) / windowLen) * 100;
        const item = document.createElement("div");
        item.className = `pv-feature-item active-site ${s.pos === selectedResidue ? "active" : ""}`;
        item.style.left = `${pct}%`;
        item.textContent = `Active Site (E${s.pos})`;
        item.title = `${s.name}: ${s.desc}`;
        item.addEventListener("click", () => selectResidueIn3D(s.pos, s.name, true));
        featuresTrack.appendChild(item);
      }
    });
  }

  // Binding sites
  if (currentData.binding_sites) {
    currentData.binding_sites.forEach((s) => {
      if (s.pos >= pvWindowStart && s.pos <= pvWindowEnd) {
        const pct = ((s.pos - pvWindowStart + 0.5) / windowLen) * 100;
        const item = document.createElement("div");
        item.className = `pv-feature-item binding ${s.pos === selectedResidue ? "active" : ""}`;
        item.style.left = `${pct}%`;
        item.textContent = `Zn²⁺ (Pos ${s.pos})`;
        item.title = `${s.name}: ${s.desc}`;
        item.addEventListener("click", () => selectResidueIn3D(s.pos, s.name, true));
        featuresTrack.appendChild(item);
      }
    });
  }

  // Variants (e.g. Leu 222)
  if (currentData.variants) {
    currentData.variants.forEach((v) => {
      if (v.pos >= pvWindowStart && v.pos <= pvWindowEnd) {
        const pct = ((v.pos - pvWindowStart + 0.5) / windowLen) * 100;
        const item = document.createElement("div");
        item.className = `pv-feature-item variant ${v.pos === selectedResidue ? "active" : ""}`;
        item.style.left = `${pct}%`;
        item.textContent = v.name.includes("222") ? "Res 222 (Leu)" : v.name;
        item.title = `${v.name}: ${v.desc}`;
        item.addEventListener("click", () => selectResidueIn3D(v.pos, v.name, true));
        featuresTrack.appendChild(item);
      }
    });
  }

  // 2. Secondary Structure & Sequence Letters within window
  for (let resNum = pvWindowStart; resNum <= pvWindowEnd; resNum++) {
    const idx = resNum - 1;
    const char = seq[idx] || "";
    const ssCode = ss[idx] || "C";
    const score = plddt[idx] || 85;

    // SS Block
    const ssBlock = document.createElement("div");
    ssBlock.className = "pv-ss-block";
    ssBlock.style.background = ssCode === "H" ? "#ff7f0e" : ssCode === "E" ? "#1f77b4" : "#cbd5e1";
    ssBlock.title = `Residue ${resNum}: ${ssCode === "H" ? "Helix" : ssCode === "E" ? "Strand" : "Loop"}`;
    ssBlock.addEventListener("click", () => selectResidueIn3D(resNum, null, true));
    ssTrack.appendChild(ssBlock);

    // Sequence Letter Cell
    const cell = document.createElement("div");
    cell.id = `pv-cell-${resNum}`;
    cell.className = "pv-seq-char";
    cell.textContent = windowLen <= 60 ? char : "";
    cell.title = `Residue ${resNum} (${char}) · pLDDT ${score.toFixed(1)}`;
    cell.style.background = getPlddtColor(score, 0.16);

    if (resNum === selectedResidue) {
      cell.classList.add("highlighted");
    }

    cell.addEventListener("mouseenter", () => {
      handleAtomHover({ resi: resNum, resn: get3Letter(char), b: score, x: 0, y: 0, z: 0 });
      if (pdbeViewer && pdbeViewer.visual) {
        try {
          pdbeViewer.visual.highlight({
            data: [{ residue_number: resNum }],
            color: { r: 250, g: 204, b: 21 }
          });
        } catch (e) {}
      }
    });
    cell.addEventListener("mouseleave", () => {
      handleAtomUnhover();
      if (pdbeViewer && pdbeViewer.visual) {
        try {
          pdbeViewer.visual.clearHighlight();
        } catch (e) {}
      }
    });
    cell.addEventListener("click", () => {
      selectResidueIn3D(resNum, null, true);
    });

    seqTrack.appendChild(cell);
  }
}

// ============================================================
// 5. UniProt Biological Features Table (Matching Video 2!)
// ============================================================
function renderUniProtFeaturesTable() {
  const tbody = document.getElementById("pv-table-body");
  if (!tbody || !currentData) return;
  tbody.innerHTML = "";

  const rows = [];

  if (currentData.active_sites) {
    currentData.active_sites.forEach((a) => {
      rows.push({
        type: "Active site",
        pillClass: "active-site",
        id: "+",
        pos: `${a.pos}`,
        posNum: a.pos,
        desc: `${a.desc} <span class="table-badge">PROSITE-ProRule</span>`,
      });
    });
  }

  if (currentData.binding_sites) {
    currentData.binding_sites.forEach((b) => {
      rows.push({
        type: "Binding site",
        pillClass: "binding",
        id: "+",
        pos: `${b.pos}`,
        posNum: b.pos,
        desc: `Zn²⁺ (UniProtKB | ChEBI) · catalytic <span class="table-badge">PROSITE-ProRule</span>`,
      });
    });
  }

  if (currentData.disulfide_bonds) {
    currentData.disulfide_bonds.forEach((d) => {
      rows.push({
        type: "Disulfide bond",
        pillClass: "disulfide",
        id: "+",
        pos: `${d.res1} ↔ ${d.res2}`,
        posNum: d.res1,
        desc: d.name,
      });
    });
  }

  if (currentData.variants) {
    currentData.variants.forEach((v) => {
      rows.push({
        type: "Natural variant",
        pillClass: "variant",
        id: "+",
        pos: `${v.pos}`,
        posNum: v.pos,
        desc: `${v.name} · ${v.desc}`,
      });
    });
  }

  rows.sort((a, b) => a.posNum - b.posNum);

  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.className = `pv-table-row ${r.posNum === selectedResidue ? "active" : ""}`;
    tr.id = `pv-row-${r.posNum}`;
    tr.innerHTML = `
      <td style="color: var(--uniprot-blue); font-weight: 700; text-align: center;">${r.id}</td>
      <td><span class="table-pill ${r.pillClass}">${r.type}</span></td>
      <td style="color: var(--text-muted); font-size: 11px;">PRO_000${r.posNum}</td>
      <td style="font-weight: 700;">${r.pos}</td>
      <td>${r.desc}</td>
    `;
    tr.addEventListener("click", () => {
      document.querySelectorAll(".pv-table-row").forEach((el) => el.classList.remove("active"));
      tr.classList.add("active");

      // Center ProtVista window on this position
      const span = pvWindowEnd - pvWindowStart;
      pvWindowStart = Math.max(1, Math.floor(r.posNum - span / 2));
      pvWindowEnd = Math.min(currentData.length, pvWindowStart + span);
      renderProtVistaOverview();
      renderProtVistaDetail();

      selectResidueIn3D(r.posNum, r.type, true);
    });
    tbody.appendChild(tr);
  });
}

function highlightUniProtTableRow(pos) {
  document.querySelectorAll(".pv-table-row").forEach((el) => el.classList.remove("active"));
  const row = document.getElementById(`pv-row-${pos}`);
  if (row) {
    row.classList.add("active");
    row.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

// ============================================================
// 6. 20 Canonical Amino Acids 3D Explorer (Tab 5)
// ============================================================
async function initAminoAcidsExplorer() {
  try {
    const res = await fetch("/api/amino_acids");
    allAminoAcids = await res.json();
    renderAminoAcidChips(allAminoAcids);

    // Setup Category filter buttons
    const filterBtns = document.querySelectorAll(".aa-filter-btn");
    filterBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        filterBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        currentAAFilter = btn.getAttribute("data-filter");
        filterAminoAcids();
      });
    });

    // Spin & Reset buttons
    const btnSpin = document.getElementById("btn-aa-spin");
    if (btnSpin) {
      btnSpin.addEventListener("click", (e) => {
        isAASpinning = !isAASpinning;
        if (viewerAA) viewerAA.spin(isAASpinning);
        e.target.classList.toggle("active", isAASpinning);
      });
    }

    const btnReset = document.getElementById("btn-aa-reset");
    if (btnReset) {
      btnReset.addEventListener("click", () => {
        if (viewerAA) {
          viewerAA.zoomTo();
          viewerAA.render();
        }
      });
    }

    // Set default selected AA
    if (allAminoAcids.length > 0) {
      selectedAminoAcid = allAminoAcids[0];
    }
  } catch (err) {
    console.error("Failed to load amino acids:", err);
  }
}

function onShowAminoAcidsTab() {
  const el = document.getElementById("viewport-aa");
  if (!el || typeof $3Dmol === "undefined") return;

  if (!viewerAA) {
    viewerAA = $3Dmol.createViewer(el, { backgroundColor: "#ffffff" });
  }

  // Defer slightly for container display:block calculation
  setTimeout(() => {
    if (viewerAA) {
      viewerAA.resize();
      const aaToRender = selectedAminoAcid || (allAminoAcids.length > 0 ? allAminoAcids[0] : null);
      if (aaToRender) {
        selectAminoAcid(aaToRender);
      }
    }
  }, 60);
}

function filterAminoAcids() {
  if (currentAAFilter === "all") {
    renderAminoAcidChips(allAminoAcids);
  } else {
    const filtered = allAminoAcids.filter((aa) => aa.category.toLowerCase().includes(currentAAFilter.toLowerCase()));
    renderAminoAcidChips(filtered);
  }
}

function renderAminoAcidChips(list) {
  const grid = document.getElementById("aa-chips-grid");
  if (!grid) return;
  grid.innerHTML = "";

  list.forEach((aa) => {
    const chip = document.createElement("div");
    chip.className = `aa-chip ${selectedAminoAcid && selectedAminoAcid.code3 === aa.code3 ? "active" : ""}`;
    chip.id = `aa-chip-${aa.code3}`;
    chip.innerHTML = `
      <div class="aa-chip-code1">${aa.code1}</div>
      <div class="aa-chip-code3">${aa.code3}</div>
      <div class="aa-chip-name">${aa.name}</div>
    `;
    chip.addEventListener("click", () => {
      document.querySelectorAll(".aa-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      selectAminoAcid(aa);
    });
    grid.appendChild(chip);
  });
}

function selectAminoAcid(aa) {
  if (!aa) return;
  selectedAminoAcid = aa;

  // Title
  const titleEl = document.getElementById("aa-3d-title");
  if (titleEl) {
    titleEl.textContent = `${aa.name} (${aa.code3} · ${aa.code1}) — 3D Ball & Stick View`;
  }

  // Highlight active chip
  document.querySelectorAll(".aa-chip").forEach((c) => c.classList.remove("active"));
  const activeChip = document.getElementById(`aa-chip-${aa.code3}`);
  if (activeChip) activeChip.classList.add("active");

  // Render in 3Dmol viewer
  if (!viewerAA) {
    const el = document.getElementById("viewport-aa");
    if (el && typeof $3Dmol !== "undefined") {
      viewerAA = $3Dmol.createViewer(el, { backgroundColor: "#ffffff" });
    }
  }

  if (viewerAA && aa.pdb) {
    viewerAA.clear();
    viewerAA.addModel(aa.pdb, "pdb");
    viewerAA.setStyle(
      {},
      {
        sphere: { radius: 0.38, colorscheme: "Jmol" },
        stick: { radius: 0.20, colorscheme: "Jmol" },
      }
    );
    viewerAA.zoomTo();
    viewerAA.render();
    if (isAASpinning) viewerAA.spin(true);
  }

  // Populate Biochemical Profile Card
  const card = document.getElementById("aa-profile-card");
  if (!card) return;
  card.innerHTML = `
    <div class="aa-profile-title-row">
      <h2 style="font-size: 22px; font-weight: 700; color: var(--text-dark); margin: 0;">${aa.name}</h2>
      <span class="meta-tag" style="background: var(--uniprot-blue-light); color: var(--uniprot-blue);">${aa.category}</span>
      <span style="font-size: 13px; color: var(--text-muted);">Codons: <strong>${(aa.codons || []).join(", ")}</strong></span>
    </div>

    <div class="aa-prop-grid">
      <div class="aa-prop-box">
        <div class="aa-prop-label">1-Letter / 3-Letter</div>
        <div class="aa-prop-val" style="color: var(--uniprot-blue);">${aa.code1} / ${aa.code3}</div>
      </div>
      <div class="aa-prop-box">
        <div class="aa-prop-label">Molecular Formula</div>
        <div class="aa-prop-val">${aa.formula}</div>
      </div>
      <div class="aa-prop-box">
        <div class="aa-prop-label">Molecular Weight</div>
        <div class="aa-prop-val">${aa.mw} g/mol</div>
      </div>
      <div class="aa-prop-box">
        <div class="aa-prop-label">Isoelectric Point (pI)</div>
        <div class="aa-prop-val">${Number(aa.pI).toFixed(2)}</div>
      </div>
      <div class="aa-prop-box">
        <div class="aa-prop-label">Hydropathy Index</div>
        <div class="aa-prop-val" style="color: ${aa.hydropathy > 0 ? '#10b981' : '#ef4444'};">${aa.hydropathy > 0 ? '+' : ''}${aa.hydropathy}</div>
      </div>
      <div class="aa-prop-box">
        <div class="aa-prop-label">Net Charge (pH 7.4)</div>
        <div class="aa-prop-val">${aa.charge}</div>
      </div>
    </div>

    <div style="margin-top: 14px;">
      <div class="aa-prop-label">Side-Chain Chemical Structure</div>
      <div style="font-family: var(--font-mono); font-size: 13px; font-weight: 600; color: var(--text-dark); margin-top: 4px; background: var(--bg-page); padding: 8px 12px; border-radius: 4px;">
        ${aa.sidechain}
      </div>
    </div>

    <div class="aa-role-box">
      <div class="aa-prop-label" style="color: var(--uniprot-blue); margin-bottom: 4px;">Role in Protein Architecture &amp; Function</div>
      <div style="font-size: 13px; line-height: 1.6; color: var(--text-dark);">
        ${aa.role}
      </div>
    </div>
  `;
}

function getPlddtColor(score, alpha = 1.0) {
  if (score >= 90) return alpha < 1 ? "rgba(0, 83, 214, 0.18)" : "#0053d6";
  if (score >= 70) return alpha < 1 ? "rgba(101, 203, 243, 0.22)" : "#65cbf3";
  if (score >= 50) return alpha < 1 ? "rgba(255, 219, 19, 0.25)" : "#ffdb13";
  return alpha < 1 ? "rgba(255, 125, 69, 0.25)" : "#ff7d45";
}

function renderFastaBox(seq, ss) {
  const box = document.getElementById("full-fasta-box");
  box.innerHTML = `
    <div style="color: var(--uniprot-blue); font-weight: 700; margin-bottom: 8px;">&gt;sp|${document.getElementById("entry-acc").textContent}|${document.getElementById("entry-gene").textContent} Length: ${seq.length} aa</div>
    <div>${seq}</div>
    <div style="color: var(--text-muted); margin-top: 10px; font-size: 11px;">Secondary Structure Assignment:</div>
    <div style="color: var(--text-muted); letter-spacing: 1px;">${ss}</div>
  `;
}

// ============================================================
// 7. Explainability Heatmap (Tab 3)
// ============================================================
function renderDistanceHeatmap(matrix) {
  const canvas = document.getElementById("heatmap-canvas");
  if (!canvas || !matrix) return;
  const ctx = canvas.getContext("2d");
  const L = matrix.length;
  const cellW = canvas.width / L;
  const cellH = canvas.height / L;

  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      const d = matrix[i][j];
      ctx.fillStyle = distanceToColor(d);
      ctx.fillRect(j * cellW, i * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const j = Math.floor((x / rect.width) * L);
    const i = Math.floor((y / rect.height) * L);

    if (i >= 0 && i < L && j >= 0 && j < L) {
      document.getElementById("explain-pair").textContent = `Residue ${i + 1} ↔ Residue ${j + 1}`;
      document.getElementById("explain-dist").textContent = `${matrix[i][j].toFixed(2)} Å`;
    }
  };
}

function distanceToColor(dist) {
  const t = Math.max(0, Math.min(1, dist / 24.0));
  const r = Math.floor(255 * Math.pow(t, 1.2));
  const g = Math.floor(255 * (0.8 * Math.sin(t * Math.PI)));
  const b = Math.floor(255 * (1.0 - t));
  return `rgb(${r}, ${g}, ${b})`;
}

// ============================================================
// 8. Variant & Mutation Lab (Tab 4)
// ============================================================
async function runMutationSimulation() {
  const seq = document.getElementById("mut-base-seq").value.trim().toUpperCase();
  const mut = document.getElementById("mut-code").value.trim().toUpperCase();
  const btn = document.getElementById("btn-run-mutation");
  const label = document.getElementById("btn-mut-label");

  if (!seq || !mut) {
    alert("Please provide both a sequence and a mutation (e.g. C22S).");
    return;
  }

  btn.disabled = true;
  label.innerHTML = `<span class="uniprot-spinner"></span> Simulating ${mut}...`;

  try {
    const res = await fetch("/api/mutate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence: seq, mutation: mut }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Simulation failed");
    }

    const data = await res.json();

    document.getElementById("mut-res-rmsd").textContent = `${data.global_rmsd} Å`;
    document.getElementById("mut-res-max").textContent = `${data.max_displacement} Å`;
    document.getElementById("mut-res-lost").textContent = `${data.lost_contacts}`;
    document.getElementById("mut-res-gained").textContent = `${data.gained_contacts}`;

    renderSplitViewers(data.wt_pdb, data.mut_pdb);
    renderDisplacementChart(data.per_res_disp, mut);
  } catch (err) {
    alert(`Simulation Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    label.textContent = "Simulate Mutation Impact";
  }
}

function renderSplitViewers(wtPdb, mutPdb) {
  const elWT = document.getElementById("viewport-wt");
  const elMut = document.getElementById("viewport-mut");

  if (!viewerWT) viewerWT = $3Dmol.createViewer(elWT, { backgroundColor: "#ffffff" });
  if (!viewerMut) viewerMut = $3Dmol.createViewer(elMut, { backgroundColor: "#ffffff" });

  viewerWT.clear();
  viewerWT.addModel(wtPdb, "pdb");
  applyStyling(viewerWT, "plddt");
  viewerWT.zoomTo();
  viewerWT.render();

  viewerMut.clear();
  viewerMut.addModel(mutPdb, "pdb");
  applyStyling(viewerMut, "plddt");
  viewerMut.zoomTo();
  viewerMut.render();
}

function renderDisplacementChart(dispArray, mutStr) {
  const ctx = document.getElementById("displacement-chart").getContext("2d");
  const labels = dispArray.map((_, idx) => `${idx + 1}`);

  if (dispChart) dispChart.destroy();

  dispChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: `Per-Residue Displacement (${mutStr})`,
          data: dispArray,
          borderColor: "#00639a",
          backgroundColor: "rgba(0, 99, 154, 0.12)",
          fill: true,
          tension: 0.3,
          pointRadius: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Residue Position" } },
        y: { title: { display: true, text: "Displacement (Å)" } },
      },
    },
  });
}

// ============================================================
// 9. Event Listeners
// ============================================================
function initEventListeners() {
  document.getElementById("btn-fold-custom").addEventListener("click", () => {
    const seq = document.getElementById("seq-custom-input").value.trim().toUpperCase();
    const relax = document.getElementById("relax-check").checked;
    foldSequence(seq, relax);
  });

  document.getElementById("btn-run-mutation").addEventListener("click", runMutationSimulation);

  // Model Predicted vs Experimental Ground Truth toggle buttons
  const btnPred = document.getElementById("btn-show-predicted");
  if (btnPred) {
    btnPred.addEventListener("click", () => {
      if (currentProteinId) {
        loadPreindexedProtein(currentProteinId, 0);
      }
    });
  }
  const btnExp = document.getElementById("btn-show-experimental");
  if (btnExp) {
    btnExp.addEventListener("click", () => {
      if (currentProteinId) {
        loadPreindexedProtein(currentProteinId, 1);
      }
    });
  }

  // Close Mol* bottom-right card
  document.getElementById("btn-close-molstar-box").addEventListener("click", () => {
    document.getElementById("molstar-info-box").style.display = "none";
    selectedResidue = null;
    if (pdbeViewer && pdbeViewer.visual) {
      try {
        pdbeViewer.visual.clearSelection();
      } catch (e) {}
    } else if (viewer3D) {
      applyStyling(viewer3D, currentColorScheme);
      viewer3D.render();
    }
    document.querySelectorAll(".pv-seq-char").forEach((el) => el.classList.remove("highlighted"));
  });

  // Color mode buttons
  document.getElementById("btn-color-plddt").addEventListener("click", (e) => {
    currentColorScheme = "plddt";
    document.querySelectorAll(".viewer-controls-overlay .ctrl-btn").forEach((b) => b.classList.remove("active"));
    e.target.classList.add("active");
    if (selectedResidue) {
      selectResidueIn3D(selectedResidue, null, false);
    } else if (viewer3D) {
      applyStyling(viewer3D, "plddt");
    }
  });

  document.getElementById("btn-color-ss").addEventListener("click", (e) => {
    currentColorScheme = "ss";
    document.querySelectorAll(".viewer-controls-overlay .ctrl-btn").forEach((b) => b.classList.remove("active"));
    e.target.classList.add("active");
    if (selectedResidue) {
      selectResidueIn3D(selectedResidue, null, false);
    } else if (viewer3D) {
      applyStyling(viewer3D, "ss");
    }
  });

  document.getElementById("btn-color-rainbow").addEventListener("click", (e) => {
    currentColorScheme = "spectrum";
    document.querySelectorAll(".viewer-controls-overlay .ctrl-btn").forEach((b) => b.classList.remove("active"));
    e.target.classList.add("active");
    if (selectedResidue) {
      selectResidueIn3D(selectedResidue, null, false);
    } else if (viewer3D) {
      applyStyling(viewer3D, "spectrum");
    }
  });

  document.getElementById("btn-spin-3d").addEventListener("click", (e) => {
    isSpinning = !isSpinning;
    if (pdbeViewer && pdbeViewer.visual) {
      try {
        pdbeViewer.visual.toggleSpin(isSpinning);
      } catch (err) {}
    } else if (viewer3D) {
      viewer3D.spin(isSpinning);
    }
    e.target.classList.toggle("active", isSpinning);
  });

  document.getElementById("btn-reset-cam").addEventListener("click", () => {
    if (pdbeViewer && pdbeViewer.visual) {
      try {
        pdbeViewer.visual.reset({ camera: true });
      } catch (err) {}
    } else if (viewer3D) {
      viewer3D.zoomTo();
      viewer3D.render();
    }
  });

  document.getElementById("btn-download-pdb").addEventListener("click", () => {
    if (!currentPDB) {
      alert("No structure has been folded yet.");
      return;
    }
    const blob = new Blob([currentPDB], { type: "chemical/x-pdb" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const acc = document.getElementById("entry-acc").textContent || "protein";
    a.href = url;
    a.download = `${acc}_mini_alphafold.pdb`;
    a.click();
    URL.revokeObjectURL(url);
  });
}
