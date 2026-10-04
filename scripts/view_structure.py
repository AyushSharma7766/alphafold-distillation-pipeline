"""
Mini-AlphaFold: 3D Structure Web Viewer
========================================
Creates an interactive, hardware-accelerated 3D HTML viewer using 3Dmol.js
and opens it in your default web browser.

Usage:
    python scripts/view_structure.py predictions/1crnA00_predicted.pdb
"""

import argparse
import sys
import webbrowser
from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Mini-AlphaFold 3D Viewer - {title}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
  <style>
    body {{
      margin: 0;
      padding: 0;
      background: #0d1117;
      color: #c9d1d9;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      overflow: hidden;
    }}
    #header {{
      position: absolute;
      top: 15px;
      left: 20px;
      z-index: 10;
      background: rgba(22, 27, 34, 0.85);
      backdrop-filter: blur(8px);
      padding: 12px 20px;
      border-radius: 10px;
      border: 1px solid #30363d;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }}
    h1 {{
      margin: 0 0 4px 0;
      font-size: 18px;
      color: #58a6ff;
    }}
    p {{
      margin: 0;
      font-size: 13px;
      color: #8b949e;
    }}
    #controls {{
      position: absolute;
      top: 15px;
      right: 20px;
      z-index: 10;
      background: rgba(22, 27, 34, 0.85);
      backdrop-filter: blur(8px);
      padding: 10px 16px;
      border-radius: 10px;
      border: 1px solid #30363d;
      display: flex;
      gap: 10px;
    }}
    button {{
      background: #238636;
      color: white;
      border: none;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: background 0.15s;
    }}
    button:hover {{
      background: #2ea043;
    }}
    #viewer {{
      width: 100vw;
      height: 100vh;
      position: absolute;
      top: 0;
      left: 0;
    }}
  </style>
</head>
<body>
  <div id="header">
    <h1>Mini-AlphaFold 3D Prediction</h1>
    <p>{title} | Left-click: Rotate | Scroll: Zoom | Right-click: Pan</p>
  </div>
  <div id="controls">
    <button onclick="setColorScheme('ss')">Secondary Structure</button>
    <button onclick="setColorScheme('spectrum')">Rainbow (N-C)</button>
    <button onclick="toggleSpin()">Toggle Spin</button>
    <button onclick="viewer.zoomTo(); viewer.render();">Reset View</button>
  </div>
  <div id="viewer"></div>

  <script>
    const pdbData = `{pdb_data}`;
    let isSpinning = false;
    let viewer = null;

    function applyCartoonStyle(scheme) {{
      viewer.removeAllLabels();
      viewer.setStyle({{}}, {{}});
      if (scheme === "ss") {{
        viewer.setStyle({{ss: "h"}}, {{cartoon: {{color: "#ff7f0e", thickness: 0.7}}}});
        viewer.setStyle({{ss: "s"}}, {{cartoon: {{color: "#1f77b4", thickness: 0.7, arrows: true}}}});
        viewer.setStyle({{ss: "c"}}, {{cartoon: {{color: "#7f7f7f", thickness: 0.4}}}});
      }} else {{
        viewer.setStyle({{}}, {{
          cartoon: {{
            color: "spectrum",
            thickness: 0.6
          }}
        }});
      }}
      viewer.render();
    }}

    function setColorScheme(scheme) {{
      applyCartoonStyle(scheme);
    }}

    document.addEventListener("DOMContentLoaded", () => {{
      const element = document.getElementById("viewer");
      viewer = $3Dmol.createViewer(element, {{
        backgroundColor: "#0d1117"
      }});

      viewer.addModel(pdbData, "pdb");
      applyCartoonStyle("ss");
      viewer.zoomTo();
      viewer.render();
    }});

    function toggleSpin() {{
      isSpinning = !isSpinning;
      viewer.spin(isSpinning);
    }}
  </script>
</body>
</html>
"""


def create_3d_viewer(pdb_file: Path | str, auto_open: bool = True) -> Path:
    pdb_path = Path(pdb_file)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    pdb_content = pdb_path.read_text(encoding="utf-8")

    html_content = HTML_TEMPLATE.format(
        title=pdb_path.stem,
        pdb_data=pdb_content.replace("\\", "\\\\").replace("`", "\\`"),
    )

    html_path = pdb_path.with_suffix(".html")
    html_path.write_text(html_content, encoding="utf-8")
    print(f"Created interactive 3D HTML viewer: {html_path}")

    if auto_open:
        webbrowser.open(html_path.resolve().as_uri())

    return html_path


def main():
    parser = argparse.ArgumentParser(description="View 3D PDB structure in browser")
    parser.add_argument("pdb_file", type=str, help="Path to .pdb file")
    parser.add_argument("--no-open", action="store_true", help="Don't automatically open browser")
    args = parser.parse_args()

    create_3d_viewer(args.pdb_file, auto_open=not args.no_open)


if __name__ == "__main__":
    main()
