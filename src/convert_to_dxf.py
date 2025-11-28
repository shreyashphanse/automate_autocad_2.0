# src/convert_to_dxf.py
"""
Main runner: processes every image in ../input/ and writes DXF files into ../output/.
Creates two outputs per image:
 - <name>.dxf          : raw merged segments from Hough+merge
 - <name>_graph.dxf    : reconstructed node-edge graph (segments split at intersections)
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import ezdxf
from hough_merge import detect_lines_hough, merge_collinear_segments
from util import length
from graph_builder import build_graph_from_segments

# -----------------------
# USER-TUNEABLE PARAMETERS
# -----------------------
MAX_DIM = 2500
RESIZE_KEEP_SCALE = True

HOUGH = {
    'rho': 1,
    'theta': np.pi/180,
    'threshold': 60,
    'min_line_length': 50,
    'max_line_gap': 25
}

ANGLE_TOL_DEG = 6.0
DIST_TOL_PX = 18.0

PIXEL_TO_REAL = 1.0
FLIP_Y = True

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

# graph snap tolerance (px). Increase to merge more nearby nodes.
GRAPH_SNAP_TOL = 8.0

# -----------------------
def preprocess_gray(img):
    blur = cv2.GaussianBlur(img, (5,5), 0)
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bw

def export_dxf_segments(segments, output_path, layer_name='AUTO_LINES'):
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    try:
        doc.layers.new(layer_name, dxfattribs={'color':7})
    except Exception:
        pass
    for seg in segments:
        (x1,y1),(x2,y2) = seg
        msp.add_line((x1,y1),(x2,y2), dxfattribs={'layer':layer_name})
    doc.saveas(str(output_path))
    print("  -> Saved:", output_path.name)

def process_image_file(img_path: Path, out_folder: Path):
    print(f"Processing: {img_path.name}")
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("  ! cannot open image, skipping")
        return

    h0, w0 = img.shape
    scale = 1.0
    if max(h0, w0) > MAX_DIM:
        scale = MAX_DIM / float(max(h0, w0))
        img = cv2.resize(img, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)
        print(f"  - resized by scale {scale:.4f}")

    bw = preprocess_gray(img)

    raw_lines = detect_lines_hough(bw,
                                  rho=HOUGH['rho'],
                                  theta=HOUGH['theta'],
                                  threshold=HOUGH['threshold'],
                                  min_line_length=HOUGH['min_line_length'],
                                  max_line_gap=HOUGH['max_line_gap'])
    print(f"  - raw Hough lines: {len(raw_lines)}")

    merged = merge_collinear_segments(raw_lines,
                                      angle_tol_deg=ANGLE_TOL_DEG,
                                      dist_tol_px=DIST_TOL_PX)
    print(f"  - merged segments count: {len(merged)}")

    if not merged:
        print("  ! no segments found; try loosening parameters")
        return

    # Map back to original scale and flip Y if needed, produce segments_out
    segments_out = []
    for ((x1,y1),(x2,y2)) in merged:
        if RESIZE_KEEP_SCALE and scale != 1.0:
            x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale
        if FLIP_Y:
            orig_h = h0
            y1 = orig_h - y1
            y2 = orig_h - y2
        if PIXEL_TO_REAL != 1.0:
            x1 *= PIXEL_TO_REAL; y1 *= PIXEL_TO_REAL; x2 *= PIXEL_TO_REAL; y2 *= PIXEL_TO_REAL
        if length((x1,y1),(x2,y2)) < 1e-6:
            continue
        segments_out.append(((float(x1), float(y1)), (float(x2), float(y2))))

    # Save raw merged segments DXF
    out_base = img_path.stem
    out_path_raw = out_folder / (out_base + ".dxf")
    export_dxf_segments(segments_out, out_path_raw, layer_name='RAW_LINES')

    # Build graph (nodes + edges) and save graph DXF
    nodes, edges = build_graph_from_segments(segments_out, snap_tol=GRAPH_SNAP_TOL)
    print(f"  - graph nodes: {len(nodes)}, graph edges: {len(edges)}")

    # optional: if coordinates are large floats, we keep them as-is
    out_path_graph = out_folder / (out_base + "_graph.dxf")
    export_dxf_segments(edges, out_path_graph, layer_name='GRAPH_LINES')

def main(input_folder: Path, output_folder: Path):
    if not input_folder.exists() or not input_folder.is_dir():
        print("Input folder does not exist:", input_folder)
        return
    output_folder.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in input_folder.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not images:
        print("No images found in", input_folder)
        return

    print(f"Found {len(images)} image(s) in {input_folder}")
    for img in images:
        process_image_file(img, output_folder)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        in_folder = Path(sys.argv[1])
        out_folder = Path(sys.argv[2])
    else:
        project_root = Path(__file__).resolve().parents[1]
        in_folder = project_root / "input"
        out_folder = project_root / "output"

    main(in_folder, out_folder)
