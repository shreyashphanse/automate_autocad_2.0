# src/convert_to_dxf.py
"""
Main runner: processes every image in ../input/ and writes DXF files into ../output/.
Usage:
    python src/convert_to_dxf.py
Or optionally:
    python src/convert_to_dxf.py path_to_input path_to_output
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import ezdxf
from hough_merge import detect_lines_hough, merge_collinear_segments
from util import length

# -----------------------
# USER-TUNEABLE PARAMETERS
# -----------------------
MAX_DIM = 2500           # if image larger than this, it will be downscaled (px)
RESIZE_KEEP_SCALE = True

# Hough params
HOUGH = {
    'rho': 1,
    'theta': np.pi/180,
    'threshold': 60,         # start lower for connected drawings; tune if too many lines
    'min_line_length': 50,   # minimum line length in px (tune per image)
    'max_line_gap': 25       # join broken lines up to this gap
}

ANGLE_TOL_DEG = 6.0
DIST_TOL_PX = 18.0

PIXEL_TO_REAL = 1.0   # keep 1.0 to keep pixel units; set to 0.01 to convert px->meters (example)
FLIP_Y = True         # flip Y to convert image coordinates to cartesian (optional)

# supported image extensions
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

# -----------------------
def preprocess_gray(img):
    blur = cv2.GaussianBlur(img, (5,5), 0)
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bw

def export_dxf(segments, output_path):
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    # create a layer
    try:
        doc.layers.new('AUTO_LINES', dxfattribs={'color':7})
    except Exception:
        pass
    for seg in segments:
        (x1,y1),(x2,y2) = seg
        msp.add_line((x1,y1),(x2,y2), dxfattribs={'layer':'AUTO_LINES'})
    doc.saveas(str(output_path))
    print("  -> Saved:", output_path.name)

def process_image_file(img_path: Path, output: Path):
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

    segments_out = []
    for ((x1,y1),(x2,y2)) in merged:
        # map back to original scale (if resized)
        if RESIZE_KEEP_SCALE and scale != 1.0:
            x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale
        # flip Y to cartesian origin bottom-left
        if FLIP_Y:
            orig_h = h0
            y1 = orig_h - y1
            y2 = orig_h - y2
        # scale to real units if desired
        if PIXEL_TO_REAL != 1.0:
            x1 *= PIXEL_TO_REAL; y1 *= PIXEL_TO_REAL; x2 *= PIXEL_TO_REAL; y2 *= PIXEL_TO_REAL

        if length((x1,y1),(x2,y2)) < 1e-3:
            continue
        segments_out.append(((float(x1), float(y1)), (float(x2), float(y2))))

    out_name = img_path.stem + ".dxf"
    out_path = output / out_name
    export_dxf(segments_out, out_path)

def main(input: Path, output: Path):
    if not input.exists() or not input.is_dir():
        print("Input folder does not exist:", input)
        return
    output.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in input.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not images:
        print("No images found in", input)
        return

    print(f"Found {len(images)} image(s) in {input}")
    for img in images:
        process_image_file(img, output)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        input = Path(sys.argv[1])
        output = Path(sys.argv[2])
    else:
        # default relative folders
        project_root = Path(__file__).resolve().parents[1]
        input = project_root / "input"
        output = project_root / "output"

    main(input, output)
