# src/hough_merge.py
"""
Robust Hough detection + merging of near-collinear segments.
Handles cases where unary_union(parts) returns a LineString, MultiLineString,
GeometryCollection, or other shapely types without raising exceptions.
"""

import cv2
import numpy as np
import math
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union
from util import angle_of_line, normalize_angle, lines_distance

def detect_lines_hough(binary_img,
                       rho=1,
                       theta=np.pi/180,
                       threshold=80,
                       min_line_length=40,
                       max_line_gap=10):
    edges = cv2.Canny(binary_img, 50, 150, apertureSize=3)
    raw = cv2.HoughLinesP(edges, rho=rho, theta=theta,
                          threshold=threshold,
                          minLineLength=min_line_length,
                          maxLineGap=max_line_gap)
    lines = []
    if raw is None:
        return lines
    for l in raw:
        x1,y1,x2,y2 = l[0]
        if math.hypot(x2-x1, y2-y1) < 2:
            continue
        lines.append((int(x1),int(y1),int(x2),int(y2)))
    return lines

def _farthest_pair(coords):
    best = (coords[0], coords[-1])
    maxd = 0.0
    n = len(coords)
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i+1, n):
            xj, yj = coords[j]
            d = (xi - xj)**2 + (yi - yj)**2
            if d > maxd:
                maxd = d
                best = (coords[i], coords[j])
    return best

def _extract_lines_from_merged(merged):
    """
    Return a list of LineString parts extracted safely from a shapely 'merged' geometry.
    Uses .geoms when available to avoid TypeError on non-iterable shapely containers.
    """
    parts = []
    if merged is None:
        return parts

    from shapely.geometry import LineString, MultiLineString, GeometryCollection

    # Single LineString
    if isinstance(merged, LineString):
        return [merged]

    # MultiLineString or GeometryCollection with .geoms
    if hasattr(merged, 'geoms'):
        try:
            for g in merged.geoms:
                if isinstance(g, LineString):
                    parts.append(g)
            return parts
        except Exception:
            pass

    # Last resort: try iterating
    try:
        for item in merged:
            if isinstance(item, LineString):
                parts.append(item)
    except TypeError:
        return parts
    except Exception:
        return parts

    return parts

def merge_collinear_segments(lines,
                             angle_tol_deg=5.0,
                             dist_tol_px=12.0):
    """
    Merge lines that are nearly collinear and close to each other.
    Input:
      lines: list of (x1,y1,x2,y2)
    Returns:
      list of merged segments: [ ((x1,y1),(x2,y2)), ... ]
    """
    if not lines:
        return []

    segments = [LineString([(l[0], l[1]), (l[2], l[3])]) for l in lines]

    angle_tol = math.radians(angle_tol_deg)
    used = [False] * len(segments)
    angles = []
    for seg in segments:
        coords = list(seg.coords)
        a = angle_of_line(coords[0][0], coords[0][1], coords[-1][0], coords[-1][1])
        angles.append(normalize_angle(a))

    groups = []
    for i, seg in enumerate(segments):
        if used[i]:
            continue
        group_indices = [i]
        used[i] = True
        ai = angles[i]
        for j in range(i+1, len(segments)):
            if used[j]:
                continue
            aj = angles[j]
            if abs(ai - aj) <= angle_tol:
                if lines_distance(seg.coords, segments[j].coords) <= dist_tol_px:
                    group_indices.append(j)
                    used[j] = True
        groups.append(group_indices)

    merged_segments = []
    from shapely.geometry import LineString as ShpLineString
    for g in groups:
        parts = [segments[idx] for idx in g]
        # safe union
        u = unary_union(parts)
        # If unary_union returns a single LineString, skip linemerge
        if isinstance(u, ShpLineString):
            merged_geom = u
        else:
            # linemerge expects a MultiLineString or iterable of lines
            try:
                merged_geom = linemerge(u)
            except Exception:
                # fallback: treat u as-is and extract any LineString parts later
                merged_geom = u

        # Extract LineString parts safely
        line_parts = _extract_lines_from_merged(merged_geom)

        for part in line_parts:
            coords = list(part.coords)
            if len(coords) < 2:
                continue
            a,b = _farthest_pair(coords)
            merged_segments.append( ( (float(a[0]), float(a[1])), (float(b[0]), float(b[1])) ) )

    # dedupe near-duplicates
    cleaned = []
    seen = []
    for seg in merged_segments:
        (x1,y1),(x2,y2) = seg
        if (x2, y2) < (x1, y1):
            seg = ((x2,y2),(x1,y1))
            (x1,y1),(x2,y2) = seg
        keep = True
        for s in seen:
            sx1,sy1 = s[0]; sx2,sy2 = s[1]
            if (abs(sx1 - x1) < 1.0 and abs(sy1 - y1) < 1.0 and
                abs(sx2 - x2) < 1.0 and abs(sy2 - y2) < 1.0):
                keep = False
                break
        if keep:
            cleaned.append(seg)
            seen.append(seg)
    return cleaned
