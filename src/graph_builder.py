# src/graph_builder.py
"""
Builds a deterministic node+edge graph from merged line segments.

Functions:
- compute_intersections(segments): returns intersection points between segments
- snap_and_merge_points(points, snap_tol): cluster close points into single nodes
- split_segments_at_nodes(segments, nodes, snap_tol): split each segment at node locations that lie on it
- build_edges_from_segments(split_segments): produce unique edges as endpoint pairs

Segments are in format: [ ((x1,y1),(x2,y2)), ... ]
Nodes are returned as list of (x,y) floats.
Edges are returned as list of ((x1,y1),(x2,y2))
"""

from shapely.geometry import LineString, Point
from shapely.ops import split
import math

def compute_intersections(segments):
    """Return intersection points of all segment pairs (excluding touching endpoints)."""
    pts = []
    shapely_lines = [LineString([s[0], s[1]]) for s in segments]
    n = len(shapely_lines)
    for i in range(n):
        for j in range(i+1, n):
            a = shapely_lines[i]
            b = shapely_lines[j]
            inter = a.intersection(b)
            if inter.is_empty:
                continue
            # handle Point or MultiPoint
            if inter.geom_type == 'Point':
                pts.append((float(inter.x), float(inter.y)))
            elif inter.geom_type == 'MultiPoint':
                for p in inter:
                    pts.append((float(p.x), float(p.y)))
            else:
                # skip line overlaps (collinear overlaps) here - endpoints handled later
                continue
    return pts

def snap_and_merge_points(points, snap_tol=15.0):
    """
    Merge points closer than snap_tol into a single representative.
    Simple O(n^2) clustering which is fine for modest counts.
    """
    merged = []
    for p in points:
        x,y = p
        found = False
        for i,(mx,my, count) in enumerate(merged):
            if math.hypot(mx - x, my - y) <= snap_tol:
                # incremental average
                new_count = count + 1
                nx = (mx*count + x) / new_count
                ny = (my*count + y) / new_count
                merged[i] = (nx, ny, new_count)
                found = True
                break
        if not found:
            merged.append((x,y,1))
    # return just coords
    return [(m[0], m[1]) for m in merged]

def _point_on_segment_strict(pt, seg, tol=1e-6):
    """Return True if pt lies on the segment (within tol)."""
    line = LineString([seg[0], seg[1]])
    p = Point(pt)
    # distance to segment
    return p.distance(line) <= tol and min(seg[0][0], seg[1][0]) - tol <= pt[0] <= max(seg[0][0], seg[1][0]) + tol and min(seg[0][1], seg[1][1]) - tol <= pt[1] <= max(seg[0][1], seg[1][1]) + tol

def split_segments_at_nodes(segments, nodes, snap_tol=6.0):
    """
    For each segment, find nodes that lie on it (within snap_tol), sort them along segment,
    and split the segment into smaller segments between consecutive nodes (including original ends).
    Returns list of segments ((x1,y1),(x2,y2)).
    """
    result = []
    for seg in segments:
        line = LineString([seg[0], seg[1]])
        # collect candidate points: segment endpoints + nodes that are close to the line
        pts = [ (float(seg[0][0]), float(seg[0][1])), (float(seg[1][0]), float(seg[1][1])) ]
        for n in nodes:
            p = Point(n)
            if p.distance(line) <= snap_tol:
                # ensure within segment bounds
                # project n onto line to get parameter t
                proj = line.project(p)
                # get projected point coords
                proj_point = line.interpolate(proj)
                px, py = float(proj_point.x), float(proj_point.y)
                # confirm point lies between endpoints (with small margin)
                if min(seg[0][0], seg[1][0]) - snap_tol <= px <= max(seg[0][0], seg[1][0]) + snap_tol and min(seg[0][1], seg[1][1]) - snap_tol <= py <= max(seg[0][1], seg[1][1]) + snap_tol:
                    pts.append((px, py))
        # dedupe and sort pts along the segment
        # compute parameter t along the line from 0..length
        base = LineString([seg[0], seg[1]])
        unique = {}
        for p in pts:
            t = base.project(Point(p))
            unique[round(t,4)] = p
        ordered = [unique[k] for k in sorted(unique.keys())]
        # create consecutive subsegments
        for i in range(len(ordered)-1):
            a = ordered[i]; b = ordered[i+1]
            # skip degenerate
            if math.hypot(b[0]-a[0], b[1]-a[1]) < 1e-6:
                continue
            result.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
    return result

def build_edges_from_segments(segments, snap_tol=6.0):
    """
    Build unique edges from given segments by snapping endpoints that are near each other (snap_tol).
    Returns a list of unique edges ((x1,y1),(x2,y2)) with endpoints ordered consistently.
    """
    # collect all endpoints
    pts = []
    for s in segments:
        pts.append((s[0][0], s[0][1]))
        pts.append((s[1][0], s[1][1]))
    # merge nearby points
    nodes = snap_and_merge_points(pts, snap_tol=snap_tol)
    # create mapping from original coord -> nearest node
    def find_node(p):
        x,y = p
        best = None; bestd = 1e12
        for n in nodes:
            d = math.hypot(n[0]-x, n[1]-y)
            if d < bestd:
                bestd = d
                best = n
        return best
    edges_set = set()
    for s in segments:
        a = find_node(s[0]); b = find_node(s[1])
        if a is None or b is None:
            continue
        if math.hypot(a[0]-b[0], a[1]-b[1]) < 1e-6:
            continue
        # order endpoints lexicographically to dedupe
        a_t = (round(a[0],6), round(a[1],6))
        b_t = (round(b[0],6), round(b[1],6))
        if b_t < a_t:
            a_t, b_t = b_t, a_t
        edges_set.add((a_t, b_t))
    edges = [ ( (e[0][0], e[0][1]), (e[1][0], e[1][1]) ) for e in edges_set ]
    # return nodes (merged) and edges
    return nodes, edges

# Convenience high-level function
def build_graph_from_segments(segments, snap_tol=6.0):
    """
    segments: [ ((x1,y1),(x2,y2)), ... ]
    Returns: nodes, edges
    """
    # endpoints + intersections
    endpoints = []
    for s in segments:
        endpoints.append((s[0][0], s[0][1]))
        endpoints.append((s[1][0], s[1][1]))
    inters = compute_intersections(segments)
    merged_points = snap_and_merge_points(endpoints + inters, snap_tol=snap_tol)
    # split segments at the nodes (intersections/projections)
    split_segs = split_segments_at_nodes(segments, merged_points, snap_tol=snap_tol)
    nodes, edges = build_edges_from_segments(split_segs, snap_tol=snap_tol)
    return nodes, edges
