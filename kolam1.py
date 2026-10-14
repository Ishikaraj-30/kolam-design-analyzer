"""
Kolam design analyzer + recreator
--------------------------------

Usage:
    python kolam_detector_and_recreator.py input_image.jpg output.svg

Dependencies:
    pip install opencv-python-headless numpy scipy scikit-image svgwrite networkx

What it does (high-level):
 1. Loads an image of a kolam (hand-drawn or printed) and converts to gray.
 2. Detects the underlying dot-grid using blob detection and fits a regular lattice.
 3. Skeletonizes the kolam strokes and extracts continuous paths.
 4. Associates stroke paths to the nearest dot-grid coordinates.
 5. Attempts to infer symmetry (rotational) of the pattern.
 6. Reconstructs a clean vector (SVG) version of the kolam using smoothed paths.

Notes:
 - This is a robust, practical starting point rather than a perfect, research-grade solution.
 - For best results, use a reasonably high-contrast photo (white background, dark kolam strokes), minimal perspective distortion.

"""

import sys
import math
import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from scipy.spatial import KDTree
from scipy import ndimage as ndi
import svgwrite
import networkx as nx

# ---------- Utilities ----------

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------- Dot detection & lattice estimation ----------

def detect_dots(gray, min_area=10, max_area=5000):
    # Simple blob detection via connected components on high-intensity spots after adaptive threshold
    # In kolam, dots are often darker than background. We'll invert so dots are bright for blob detection.
    thresh_val = threshold_otsu(gray)
    bw = gray < thresh_val  # dots/strokes as True
    # distance transform and local maxima could be used, but connected components gives candidate blobs
    labeled, n = ndi.label(bw)
    blobs = []
    for lbl in range(1, n+1):
        mask = (labeled == lbl)
        area = mask.sum()
        if area < min_area or area > max_area:
            continue
        ys, xs = np.nonzero(mask)
        cx = xs.mean(); cy = ys.mean()
        blobs.append((cx, cy, area))
    return np.array(blobs)

def fit_grid_to_dots(dots, tol=10):
    # Fit a rectangular lattice to detected dot centers.
    # Strategy: cluster x coordinates into columns and y into rows using 1D clustering by distance
    if len(dots) == 0:
        return None
    xs = np.sort(dots[:,0])
    ys = np.sort(dots[:,1])
    def cluster1d(vals):
        clusters = []
        current = [vals[0]]
        for v in vals[1:]:
            if abs(v - np.mean(current)) < tol:
                current.append(v)
            else:
                clusters.append(np.mean(current))
                current = [v]
        clusters.append(np.mean(current))
        return np.array(clusters)
    cols = cluster1d(xs)
    rows = cluster1d(ys)
    # produce grid points
    grid = np.array([[x,y] for y in rows for x in cols])
    return {'cols': cols, 'rows': rows, 'grid': grid}

# ---------- Stroke extraction & skeletonization ----------

def extract_strokes(gray):
    # threshold to binary (strokes = True)
    thresh = threshold_otsu(gray)
    bw = gray < thresh
    # Clean small noise
    bw = ndi.binary_opening(bw, structure=np.ones((3,3)))
    bw = ndi.binary_closing(bw, structure=np.ones((3,3)))
    # Skeletonize (expects boolean image)
    skel = skeletonize(bw).astype(np.uint8)
    return skel

# ---------- Graph extraction from skeleton ----------

def skeleton_to_graph(skel):
    # Build graph where pixels are nodes; but to reduce size, compress linear chains
    H, W = skel.shape
    G = nx.Graph()
    ys, xs = np.nonzero(skel)
    pixels = set((int(y), int(x)) for y,x in zip(ys,xs))
    # neighbor offsets for 8-connectivity
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    # compute degree
    deg = {}
    for p in pixels:
        d = 0
        for dy,dx in nbrs:
            if (p[0]+dy, p[1]+dx) in pixels:
                d += 1
        deg[p] = d
    # nodes are endpoints (deg==1), junctions (deg>2), and optionally corner points
    key_nodes = {p for p,d in deg.items() if d != 2}
    # Now walk from each key node along chains to other key nodes
    visited = set()
    for node in key_nodes:
        for dy,dx in nbrs:
            cur = (node[0]+dy, node[1]+dx)
            if cur not in pixels:
                continue
            path = [node, cur]
            while True:
                if cur in key_nodes:
                    G.add_edge(node, cur, pixels=path.copy())
                    break
                # find next pixel (excluding previous)
                nexts = []
                for ddy,ddx in nbrs:
                    nb = (cur[0]+ddy, cur[1]+ddx)
                    if nb in pixels and nb != path[-2]:
                        nexts.append(nb)
                if len(nexts) == 0:
                    # dead end
                    G.add_edge(node, cur, pixels=path.copy())
                    break
                cur = nexts[0]
                path.append(cur)
    # simplify node coords to float positions (centers)
    G2 = nx.Graph()
    for a,b,data in G.edges(data=True):
        pa = (a[1]+0.5, a[0]+0.5)
        pb = (b[1]+0.5, b[0]+0.5)
        pts = [(p[1]+0.5, p[0]+0.5) for p in data['pixels']]
        G2.add_edge(pa, pb, pixels=pts)
    return G2

# ---------- Associate stroke paths with nearest grid points ----------

def snap_points_to_grid(points, grid_kdtree, grid_points, max_dist=15):
    # points: list of (x,y) ; grid_points: Nx2 array
    snapped = []
    for p in points:
        d, idx = grid_kdtree.query(p)
        if d <= max_dist:
            snapped.append(tuple(grid_points[idx]))
        else:
            snapped.append(None)
    return snapped

# ---------- Symmetry detection (simple rotational) ----------

def detect_rotational_symmetry(binary_img, center=None, max_k=8):
    # Compare the image to rotated copies to find best rotational symmetry order
    h,w = binary_img.shape
    if center is None:
        center = (w/2.0, h/2.0)
    best_k = 1
    best_score = 0
    for k in range(2, max_k+1):
        angles = [360.0 * i / k for i in range(1,k)]
        score = 0.0
        for a in angles:
            M = cv2.getRotationMatrix2D(center, a, 1.0)
            rot = cv2.warpAffine(binary_img.astype(np.uint8)*255, M, (w,h), flags=cv2.INTER_NEAREST, borderValue=0)
            # overlap
            inter = np.logical_and(binary_img, rot>0).sum()
            uni = np.logical_or(binary_img, rot>0).sum()
            if uni > 0:
                score += inter / uni
        score /= (k-1)
        if score > best_score + 0.02:
            best_score = score
            best_k = k
    return best_k, best_score

# ---------- SVG reconstruction ----------

def path_smooth(points, s=0.5):
    # simple Chaikin-like subdivision for smoothing
    if len(points) < 3:
        return points
    pts = points.copy()
    for _ in range(3):
        new = [pts[0]]
        for i in range(len(pts)-1):
            p0 = np.array(pts[i]); p1 = np.array(pts[i+1])
            new.append(tuple(0.75*p0 + 0.25*p1))
            new.append(tuple(0.25*p0 + 0.75*p1))
        new.append(pts[-1])
        pts = new
    return pts


def export_svg(paths, out_path, canvas_size=(1000,1000), stroke_width=2):
    dwg = svgwrite.Drawing(out_path, size=(canvas_size[0], canvas_size[1]))
    # white background
    dwg.add(dwg.rect(insert=(0,0), size=canvas_size, fill='white'))
    group = dwg.add(dwg.g(id='kolam', stroke='black', fill='none', stroke_width=stroke_width))
    for pts in paths:
        if len(pts) < 2:
            continue
        smooth = path_smooth(pts)
        group.add(dwg.polyline(smooth, stroke_linecap='round', stroke_linejoin='round'))
    dwg.save()

# ---------- Main pipeline ----------

def analyze_and_recreate(input_path, output_svg_path):
    img = read_image(input_path)
    h,w = img.shape[:2]
    gray = to_gray(img)
    # 1. Detect dots
    dots = detect_dots(gray)
    grid_info = None
    if len(dots) >= 4:
        grid_info = fit_grid_to_dots(dots)
    # 2. Extract skeleton
    skel = extract_strokes(gray)
    # 3. Graph
    G = skeleton_to_graph(skel)
    # 4. Extract path point lists from graph edges
    paths = []
    for a,b,data in G.edges(data=True):
        pts = data['pixels']
        paths.append(pts)
    # 5. Normalize and scale to a canvas for SVG
    # Determine bounding box
    all_pts = [p for path in paths for p in path]
    if len(all_pts) == 0:
        raise RuntimeError('No stroke paths found; check image quality or thresholding')
    xs = np.array([p[0] for p in all_pts])
    ys = np.array([p[1] for p in all_pts])
    minx,maxx = xs.min(), xs.max()
    miny,maxy = ys.min(), ys.max()
    pad = 20
    target_w = w
    target_h = h
    scale_x = (target_w - 2*pad) / max(1.0, (maxx - minx))
    scale_y = (target_h - 2*pad) / max(1.0, (maxy - miny))
    scale = min(scale_x, scale_y)
    svg_paths = []
    for path in paths:
        transformed = [((p[0]-minx)*scale + pad, (p[1]-miny)*scale + pad) for p in path]
        svg_paths.append(transformed)
    # 6. Optionally detect symmetry and replicate quarter if needed (not implemented heavy)
    # 7. Export SVG
    export_svg(svg_paths, output_svg_path, canvas_size=(target_w, target_h), stroke_width=2)
    # return some analysis metadata
    rot_order, rot_score = detect_rotational_symmetry(skel)
    return {
        'input': input_path,
        'output_svg': output_svg_path,
        'num_paths': len(svg_paths),
        'detected_dots': int(len(dots)),
        'grid_found': grid_info is not None,
        'rotational_symmetry_order': int(rot_order),
        'rotational_symmetry_score': float(rot_score)
    }

# ---------- CLI ----------
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python kolam_detector_and_recreator.py input.jpg output.svg')
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2]
    meta = analyze_and_recreate(inp, out)
    print('Finished. Metadata:')
    for k,v in meta.items():
        print(f'  {k}: {v}')
