"""From-scratch matching, 4D voting, affine RANSAC, and polygon NMS.

OpenCV is used only by the runner for image I/O, SIFT, and drawing.
Coordinates are (x, y); affine matrices are 2 x 3, mapping template to scene.
"""
from dataclasses import dataclass
from itertools import product
import math

import numpy as np


@dataclass
class Features:
    points: np.ndarray
    sizes: np.ndarray
    angles: np.ndarray  # radians
    descriptors: np.ndarray


@dataclass
class Matches:
    scene: np.ndarray
    template: np.ndarray
    ratios: np.ndarray
    distances: np.ndarray


@dataclass
class Detection:
    affine: np.ndarray
    corners: np.ndarray
    inliers: np.ndarray  # indices into the original Matches arrays
    median_error: float
    rmse: float
    coverage: float


def match_descriptors(scene, template, ratio=0.78, chunk_size=256):
    """Exact Euclidean scene -> template nearest two, using bounded memory.

    ||s-t||^2 = ||s||^2 + ||t||^2 - 2 s.t. Compare d1^2 < tau^2 d2^2
    to avoid square roots in the test. Multiple scene points may use one
    template descriptor. Equal/zero second distances fail the strict test.
    """
    if not 0 < ratio < 1 or chunk_size < 1:
        raise ValueError("ratio must be in (0, 1); chunk_size must be positive")
    scene, template = np.asarray(scene, float), np.asarray(template, float)
    if scene.ndim != 2 or template.ndim != 2 or scene.shape[1] != template.shape[1]:
        raise ValueError("Descriptor arrays must have matching feature dimensions")
    empty = Matches(np.array([], int), np.array([], int), np.array([]), np.array([]))
    if len(template) < 2 or len(scene) == 0:
        return empty
    accepted = []
    tnorm = np.sum(template * template, axis=1)
    for start in range(0, len(scene), chunk_size):
        block = scene[start:start + chunk_size]
        d2 = np.maximum(np.sum(block * block, axis=1)[:, None] + tnorm - 2 * block @ template.T, 0)
        nearest = np.argpartition(d2, 1, axis=1)[:, :2]
        values = np.take_along_axis(d2, nearest, axis=1)
        order = np.argsort(values, axis=1)
        nearest = np.take_along_axis(nearest, order, axis=1)
        values = np.take_along_axis(values, order, axis=1)
        ok = (values[:, 1] > 1e-12) & (values[:, 0] < ratio ** 2 * values[:, 1])
        rows = np.flatnonzero(ok)
        accepted.extend(zip(start + rows, nearest[rows, 0],
                            np.sqrt(values[rows, 0] / values[rows, 1]), np.sqrt(values[rows, 0])))
    if not accepted:
        return empty
    data = np.asarray(accepted)
    return Matches(data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2], data[:, 3])


def hough_clusters(template, scene, matches, active, center, cfg):
    """Sparse 4D accumulator (cx, cy, log2 scale, periodic rotation).

    Each vote enters both adjacent bins in each dimension (16 cells), so
    a boundary cannot split a tight group. Each correspondence counts once
    per cell. Rotation bins wrap at 2*pi. Identical candidate sets are merged.
    """
    ti, si = matches.template[active], matches.scene[active]
    scale = scene.sizes[si] / template.sizes[ti]
    theta = (scene.angles[si] - template.angles[ti]) % (2 * np.pi)
    delta = center - template.points[ti]
    co, sn = np.cos(theta), np.sin(theta)
    rotated = np.column_stack((co * delta[:, 0] - sn * delta[:, 1],
                               sn * delta[:, 0] + co * delta[:, 1]))
    centers = scene.points[si] + scale[:, None] * rotated
    coords = np.column_stack((centers / cfg['hough_position_bin'],
                              np.log2(scale) / cfg['hough_log_scale_bin'],
                              theta * cfg['hough_angle_bins'] / (2 * np.pi)))
    base = np.floor(coords).astype(int)
    bins = {}
    for i, origin in enumerate(base):
        if not cfg['min_scale'] <= scale[i] <= cfg['max_scale']:
            continue
        for offset in product((0, 1), repeat=4):
            cell = origin + offset
            cell[3] %= cfg['hough_angle_bins']
            bins.setdefault(tuple(cell), []).append(int(active[i]))
    groups = {tuple(ids) for ids in bins.values() if len(ids) >= cfg['min_cluster_votes']}
    return [np.asarray(ids, int) for ids in sorted(groups, key=lambda g: (-len(g), g))]


def transform(points, affine):
    return np.asarray(points) @ affine[:, :2].T + affine[:, 2]


def fit_affine(source, target):
    """Centered least squares; reject repeated/collinear source or target.

    Solve [x-mean(x), y-mean(y), 1] B = [x', y'] and undo centering.
    """
    source, target = np.asarray(source, float), np.asarray(target, float)
    if len(source) < 3 or len(source) != len(target):
        return None
    mean = source.mean(axis=0)
    design = np.column_stack((source - mean, np.ones(len(source))))
    if np.linalg.matrix_rank(design) < 3 or np.linalg.matrix_rank(target - target.mean(axis=0)) < 2:
        return None
    coeff, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    affine = coeff.T
    affine[:, 2] -= affine[:, :2] @ mean
    return affine


def plausible(affine, cfg):
    if affine is None or not np.isfinite(affine).all():
        return False
    linear = affine[:, :2]
    if np.linalg.det(linear) <= 1e-8:
        return False  # collapsed geometry and mirror flips
    singular = np.linalg.svd(linear, compute_uv=False)
    return (singular[-1] >= cfg['min_scale'] and singular[0] <= cfg['max_scale']
            and singular[0] / singular[-1] <= cfg['max_anisotropy'])


def independent_inliers(source, target, errors, threshold):
    """Count each physical feature once, including SIFT orientation duplicates.

    Prefer the smallest residual when multiple matches share coordinates.
    Distinct cartons may reuse template points because deduplication is local
    to each fitted model, never global in the descriptor matcher.
    """
    selected, seen_source, seen_target = [], set(), set()
    for i in np.argsort(errors, kind='stable'):
        if errors[i] > threshold:
            break
        p, q = tuple(np.round(source[i], 1)), tuple(np.round(target[i], 1))
        if p not in seen_source and q not in seen_target:
            selected.append(i)
            seen_source.add(p)
            seen_target.add(q)
    return np.asarray(selected, int)


def ransac_affine(source, target, cfg, rng):
    """Three-point sampling, geometric rejection, robust scoring and refit."""
    if len(source) < 3:
        return None
    best, best_ids, best_score = None, np.array([], int), (0, -math.inf)
    limit, iteration = cfg['ransac_iterations'], 0
    while iteration < limit:
        iteration += 1
        sample = rng.choice(len(source), 3, replace=False)
        # A small triangle is numerically unstable even if technically full rank.
        if any(abs(np.linalg.det((points[sample[1:]] - points[sample[0]]))) < 2.0
               for points in (source, target)):
            continue
        affine = fit_affine(source[sample], target[sample])
        if not plausible(affine, cfg):
            continue
        errors = np.linalg.norm(transform(source, affine) - target, axis=1)
        ids = independent_inliers(source, target, errors, cfg['reprojection_threshold'])
        score = (len(ids), -float(np.median(errors[ids]))) if len(ids) else (0, -math.inf)
        if score > best_score:
            best, best_ids, best_score = affine, ids, score
            success = (len(ids) / len(source)) ** 3
            if success > 0:
                needed = 1 if success >= 1 else math.ceil(math.log(0.005) / math.log1p(-success))
                limit = min(limit, max(50, needed))
    if best is None or len(best_ids) < 3:
        return None
    return refine_affine(source, target, best, cfg)


def refine_affine(source, target, affine, cfg):
    for _ in range(10):
        errors = np.linalg.norm(transform(source, affine) - target, axis=1)
        ids = independent_inliers(source, target, errors, cfg['reprojection_threshold'])
        if len(ids) < 3:
            return None
        fitted = fit_affine(source[ids], target[ids])
        if not plausible(fitted, cfg):
            return None
        converged = np.allclose(affine, fitted, atol=1e-7)
        affine = fitted
        if converged:
            break
    errors = np.linalg.norm(transform(source, affine) - target, axis=1)
    ids = independent_inliers(source, target, errors, cfg['reprojection_threshold'])
    return affine, ids, errors


def polygon_area(polygon):
    p = np.asarray(polygon)
    if len(p) < 3:
        return 0.0
    return float(abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2)


def convex_hull(points):
    """Andrew's monotone chain, used for spatial support coverage."""
    points = sorted(set(map(tuple, points)))
    if len(points) < 3:
        return np.asarray(points)
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    def half(seq):
        result = []
        for p in seq:
            while len(result) >= 2 and cross(result[-2], result[-1], p) <= 0:
                result.pop()
            result.append(p)
        return result
    return np.asarray(half(points)[:-1] + half(points[::-1])[:-1])


def convex_intersection(subject, clip):
    """Sutherland-Hodgman clipping, with either polygon winding accepted."""
    output = [np.asarray(p, float) for p in subject]
    clip = np.asarray(clip, float)
    signed = np.sum(clip[:, 0] * np.roll(clip[:, 1], -1) - clip[:, 1] * np.roll(clip[:, 0], -1))
    orientation = 1 if signed >= 0 else -1
    def cross(a, b):
        return a[0]*b[1] - a[1]*b[0]
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        inputs, output = output, []
        if not inputs:
            break
        previous = inputs[-1]
        prev_side = orientation * cross(b-a, previous-a)
        for current in inputs:
            side = orientation * cross(b-a, current-a)
            if (side >= -1e-9) != (prev_side >= -1e-9):
                denominator = prev_side - side
                if abs(denominator) > 1e-12:
                    output.append(previous + (current-previous) * prev_side / denominator)
            if side >= -1e-9:
                output.append(current)
            previous, prev_side = current, side
    return np.asarray(output, float).reshape(-1, 2)


def polygon_iou(a, b):
    intersection = polygon_area(convex_intersection(a, b))
    union = polygon_area(a) + polygon_area(b) - intersection
    return float(np.clip(intersection / union, 0, 1)) if union > 1e-12 else 0.0


def nms(detections, threshold):
    ranked = sorted(detections, key=lambda d: (-len(d.inliers), d.median_error))
    kept = []
    for detection in ranked:
        if all(polygon_iou(detection.corners, other.corners) <= threshold for other in kept):
            kept.append(detection)
    return kept


def detect(template, scene, matches, template_shape, scene_shape, cfg):
    """Greedy extraction: cluster, verify, refine globally, remove only inliers."""
    h, w = template_shape[:2]
    corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], float)
    center = np.array([(w-1)/2, (h-1)/2])
    sh, sw = scene_shape[:2]
    scene_boundary = np.array([[0, 0], [sw-1, 0], [sw-1, sh-1], [0, sh-1]], float)
    active = np.arange(len(matches.scene))
    source, target = template.points[matches.template], scene.points[matches.scene]
    detections, rounds = [], []
    rng = np.random.default_rng(cfg['seed'])
    while len(active) >= cfg['min_inliers']:
        groups = hough_clusters(template, scene, matches, active, center, cfg)
        accepted = None
        tried = 0
        for group in groups:
            tried += 1
            result = ransac_affine(source[group], target[group], cfg, rng)
            if result is None or len(result[1]) < 3:
                continue
            # Hough only seeds the fit. Recruit all remaining geometric inliers.
            result = refine_affine(source[active], target[active], result[0], cfg)
            if result is None:
                continue
            affine, local_ids, errors = result
            if len(local_ids) < cfg['min_inliers']:
                continue
            ids = active[local_ids]
            coverage = polygon_area(convex_hull(source[ids])) / polygon_area(corners)
            boundary = transform(corners, affine)
            visible = polygon_area(convex_intersection(boundary, scene_boundary)) / polygon_area(boundary)
            if coverage < cfg['min_coverage'] or visible < 0.25:
                continue
            inlier_errors = errors[local_ids]
            accepted = Detection(affine, boundary, ids, float(np.median(inlier_errors)),
                                 float(np.sqrt(np.mean(inlier_errors**2))), coverage)
            detections.append(accepted)
            # Remove even duplicate-orientation inliers, but no enclosing-box matches.
            active = active[errors > cfg['reprojection_threshold']]
            break
        rounds.append({'candidate_clusters': len(groups), 'clusters_tested': tried,
                       'accepted_inliers': len(accepted.inliers) if accepted else 0,
                       'remaining_matches': len(active)})
        if accepted is None:
            break
    return nms(detections, cfg['nms_iou']), {'rounds': rounds, 'before_nms': len(detections)}
