"""
Cost Matrix Computation and Optimal Assignment for Multi-Object Tracking (Phase 21A)

Provides:
- Vectorized IoU cost matrix computation
- Centroid distance cost matrix
- Optimal (Hungarian/Jonker-Volgenant) assignment via scipy if available
- Greedy fallback assignment when scipy is not installed

SAFETY: No biometrics — purely geometric matching.
"""
import numpy as np
from typing import List, Tuple

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of bounding boxes.

    Args:
        bboxes_a: (N, 4) array of [x1, y1, x2, y2]
        bboxes_b: (M, 4) array of [x1, y1, x2, y2]

    Returns:
        (N, M) IoU matrix
    """
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)), dtype=np.float64)

    a = bboxes_a[:, np.newaxis, :]  # (N, 1, 4)
    b = bboxes_b[np.newaxis, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_area = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(
        0.0, inter_y2 - inter_y1
    )

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union_area = area_a + area_b - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0)


def iou_cost_matrix(
    track_bboxes: np.ndarray, det_bboxes: np.ndarray
) -> np.ndarray:
    """
    Compute IoU-based cost matrix: cost = 1 - IoU.

    Args:
        track_bboxes: (N, 4) predicted track bboxes [x1, y1, x2, y2]
        det_bboxes: (M, 4) detection bboxes [x1, y1, x2, y2]

    Returns:
        (N, M) cost matrix where 0 = perfect overlap, 1 = no overlap
    """
    return 1.0 - iou_batch(track_bboxes, det_bboxes)


def centroid_distance_matrix(
    track_bboxes: np.ndarray,
    det_bboxes: np.ndarray,
    frame_diag: float = 1.0,
) -> np.ndarray:
    """
    Compute normalized centroid distance matrix.

    Args:
        track_bboxes: (N, 4) predicted track bboxes [x1, y1, x2, y2]
        det_bboxes: (M, 4) detection bboxes [x1, y1, x2, y2]
        frame_diag: diagonal length of the frame for normalization

    Returns:
        (N, M) normalized distance matrix [0, 1+]
    """
    if len(track_bboxes) == 0 or len(det_bboxes) == 0:
        return np.zeros((len(track_bboxes), len(det_bboxes)), dtype=np.float64)

    # Compute centroids
    track_cx = (track_bboxes[:, 0] + track_bboxes[:, 2]) / 2.0
    track_cy = (track_bboxes[:, 1] + track_bboxes[:, 3]) / 2.0
    det_cx = (det_bboxes[:, 0] + det_bboxes[:, 2]) / 2.0
    det_cy = (det_bboxes[:, 1] + det_bboxes[:, 3]) / 2.0

    dx = track_cx[:, np.newaxis] - det_cx[np.newaxis, :]
    dy = track_cy[:, np.newaxis] - det_cy[np.newaxis, :]
    dist = np.sqrt(dx ** 2 + dy ** 2)

    diag = max(frame_diag, 1.0)
    return dist / diag


def linear_assignment(
    cost_matrix: np.ndarray, threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve the linear assignment problem on a cost matrix.

    Uses scipy's Jonker-Volgenant if available, otherwise greedy fallback.

    Args:
        cost_matrix: (N, M) cost matrix where lower = better match
        threshold: maximum cost for a valid assignment

    Returns:
        (matches, unmatched_rows, unmatched_cols)
    """
    n_rows, n_cols = cost_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    if _HAS_SCIPY:
        row_idx, col_idx = _scipy_lsa(cost_matrix)
    else:
        row_idx, col_idx = _greedy_assignment(cost_matrix)

    matches = []
    matched_rows = set()
    matched_cols = set()

    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] <= threshold:
            matches.append((int(r), int(c)))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_rows = sorted(set(range(n_rows)) - matched_rows)
    unmatched_cols = sorted(set(range(n_cols)) - matched_cols)
    return matches, unmatched_rows, unmatched_cols


def _greedy_assignment(
    cost_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy minimum-cost assignment fallback when scipy is unavailable.

    Iterates over costs in ascending order and greedily assigns
    the cheapest unassigned (row, col) pair.
    """
    n_rows, n_cols = cost_matrix.shape
    flat = cost_matrix.ravel()
    order = flat.argsort()

    used_rows: set = set()
    used_cols: set = set()
    row_list: list = []
    col_list: list = []

    for idx in order:
        r = int(idx // n_cols)
        c = int(idx % n_cols)
        if r not in used_rows and c not in used_cols:
            row_list.append(r)
            col_list.append(c)
            used_rows.add(r)
            used_cols.add(c)
        if len(used_rows) >= min(n_rows, n_cols):
            break

    return np.array(row_list, dtype=int), np.array(col_list, dtype=int)


def has_scipy() -> bool:
    """Check if scipy is available for optimal assignment."""
    return _HAS_SCIPY


def centroid_distance_assignment(
    cost_matrix: np.ndarray,
    distance_threshold: float = 0.25,
    ambiguity_threshold: float = 0.01,
    relative_ambiguity_margin: float = 0.10,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve linear assignment on normalized centroid distance matrix with ambiguity check.

    If multiple candidate detections are within close distance to the same track
    (or multiple candidate tracks are within close distance to the same detection),
    and their distance difference is within ambiguity_threshold or relative_ambiguity_margin,
    the match is deemed ambiguous and rejected (abstention) to prevent false identity swaps.

    Args:
        cost_matrix: (N, M) normalized distance matrix, with gated entries >= 1e5
        distance_threshold: maximum allowed normalized distance
        ambiguity_threshold: minimum absolute distance margin required to distinguish competing matches
        relative_ambiguity_margin: minimum relative distance margin ((d2 - d1) / d1) required

    Returns:
        (matches, unmatched_rows, unmatched_cols)
    """
    n_rows, n_cols = cost_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    # Run Hungarian/greedy assignment
    initial_matches, _, _ = linear_assignment(cost_matrix, threshold=distance_threshold)

    matches = []
    matched_rows = set()
    matched_cols = set()

    for r, c in initial_matches:
        best_cost = cost_matrix[r, c]
        if best_cost > distance_threshold:
            continue

        # Check row ambiguity: does track r have another candidate detection c2 with similar cost?
        is_ambiguous = False
        for c2 in range(n_cols):
            if c2 != c and cost_matrix[r, c2] <= distance_threshold:
                cost_diff = cost_matrix[r, c2] - best_cost
                rel_diff = cost_diff / max(best_cost, 1e-6)
                if cost_diff < ambiguity_threshold or rel_diff < relative_ambiguity_margin:
                    is_ambiguous = True
                    break

        # Check column ambiguity: does detection c have another candidate track r2 with similar cost?
        if not is_ambiguous:
            for r2 in range(n_rows):
                if r2 != r and cost_matrix[r2, c] <= distance_threshold:
                    cost_diff = cost_matrix[r2, c] - best_cost
                    rel_diff = cost_diff / max(best_cost, 1e-6)
                    if cost_diff < ambiguity_threshold or rel_diff < relative_ambiguity_margin:
                        is_ambiguous = True
                        break

        if is_ambiguous:
            # Prefer abstention over forcing a false association
            continue

        matches.append((int(r), int(c)))
        matched_rows.add(r)
        matched_cols.add(c)

    unmatched_rows = sorted(set(range(n_rows)) - matched_rows)
    unmatched_cols = sorted(set(range(n_cols)) - matched_cols)
    return matches, unmatched_rows, unmatched_cols

