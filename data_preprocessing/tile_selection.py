"""Label-free spatial diversification of high-attention candidate tiles.

This controls spatial redundancy; it does not certify tumor content or domain robustness.
Coordinates and tile_size must use the same level-0 pixel units.
"""
import numpy as np


def select_tiles(weights, coords, k, *, method="top", tile_size=1,
                 candidate_pool=128, min_distance_tiles=2.0):
    weights = np.asarray(weights, dtype=float)
    coords = np.asarray(coords, dtype=float)
    if weights.ndim != 1 or coords.shape != (len(weights), 2):
        raise ValueError("Attention/coordinate shape mismatch")
    if not np.isfinite(weights).all() or not np.isfinite(coords).all():
        raise ValueError("Nonfinite attention or coordinates")
    if tile_size <= 0 or min_distance_tiles < 0 or candidate_pool < 1 or k < 1:
        raise ValueError("Invalid selection parameters")
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("Duplicate tile coordinates")
    order = np.argsort(-weights, kind="stable")
    target = min(k, len(order))
    if method == "top":
        return order[:target], {"method": method, "candidate_count": len(order), "relaxed_count": 0}
    if method != "spatial":
        raise ValueError(f"Unknown selector: {method}")
    candidates = order[:max(target, candidate_pool)]
    chosen = []
    levels = []
    # Never silently select fewer tiles: relax within the fixed high-attention pool,
    # recording how many slots could not satisfy the requested separation.
    for multiplier in (1.0, 0.5, 0.0):
        minimum = tile_size * min_distance_tiles * multiplier
        for i in candidates:
            if len(chosen) == target:
                break
            if int(i) in chosen:
                continue
            if not chosen or np.min(np.linalg.norm(coords[chosen] - coords[i], axis=1)) >= minimum:
                chosen.append(int(i)); levels.append(multiplier)
    chosen = sorted(chosen, key=lambda i: (-weights[i], i))
    return np.asarray(chosen, dtype=int), dict(method=method, candidate_count=len(candidates),
        requested_min_distance_tiles=min_distance_tiles,
        relaxed_count=sum(x < 1 for x in levels),
        zero_distance_fallback_count=sum(x == 0 for x in levels))


def spatial_metrics(coords, selected, tile_size):
    coords = np.asarray(coords, dtype=float)
    points = coords[selected]
    if len(points) < 2:
        return dict(n_selected=len(points), median_nearest_neighbor_tiles=None,
                    min_distance_tiles=None, largest_adjacent_component=len(points))
    distance = np.linalg.norm(points[:, None] - points[None, :], axis=-1) / tile_size
    np.fill_diagonal(distance, np.inf)
    remaining = set(range(len(points))); largest = 0
    while remaining:
        queue = [remaining.pop()]; component = 0
        while queue:
            i = queue.pop(); component += 1
            neighbors = {j for j in remaining if distance[i, j] <= 1.5}
            remaining -= neighbors; queue.extend(neighbors)
        largest = max(largest, component)
    span = np.maximum(np.ptp(coords, axis=0), tile_size)
    cells = np.minimum(((points - coords.min(axis=0)) / span * 4).astype(int), 3)
    return dict(n_selected=len(points), min_distance_tiles=float(distance.min()),
        median_nearest_neighbor_tiles=float(np.median(distance.min(axis=1))),
        largest_adjacent_component=largest, occupied_grid_cells=len(np.unique(cells, axis=0)),
        selected_bbox_fraction=float(np.prod(np.ptp(points, axis=0) + tile_size) /
                                     np.prod(np.ptp(coords, axis=0) + tile_size)))
