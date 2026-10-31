"""Audit existing and diversified selections from saved H5 coordinates; no SVS/GPU needed."""
import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_preprocessing.tile_selection import select_tiles, spatial_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--embeddings', required=True)
    p.add_argument('--abmil', required=True)
    p.add_argument('--prepared', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--existing-index', required=True)
    p.add_argument('--candidate-pool', type=int, default=128)
    p.add_argument('--min-distance-tiles', type=float, default=2)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    with (Path(a.prepared) / 'splits.tsv').open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    with Path(a.existing_index).open() as f:
        index = list(csv.DictReader(f, delimiter='\t'))
    results = []; plotting = []
    for row in rows:
        slide = row['slide_id']
        weights = np.load(Path(a.abmil) / 'attention' / f'{slide}.npy')
        with h5py.File(Path(a.embeddings) / f'{slide}.h5') as f:
            coords = f['coords'][:]; size = float(f.attrs['tile_level0'])
        top, _ = select_tiles(weights, coords, 16, tile_size=size)
        old = sorted((r for r in index if r['slide_id'] == slide), key=lambda r: int(r['position']))
        if [(int(r['x']), int(r['y'])) for r in old] != [tuple(map(int, c)) for c in coords[top]]:
            raise ValueError(f'Existing montage does not match top-attention selection: {slide}')
        diverse, detail = select_tiles(weights, coords, 16, method='spatial', tile_size=size,
            candidate_pool=a.candidate_pool, min_distance_tiles=a.min_distance_tiles)
        for method, chosen in [('top', top), ('spatial', diverse)]:
            results.append(dict(slide_id=slide, split=row['split'], canonical_uid=row['canonical_uid'],
                method=method, candidate_tiles=len(coords), **spatial_metrics(coords, chosen, size),
                attention_mass=float(weights[chosen].sum()),
                relaxed_count=detail['relaxed_count'] if method == 'spatial' else 0,
                selected_coordinates=coords[chosen].tolist()))
        # Illustrative plots use train/validation only and never outcome labels.
        if row['split'] != 'test':
            plotting.append((spatial_metrics(coords, top, size)['largest_adjacent_component'],
                             slide, row['split'], coords, top, diverse, size))
    summary = {}
    for method in ['top', 'spatial']:
        group = [r for r in results if r['method'] == method]
        summary[method] = dict(slides=len(group),
            slides_with_8_or_more_adjacent_tiles=sum(r['largest_adjacent_component'] >= 8 for r in group),
            median_largest_adjacent_component=float(np.median([r['largest_adjacent_component'] for r in group])),
            median_nearest_neighbor_tiles=float(np.median([r['median_nearest_neighbor_tiles'] for r in group])),
            slides_requiring_relaxation=sum(r['relaxed_count'] > 0 for r in group))
    report = dict(arguments=vars(a), summary=summary, slides=results,
        note='Adjacency <=1.5 tile widths. Spatial metrics use sampled eligible coordinates, not full slide or tumor extent. '
             '8 adjacent tiles is a descriptive flag, not a validated QC threshold. '
             'No labels used for selection or plotting. No claim of improved classification or external-site robustness.')
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples = sorted(plotting, key=lambda r: (-r[0], r[1]))[:4]
    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 3 * len(examples)), squeeze=False)
    for line, (_, slide, split, coords, top, diverse, size) in enumerate(examples):
        for col, (name, chosen) in enumerate([('Top attention', top), ('Spatial diversity', diverse)]):
            ax = axes[line, col]
            ax.scatter(coords[:, 0]/size, coords[:, 1]/size, s=2, c='#cbd5e1')
            ax.scatter(coords[chosen, 0]/size, coords[chosen, 1]/size, s=30, c='#e11d48')
            ax.set_title(f'{slide} ({split}) | {name}')
            ax.set_aspect('equal'); ax.invert_yaxis()
            ax.set_xlabel('x / tile width'); ax.set_ylabel('y / tile width')
    fig.suptitle('Selected 16 tiles (red) within sampled tile coordinates (gray)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(out / 'spatial_comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
