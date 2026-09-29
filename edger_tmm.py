"""TMM + log-CPM via edgePython (Python port of Bioconductor edgeR).

Used by prepare_rnaseq.py.
"""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import edgepython as ep


def log2_tmm_cpm(
    counts: np.ndarray,
    sample_ids: list[str] | None = None,
    gene_ids: list[str] | None = None,
    work_dir=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (log-CPM, TMM norm factors) from edgePython.

    counts is genes x samples. RSEM values are rounded to integers.
    log-CPM is ``ep.cpm(y, log=True, prior_count=1)``.
    ``sample_ids`` / ``gene_ids`` / ``work_dir`` are accepted for call compatibility.
    """
    del sample_ids, gene_ids, work_dir
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError("counts must be genes x samples")
    counts = np.round(np.maximum(counts, 0.0))

    y = ep.make_dgelist(counts=counts)
    y = ep.calc_norm_factors(y, method="TMM")
    logcpm = np.asarray(ep.cpm(y, log=True, prior_count=1), dtype=np.float64)
    factors = np.asarray(y["samples"]["norm.factors"], dtype=np.float64)
    print(
        f"edgePython TMM: {counts.shape[0]} genes x {counts.shape[1]} samples; "
        f"norm.factors min={factors.min():.3f} median={np.median(factors):.3f} "
        f"max={factors.max():.3f}"
    )
    return logcpm, factors
