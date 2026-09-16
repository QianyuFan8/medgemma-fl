"""Exploratory tissue-label associations for paired, frozen-SAE activations.

Labels are used only after inference. Rankings are descriptive: no feature names,
statistical significance, independent validation or causal effects are inferred.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from data_utils import RAW_TISSUE_CODES, TISSUE_CLASSES


def class_statistics(features, labels, num_classes=9):
    """Mean, prevalence and tie-aware one-vs-rest AUROC for each feature.

    AUROC = P(z_positive > z_negative) + .5 P(equal). Rest samples are weighted
    equally, not rest classes; balanced sampling makes the two equivalent.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if (features.ndim != 2 or not features.shape[1] or labels.shape != (len(features),)
            or labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= num_classes)
            or not np.isfinite(features).all() or np.any(features < 0)):
        raise ValueError("Expected finite nonnegative [samples, features] and integer tissue labels")
    counts = np.bincount(labels, minlength=num_classes)
    if num_classes < 2 or np.any(counts < 2):
        raise ValueError("Class association requires at least two samples in every class")
    means = np.stack([features[labels == c].mean(axis=0) for c in range(num_classes)])
    active_counts = np.stack([(features[labels == c] > 0).sum(axis=0) for c in range(num_classes)])
    rest_counts = len(labels) - counts
    rest_means = (features.sum(axis=0)[None, :] - means * counts[:, None]) / rest_counts[:, None]
    rest_active = (active_counts.sum(axis=0)[None, :] - active_counts) / rest_counts[:, None]
    auc = np.full_like(means, 0.5)
    # Constant features have AUC .5. Only rank varying columns, usually a small
    # subset of the 16k SAE bank. Average tied ranks also handle sparse zeros.
    for j in np.flatnonzero(np.ptp(features, axis=0) > 0):
        values = features[:, j]
        _, inverse, tie_counts = np.unique(values, return_inverse=True, return_counts=True)
        ends = np.cumsum(tie_counts)
        ranks = (ends - (tie_counts - 1) / 2)[inverse]
        rank_sums = np.bincount(labels, weights=ranks, minlength=num_classes)
        auc[:, j] = (rank_sums - counts * (counts + 1) / 2) / (counts * rest_counts)
    return {"counts": counts, "mean": means, "rest_mean": rest_means,
            "mean_difference": means - rest_means, "active_count": active_counts,
            "active_fraction": active_counts / counts[:, None], "rest_active_fraction": rest_active,
            "auroc": auc}


def rank_class_features(stats, class_id, top_k=10, min_active=3):
    """Positive associations only; never fill a Top K with constant/dead features."""
    if top_k < 1 or min_active < 1:
        raise ValueError("top_k and min_active must be positive")
    eligible = np.flatnonzero((stats["auroc"][class_id] > 0.5)
                             & (stats["mean_difference"][class_id] > 0)
                             & (stats["active_count"][class_id] >= min_active))
    order = np.lexsort((eligible, -stats["mean_difference"][class_id, eligible],
                       -stats["auroc"][class_id, eligible]))
    return eligible[order[:top_k]]


def _csv(path, rows, fields):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_plotting():
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("Class heatmaps require: python -m pip install -r requirements-scope2.txt") from exc
    return matplotlib


def _heatmaps(stats, feature_ids, output_dir, stem, metric, title):
    """Shared color scales and feature IDs across models; paginate long rankings."""
    if not len(feature_ids):
        return []
    matplotlib = require_plotting()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for page, start in enumerate(range(0, len(feature_ids), 30), start=1):
        selected = np.asarray(feature_ids[start:start + 30], dtype=int)
        arrays = [stats[tag][metric][:, selected].T.copy() for tag in ("base", "tuned")]
        if metric == "mean":
            # One denominator per feature, shared across classes AND both models.
            scale = np.maximum(arrays[0].max(axis=1), arrays[1].max(axis=1))
            arrays = [np.divide(a, scale[:, None], out=np.zeros_like(a), where=scale[:, None] > 0)
                      for a in arrays]
        fig, axes = plt.subplots(1, 2, figsize=(12, max(4, .27 * len(selected) + 2)),
                                 sharey=True, layout="constrained")
        for ax, tag, values in zip(axes, ("Base", "Tuned"), arrays):
            mesh = ax.imshow(values, aspect="auto", vmin=0, vmax=1,
                             cmap="viridis" if metric == "mean" else "coolwarm")
            ax.set_xticks(range(9), RAW_TISSUE_CODES, rotation=45, ha="right")
            ax.set_yticks(range(len(selected)), [str(j) for j in selected])
            ax.set_title(tag)
            ax.set_xlabel("True tissue class")
        axes[0].set_ylabel("SAE feature ID (shared order)")
        color_label = "Mean activation / shared feature maximum" if metric == "mean" else "One-vs-rest AUROC (0.5 = no separation)"
        fig.colorbar(mesh, ax=axes, label=color_label, shrink=.85)
        fig.suptitle(f"{title}\nExploratory associations; no medical concept or causal claim", fontsize=11)
        for extension in ("png", "svg"):
            name = f"{stem}_{page:02d}.{extension}"
            fig.savefig(Path(output_dir) / name, dpi=150)
            paths.append(name)
        plt.close(fig)
    return paths


def export_class_report(base, tuned, output_dir, top_k=10, min_active=3):
    """Export class statistics for all features active in either paired model."""
    from scope2_utils import feature_comparison

    if len(base["samples"]) != len(tuned["samples"]):
        raise ValueError("Paired sample counts differ")
    for a, b in zip(base["samples"], tuned["samples"]):
        if any(a[key] != b[key] for key in ("sample_id", "label", "input_sha256")):
            raise ValueError("Class analysis requires paired sample IDs, labels and processed inputs")
    labels = np.array([row["label"] for row in base["samples"]], dtype=int)
    if any(len(model["features"]) != len(labels) for model in (base, tuned)):
        raise ValueError("Feature rows must match sample metadata")
    if len(set(row["sample_id"] for row in base["samples"])) != len(labels):
        raise ValueError("Duplicate sample IDs")
    comparison = feature_comparison(base["features"], tuned["features"])
    counts = np.bincount(labels, minlength=9)
    # Preserve legacy small smoke runs. Do not manufacture scores for absent classes.
    if len(counts) != 9 or np.any(counts < 2):
        return {"status": "skipped", "class_counts": dict(zip(RAW_TISSUE_CODES, map(int, counts))),
                "reason": "Need at least two samples in every class; use --samples_per_class 20."}
    stats = {tag: class_statistics(model["features"], labels) for tag, model in (("base", base), ("tuned", tuned))}
    active_ids = np.flatnonzero(np.maximum(base["features"].max(axis=0), tuned["features"].max(axis=0)) > 0)
    change_order = np.argsort(-comparison["mean_abs_delta"], kind="stable")
    change_order = change_order[comparison["mean_abs_delta"][change_order] > 0]
    change_ranks = {int(j): rank for rank, j in enumerate(change_order, start=1)}
    change_top = set(map(int, change_order[:top_k]))
    selected = {tag: [rank_class_features(s, c, top_k, min_active) for c in range(9)] for tag, s in stats.items()}
    class_change = stats["tuned"]["mean"] - stats["base"]["mean"]
    class_abs_change = np.stack([np.abs(tuned["features"][labels == c] - base["features"][labels == c]).mean(axis=0)
                                 for c in range(9)])

    def row(tag, c, j):
        s = stats[tag]
        return {"model": tag, "label": c, "class_code": RAW_TISSUE_CODES[c], "class_name": TISSUE_CLASSES[c],
                "feature_id": int(j), "class_samples": int(counts[c]), "rest_samples": int(len(labels) - counts[c]),
                "class_mean": float(s["mean"][c, j]), "rest_mean": float(s["rest_mean"][c, j]),
                "mean_difference": float(s["mean_difference"][c, j]),
                "class_active_count": int(s["active_count"][c, j]),
                "class_active_fraction": float(s["active_fraction"][c, j]),
                "rest_active_fraction": float(s["rest_active_fraction"][c, j]), "auroc": float(s["auroc"][c, j]),
                "base_auroc": float(stats["base"]["auroc"][c, j]), "tuned_auroc": float(stats["tuned"]["auroc"][c, j]),
                "auroc_delta": float(stats["tuned"]["auroc"][c, j] - stats["base"]["auroc"][c, j]),
                "class_mean_delta": float(class_change[c, j]), "class_mean_abs_delta": float(class_abs_change[c, j]),
                "global_mean_abs_delta": float(comparison["mean_abs_delta"][j]),
                "global_change_rank": change_ranks.get(int(j), ""), "in_global_top_changes": int(j) in change_top}

    fields = list(row("base", 0, 0))
    output_dir = Path(output_dir)
    _csv(output_dir / "class_feature_summary.csv",
         (row(tag, c, j) for tag in stats for c in range(9) for j in active_ids), fields)
    _csv(output_dir / "class_top_features.csv",
         ({"rank": rank, **row(tag, c, j)} for tag in stats for c in range(9)
          for rank, j in enumerate(selected[tag][c], start=1)), ["rank"] + fields)
    performance = []
    for tag, model in (("base", base), ("tuned", tuned)):
        for c in range(9):
            samples = [r for r in model["samples"] if r["label"] == c]
            errors = [r["relative_squared_error"] for r in samples if r["relative_squared_error"] is not None]
            performance.append({"model": tag, "class_code": RAW_TISSUE_CODES[c], "samples": len(samples),
                                "accuracy": sum(r["correct"] for r in samples) / len(samples),
                                "unparsed": sum(r["predicted_label"] < 0 for r in samples),
                                "mean_l0": float(np.mean([r["l0"] for r in samples])),
                                "mean_sample_relative_squared_error": float(np.mean(errors)) if errors else None})
    _csv(output_dir / "class_performance.csv", performance, list(performance[0]))
    # Same union and ordering for both models; no independent auto-scaling.
    union = sorted({int(j) for tag in selected for ids in selected[tag] for j in ids})
    plots = _heatmaps(stats, union, output_dir, "class_association_heatmap", "auroc", "Class-associated candidates")
    plots += _heatmaps(stats, union, output_dir, "class_activation_heatmap", "mean", "Class-associated candidates: mean activation")
    plots += _heatmaps(stats, list(map(int, change_order[:top_k])), output_dir, "top_changes_activation_heatmap",
                       "mean", "Largest paired feature changes: class mean activation")
    return {"status": "complete", "analysis_type": "exploratory, same-sample association; not validated or causal",
            "group_by": "ground_truth_label", "class_counts": dict(zip(RAW_TISSUE_CODES, map(int, counts))),
            "balanced": bool(np.all(counts == counts[0])),
            "rest_weighting": "equal weight per sample; class frequencies matter for unbalanced manifests",
            "ranking": "AUROC descending, then mean difference descending, then feature ID ascending",
            "candidate_filter": {"auroc_above": .5, "mean_difference_above": 0, "min_class_active_samples": min_active},
            "requested_top_k": top_k, "active_features_exported": len(active_ids),
            "candidate_counts": {tag: dict(zip(RAW_TISSUE_CODES, map(len, groups))) for tag, groups in selected.items()},
            "plots": plots, "notes": ["A short or empty Top K is valid; dead/constant features are not used to fill it.",
                                      "AUROC is descriptive, not a significance test or feature name.",
                                      "Inspect reconstruction diagnostics before interpreting candidates."]}
