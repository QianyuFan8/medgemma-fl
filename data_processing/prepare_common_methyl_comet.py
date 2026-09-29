"""450K ∩ EPIC ∩ COMET CpGs for MedGemma DNA methylation + COMET RMS.

COMET keeps Sample.Type == Patient tumor only. TARGET-ALL-P3 tumors are ALL.
Writes under data_processing/outputs/:
    common_eligible_cpgs.csv   # 3-way common CpGs (no sample labels)
    common_top5pct_cpgs.csv    # top 5% by variance, used for UMAP
    umap_top5pct.csv           # TARGET tumors + COMET patient tumors
    umap_top5pct.png
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap.umap_ as umap

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_ROOT = REPO_ROOT / "rawdata"
METH_DIR = DATA_ROOT / "methyl"
COMET_DIR = DATA_ROOT / "COMET_RMS"
OUT_DIR = HERE / "outputs"

METH_CHUNK_ROWS = 1500
METH_MIN_PROBE_COVERAGE = 0.95
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
RANDOM_STATE = 42
FILE_PROJECT_DISEASE = {
    "TARGET-ALL-P3": "ALL",
    "TARGET-AML": "AML",
    "TARGET-CCSK": "CCSK",
    "TARGET-NBL": "NBL",
    "TARGET-OS": "OS",
    "TARGET-RT": "RT",
    "TARGET-WT": "WT",
}
NORMAL_STYPES = {"10", "11", "14", "15", "17"}
BARCODE_RE = re.compile(r"TARGET-(\d\d)-([A-Za-z0-9]+)-(\d\d)")


def is_normal_barcode(barcode: str) -> bool:
    match = BARCODE_RE.match(str(barcode))
    return match is not None and match.group(3) in NORMAL_STYPES


def inspect_ave_beta(path: Path, platform: str, file_project: str) -> dict:
    with open(path, encoding="utf-8-sig") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    beta_cols = [c for c in header[1:] if c.endswith(".Ave_Beta")]
    if not beta_cols:
        raise ValueError(f"No .Ave_Beta columns in {path.name}")
    return {
        "path": path,
        "platform": platform,
        "file_project": file_project,
        "id_col": header[0],
        "beta_cols": beta_cols,
        "sample_ids": [c[: -len(".Ave_Beta")] for c in beta_cols],
        "n_samples": len(beta_cols),
    }


def inspect_target_beta(path: Path) -> dict | None:
    match = re.search(r"_(450K|EPIC|27K)_beta\.txt$", path.name, flags=re.I)
    if match is None:
        return None
    project = re.search(r"(TARGET-[A-Z0-9-]+?)_+(?:450K|EPIC|27K)_beta\.txt$", path.name, flags=re.I)
    return inspect_ave_beta(
        path,
        platform=match.group(1).upper(),
        file_project=project.group(1).upper() if project else path.name,
    )


def keep_columns(info: dict, keep_ids: set[str]) -> dict:
    pairs = [(s, c) for s, c in zip(info["sample_ids"], info["beta_cols"]) if s in keep_ids]
    out = dict(info)
    out["sample_ids"] = [s for s, _ in pairs]
    out["beta_cols"] = [c for _, c in pairs]
    out["n_samples"] = len(pairs)
    return out


def probe_ids(path: Path, id_col: str) -> pd.Index:
    ids = pd.read_csv(path, sep="\t", usecols=[id_col], dtype=str, encoding="utf-8-sig")[id_col].str.strip()
    return pd.Index(ids.to_numpy(), name="cpg_id")


def shared_probe_ids(files: list[dict], label: str) -> pd.Index:
    common = probe_ids(files[0]["path"], files[0]["id_col"])
    for f in files[1:]:
        common = common[common.isin(probe_ids(f["path"], f["id_col"]))]
    print(f"{label}: {len(common):,} shared CpGs across {len(files)} files")
    return common


def comet_probe_ids(beta_path: Path) -> pd.Index:
    ids = []
    with open(beta_path, encoding="utf-8-sig") as handle:
        next(handle)
        for line in handle:
            ids.append(line.split("\t", 1)[0].strip())
    return pd.Index(ids, name="cpg_id")


def stream_variance(files: list[dict], universe: pd.Index) -> pd.DataFrame:
    common = pd.Index(universe.astype(str).to_numpy(), name="cpg_id")
    n_probes = len(common)
    n_samples = sum(f["n_samples"] for f in files)
    observed = np.zeros(n_probes, dtype=np.int32)
    sums = np.zeros(n_probes, dtype=np.float64)
    sums_sq = np.zeros(n_probes, dtype=np.float64)
    for f in files:
        print(f"  scanning {f['path'].name} ({f['n_samples']} arrays)")
        id_col = f["id_col"]
        for chunk in pd.read_csv(
            f["path"],
            sep="\t",
            usecols=[id_col] + f["beta_cols"],
            chunksize=METH_CHUNK_ROWS,
            encoding="utf-8-sig",
            na_values=["NA", "NaN", "nan", ""],
            low_memory=False,
        ):
            ids = chunk[id_col].astype(str).str.strip().to_numpy()
            vals = chunk[f["beta_cols"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            bad = np.isfinite(vals) & ((vals < -1e-6) | (vals > 1 + 1e-6))
            if bad.any():
                raise ValueError(f"beta outside [0, 1] in {f['path'].name}")
            positions = common.get_indexer(ids)
            keep = positions >= 0
            if not keep.any():
                continue
            positions, vals = positions[keep], vals[keep]
            finite = np.isfinite(vals)
            cleaned = np.where(finite, vals, 0.0).astype(np.float64)
            observed[positions] += finite.sum(axis=1)
            sums[positions] += cleaned.sum(axis=1)
            sums_sq[positions] += np.square(cleaned).sum(axis=1)
    denom = np.maximum(observed, 1)
    variance = np.maximum(sums_sq / denom - (sums / denom) ** 2, 0.0)
    stats = pd.DataFrame({"cpg_id": common.to_numpy(), "beta_variance": variance, "n_observed": observed})
    min_obs = max(4, math.ceil(n_samples * METH_MIN_PROBE_COVERAGE))
    eligible = stats.loc[(stats.n_observed >= min_obs) & (stats.beta_variance > 1e-10)]
    eligible = eligible.sort_values("beta_variance", ascending=False).reset_index(drop=True)
    print(f"eligible {len(eligible):,} CpGs (coverage ≥ {METH_MIN_PROBE_COVERAGE:.0%}, n={n_samples} arrays)")
    return eligible


def load_selected_beta(files: list[dict], cpg_ids: list[str]) -> np.ndarray:
    wanted = set(cpg_ids)
    blocks = []
    for f in files:
        id_col = f["id_col"]
        parts = []
        print(f"  loading top 5% from {f['path'].name}")
        for chunk in pd.read_csv(
            f["path"],
            sep="\t",
            usecols=[id_col] + f["beta_cols"],
            chunksize=METH_CHUNK_ROWS,
            encoding="utf-8-sig",
            na_values=["NA", "NaN", "nan", ""],
            low_memory=False,
        ):
            ids = chunk[id_col].astype(str).str.strip()
            hit = ids.isin(wanted)
            if hit.any():
                sub = chunk.loc[hit, f["beta_cols"]].apply(pd.to_numeric, errors="coerce")
                sub.index = ids.loc[hit]
                parts.append(sub)
        selected = pd.concat(parts)
        selected = selected.loc[~selected.index.duplicated()].reindex(cpg_ids)
        blocks.append(selected.to_numpy(dtype=np.float32))
    beta = np.concatenate(blocks, axis=1)
    medians = np.nanmedian(beta, axis=1)
    missing = ~np.isfinite(beta)
    beta[missing] = medians[np.where(missing)[0]]
    return beta


def source_label(cohort: str, platform: str) -> str:
    if cohort == "COMET":
        return "COMET"
    return platform


def scatter_by_label(ax, xy: np.ndarray, labels: pd.Series, title: str, order: list[str] | None = None) -> None:
    labels = labels.fillna("UNKNOWN").astype(str)
    codes = order if order is not None else sorted(labels.unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(codes), 1)))
    for i, lab in enumerate(codes):
        idx = labels.to_numpy() == lab
        if not idx.any():
            continue
        ax.scatter(xy[idx, 0], xy[idx, 1], s=12, color=palette[i % len(palette)], alpha=0.85, label=lab, linewidths=0)
    ax.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend(loc="best", fontsize=7, markerscale=1.4, frameon=False)


def main() -> None:
    comet_beta = COMET_DIR / "RMSs_beta.txt"
    comet_meta_path = COMET_DIR / "RMSs_meta.txt"
    for path in (METH_DIR, COMET_DIR, comet_beta, comet_meta_path):
        if not path.exists():
            raise FileNotFoundError(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    target_files = [inspect_target_beta(p) for p in sorted(METH_DIR.glob("*_beta.txt"))]
    target_files = [f for f in target_files if f is not None]
    target_tumor_files = []
    target_rows = []
    n_normal = 0
    for f in target_files:
        keep = []
        for sid, col in zip(f["sample_ids"], f["beta_cols"]):
            if is_normal_barcode(sid):
                n_normal += 1
                continue
            keep.append(sid)
            target_rows.append(
                {
                    "sample_id": sid,
                    "cohort": "TARGET",
                    "platform": f["platform"],
                    "source": source_label("TARGET", f["platform"]),
                    "file_project": f["file_project"],
                    "original_column": col,
                    "disease": FILE_PROJECT_DISEASE.get(f["file_project"], "UNKNOWN"),
                }
            )
        kept = keep_columns(f, set(keep))
        if kept["n_samples"]:
            target_tumor_files.append(kept)
    print(f"TARGET tumor arrays: {len(target_rows):,}  (dropped {n_normal} normals)")
    print(pd.crosstab(pd.DataFrame(target_rows).file_project, pd.DataFrame(target_rows).disease).to_string())

    comet_info = inspect_ave_beta(comet_beta, platform="EPIC", file_project="COMET-RMS")
    comet_meta = pd.read_csv(comet_meta_path, sep="\t")
    if "Sample.Type" not in comet_meta.columns or "Sample_Name" not in comet_meta.columns:
        raise ValueError(f"Expected Sample_Name and Sample.Type in {comet_meta_path}")
    print("COMET Sample.Type:")
    print(comet_meta["Sample.Type"].value_counts().to_string())
    tumor_names = set(comet_meta.loc[comet_meta["Sample.Type"].eq("Patient tumor"), "Sample_Name"].astype(str))
    n_drop = int((~comet_meta["Sample.Type"].eq("Patient tumor")).sum())
    comet_tumor = keep_columns(comet_info, tumor_names)
    print(f"Dropping {n_drop} xenograft/cell-line arrays")
    print(f"COMET patient tumor arrays: {comet_tumor['n_samples']}")
    comet_rows = [
        {
            "sample_id": sid,
            "cohort": "COMET",
            "platform": "EPIC",
            "source": "COMET",
            "file_project": "COMET-RMS",
            "original_column": col,
            "disease": "RMS",
        }
        for sid, col in zip(comet_tumor["sample_ids"], comet_tumor["beta_cols"])
    ]

    files_450k = [f for f in target_files if f["platform"] == "450K"]
    files_epic = [f for f in target_files if f["platform"] == "EPIC"]
    ids_450k = shared_probe_ids(files_450k, "450K")
    ids_epic = shared_probe_ids(files_epic, "EPIC")
    ids_comet = comet_probe_ids(comet_beta)
    print(f"COMET: {len(ids_comet):,} probes")
    two = ids_450k.intersection(ids_epic)
    three_way = two.intersection(ids_comet)
    print(f"450K ∩ EPIC: {len(two):,}")
    print(f"450K ∩ EPIC ∩ COMET: {len(three_way):,}  (dropped {len(two) - len(three_way):,} not on COMET)")
    if len(three_way) == 0:
        raise ValueError("No CpGs shared by 450K, EPIC, and COMET")

    pooled = target_tumor_files + [comet_tumor]
    print("Variance on TARGET tumors + COMET patient tumors, 3-way CpGs")
    eligible = stream_variance(pooled, three_way)

    n5 = max(2, math.ceil(len(eligible) * 0.05))
    top5 = eligible.iloc[:n5].copy()
    print(f"top 5% of 3-way eligible: {len(top5):,}")

    cpg_path = OUT_DIR / "common_eligible_cpgs.csv"
    top5_path = OUT_DIR / "common_top5pct_cpgs.csv"
    eligible.to_csv(cpg_path, index=False)
    top5.to_csv(top5_path, index=False)
    print(f"wrote {cpg_path}")

    beta = load_selected_beta(pooled, top5.cpg_id.astype(str).tolist())
    meta = pd.DataFrame(target_rows + comet_rows)
    sample_ids = [sid for f in pooled for sid in f["sample_ids"]]
    meta = meta.set_index("sample_id").loc[sample_ids].reset_index()
    if beta.shape[1] != len(meta):
        raise ValueError(f"{beta.shape[1]} arrays vs {len(meta)} metadata rows")

    print(f"UMAP input: {beta.shape[1]} samples x {beta.shape[0]} CpGs")
    xy = umap.UMAP(
        n_neighbors=min(UMAP_N_NEIGHBORS, beta.shape[1] - 1),
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric="euclidean",
        random_state=RANDOM_STATE,
    ).fit_transform(np.asarray(beta, dtype=np.float64).T)
    meta["UMAP1"] = xy[:, 0]
    meta["UMAP2"] = xy[:, 1]
    umap_path = OUT_DIR / "umap_top5pct.csv"
    meta.to_csv(umap_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    scatter_by_label(axes[0], xy, meta.disease, f"Top 5% 3-way CpGs (n={len(top5)})")
    scatter_by_label(axes[1], xy, meta.source, "Same embedding, 450K / EPIC / COMET", order=["450K", "EPIC", "COMET"])
    fig.suptitle("450K ∩ EPIC ∩ COMET UMAP (TARGET tumor + COMET patient tumor)", fontsize=13)
    fig.tight_layout()
    fig_path = OUT_DIR / "umap_top5pct.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {umap_path}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
