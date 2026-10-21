"""COMET RMS patient-tumor UMAP (EDA).

Read only *.Ave_Beta columns. Keep Sample.Type == Patient tumor.
Rank CpGs by variance (no subtype labels). Color after the fit by
Subtype Use For Clustering from COMET_RMS.xlsx.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BETA_CANDIDATES = [
    ROOT / "COMET_RMS_Data",
    Path("/Users/rita/medgemma-fl_old/rawdata/COMET_RMS"),
]
LABEL_CANDIDATES = [
    ROOT / "COMET_RMS_Data" / "COMET_RMS.xlsx",
    ROOT / "COMET_RMS_Data" / "COMET_RMS_autorecover.xlsb",
]
RISK_CANDIDATES = [
    ROOT / "COMET_RMS_Data" / "COMET_Risk.xlsx",
    ROOT / "COMET_RMS_Data" / "RMSs_risk.xlsx",
    Path("/Users/rita/medgemma-fl_old/COMET.xlsx"),
]
SAMPLE_ID_COL = "Sample ID"
RISK_COL = "Risk"
RISK_ID_COL = "COMET_ID"
RISK_ORDER = ("Low", "Int", "High")
RISK_COLORS = {"Low": "#4C9A2A", "Int": "#E6A800", "High": "#C0392B", "No risk label": "0.75"}
OUT_DIR = HERE / "umap_outputs"
SAMPLE_TYPE = "Patient tumor"
CHUNK_ROWS = 20000
MIN_COVERAGE = 0.95
TOP_FRACTION = 0.05
MIN_SUBTYPE_N = 10
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
RANDOM_STATE = 42
SUBTYPE_COL = "Subtype Use For Clustering"
JOIN_COL = "X850k Tumor File Name"

SUBTYPE_COLORS = {
    "ARMS": "#1f77b4",
    "ERMS": "#2ca02c",
    "RMS NOS": "#8c564b",
    "Spindle": "#7f7f7f",
    "Unknown": "#17becf",
}
SUBTYPE_DRAW = ["ERMS", "ARMS", "RMS NOS", "Spindle", "Unknown"]
SUBTYPE_LEGEND = ["ARMS", "ERMS", "RMS NOS", "Spindle", "Unknown"]


def find_beta_dir() -> Path:
    for path in BETA_CANDIDATES:
        if (path / "RMSs_beta.txt").is_file() and (path / "RMSs_meta.txt").is_file():
            return path
    raise FileNotFoundError("RMSs_beta.txt and RMSs_meta.txt were not found")


def load_labels() -> pd.DataFrame:
    for path in LABEL_CANDIDATES:
        if not path.is_file():
            continue
        if path.suffix.lower() == ".xlsb":
            frame = pd.read_excel(path, engine="pyxlsb")
        else:
            frame = pd.read_excel(path)
        keep = [JOIN_COL, SUBTYPE_COL, SAMPLE_ID_COL]
        missing = [col for col in keep if col not in frame.columns]
        if missing:
            raise ValueError(f"{path.name} is missing {missing}")
        print(f"labels: {path}")
        return frame
    raise FileNotFoundError("COMET_RMS.xlsx was not found")


def inspect_beta(path: Path) -> dict:
    with open(path, encoding="utf-8-sig") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    beta_cols = [c for c in header[1:] if c.endswith(".Ave_Beta")]
    if not beta_cols:
        raise ValueError(f"No .Ave_Beta columns in {path.name}")
    return {
        "id_col": header[0],
        "beta_cols": beta_cols,
        "sample_ids": [c[: -len(".Ave_Beta")] for c in beta_cols],
    }


def stream_variance(path: Path, id_col: str, beta_cols: list[str]) -> pd.DataFrame:
    n_samples = len(beta_cols)
    cpg_parts: list[np.ndarray] = []
    observed_parts: list[np.ndarray] = []
    sum_parts: list[np.ndarray] = []
    sumsq_parts: list[np.ndarray] = []
    print(f"Scanning variance on {n_samples} patient-tumor arrays", flush=True)
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[id_col] + beta_cols,
        chunksize=CHUNK_ROWS,
        encoding="utf-8-sig",
        na_values=["NA", "NaN", "nan", ""],
        low_memory=False,
    ):
        ids = chunk[id_col].astype(str).str.strip().to_numpy()
        vals = chunk[beta_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        finite = np.isfinite(vals)
        cleaned = np.where(finite, vals, 0.0).astype(np.float64)
        cpg_parts.append(ids)
        observed_parts.append(finite.sum(axis=1).astype(np.int32))
        sum_parts.append(cleaned.sum(axis=1))
        sumsq_parts.append(np.square(cleaned).sum(axis=1))
        print(f"  scanned {sum(len(p) for p in cpg_parts):,} CpGs", flush=True)
    observed = np.concatenate(observed_parts)
    sums = np.concatenate(sum_parts)
    sums_sq = np.concatenate(sumsq_parts)
    denom = np.maximum(observed, 1)
    variance = np.maximum(sums_sq / denom - (sums / denom) ** 2, 0.0)
    stats = pd.DataFrame(
        {
            "cpg_id": np.concatenate(cpg_parts),
            "beta_variance": variance,
            "n_observed": observed,
        }
    )
    stats = stats.drop_duplicates("cpg_id", keep="first")
    min_obs = max(4, math.ceil(n_samples * MIN_COVERAGE))
    eligible = stats.loc[(stats.n_observed >= min_obs) & (stats.beta_variance > 1e-10)]
    eligible = eligible.sort_values("beta_variance", ascending=False).reset_index(drop=True)
    print(f"scanned {len(stats):,} CpGs; eligible {len(eligible):,} (coverage ≥ {MIN_COVERAGE:.0%})", flush=True)
    return eligible


def load_selected_beta(path: Path, id_col: str, beta_cols: list[str], cpg_ids: list[str]) -> np.ndarray:
    wanted = set(cpg_ids)
    parts = []
    print(f"Loading {len(cpg_ids):,} CpGs", flush=True)
    seen = 0
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[id_col] + beta_cols,
        chunksize=CHUNK_ROWS,
        encoding="utf-8-sig",
        na_values=["NA", "NaN", "nan", ""],
        low_memory=False,
    ):
        ids = chunk[id_col].astype(str).str.strip()
        hit = ids.isin(wanted)
        seen += len(chunk)
        if hit.any():
            sub = chunk.loc[hit, beta_cols].apply(pd.to_numeric, errors="coerce")
            sub.index = ids.loc[hit]
            parts.append(sub)
        if seen % (CHUNK_ROWS * 5) == 0:
            print(f"  read {seen:,} rows", flush=True)
    selected = pd.concat(parts)
    selected = selected.loc[~selected.index.duplicated()].reindex(cpg_ids)
    beta = np.array(selected.to_numpy(dtype=np.float32), copy=True)
    medians = np.nanmedian(beta, axis=1)
    missing = ~np.isfinite(beta)
    if missing.any():
        beta[missing] = medians[np.where(missing)[0]]
    return beta


def load_risk() -> pd.DataFrame:
    for path in RISK_CANDIDATES:
        if not path.is_file():
            continue
        frame = pd.read_excel(path)
        if RISK_ID_COL not in frame.columns or RISK_COL not in frame.columns:
            frame = pd.read_excel(path, header=2)
        if RISK_ID_COL not in frame.columns or RISK_COL not in frame.columns:
            raise ValueError(f"{path.name} has no {RISK_ID_COL} and {RISK_COL}")
        print(f"risk: {path}", flush=True)
        return frame
    raise FileNotFoundError("COMET risk workbook was not found")


def to_rms_id(value) -> str:
    text = str(value).strip()
    if text.startswith("RMS_"):
        return text
    if text.replace(".", "", 1).isdigit():
        return f"RMS_{int(float(text))}"
    return text


def add_risk(table: pd.DataFrame, labels: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    file_to_sid = (
        labels.dropna(subset=[JOIN_COL, SAMPLE_ID_COL])
        .assign(**{JOIN_COL: lambda d: d[JOIN_COL].astype(str).str.strip(), SAMPLE_ID_COL: lambda d: d[SAMPLE_ID_COL].astype(str).str.strip()})
        .drop_duplicates(JOIN_COL, keep="first")
        .set_index(JOIN_COL)[SAMPLE_ID_COL]
    )
    risk_map = (
        risk.dropna(subset=[RISK_ID_COL])
        .assign(**{RISK_ID_COL: lambda d: d[RISK_ID_COL].map(to_rms_id)})
        .drop_duplicates(RISK_ID_COL, keep="first")
        .set_index(RISK_ID_COL)[RISK_COL]
    )
    out = table.copy()
    out[SAMPLE_ID_COL] = out["sample_id"].map(file_to_sid)
    out[RISK_COL] = out[SAMPLE_ID_COL].map(risk_map)
    print("Risk on UMAP samples:", flush=True)
    print(out[RISK_COL].value_counts(dropna=False).to_string(), flush=True)
    return out


def normalize_subtype(labels: pd.Series) -> pd.Series:
    return labels.fillna("No subtype label").astype(str).replace({"": "No subtype label", "nan": "No subtype label"})


def drop_rare_subtypes(table: pd.DataFrame) -> pd.DataFrame:
    labels = normalize_subtype(table[SUBTYPE_COL])
    counts = labels.value_counts()
    print("subtype counts:", flush=True)
    print(counts.to_string(), flush=True)
    rare = counts[counts < MIN_SUBTYPE_N]
    if len(rare):
        print(f"dropping subtypes with n < {MIN_SUBTYPE_N}:", flush=True)
        print(rare.to_string(), flush=True)
    keep = labels.isin(counts[counts >= MIN_SUBTYPE_N].index)
    kept = table.loc[keep].reset_index(drop=True)
    kept[SUBTYPE_COL] = normalize_subtype(kept[SUBTYPE_COL])
    return kept


def _scatter(ax, xy: np.ndarray, labels: pd.Series, draw_order: list[str], legend_order: list[str], colors: dict, title: str) -> None:
    labels = labels.where(labels.notna(), "No risk label").astype(str).replace({"": "No risk label"})
    present = [lab for lab in draw_order if (labels == lab).any()]
    extras = [lab for lab in sorted(labels.unique(), key=str) if lab not in present]
    handles = {}
    for lab in present + extras:
        idx = labels.to_numpy() == lab
        handle = ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=16,
            color=colors.get(lab, "#555555"),
            alpha=0.9,
            label=f"{lab} (n={int(idx.sum())})",
            linewidths=0,
        )
        handles[lab] = handle
    ordered = [lab for lab in legend_order if lab in handles]
    ordered += [lab for lab in handles if lab not in ordered]
    ax.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend(
        [handles[lab] for lab in ordered],
        [handles[lab].get_label() for lab in ordered],
        loc="upper right",
        fontsize=9,
        markerscale=1.3,
        frameon=False,
    )


def scatter_by_subtype(ax, xy: np.ndarray, labels: pd.Series, title: str) -> None:
    _scatter(ax, xy, labels, SUBTYPE_DRAW, SUBTYPE_LEGEND, SUBTYPE_COLORS, title)


def scatter_by_risk(ax, xy: np.ndarray, labels: pd.Series, title: str) -> None:
    _scatter(
        ax,
        xy,
        labels,
        ["No risk label", *RISK_ORDER],
        [*RISK_ORDER, "No risk label"],
        RISK_COLORS,
        title,
    )


def write_figures(table: pd.DataFrame, n_cpgs: int) -> None:
    xy = table[["UMAP1", "UMAP2"]].to_numpy()
    fig, ax = plt.subplots(figsize=(8.2, 6.3))
    scatter_by_subtype(
        ax,
        xy,
        table[SUBTYPE_COL],
        f"COMET patient tumor, top 5% variance, subtypes n≥10 (n={n_cpgs} CpGs)",
    )
    fig.suptitle("COMET RMS UMAP (Patient tumor only)", fontsize=14)
    fig.tight_layout()
    fig_path = OUT_DIR / "comet_patient_tumor_umap_subtype.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}", flush=True)

    fig, ax = plt.subplots(figsize=(8.2, 6.3))
    scatter_by_risk(ax, xy, table[RISK_COL], "COMET patient tumor, top 5% variance, subtypes n≥10")
    fig.suptitle("COMET RMS UMAP colored by Risk", fontsize=14)
    fig.tight_layout()
    risk_fig = OUT_DIR / "comet_patient_tumor_umap_risk.png"
    fig.savefig(risk_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {risk_fig}", flush=True)


def main() -> None:
    data_dir = find_beta_dir()
    beta_path = data_dir / "RMSs_beta.txt"
    meta_path = data_dir / "RMSs_meta.txt"
    print(f"beta: {beta_path}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    info = inspect_beta(beta_path)
    meta = pd.read_csv(meta_path, sep="\t")
    tumor_names = set(meta.loc[meta["Sample.Type"].eq(SAMPLE_TYPE), "Sample_Name"].astype(str).str.strip())
    keep_pairs = [(sid, col) for sid, col in zip(info["sample_ids"], info["beta_cols"]) if sid in tumor_names]
    print(f"patient-tumor arrays in beta: {len(keep_pairs)}", flush=True)

    labels = load_labels()
    file_to_subtype = (
        labels.dropna(subset=[JOIN_COL])
        .assign(**{JOIN_COL: lambda d: d[JOIN_COL].astype(str).str.strip()})
        .drop_duplicates(JOIN_COL, keep="first")
        .set_index(JOIN_COL)[SUBTYPE_COL]
    )
    table = pd.DataFrame(
        {
            "sample_id": [sid for sid, _ in keep_pairs],
            "sample_type": SAMPLE_TYPE,
        }
    )
    table[SUBTYPE_COL] = table["sample_id"].map(file_to_subtype)
    table = drop_rare_subtypes(table)
    kept_ids = table["sample_id"].tolist()
    beta_cols = [col for sid, col in keep_pairs if sid in set(kept_ids)]
    # Preserve the filtered table order.
    col_by_id = {sid: col for sid, col in keep_pairs}
    beta_cols = [col_by_id[sid] for sid in kept_ids]

    cached_cpgs = OUT_DIR / "comet_patient_tumor_top5pct_cpgs.csv"
    if cached_cpgs.is_file():
        top = pd.read_csv(cached_cpgs)
        print(f"reusing {cached_cpgs.name}: {len(top):,} CpGs", flush=True)
    else:
        eligible = stream_variance(beta_path, info["id_col"], beta_cols)
        n_top = max(2, math.ceil(len(eligible) * TOP_FRACTION))
        top = eligible.iloc[:n_top].copy()
        top.to_csv(cached_cpgs, index=False)
        print(f"top {TOP_FRACTION:.0%}: {len(top):,} CpGs -> {cached_cpgs}", flush=True)

    beta = load_selected_beta(beta_path, info["id_col"], beta_cols, top["cpg_id"].astype(str).tolist())
    if beta.shape[1] != len(table):
        raise ValueError(f"{beta.shape[1]} arrays vs {len(table)} metadata rows")
    print(f"UMAP input: {beta.shape[1]} samples x {beta.shape[0]} CpGs", flush=True)
    import umap.umap_ as umap

    xy = umap.UMAP(
        n_neighbors=min(UMAP_N_NEIGHBORS, beta.shape[1] - 1),
        min_dist=UMAP_MIN_DIST,
        n_components=2,
        metric="euclidean",
        random_state=RANDOM_STATE,
    ).fit_transform(np.asarray(beta, dtype=np.float64).T)
    table["UMAP1"] = xy[:, 0]
    table["UMAP2"] = xy[:, 1]
    labeled = add_risk(table, labels, load_risk())
    coord_path = OUT_DIR / "comet_patient_tumor_umap_subtype.csv"
    labeled.drop(columns=[SAMPLE_ID_COL, RISK_COL]).to_csv(coord_path, index=False)
    risk_path = OUT_DIR / "comet_patient_tumor_umap_risk.csv"
    labeled.to_csv(risk_path, index=False)
    print(f"wrote {coord_path} ({len(labeled)} samples)", flush=True)
    print(f"wrote {risk_path}", flush=True)
    write_figures(labeled, len(top))


def plot_saved() -> None:
    coord_path = OUT_DIR / "comet_patient_tumor_umap_subtype.csv"
    table = pd.read_csv(coord_path)
    labeled = add_risk(table, load_labels(), load_risk())
    labeled.drop(columns=[SAMPLE_ID_COL, RISK_COL]).to_csv(coord_path, index=False)
    risk_path = OUT_DIR / "comet_patient_tumor_umap_risk.csv"
    labeled.to_csv(risk_path, index=False)
    print(f"wrote {coord_path}", flush=True)
    print(f"wrote {risk_path}", flush=True)
    n_cpgs = 43293
    cached = OUT_DIR / "comet_patient_tumor_top5pct_cpgs.csv"
    if cached.is_file():
        n_cpgs = len(pd.read_csv(cached, usecols=["cpg_id"]))
    write_figures(labeled, n_cpgs)


if __name__ == "__main__":
    import sys

    if "--plot-only" in sys.argv:
        plot_saved()
    else:
        main()
