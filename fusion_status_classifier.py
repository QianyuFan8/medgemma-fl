"""Predict FOXO1 fusion status from EPIC methylation.

Final Fusion Status is missing for most patient tumors. Missing is not a class.
The model trains only on POS versus NEG. Samples without a call are scored
afterward and are not used as labels.

Folds are grouped by patient (RMSs_meta.txt UID.Subject), so every
array from one patient sits in the same fold. CpGs are chosen inside each
training fold by variance, so the held-out patients do not choose the features.
C is chosen by an inner 3-fold CV that is also grouped by patient.

Only ARMS and ERMS tumors are used. Spindle, RMS NOS, Infantile, Sclerosing and
Unknown are dropped before anything else, from training and from prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "COMET_RMS_Data"
OUT = ROOT / "fusion_classifier_outputs"
BETA = DATA / "RMSs_beta.txt"
TOP_CPGS = 1000
MIN_COVERAGE = 0.95
N_SPLITS = 5
CHUNK_ROWS = 20000
RANDOM_STATE = 42
CS_GRID = np.logspace(-3, 2, 10)
KEEP_SUBTYPES = ["ARMS", "ERMS"]
FUSION_DETAIL = ["PAX3 Fusion", "PAX7 Fusion", "FKHR/FOXO1 FISH"]
CLINICAL_DETAIL = [
    "Subtype",
    "Site",
    "Age At Collection (yr)",
    "Age At Dx",
    "Sex",
    "Clinical Presentation",
    "Notes 1",
]


def load_cohort() -> pd.DataFrame:
    clinical = pd.read_excel(DATA / "COMET_RMS.xlsx")
    fusion = pd.read_excel(DATA / "COMET_RMS_fusion_status.xlsx")
    meta = pd.read_csv(DATA / "RMSs_meta.txt", sep="\t", dtype=str)
    tumors = meta.loc[meta["Sample.Type"].str.strip().eq("Patient tumor"), ["Sentrix_ID", "UID.Subject"]].copy()
    tumors["Sentrix_ID"] = tumors["Sentrix_ID"].str.strip()
    tumors["UID.Subject"] = tumors["UID.Subject"].fillna("").str.strip()
    if tumors["Sentrix_ID"].duplicated().any():
        raise ValueError("RMSs_meta.txt has duplicate Sentrix_ID values")

    clinical = clinical.copy()
    clinical["Sample ID"] = clinical["Sample ID"].astype(str).str.strip()
    clinical["X850k Tumor File Name"] = clinical["X850k Tumor File Name"].astype(str).str.strip()
    clinical["Comp Bio ID"] = clinical["Comp Bio ID"].astype(str).str.strip()
    cohort = clinical.merge(tumors, left_on="X850k Tumor File Name", right_on="Sentrix_ID", how="inner")
    if cohort["UID.Subject"].eq("").any():
        raise ValueError("Some patient-tumor arrays lack RMSs_meta.txt UID.Subject")

    fusion = fusion.copy()
    fusion["Sample ID"] = fusion["Sample ID"].astype(str).str.strip()
    fusion["Final Fusion Status"] = fusion["Final Fusion Status"].fillna("missing")
    cohort = cohort.merge(fusion[["Sample ID", "Final Fusion Status"] + FUSION_DETAIL], on="Sample ID", how="left")
    cohort["Final Fusion Status"] = cohort["Final Fusion Status"].fillna("missing")
    cohort["Subtype Use For Clustering"] = cohort["Subtype Use For Clustering"].fillna("Unknown")
    cohort = cohort.loc[cohort["Subtype Use For Clustering"].isin(KEEP_SUBTYPES)]
    # Canonical patient identity is RMSs_meta.txt UID.Subject, not a Comp Bio ID prefix
    # (old and new Comp Bio numbering, e.g. SJRHB022 and SJRHB003922, are one patient).
    if not cohort["Final Fusion Status"].isin(["POS", "NEG", "missing"]).all():
        unexpected = sorted(set(cohort["Final Fusion Status"]) - {"POS", "NEG", "missing"})
        raise ValueError(f"Unexpected fusion labels: {unexpected}")
    return cohort.reset_index(drop=True)


def beta_columns(sentrix_ids: list[str]) -> list[str]:
    with open(BETA, encoding="utf-8-sig") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    available = set(header)
    columns = [f"{sample}.Ave_Beta" for sample in sentrix_ids]
    absent = [column for column in columns if column not in available]
    if absent:
        raise ValueError(f"{len(absent)} Sentrix columns are absent from RMSs_beta.txt, e.g. {absent[:3]}")
    return columns


def _eligible(cpg_ids: np.ndarray, count: np.ndarray, total: np.ndarray, total_sq: np.ndarray, n_samples: int) -> pd.DataFrame:
    denom = np.maximum(count, 1)
    variance = np.maximum(total_sq / denom - (total / denom) ** 2, 0.0)
    minimum = max(4, int(np.ceil(n_samples * MIN_COVERAGE)))
    keep = (count >= minimum) & (variance > 1e-10)
    ranked = pd.DataFrame({"cpg_id": cpg_ids[keep], "variance": variance[keep]})
    return ranked.sort_values("variance", ascending=False).head(TOP_CPGS)


def select_cpgs(cohort: pd.DataFrame, columns: list[str], fold_of_labeled: np.ndarray) -> dict[str, list[str]]:
    labeled = cohort["Final Fusion Status"].ne("missing").to_numpy()
    groups = {"all": np.flatnonzero(labeled)}
    for fold in range(N_SPLITS):
        # Variance for fold k uses the training patients, which are the labeled patients outside fold k.
        groups[f"train{fold}"] = np.flatnonzero(labeled & (fold_of_labeled != fold))

    state = {
        name: {
            "idx": idx,
            "count": [],
            "total": [],
            "total_sq": [],
        }
        for name, idx in groups.items()
    }
    cpg_parts: list[np.ndarray] = []
    print(f"Pass 1: variance on {len(columns)} arrays", flush=True)
    seen = 0
    for chunk in pd.read_csv(
        BETA,
        sep="\t",
        usecols=["TargetID"] + columns,
        chunksize=CHUNK_ROWS,
        encoding="utf-8-sig",
        na_values=["NA", "NaN", "nan", ""],
        low_memory=False,
    ):
        values = chunk[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        cpg_parts.append(chunk["TargetID"].astype(str).str.strip().to_numpy())
        finite = np.isfinite(values)
        filled = np.where(finite, values, 0.0)
        for item in state.values():
            block = filled[:, item["idx"]]
            observed = finite[:, item["idx"]]
            item["count"].append(observed.sum(axis=1).astype(np.int32))
            item["total"].append(block.sum(axis=1))
            item["total_sq"].append(np.square(block).sum(axis=1))
        seen += len(chunk)
        print(f"  scanned {seen:,} CpGs", flush=True)

    cpg_ids = np.concatenate(cpg_parts)
    selected: dict[str, list[str]] = {}
    for name, item in state.items():
        if not name.startswith("train") and name != "all":
            continue
        n_samples = len(item["idx"])
        ranked = _eligible(
            cpg_ids,
            np.concatenate(item["count"]),
            np.concatenate(item["total"]),
            np.concatenate(item["total_sq"]),
            n_samples,
        )
        selected[name] = ranked["cpg_id"].tolist()
        print(f"  {name}: {len(selected[name])} CpGs from {n_samples} arrays", flush=True)
    return selected


def load_beta(columns: list[str], cpg_ids: list[str]) -> pd.DataFrame:
    wanted = set(cpg_ids)
    parts = []
    print(f"Pass 2: loading {len(cpg_ids):,} CpGs", flush=True)
    seen = 0
    for chunk in pd.read_csv(
        BETA,
        sep="\t",
        usecols=["TargetID"] + columns,
        chunksize=CHUNK_ROWS,
        encoding="utf-8-sig",
        na_values=["NA", "NaN", "nan", ""],
        low_memory=False,
    ):
        ids = chunk["TargetID"].astype(str).str.strip()
        hit = ids.isin(wanted)
        seen += len(chunk)
        if hit.any():
            block = chunk.loc[hit, columns].apply(pd.to_numeric, errors="coerce")
            block.index = ids.loc[hit]
            parts.append(block)
        if seen % (CHUNK_ROWS * 5) == 0:
            print(f"  read {seen:,} rows", flush=True)
    beta = pd.concat(parts)
    beta = beta.loc[~beta.index.duplicated()].reindex(cpg_ids)
    return beta.T


def prepare_matrix(beta: pd.DataFrame, cpg_ids: list[str], train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    block = beta.iloc[:, beta.columns.get_indexer(cpg_ids)].to_numpy(dtype=np.float64)
    medians = np.nanmedian(block[train_rows], axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.5)
    missing = ~np.isfinite(block)
    if missing.any():
        block[missing] = np.take(medians, np.where(missing)[1])
    scaler = StandardScaler()
    scaler.fit(block[train_rows])
    return scaler.transform(block), block, scaler


def fit_model(x_train: np.ndarray, y_train: np.ndarray, patients: np.ndarray) -> LogisticRegressionCV:
    # The inner CV that picks C is grouped by patient too, so a patient's diagnosis
    # and relapse arrays never sit on both sides of an inner split.
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    model = LogisticRegressionCV(
        Cs=CS_GRID,
        cv=list(inner.split(x_train, y_train, patients)),
        penalty="l2",
        # The deployed decision is probability >= 0.5, so select C using the
        # same thresholded objective rather than AUC (which only evaluates
        # ranking and can prefer probabilities compressed around 0.5).
        scoring="balanced_accuracy",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    return model


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, calls: np.ndarray) -> dict:
    metrics = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((y_true == 0).sum()),
        "roc_auc": None,
        "average_precision": None,
        "balanced_accuracy": None,
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["average_precision"] = float(average_precision_score(y_true, scores))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, calls))
        matrix = confusion_matrix(y_true, calls, labels=[0, 1])
        metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = (int(v) for v in matrix.ravel())
    return metrics


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cohort = load_cohort()
    columns = beta_columns(cohort["Sentrix_ID"].tolist())
    labeled_positions = np.flatnonzero(cohort["Final Fusion Status"].ne("missing").to_numpy())
    y_labeled = cohort.loc[labeled_positions, "Final Fusion Status"].eq("POS").to_numpy().astype(int)
    fold_of_labeled = np.full(len(cohort), -1, dtype=int)
    groups = cohort.loc[labeled_positions, "UID.Subject"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (_, test_index) in enumerate(splitter.split(labeled_positions, y_labeled, groups)):
        fold_of_labeled[labeled_positions[test_index]] = fold
    folds_per_patient = pd.Series(fold_of_labeled[labeled_positions]).groupby(groups).nunique()
    if (folds_per_patient > 1).any():
        raise ValueError("A patient's arrays were split across folds")

    patient_ids = cohort["UID.Subject"].to_numpy()
    selected = select_cpgs(cohort, columns, fold_of_labeled)
    union = list(dict.fromkeys(cpg for name in selected for cpg in selected[name]))
    beta = load_beta(columns, union)

    oof_score = np.full(len(cohort), np.nan)
    oof_call = np.full(len(cohort), -1, dtype=int)
    for fold in range(N_SPLITS):
        train_rows = np.flatnonzero((fold_of_labeled != fold) & (fold_of_labeled >= 0))
        test_rows = np.flatnonzero(fold_of_labeled == fold)
        transformed, _, _ = prepare_matrix(beta, selected[f"train{fold}"], train_rows)
        model = fit_model(transformed[train_rows], y_labeled_from(cohort, train_rows), patient_ids[train_rows])
        oof_score[test_rows] = model.predict_proba(transformed[test_rows])[:, 1]
        oof_call[test_rows] = (oof_score[test_rows] >= 0.5).astype(int)
        print(f"fold {fold}: C={model.C_[0]:.4g}", flush=True)

    labeled = fold_of_labeled >= 0
    y_all = cohort["Final Fusion Status"].eq("POS").to_numpy().astype(int)
    metrics = {"methylation_oof": binary_metrics(y_all[labeled], oof_score[labeled], oof_call[labeled])}

    train_rows = np.flatnonzero(labeled)
    transformed, _, _ = prepare_matrix(beta, selected["all"], train_rows)
    final = fit_model(transformed[train_rows], y_labeled_from(cohort, train_rows), patient_ids[train_rows])
    missing_rows = np.flatnonzero(cohort["Final Fusion Status"].eq("missing").to_numpy())
    missing_score = final.predict_proba(transformed[missing_rows])[:, 1]

    ids = ["Sample ID", "Comp Bio ID", "UID.Subject", "Sentrix_ID", "Subtype Use For Clustering"]
    review = cohort.loc[labeled, ids + ["Final Fusion Status"]].copy()
    review["fold"] = fold_of_labeled[labeled]
    review["predicted_pos_probability"] = oof_score[labeled]
    review["predicted_fusion_status"] = np.where(oof_call[labeled] == 1, "POS", "NEG")
    review.to_csv(OUT / "labeled_oof_predictions.csv", index=False)

    wrong = review["predicted_fusion_status"].ne(review["Final Fusion Status"])
    errors = review.loc[wrong].join(cohort[FUSION_DETAIL + CLINICAL_DETAIL])
    errors.insert(0, "error", np.where(errors["Final Fusion Status"].eq("NEG"), "false positive", "false negative"))
    errors.sort_values(["error", "UID.Subject"]).to_csv(OUT / "mispredictions.csv", index=False)

    imputed = cohort.loc[missing_rows, ids].copy()
    imputed["predicted_pos_probability"] = missing_score
    imputed["predicted_fusion_status"] = np.where(missing_score >= 0.5, "POS", "NEG")
    imputed["note"] = "prediction only; not a Final Fusion Status label"
    imputed.to_csv(OUT / "missing_fusion_predictions.csv", index=False)

    summary = {
        "n_patient_tumors": int(len(cohort)),
        "n_labeled": int(labeled.sum()),
        "n_labeled_patients": int(cohort.loc[labeled, "UID.Subject"].nunique()),
        "cv": f"StratifiedGroupKFold({N_SPLITS}) grouped by UID.Subject",
        "n_missing_fusion": int((~labeled).sum()),
        "subtypes": KEEP_SUBTYPES,
        "features": "EPIC Ave_Beta only; subtype is not a model input",
        "cpgs_per_model": TOP_CPGS,
        "metrics": metrics,
        "final_model_C": float(final.C_[0]),
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def y_labeled_from(cohort: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    return cohort.iloc[rows]["Final Fusion Status"].eq("POS").to_numpy().astype(int)


if __name__ == "__main__":
    main()
