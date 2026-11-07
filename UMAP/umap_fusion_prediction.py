"""Project FOXO1 fusion status onto the COMET patient-tumor UMAP.

Uses the coordinates already saved by umap_patient_tumor_subtype.py and the
outputs of fusion_status_classifier.py. Only ARMS and ERMS are drawn.

  circle    labeled sample, colored by its Final Fusion Status
  triangle  unlabeled sample, colored by the final model's prediction
  blue      labeled sample whose out-of-fold prediction was wrong
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
COORDS = HERE / "umap_outputs" / "comet_patient_tumor_umap_subtype.csv"
FUSION = ROOT / "fusion_classifier_outputs"
OUT_PNG = HERE / "umap_outputs" / "comet_umap_fusion_prediction.png"
OUT_CSV = HERE / "umap_outputs" / "comet_umap_fusion_prediction.csv"
SUBTYPE_PNG = HERE / "umap_outputs" / "comet_umap_subtype_projection.png"
KEEP_SUBTYPES = ["ARMS", "ERMS"]
SUBTYPE_COLORS = {"ARMS": "#1f77b4", "ERMS": "#2ca02c"}

GREY = "#B4B4B4"
RED = "#D62728"
WRONG = "#1F5FD6"


def load_table() -> pd.DataFrame:
    coords = pd.read_csv(COORDS, dtype={"sample_id": str})
    coords = coords.loc[coords["Subtype Use For Clustering"].isin(KEEP_SUBTYPES), ["sample_id", "UMAP1", "UMAP2"]]

    labeled = pd.read_csv(FUSION / "labeled_oof_predictions.csv", dtype={"Sentrix_ID": str})
    labeled["group"] = "labeled"
    labeled["status"] = labeled["Final Fusion Status"]
    labeled["wrong"] = labeled["predicted_fusion_status"].ne(labeled["Final Fusion Status"])

    missing = pd.read_csv(FUSION / "missing_fusion_predictions.csv", dtype={"Sentrix_ID": str})
    missing["group"] = "predicted"
    missing["status"] = missing["predicted_fusion_status"]
    missing["wrong"] = False

    columns = [
        "Sample ID",
        "Comp Bio ID",
        "Sentrix_ID",
        "Subtype Use For Clustering",
        "group",
        "status",
        "predicted_pos_probability",
        "wrong",
    ]
    samples = pd.concat([labeled[columns], missing[columns]], ignore_index=True)
    table = samples.merge(coords, left_on="Sentrix_ID", right_on="sample_id", how="left")
    absent = table["UMAP1"].isna()
    if absent.any():
        raise ValueError(f"{absent.sum()} samples have no UMAP coordinates, e.g. {table.loc[absent, 'Sample ID'].head(3).tolist()}")
    if len(table) != len(coords):
        raise ValueError(f"{len(coords)} ARMS/ERMS points on the UMAP but {len(table)} classifier samples")
    return table.drop(columns="sample_id")


def plot(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    layers = [
        ("labeled", "NEG", "o", 16, GREY),
        ("predicted", "NEG", "^", 46, GREY),
        ("labeled", "POS", "o", 16, RED),
        ("predicted", "POS", "^", 46, RED),
    ]
    for group, status, marker, size, color in layers:
        rows = table["group"].eq(group) & table["status"].eq(status) & ~table["wrong"]
        ax.scatter(
            table.loc[rows, "UMAP1"],
            table.loc[rows, "UMAP2"],
            s=size,
            marker=marker,
            color=color,
            edgecolors="white" if group == "predicted" else "none",
            linewidths=0.4,
            alpha=0.9,
        )

    wrong = table.loc[table["wrong"]]
    ax.scatter(wrong["UMAP1"], wrong["UMAP2"], s=44, marker="o", color=WRONG, edgecolors="black", linewidths=0.8, zorder=5)
    # Neighbouring errors (same patient, or close in UMAP) alternate above and below.
    for i, (_, row) in enumerate(wrong.sort_values("UMAP1").iterrows()):
        ax.annotate(
            f"{row['Sample ID']} (label {row['status']}, p={row['predicted_pos_probability']:.2f})",
            (row["UMAP1"], row["UMAP2"]),
            xytext=(6, 7 if i % 2 == 0 else -13),
            textcoords="offset points",
            fontsize=7.5,
            color=WRONG,
        )

    def count(group: str, status: str) -> int:
        return int((table["group"].eq(group) & table["status"].eq(status) & ~table["wrong"]).sum())

    handles = [
        Line2D([], [], marker="o", ls="", color=GREY, markersize=5, label=f"Known NEG (n={count('labeled', 'NEG')})"),
        Line2D([], [], marker="o", ls="", color=RED, markersize=5, label=f"Known POS (n={count('labeled', 'POS')})"),
        Line2D([], [], marker="^", ls="", color=GREY, markersize=8, label=f"Predicted NEG (n={count('predicted', 'NEG')})"),
        Line2D([], [], marker="^", ls="", color=RED, markersize=8, label=f"Predicted POS (n={count('predicted', 'POS')})"),
        Line2D([], [], marker="o", ls="", color=WRONG, markeredgecolor="black", markersize=7, label=f"Mispredicted in CV (n={len(wrong)})"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
    ax.set(
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        title="Top 5% variance CpGs; subtype n≥10 coordinates; Color by fusion status",
    )
    fig.suptitle("COMET RMS UMAP (Patient tumor)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_by_subtype(table: pd.DataFrame) -> None:
    """Show the same coordinates colored by subtype, matching the source UMAP."""
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    for subtype in KEEP_SUBTYPES:
        rows = table["Subtype Use For Clustering"].eq(subtype)
        ax.scatter(
            table.loc[rows, "UMAP1"],
            table.loc[rows, "UMAP2"],
            s=22,
            marker="o",
            color=SUBTYPE_COLORS[subtype],
            alpha=0.9,
            linewidths=0,
            label=f"{subtype} (n={int(rows.sum())})",
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
    ax.set(
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        title="Top 5% variance CpGs; subtype n≥10 coordinates; Color by subtype",
    )
    fig.suptitle("COMET RMS UMAP (Patient tumor)", fontsize=14)
    fig.tight_layout()
    fig.savefig(SUBTYPE_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    table = load_table()
    table.to_csv(OUT_CSV, index=False)
    plot(table)
    plot_by_subtype(table)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {SUBTYPE_PNG}")


if __name__ == "__main__":
    main()
