"""Label the three groups on the existing patient-tumor UMAP.

The embedding is the top 5% variance CpGs, ranked without subtype labels.
Clusters are the three separated groups in that plane. This does not use the
limma CpGs chosen for ERMS versus ARMS.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "umap_outputs"
COORD = OUT / "comet_patient_tumor_umap_subtype.csv"
ROOT = HERE.parent
CLINICAL = ROOT / "COMET_RMS_Data" / "COMET_RMS.xlsx"
FUSION = ROOT / "COMET_RMS_Data" / "COMET_RMS_fusion_status.xlsx"
SUBTYPE_COL = "Subtype Use For Clustering"
JOIN_COL = "X850k Tumor File Name"


def assign_cluster(umap1: float, umap2: float) -> int:
    """1 left group, 2 right group, 3 small top group."""
    if umap2 >= 7.8 and umap1 < 5:
        return 3
    if umap1 >= 5.0:
        return 2
    return 1


def main() -> None:
    coords = pd.read_csv(COORD)
    clinical = pd.read_excel(CLINICAL)
    fusion = pd.read_excel(FUSION)
    file_to_sid = (
        clinical.dropna(subset=[JOIN_COL, "Sample ID"])
        .assign(**{JOIN_COL: lambda d: d[JOIN_COL].astype(str).str.strip()})
        .drop_duplicates(JOIN_COL, keep="first")
        .set_index(JOIN_COL)["Sample ID"]
        .astype(str)
        .str.strip()
    )
    table = coords.copy()
    table["Sample ID"] = table["sample_id"].map(file_to_sid)
    if table["Sample ID"].isna().any() or table["Sample ID"].duplicated().any():
        raise ValueError("Sample ID join is incomplete or duplicated")
    table["cluster"] = [assign_cluster(x, y) for x, y in zip(table.UMAP1, table.UMAP2)]
    fusion = fusion.copy()
    fusion["Sample ID"] = fusion["Sample ID"].astype(str).str.strip()
    fusion["Final Fusion Status"] = fusion["Final Fusion Status"].fillna("missing")
    table = table.merge(fusion[["Sample ID", "Final Fusion Status"]], on="Sample ID", how="left")
    table["Final Fusion Status"] = table["Final Fusion Status"].fillna("missing")

    labels = table[["Sample ID", "cluster"]].sort_values(["cluster", "Sample ID"])
    label_path = OUT / "comet_umap_cluster_labels.csv"
    labels.to_csv(label_path, index=False)

    detail_path = OUT / "comet_umap_cluster_subtype_fusion.csv"
    detail_cols = [
        "Sample ID",
        "sample_id",
        SUBTYPE_COL,
        "Final Fusion Status",
        "cluster",
        "UMAP1",
        "UMAP2",
    ]
    table[detail_cols].sort_values(["cluster", "Sample ID"]).to_csv(detail_path, index=False)

    subtype_by_cluster = pd.crosstab(table[SUBTYPE_COL], table["cluster"], margins=True)
    fusion_by_cluster = pd.crosstab(table["Final Fusion Status"], table["cluster"], margins=True)
    focus = table[table[SUBTYPE_COL].isin(["ERMS", "ARMS"])]
    three_way = pd.crosstab(
        [focus["cluster"], focus[SUBTYPE_COL]],
        focus["Final Fusion Status"],
        margins=True,
    )
    print(f"wrote {label_path} ({len(labels)} samples)")
    print(f"wrote {detail_path}")
    print("\ncluster x subtype")
    print(subtype_by_cluster.to_string())
    print("\ncluster x fusion")
    print(fusion_by_cluster.to_string())
    print("\nERMS/ARMS: cluster x subtype x fusion")
    print(three_way.to_string())


if __name__ == "__main__":
    main()
