"""Export downstream image linkage; this does NOT train a tile selector."""
import pandas as pd
from ..cohort.policy import assert_training_policy


def export_manifest(cohort, silver_file, destination):
    p = pd.read_csv(cohort / "patients.csv", keep_default_na=False)
    assert_training_policy(p)
    s = pd.read_csv(silver_file, keep_default_na=False)
    accepted = set(s.loc[s.included, "patient_id"])
    if not accepted <= set(p.loc[p.split.eq("train_silver_candidate"), "patient_id"]):
        raise ValueError("Silver labels contain a patient outside the allowed training candidates")
    chosen = p[p.split.isin(["train_gold", "validation", "test"]) | p.patient_id.isin(accepted)]
    rows = chosen[["patient_id", "sample_id", "slide_id", "split", "gold_label"]].copy()
    rows["svs_filename"] = rows.slide_id.astype(str) + ".svs"
    rows["label_source"] = rows.split.map(lambda v: "silver" if v=="train_silver_candidate" else "gold")
    rows["tile_selector_status"] = "refit_on_new_training_split_required"
    rows.to_csv(destination, index=False)
