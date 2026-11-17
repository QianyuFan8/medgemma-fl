import json
import joblib
import numpy as np
import pandas as pd
from ..common import digest
from ..evaluation.metrics import metrics
from ..cohort.policy import assert_training_policy

def load_model(args, p, receipt):
    artifact = joblib.load(args.model)  # Load only a trusted locally trained artifact.
    if artifact["cohort_sha256"] != digest(args.cohort / "patients.csv") or artifact["matrix_receipt"] != receipt:
        raise ValueError("Model, matrix and cohort provenance differ")
    if set(artifact["training_patient_ids"]) != set(p.loc[p.teacher_train, "patient_id"]):
        raise ValueError("Teacher training membership differs")
    return artifact["model"]


def predict(args, p, X, receipt):
    assert_training_policy(p)
    model = load_model(args, p, receipt)
    if args.action == "evaluate":
        if not args.split:
            raise ValueError("Specify --split explicitly; test is never evaluated automatically")
        selected = p.split.eq(args.split).to_numpy()
    else:
        if not .5 <= args.min_confidence <= 1:
            raise ValueError("--min-confidence must be between 0.5 and 1")
        selected = p.split.eq("train_silver_candidate").to_numpy()
    result = p.loc[selected, ["patient_id", "sample_id", "sentrix_id", "slide_id", "subtype", "gold_label", "split"]].copy()
    score = model.predict_proba(X[selected])[:, 1]
    result["score_pos"], result["predicted_label"] = score, np.where(score >= .5, "POS", "NEG")
    result["teacher_model_sha256"] = digest(args.model)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.action == "silver":
        if not result.gold_label.eq("UNKNOWN").all():
            raise ValueError("Refusing to overwrite a real label")
        result["label_source"] = "silver"
        result["confidence"] = np.maximum(score, 1-score)
        result["included"] = result.confidence >= args.min_confidence
        result["confidence_cutoff"] = args.min_confidence
        result["student_loss_weight"] = np.where(result.included, .3, 0.)
        result["teacher_training_patient_count"] = int(p.teacher_train.sum())
        result["cohort_sha256"] = digest(args.cohort / "patients.csv")
        result.to_csv(args.out / "silver_labels.csv", index=False)
        gold = p.loc[p.split.eq("train_gold"), ["patient_id", "sample_id", "sentrix_id", "slide_id", "subtype", "gold_label"]].copy()
        gold["training_label"], gold["label_source"], gold["student_loss_weight"] = gold.gold_label, "gold", 1.
        silver = result.loc[result.included, ["patient_id", "sample_id", "sentrix_id", "slide_id", "subtype", "predicted_label", "label_source", "student_loss_weight"]].rename(columns={"predicted_label":"training_label"})
        pd.concat([gold.drop(columns="gold_label"), silver]).to_csv(args.out / "student_train.csv", index=False)
    else:
        result.to_csv(args.out / f"{args.split}_predictions.csv", index=False)
        (args.out / "metrics.json").write_text(json.dumps(metrics(result.gold_label.eq("POS"), score), indent=2))

