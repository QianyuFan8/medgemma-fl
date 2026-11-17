import json
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from ..common import digest
from ..evaluation.metrics import metrics
from .search import make_search
from ..cohort.policy import assert_training_policy, POLICY_VERSION

def fit(args, p, X, receipt):
    assert_training_policy(p)
    args.out.mkdir(parents=True, exist_ok=False)
    ids = np.flatnonzero(p.teacher_train.to_numpy(dtype=bool))
    y = p.loc[ids, "gold_label"].eq("POS").astype(int).to_numpy()
    if set(y) != {0,1} or min(np.bincount(y)) < 5:
        raise ValueError("Need at least five gold teacher patients per class")
    scores = np.zeros(len(ids))
    fold_id = np.zeros(len(ids), dtype=int)
    chosen = []
    # One patient per row: stratification is already patient-grouped.
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=args.seed).split(ids, y)):
        search = make_search(args.seed + fold, args.top_k, cache=str(args.out / "cv_cache"), selector=args.selector, rscript=args.rscript)
        search.fit(X[ids[tr]], y[tr])
        scores[va] = search.predict_proba(X[ids[va]])[:, 1]
        fold_id[va] = fold
        chosen.append(search.best_params_)
        print(f"Completed outer fold {fold+1}/5", flush=True)
    oof = p.loc[ids, ["patient_id", "sample_id", "sentrix_id", "gold_label", "split"]].copy()
    oof["score_pos"], oof["outer_fold"] = scores, fold_id
    oof["prediction_source"] = "nested_patient_oof"
    oof.to_csv(args.out / "teacher_oof.csv", index=False)
    final = make_search(args.seed, args.top_k, cache=str(args.out / "cv_cache"), selector=args.selector, rscript=args.rscript)
    final.fit(X[ids], y)
    artifact = dict(model=final.best_estimator_, training_patient_ids=p.loc[ids, "patient_id"].tolist(),
                    cohort_sha256=digest(args.cohort / "patients.csv"), matrix_receipt=receipt)
    joblib.dump(artifact, args.out / "teacher.joblib")
    cpgs = (args.matrix / "cpg_ids.txt").read_text().splitlines()
    selected = final.best_estimator_.named_steps["cpg"].indices_
    (args.out / "selected_cpgs.txt").write_text("\n".join(cpgs[i] for i in selected) + "\n")
    selector = final.best_estimator_.named_steps["cpg"]
    if hasattr(selector, "statistics_"):
        stats = selector.statistics_.copy()
        stats.insert(0, "cpg_id", [cpgs[i] for i in stats.index0.astype(int)])
        stats.to_csv(args.out / "selected_cpg_statistics.tsv", sep="\t", index=False)
    report = dict(oof=metrics(y, scores), outer_chosen_parameters=chosen,
                  final_chosen_parameters=final.best_params_, seed=args.seed,
                  teacher_sha256=digest(args.out / "teacher.joblib"),
                  label_scope="Audited ERMS / ARMS gold training patients only, one array/patient",
                  training_policy=POLICY_VERSION,
                  selector=args.selector, top_k=args.top_k,
                  preprocessing="Fold-specific selection + median + StandardScaler + L2 logistic on beta values",
                  probabilities="Uncalibrated; confidence threshold is an experimental filter, not guaranteed precision")
    (args.out / "fit_report.json").write_text(json.dumps(report, indent=2))


