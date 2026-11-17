import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from ..features.selection import FoldCpGSelector, LimmaCpGSelector

def make_search(seed, top_k=1000, folds=3, cs=None, cache=None, selector="variance", rscript="Rscript"):
    if selector not in {"variance", "limma"}:
        raise ValueError("Unknown CpG selector")
    estimator = Pipeline([
        ("cpg", FoldCpGSelector(top_k=top_k) if selector == "variance" else LimmaCpGSelector(top_k=top_k, rscript=rscript)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=3000, random_state=seed)),
    ], memory=cache)
    return GridSearchCV(estimator, {"ridge__C": np.logspace(-3, 2, 10) if cs is None else cs},
                        scoring="balanced_accuracy", cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                        n_jobs=1, refit=True, error_score="raise")


