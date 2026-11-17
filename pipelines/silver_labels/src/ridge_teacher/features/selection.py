"""Feature selection fitted on training patients only."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import subprocess
import tempfile
from pathlib import Path
import pandas as pd

class FoldCpGSelector(TransformerMixin, BaseEstimator):
    def __init__(self, top_k=1000, coverage=.95):
        self.top_k, self.coverage = top_k, coverage

    def fit(self, X, y=None):
        finite = np.isfinite(X)
        n = finite.sum(axis=0)
        values = np.where(finite, X, 0.).astype(np.float64)
        mean = values.sum(axis=0) / np.maximum(n, 1)
        var = np.maximum((values ** 2).sum(axis=0) / np.maximum(n, 1) - mean ** 2, 0.)
        valid = (n >= max(4, int(np.ceil(len(X) * self.coverage)))) & (var > 1e-10)
        eligible = np.flatnonzero(valid)
        self.indices_ = eligible[np.argsort(-var[eligible], kind="stable")[:self.top_k]]
        if not len(self.indices_):
            raise ValueError("No CpGs pass training-fold QC")
        self.n_features_in_ = X.shape[1]
        self.n_fit_samples_ = len(X)
        return self

    def transform(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature universe changed")
        return X[:, self.indices_]


class LimmaCpGSelector(TransformerMixin, BaseEstimator):
    """R limma on training-only, median-filled M-values; return selected betas.

    Ridge receives standardized beta features for BOTH selectors, so the
    comparison changes the selector rather than changing its input transform.
    """
    def __init__(self, top_k=1000, coverage=.95, rscript="Rscript"):
        self.top_k, self.coverage, self.rscript = top_k, coverage, rscript

    def fit(self, X, y):
        if set(np.unique(y)) != {0, 1} or len(y) != len(X):
            raise ValueError("LIMMA needs binary gold labels for this training fold")
        script = Path(__file__).with_name("limma_select.R")
        with tempfile.TemporaryDirectory(prefix="ridge-limma-") as temp:
            d = Path(temp)
            # Sample-major float32 binary; avoids enormous TSV intermediates.
            np.asarray(X, dtype="<f4").tofile(d / "beta.bin")
            np.savetxt(d / "labels.txt", np.asarray(y, dtype=int), fmt="%d")
            command = [self.rscript, "--vanilla", str(script), str(d / "beta.bin"),
                       str(X.shape[0]), str(X.shape[1]), str(d / "labels.txt"),
                       str(d / "rank.tsv"), str(self.top_k), str(self.coverage)]
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError("LIMMA failed (no silent fallback to variance): " + result.stderr[-4000:])
            self.statistics_ = pd.read_csv(d / "rank.tsv", sep="\t")
        self.indices_ = self.statistics_.index0.to_numpy(dtype=int)
        if not len(self.indices_) or len(set(self.indices_)) != len(self.indices_):
            raise ValueError("LIMMA returned an empty or duplicated panel")
        self.n_features_in_, self.n_fit_samples_ = X.shape[1], len(X)
        return self

    def transform(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Feature universe changed")
        return X[:, self.indices_]


