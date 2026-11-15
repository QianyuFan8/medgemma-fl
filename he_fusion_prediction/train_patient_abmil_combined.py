"""Patient-grouped 5-fold ABMIL on H-Optimus-1 + EmbeddingGemma 2 bags."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, confusion_matrix


class GatedABMIL(torch.nn.Module):
    def __init__(self, dim=2304, hidden=512, attn=256, dropout=0.25):
        super().__init__()
        self.embed = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout))
        self.att_v = torch.nn.Sequential(torch.nn.Linear(hidden, attn), torch.nn.Tanh())
        self.att_u = torch.nn.Sequential(torch.nn.Linear(hidden, attn), torch.nn.Sigmoid())
        self.att_w = torch.nn.Linear(attn, 1)
        self.head = torch.nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.embed(x)
        a = self.att_w(self.att_v(h) * self.att_u(h)).squeeze(-1)
        w = a.softmax(dim=0)
        return self.head((w[:, None] * h).sum(0)), w


def load_bag(sid, hdir, edir):
    with h5py.File(hdir / f"{sid}.h5", "r") as h, h5py.File(edir / f"{sid}.h5", "r") as e:
        hc, ec = h["coords"][:], e["coords"][:]
        if not np.array_equal(hc, ec):
            raise ValueError(f"coordinate mismatch for {sid}")
        x = np.concatenate([h["feats"][:].astype("float32"), e["feats"][:].astype("float32")], axis=1)
    return x


def metrics(rows):
    y = np.array([r["label"] for r in rows])
    p = np.array([r["prob_pos"] for r in rows])
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n_patients": int(len(y)), "accuracy": float((pred == y).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            "pr_auc": float(average_precision_score(y, p)) if len(set(y)) == 2 else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hoptimus", required=True)
    ap.add_argument("--embeddinggemma", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    df = pd.read_csv(args.manifest).copy()
    df["label_num"] = df["label"].map({"NEG": 0, "POS": 1})
    if df["label_num"].isna().any(): raise ValueError("labels must be NEG/POS")
    hdir, edir = Path(args.hoptimus), Path(args.embeddinggemma)
    bags = {sid: load_bag(sid, hdir, edir) for sid in sorted(df["Slide ID"].astype(str).unique())}
    # Training-only per-encoder standardization is intentionally omitted here; frozen
    # embeddings are concatenated directly, while the ABMIL projection learns scale.
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    patient_info = df.groupby("patient_id", as_index=False).agg(label_num=("label_num", "first"), fold=("fold", "first"))
    oof = []
    for fold in sorted(patient_info["fold"].unique()):
        train_pat = set(patient_info.loc[patient_info.fold != fold, "patient_id"])
        test_pat = set(patient_info.loc[patient_info.fold == fold, "patient_id"])
        train = df[df.patient_id.isin(train_pat)]
        test = df[df.patient_id.isin(test_pat)]
        model = GatedABMIL().to(device)
        pos = int(train.label_num.sum()); n = len(train)
        weight = torch.tensor([n / (2 * (n - pos)), n / (2 * pos)], device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        for _ in range(args.epochs):
            model.train()
            for _, row in train.sample(frac=1, random_state=args.seed + fold).iterrows():
                logits, _ = model(torch.from_numpy(bags[str(row["Slide ID"])]).to(device))
                loss = loss_fn(logits[None, :], torch.tensor([int(row["label_num"])], device=device))
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); slide_scores = {}
        with torch.no_grad():
            for _, row in test.iterrows():
                sid = str(row["Slide ID"])
                slide_scores.setdefault(row["patient_id"], []).append(float(model(torch.from_numpy(bags[sid]).to(device))[0].softmax(-1)[1]))
        for pid, scores in slide_scores.items():
            label = int(patient_info.loc[patient_info.patient_id == pid, "label_num"].iloc[0])
            oof.append({"patient_id": pid, "fold": int(fold), "label": label, "prob_pos": float(np.mean(scores))})
    pd.DataFrame(oof).to_csv(out / "patient_oof_predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics(oof), indent=2) + "\n")
    print(json.dumps(metrics(oof), indent=2))


if __name__ == "__main__":
    main()
