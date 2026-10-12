"""Attention-based MIL (Ilse et al. 2018, gated) on tile embeddings, then top-k montages.

`train`  fits ABMIL for ERMS vs ARMS on TRAINING slides only (fixed epochs, no
         validation-based model selection) and scores attention for every slide.
`montage` reads the top-k attention tiles from each slide and writes a
         grid x grid PNG (default 4 x 4 of 224 px tiles = 896 x 896, MedGemma's input size).

Validation/test labels are never used to fit or select the attention model, so the
montage of a held-out slide depends only on training data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np

from subtype_utils import SUBTYPE_CLASSES, binary_auroc, binary_auprc


def read_split_table(prepared):
    rows = []
    with (Path(prepared) / "splits.tsv").open(newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("qc_pass", "TRUE").upper() == "TRUE" and r.get("slide_status") == "available":
                rows.append(r)
    if not rows:
        raise ValueError("No QC-passing patients with an available slide in splits.tsv")
    return rows


def load_bag(embeddings, slide_id):
    import h5py

    with h5py.File(Path(embeddings) / f"{slide_id}.h5", "r") as f:
        return f["feats"][:].astype(np.float32), f["coords"][:], dict(f.attrs)


def build_model(dim, hidden=512, attn=256, dropout=0.25):
    import torch.nn as nn

    class GatedABMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout))
            self.att_v = nn.Sequential(nn.Linear(hidden, attn), nn.Tanh())
            self.att_u = nn.Sequential(nn.Linear(hidden, attn), nn.Sigmoid())
            self.att_w = nn.Linear(attn, 1)
            self.head = nn.Linear(hidden, len(SUBTYPE_CLASSES))

        def forward(self, x):
            h = self.embed(x)
            a = self.att_w(self.att_v(h) * self.att_u(h)).squeeze(-1)
            weights = a.softmax(dim=0)
            return self.head((weights.unsqueeze(-1) * h).sum(0)), weights

    return GatedABMIL()


def evaluate(model, bags, device):
    import torch

    model.eval()
    truth, scores, preds = [], [], []
    with torch.no_grad():
        for feats, label in bags:
            logits, _ = model(torch.from_numpy(feats).to(device))
            prob = logits.softmax(-1)[1].item()
            truth.append(label)
            scores.append(prob)
            preds.append(int(prob >= 0.5))
    acc = float(np.mean([t == p for t, p in zip(truth, preds)]))
    out = dict(n=len(truth), accuracy=acc)
    if len(set(truth)) == 2:
        out.update(roc_auc=binary_auroc(truth, scores), pr_auc=binary_auprc(truth, scores))
    return out


def train(a):
    import torch

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    rows = read_split_table(a.prepared)
    bags = {s: [] for s in ("train", "validation", "test")}
    dim = None
    for r in rows:
        feats, _, _ = load_bag(a.embeddings, r["slide_id"])
        if a.max_tiles and len(feats) > a.max_tiles:
            feats = feats[np.sort(np.random.default_rng(a.seed).choice(len(feats), a.max_tiles, replace=False))]
        dim = feats.shape[1]
        bags[r["split"]].append((feats, SUBTYPE_CLASSES.index(r["diagnosis"]), r["slide_id"]))
    if not bags["train"] or len({b[1] for b in bags["train"]}) < 2:
        raise ValueError("Training split needs slides from both subtypes")
    model = build_model(dim, a.hidden, a.attn, a.dropout).to(device)
    n_pos = sum(b[1] for b in bags["train"])
    weight = torch.tensor([len(bags["train"]) / (2 * (len(bags["train"]) - n_pos)), len(bags["train"]) / (2 * n_pos)], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    history = []
    train_bags = [(f, l) for f, l, _ in bags["train"]]
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = list(range(len(train_bags)))
        random.shuffle(order)
        total = 0.0
        for i in order:
            feats, label = train_bags[i]
            logits, _ = model(torch.from_numpy(feats).to(device))
            loss = loss_fn(logits.unsqueeze(0), torch.tensor([label], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        record = dict(epoch=epoch, train_loss=total / len(train_bags))
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            record["train"] = evaluate(model, train_bags, device)
            if bags["validation"]:
                record["validation"] = evaluate(model, [(f, l) for f, l, _ in bags["validation"]], device)
        history.append(record)
        print(json.dumps(record))
    out.mkdir(parents=True)
    (out / "attention").mkdir()
    torch.save(dict(state_dict=model.state_dict(), dim=dim, hidden=a.hidden, attn=a.attn, dropout=a.dropout,
                    classes=SUBTYPE_CLASSES, epochs=a.epochs, seed=a.seed), out / "model.pt")
    model.eval()
    scored = []
    with torch.no_grad():
        for split, items in bags.items():
            for feats, label, slide_id in items:
                full, coords, _ = load_bag(a.embeddings, slide_id)
                logits, weights = model(torch.from_numpy(full).to(device))
                np.save(out / "attention" / f"{slide_id}.npy", weights.cpu().numpy())
                scored.append(dict(slide_id=slide_id, split=split, truth=SUBTYPE_CLASSES[label],
                                   score_ARMS=logits.softmax(-1)[1].item(), tiles=len(full)))
    with (out / "slide_scores.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0]), delimiter="\t")
        w.writeheader()
        w.writerows(scored)
    (out / "training.json").write_text(json.dumps(dict(arguments=vars(a), history=history,
                                                       n_slides={k: len(v) for k, v in bags.items()},
                                                       note="Fixed-epoch training on train slides only; validation metrics are reported, not used for selection."), indent=2) + "\n")
    print(f"Saved ABMIL model and attention for {len(scored)} slides to {out}")


def montage(a):
    import openslide
    from PIL import Image

    k = a.grid * a.grid
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for att_path in sorted(Path(a.abmil).glob("attention/*.npy")):
        slide_id = att_path.stem
        target = out / f"{slide_id}.png"
        if target.exists():
            continue
        weights = np.load(att_path)
        _, coords, attrs = load_bag(a.embeddings, slide_id)
        if len(weights) != len(coords):
            raise ValueError(f"Attention/coordinate mismatch for {slide_id}")
        top = np.argsort(-weights)[:k]
        slide_path = a.slide_dir and Path(a.slide_dir) / f"{slide_id}.svs"
        if not slide_path or not slide_path.is_file():
            slide_path = Path(attrs["slide"])
        slide = openslide.OpenSlide(str(slide_path))
        size, tile = int(attrs["tile_level0"]), int(attrs["tile"])
        canvas = Image.new("RGB", (a.grid * tile, a.grid * tile), "white")
        for j, i in enumerate(top):
            x, y = (int(v) for v in coords[i])
            region = slide.read_region((x, y), 0, (size, size)).convert("RGB").resize((tile, tile), Image.LANCZOS)
            canvas.paste(region, ((j % a.grid) * tile, (j // a.grid) * tile))
            index.append(dict(slide_id=slide_id, position=j, x=x, y=y, attention=float(weights[i])))
        slide.close()
        canvas.save(target)
        print(f"{slide_id}: montage from {len(top)} of {len(weights)} tiles")
    if index:
        with (out / "montage_index.tsv").open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(index[0]), delimiter="\t")
            if f.tell() == 0:
                w.writeheader()
            w.writerows(index)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--embeddings", required=True, help="Directory of <slide>.h5 from wsi_embed.py")
    tr.add_argument("--prepared", required=True, help="prepare_comet_subtype.py prepare output (splits.tsv)")
    tr.add_argument("--output", required=True)
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--eval-every", type=int, default=5)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--weight-decay", type=float, default=1e-5)
    tr.add_argument("--hidden", type=int, default=512)
    tr.add_argument("--attn", type=int, default=256)
    tr.add_argument("--dropout", type=float, default=0.25)
    tr.add_argument("--max-tiles", type=int, default=0, help="Random cap per training bag (0 = all tiles)")
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--seed", type=int, default=20261011)
    mo = sub.add_parser("montage")
    mo.add_argument("--embeddings", required=True)
    mo.add_argument("--abmil", required=True, help="train output directory (attention/*.npy)")
    mo.add_argument("--output", required=True)
    mo.add_argument("--slide-dir", help="Override the slide path recorded in the embeddings")
    mo.add_argument("--grid", type=int, default=4, help="grid x grid tiles; 4 = 16 tiles")
    a = p.parse_args()
    train(a) if a.command == "train" else montage(a)


if __name__ == "__main__":
    main()
