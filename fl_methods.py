"""Non-IID federated methods for the LoRA clients: FedProx, SCAFFOLD and FedLC.

All three act on the client's local training; the server keeps FedAvg aggregation.
  fedprox  : local loss += mu/2 * ||w - w_global||^2 over trainable parameters.
  scaffold : gradient correction (c - c_i) added as a linear term <c - c_i, w>; the client
             control variate c_i is updated with SCAFFOLD option II and shipped to the
             server under "model.scaffold.<param>" keys, where it is averaged into c.
             With every client participating each round, c = mean(c_i) exactly.
  fedlc    : Federated Learning with Label Distribution Skew via Logits Calibration
             (Zhang et al., ICML 2022). At the first answer token the logit of each class
             option is reduced by tau * n_c^(-1/4), n_c = local training count of class c,
             so a client does not learn to suppress classes it rarely (or never) sees.
             The cross-entropy is computed only over answer positions via the backbone +
             lm_head, avoiding full-sequence vocabulary logits.
Caveats: the optimizer is AdamW, so the SCAFFOLD update (x - y)/(sum of learning rates)
is the usual heuristic rather than the exact SGD derivation; SCAFFOLD doubles upload size.
"""
from __future__ import annotations

from collections import Counter
import math

import torch

METHODS = ("fedavg", "fedprox", "scaffold", "fedlc")
SCAFFOLD_WIRE_PREFIX = "model.scaffold."


def is_scaffold_key(key: str) -> bool:
    return key.startswith("scaffold.") or ".scaffold." in key or key.startswith(SCAFFOLD_WIRE_PREFIX)


def split_scaffold_params(params: dict) -> tuple[dict, dict]:
    """Separate adapter weights from server control variates (keyed by parameter name)."""
    adapter, control = {}, {}
    for key, value in params.items():
        if key.startswith(SCAFFOLD_WIRE_PREFIX):
            control[key[len(SCAFFOLD_WIRE_PREFIX):]] = value
        else:
            adapter[key] = value
    return adapter, control


def trainable_named_params(model) -> dict[str, torch.nn.Parameter]:
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def snapshot(params: dict, device=None, dtype=None) -> dict[str, torch.Tensor]:
    out = {}
    for name, p in params.items():
        t = p.detach().clone()
        if device is not None or dtype is not None:
            t = t.to(device=device or t.device, dtype=dtype or t.dtype)
        out[name] = t
    return out


def prox_penalty(params: dict, anchor: dict, mu: float) -> torch.Tensor:
    """mu/2 * sum ||p - p0||^2, accumulated in fp32."""
    total = None
    for name, p in params.items():
        d = (p.float() - anchor[name].float()).pow(2).sum()
        total = d if total is None else total + d
    if total is None:
        raise ValueError("No trainable parameters for the proximal term")
    return 0.5 * mu * total


def scaffold_linear_term(params: dict, correction: dict) -> torch.Tensor:
    """sum <c - c_i, p>; its gradient adds (c - c_i) to every parameter gradient."""
    total = None
    for name, p in params.items():
        corr = correction.get(name)
        if corr is None:
            continue
        d = (p.float() * corr.float()).sum()
        total = d if total is None else total + d
    return total if total is not None else torch.zeros((), device=next(iter(params.values())).device)


def scaffold_update_ci(ci: dict | None, c: dict | None, x: dict, y: dict, lr_sum: float) -> dict:
    """Option II of SCAFFOLD: c_i <- c_i - c + (x - y) / (K * eta), with K*eta = sum of step lrs."""
    if lr_sum <= 0:
        raise ValueError("SCAFFOLD needs a positive accumulated learning rate (no optimizer steps ran?)")
    out = {}
    for name in x:
        base = ci[name].float() if ci else torch.zeros_like(x[name], dtype=torch.float32)
        serv = c[name].float() if c else torch.zeros_like(x[name], dtype=torch.float32)
        out[name] = (base - serv + (x[name].float() - y[name].float()) / lr_sum).cpu()
    return out


def answer_offsets(labels: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Per-sample index of the first supervised token, the shared logits offset and the kept length."""
    supervised = labels != -100
    if not bool(supervised.any(dim=1).all()):
        raise ValueError("Every example must contain supervised answer tokens")
    first = supervised.float().argmax(dim=1)
    offset = int(first.min().item()) - 1
    if offset < 0:
        raise ValueError("Answer cannot start at position 0")
    return first, offset, labels.shape[1] - offset


def shifted_answer_ce(logits: torch.Tensor, labels: torch.Tensor, offset: int) -> torch.Tensor:
    """Cross-entropy over supervised positions when logits cover sequence positions offset..L-1."""
    targets = labels[:, offset + 1:]
    pred = logits[:, :-1, :]
    return torch.nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]).float(), targets.reshape(-1),
                                             ignore_index=-100, reduction="mean")


def fedlc_margins(counts: dict[str, int], tau: float, floor: float = 1e-2) -> dict[str, float]:
    """tau * n_c^(-1/4); absent classes use a small floor count so the margin stays finite."""
    return {c: tau * max(float(n), floor) ** -0.25 for c, n in counts.items()}


def fedlc_adjust(row: torch.Tensor, class_token_ids: dict[str, int], margins: dict[str, float]) -> torch.Tensor:
    """Subtract each class margin from that class's option-token logit (one position, in place)."""
    for cls, tok in class_token_ids.items():
        row[tok] = row[tok] - margins[cls]
    return row


def class_token_ids(tokenizer, classes) -> dict[str, int]:
    ids = {}
    for cls in classes:
        toks = tokenizer(cls, add_special_tokens=False)["input_ids"]
        if not toks:
            raise ValueError(f"Class {cls!r} tokenizes to nothing")
        ids[cls] = int(toks[0])
    if len(set(ids.values())) != len(ids):
        raise ValueError("Class options share a first token; FedLC needs distinct first tokens")
    return ids


class FLMethodState:
    """Per-client state for the chosen method, living across rounds inside client.py."""

    def __init__(self, method: str, mu: float = 0.0, tau: float = 1.0, classes=None):
        if method not in METHODS:
            raise ValueError(f"Unknown fl_method {method}; choose from {METHODS}")
        self.method, self.mu, self.tau, self.classes = method, mu, tau, list(classes or [])
        self.ci = None            # SCAFFOLD client control variate (CPU fp32)
        self.params = None        # trainable parameters of the live model
        self.anchor = None        # fedprox: global weights at round start (model device)
        self.correction = None    # scaffold: c - c_i (model device)
        self.x_cpu = None         # scaffold: global weights at round start (CPU fp32)
        self.server_c = None
        self.lr_sum = 0.0
        self.counts = None
        self.token_ids = None

    # ---- round boundaries -------------------------------------------------
    def start_round(self, model, server_control: dict | None, train_labels=None, tokenizer=None):
        self.params = trainable_named_params(model)
        self.lr_sum = 0.0
        if self.method == "fedprox":
            self.anchor = snapshot(self.params)
        elif self.method == "scaffold":
            self.x_cpu = snapshot(self.params, device="cpu", dtype=torch.float32)
            self.server_c = {k: v.detach().cpu().float() for k, v in server_control.items()} if server_control else None
            if self.server_c is not None and set(self.server_c) != set(self.params):
                raise ValueError("Server control variate keys do not match trainable parameters")
            if self.server_c is None and self.ci is None:
                self.correction = None
            else:
                self.correction = {}
                for name, p in self.params.items():
                    c = self.server_c[name] if self.server_c else torch.zeros_like(p, dtype=torch.float32, device="cpu")
                    ci = self.ci[name] if self.ci else torch.zeros_like(p, dtype=torch.float32, device="cpu")
                    self.correction[name] = (c - ci).to(device=p.device, dtype=p.dtype)
        elif self.method == "fedlc":
            if train_labels is None or tokenizer is None or not self.classes:
                raise ValueError("FedLC needs the local training labels, the tokenizer and the class list")
            self.counts = {c: 0 for c in self.classes}
            self.counts.update(Counter(train_labels))
            self.margins = fedlc_margins(self.counts, self.tau)
            self.token_ids = class_token_ids(tokenizer, self.classes)
            print(f"FedLC local counts={self.counts} margins={{{', '.join(f'{k}: {v:.3f}' for k, v in self.margins.items())}}}")

    def end_round(self, model) -> dict:
        """Returns extra wire params (SCAFFOLD control variate), empty for other methods."""
        if self.method != "scaffold":
            return {}
        y = snapshot(self.params, device="cpu", dtype=torch.float32)
        self.ci = scaffold_update_ci(self.ci, self.server_c, self.x_cpu, y, self.lr_sum)
        return {SCAFFOLD_WIRE_PREFIX + name: self.ci[name].to(self.params[name].dtype).contiguous()
                for name in self.ci}

    # ---- loss pieces --------------------------------------------------------
    def extra_term(self):
        if self.method == "fedprox":
            return prox_penalty(self.params, self.anchor, self.mu)
        if self.method == "scaffold" and self.correction is not None:
            return scaffold_linear_term(self.params, self.correction)
        return None

    def fedlc_loss(self, unwrapped_model, inputs):
        labels = inputs["labels"]
        feats = {k: v for k, v in inputs.items() if k != "labels"}
        first, offset, _ = answer_offsets(labels)
        base = unwrapped_model.get_base_model() if hasattr(unwrapped_model, "get_base_model") else unwrapped_model
        backbone, lm_head = base.model, base.lm_head
        device_type = labels.device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            hidden = backbone(**feats, use_cache=False).last_hidden_state[:, offset:, :]
            logits = lm_head(hidden)
        logits = logits.float()
        cap = getattr(base.config.get_text_config(), "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        for i in range(labels.shape[0]):
            pos = int(first[i].item()) - 1 - offset
            own = int(labels[i, int(first[i].item())].item())
            if own not in self.token_ids.values():
                raise ValueError(f"First answer token {own} is not a class option token {self.token_ids}; check the chat template")
            logits[i, pos] = fedlc_adjust(logits[i, pos], self.token_ids, self.margins)
        return shifted_answer_ce(logits, labels, offset)

    # ---- trainer factory ------------------------------------------------------
    def make_trainer(self, trainer_cls, **kwargs):
        state = self
        from transformers import TrainerCallback

        class _LrSum(TrainerCallback):
            def on_optimizer_step(self, args, trainer_state, control, optimizer=None, **kw):
                if optimizer is not None:
                    state.lr_sum += float(optimizer.param_groups[0]["lr"])

        class _MethodTrainer(trainer_cls):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                if state.method == "fedlc":
                    loss = state.fedlc_loss(self.accelerator.unwrap_model(model), inputs)
                    outputs = {"loss": loss}  # dict keeps Trainer.prediction_step happy (no logits)
                else:
                    loss, outputs = super().compute_loss(model, inputs, return_outputs=True,
                                                         num_items_in_batch=num_items_in_batch)
                extra = state.extra_term()
                if extra is not None:
                    loss = loss + extra
                return (loss, outputs) if return_outputs else loss

        trainer = _MethodTrainer(**kwargs)
        if self.method == "scaffold":
            trainer.add_callback(_LrSum())
        return trainer
