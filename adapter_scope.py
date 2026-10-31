"""Explicit adapter scope shared by training and checkpoint inference (no GPU imports)."""
SCOPES = ("legacy", "lora-only")


def saved_modules(scope):
    if scope not in SCOPES:
        raise ValueError(f"Unknown adapter scope: {scope}")
    return ["lm_head", "embed_tokens"] if scope == "legacy" else None


def infer_adapter_scope(state):
    """Old NVFlare checkpoints have full embedding/head weights; new ones do not."""
    full = set()
    for key in state:
        if "lora_" in key or key.startswith("scaffold."):
            continue
        for name in ("lm_head", "embed_tokens"):
            if f".{name}." in key and key.endswith(".weight"):
                full.add(name)
    # Tied embedding/output weights may be serialized only once.
    return "legacy" if full else "lora-only"


def trainable_summary(model, scope):
    saved_modules(scope)
    rows = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    if scope == "lora-only":
        extra = [name for name, _ in rows if ".lora_A." not in name and ".lora_B." not in name]
        if extra:
            raise ValueError(f"lora-only contains non-adapter trainable parameters: {extra[:10]}")
    if not rows:
        raise ValueError("No trainable parameters")
    return dict(adapter_scope=scope, trainable_parameters=sum(n for _, n in rows),
                trainable_tensors=[name for name, _ in rows])
