"""Explicit adapter scope shared by training and checkpoint inference (no GPU imports)."""
SCOPES = ("legacy", "lora-only", "language-only")
LANGUAGE_TARGETS = r'.*\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'


def target_modules(scope):
    saved_modules(scope)
    return LANGUAGE_TARGETS if scope == 'language-only' else 'all-linear'


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
    if full:
        return 'legacy'
    adapters = [k for k in state if '.lora_A.' in k or '.lora_B.' in k]
    if adapters and all('.language_model.' in k for k in adapters):
        return 'language-only'
    return 'lora-only'


def trainable_summary(model, scope):
    saved_modules(scope)
    rows = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    if scope in ("lora-only", "language-only"):
        extra = [name for name, _ in rows if ".lora_A." not in name and ".lora_B." not in name]
        if extra:
            raise ValueError(f"lora-only contains non-adapter trainable parameters: {extra[:10]}")
    if not rows:
        raise ValueError("No trainable parameters")
    if scope == 'language-only' and any('.language_model.' not in name for name, _ in rows):
        raise ValueError('language-only must freeze every parameter outside the language model')
    return dict(adapter_scope=scope, trainable_parameters=sum(n for _, n in rows),
                vision_trainable_parameters=sum(n for name, n in rows if '.vision_tower.' in name),
                trainable_tensors=[name for name, _ in rows])
