"""Patient-mean assistant-token CE, with explicit gold/silver multipliers.

L = mean_i w_i * mean_{answer tokens of i} CE. Never divide by sum(w):
with microbatch=1 that would cancel the silver multiplier entirely.
"""
import math
import torch
from medgemma_fl.fl_methods import answer_offsets


class WeightedCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, examples):
        weights = [float(x['student_loss_weight']) for x in examples]
        if any(not math.isfinite(w) or not 0 < w <= 1 for w in weights):
            raise ValueError('Expected finite student weights in (0,1]')
        batch = self.collator(examples)
        batch['student_loss_weight'] = torch.tensor(weights, dtype=torch.float32)
        return batch


def patient_weighted_ce(logits, labels, offset, weights):
    targets = labels[:, offset + 1:]
    pred = logits[:, :-1, :].float()
    if pred.shape[:2] != targets.shape or weights.shape != (labels.shape[0],):
        raise ValueError('Weighted loss dimensions disagree')
    mask = targets.ne(-100)
    if not bool(mask.any(dim=1).all()):
        raise ValueError('Every patient must have supervised answer tokens')
    losses = torch.nn.functional.cross_entropy(
        pred.reshape(-1, pred.shape[-1]), targets.reshape(-1),
        ignore_index=-100, reduction='none').view_as(targets)
    return ((losses.sum(1) / mask.sum(1)) * weights.to(logits.device)).mean()


def weighted_forward(model, inputs):
    labels, weights = inputs['labels'], inputs['student_loss_weight']
    _, offset, kept = answer_offsets(labels)
    feats = {k: v for k, v in inputs.items() if k not in ('labels', 'student_loss_weight')}
    # Go through the DDP wrapper, not its .module. Only materialize vocabulary
    # logits for the short answer suffix, not the thousands of prompt tokens.
    result = model(**feats, use_cache=False, logits_to_keep=kept)
    return patient_weighted_ce(result.logits, labels, offset, weights)


def make_weighted_trainer(trainer_cls, **kwargs):
    class WeightedTrainer(trainer_cls):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = weighted_forward(model, inputs)
            return (loss, {'loss': loss}) if return_outputs else loss

    trainer = WeightedTrainer(**kwargs)
    # Our loss is a microbatch patient mean, not a token-count-normalized sum.
    # Trainer/Accelerate must apply gradient-accumulation scaling exactly once.
    trainer.model_accepts_loss_kwargs = False
    return trainer
