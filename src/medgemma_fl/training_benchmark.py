"""Opt-in synchronized optimizer-step timings (warmup excluded)."""
import json
import statistics
import time
from pathlib import Path


def timing_callback(output, rank, warmup_steps=5):
    import torch
    from transformers import TrainerCallback

    class Timing(TrainerCallback):
        def __init__(self):
            self.start = None
            self.steps = []

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.start = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.steps.append({'step': state.global_step, 'seconds': time.perf_counter()-self.start})

        def on_train_end(self, args, state, control, **kwargs):
            measured = [s['seconds'] for s in self.steps if s['step'] > warmup_steps]
            if not measured:
                raise ValueError('Benchmark needs more steps than its warmup')
            report = dict(rank=rank, warmup_steps=warmup_steps, steps=self.steps,
                          mean_step_sec=statistics.mean(measured), median_step_sec=statistics.median(measured),
                          measured_optimizer_steps=len(measured),
                          note='CUDA synchronized each step; no evaluation/checkpoint time included.')
            (Path(output)/f'rank-{rank}_timing.json').write_text(json.dumps(report, indent=2)+'\n')

    return Timing()
