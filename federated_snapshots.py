"""Save explicitly requested post-aggregation global adapters, never local adapters."""
import hashlib
import json
from pathlib import Path
import torch

from custom_aggregators import NaiveMaxRankAggregator


class SnapshotNaiveAggregator(NaiveMaxRankAggregator):
    def __init__(self, global_lora_rank, snapshot_dir, save_rounds, expected_clients=3):
        super().__init__(global_lora_rank=global_lora_rank)
        # FLARE serializes constructor-named attributes into JSON job config.
        self.snapshot_dir = str(snapshot_dir)
        self.save_rounds = sorted(set(save_rounds))
        self.expected_clients = expected_clients
        self.completed_rounds = 0
        if not self.save_rounds or min(self.save_rounds) < 1:
            raise ValueError('Snapshot rounds must be positive')

    def aggregate_model(self):
        if len(self.client_lora_params) != self.expected_clients:
            raise ValueError('Refusing an incomplete-client aggregate')
        model = super().aggregate_model()
        self.completed_rounds += 1
        if self.completed_rounds in self.save_rounds:
            directory = Path(self.snapshot_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'round-{self.completed_rounds:03d}.pt'
            if path.exists() or path.with_suffix('.json').exists():
                raise FileExistsError('Refusing to replace an existing global round checkpoint')
            temporary = path.with_suffix('.tmp')
            torch.save({'model': model.params}, temporary)
            temporary.replace(path)
            info = dict(completed_rounds=self.completed_rounds, stage='post_aggregation',
                        n_clients=len(self.client_lora_params),
                        training_record_weights=[float(w) for w, _ in self.client_lora_params],
                        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            path.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n')
        return model
