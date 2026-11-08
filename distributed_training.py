"""Replicated DDP helpers; FLARE communication belongs exclusively to rank zero."""
from contextlib import contextmanager
from datetime import timedelta

from training_performance import distributed_layout
from client_gpu_lock import client_gpu_slot


class DistributedClient:
    def __init__(self, flare_api=None, backend='nccl'):
        import torch
        import torch.distributed as dist
        self.world, self.rank, self.local_rank = distributed_layout()
        self.dist, self.flare, self.group = dist, flare_api, None
        if backend == 'nccl':
            if self.world == 1 and torch.cuda.device_count() > 1:
                raise ValueError('Multiple GPUs are visible without DDP. Use torchrun, or set '
                                 'CUDA_VISIBLE_DEVICES to one GPU; implicit DataParallel is unsupported.')
            if self.local_rank >= torch.cuda.device_count():
                raise ValueError('LOCAL_RANK exceeds visible GPUs; use one torchrun worker per GPU')
            torch.cuda.set_device(self.local_rank)
        if self.world > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend=backend, timeout=timedelta(minutes=60))
            # CPU transport prevents large serialized adapter messages occupying VRAM.
            self.group = dist.new_group(backend='gloo', timeout=timedelta(minutes=60))
        if self.flare is not None and self.rank == 0:
            self.flare.init()

    def broadcast(self, value):
        if self.world == 1:
            return value
        box = [value if self.rank == 0 else None]
        self.dist.broadcast_object_list(box, src=0, group=self.group)
        return box[0]

    def barrier(self):
        if self.world > 1:
            self.dist.barrier(group=self.group)

    def site_name(self):
        name = self.flare.system_info().get('site_name', 'unknown') if self.rank == 0 else None
        return self.broadcast(name)

    def is_running(self):
        return self.broadcast(self.flare.is_running() if self.rank == 0 else None)

    def receive(self):
        return self.broadcast(self.flare.receive() if self.rank == 0 else None)

    def send(self, model):
        if self.rank == 0:
            self.flare.send(model)
        self.barrier()

    @contextmanager
    def gpu_slot(self, path, site, phase):
        # One lock per client GPU group, not one lock per DDP rank (deadlock).
        with client_gpu_slot(path if self.rank == 0 else None, site, phase):
            self.barrier()
            yield
            self.barrier()

    def close(self):
        if self.world > 1 and self.dist.is_initialized():
            self.dist.destroy_process_group()
