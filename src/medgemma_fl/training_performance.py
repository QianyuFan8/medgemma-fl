"""Shared, opt-in training performance settings; defaults preserve old runs."""
import argparse
import os
import re


def add_performance_args(parser):
    parser.add_argument('--gradient_checkpointing', action=argparse.BooleanOptionalAction, default=True,
                        help='Recompute activations to save VRAM; disable only when the batch fits.')
    parser.add_argument('--quantized', action=argparse.BooleanOptionalAction, default=True,
                        help='4-bit QLoRA. Disable for BF16 LoRA (more VRAM; benchmark speed).')
    parser.add_argument('--local_eval', action=argparse.BooleanOptionalAction, default=True,
                        help='Local validation loss during/after training; global scoring is separate.')
    parser.add_argument('--preprocess_cache_dir', default=None,
                        help='Local SSD cache of deterministic processor tensors, NOT model activations.')
    parser.add_argument('--preprocess_cache_max_gb', type=float, default=16.0,
                        help='Disk budget shared by cache entries in this directory; stop writing when full.')


def performance_cli(args):
    result = []
    for name in ('gradient_checkpointing', 'quantized', 'local_eval'):
        if not getattr(args, name, True):
            result.append('--no-' + name)
    cache = getattr(args, 'preprocess_cache_dir', None)
    if cache:
        result += ['--preprocess_cache_dir', os.path.abspath(cache),
                   '--preprocess_cache_max_gb', str(args.preprocess_cache_max_gb)]
    return result


def parse_gpu_ids(value):
    parts = value.split(',')
    if not parts or any(not re.fullmatch(r'\d+', p.strip()) for p in parts):
        raise ValueError('GPU IDs must be comma-separated nonnegative integers, e.g. 0,1,2')
    ids = [int(p) for p in parts]
    if len(ids) != len(set(ids)):
        raise ValueError('GPU IDs must not be repeated')
    return ids


def client_gpu_plan(gpu_ids, clients=3):
    ids = parse_gpu_ids(gpu_ids)
    if clients < 1:
        raise ValueError('Client count must be positive')
    if len(ids) < clients:
        groups = [[ids[i % len(ids)]] for i in range(clients)]
    else:
        # Prefer powers of two: 4 GPUs -> 2/1/1; 8 GPUs -> 4/2/2.
        # This permits the same effective batch (4) at every client.
        sizes = [1]*clients
        left = len(ids)-clients
        while left:
            i = min(range(clients), key=lambda j: sizes[j])
            extra = min(left, sizes[i])
            sizes[i] += extra
            left -= extra
        groups, offset = [], 0
        for size in sizes:
            groups.append(ids[offset:offset+size]); offset += size
    return ','.join('['+','.join(map(str,g))+']' for g in groups), min(len(ids), clients)


def parse_client_gpu_groups(value, clients):
    raw = re.findall(r'\[([^\]]+)\]', value)
    if len(raw) != clients or ','.join(f'[{g}]' for g in raw) != value:
        raise ValueError('Expected one GPU group per client, e.g. "[0,1],[2],[3]"')
    groups = [parse_gpu_ids(g) for g in raw]
    for i, g in enumerate(groups):
        for other in groups[:i]:
            if set(g) & set(other) and g != other:
                raise ValueError('GPU groups must be disjoint or identical; partial overlap is unsafe')
    return groups


def accumulation_for_world(effective_batch, micro_batch, world):
    if min(effective_batch, micro_batch, world) < 1 or effective_batch % (micro_batch * world):
        raise ValueError('Effective batch must be a positive multiple of micro batch × GPUs per client')
    return effective_batch // (micro_batch * world)


def distributed_layout(environ=None):
    env = os.environ if environ is None else environ
    world = int(env.get('WORLD_SIZE', '1'))
    rank = int(env.get('RANK', '0'))
    local = int(env.get('LOCAL_RANK', '0'))
    if world < 1 or not 0 <= rank < world or local < 0:
        raise ValueError('Invalid torchrun rank environment')
    return world, rank, local


def effective_batch_size(args, world_size=1):
    return args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
