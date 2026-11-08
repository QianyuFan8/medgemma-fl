"""Exact CPU tensor cache for deterministic collators (no learned features).

Keys cover ordered examples, decoded image pixels, processor/tokenizer state,
package versions and preprocessing code. Do not use with random augmentation.
Whole-batch caching preserves all processor-specific padding and image layouts.
It is most effective with the project's batch size of one and a local SSD.
"""
import fcntl
import hashlib
import importlib.metadata
import json
import os
import pickle
from pathlib import Path
import tempfile
import time


def _hash_value(h, value):
    if isinstance(value, dict):
        h.update(b'dict')
        for key in sorted(value):
            _hash_value(h, key)
            _hash_value(h, value[key])
    elif isinstance(value, (tuple, list)):
        h.update(f'list:{len(value)}:'.encode())
        for item in value:
            _hash_value(h, item)
    elif hasattr(value, 'tobytes') and hasattr(value, 'mode') and hasattr(value, 'size'):
        _hash_value(h, (value.mode, value.size))
        h.update(value.tobytes())
    else:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        h.update(str(len(encoded)).encode() + b':' + encoded)


def processor_fingerprint(processor, task, max_length):
    root = Path(__file__).parent
    tokenizer = processor.tokenizer
    metadata = dict(schema=1, task=task, max_length=max_length,
                    processor=processor.to_dict(), chat_template=getattr(processor, 'chat_template', None),
                    tokenizer_chat_template=getattr(tokenizer, 'chat_template', None),
                    special_tokens=tokenizer.special_tokens_map, padding_side=tokenizer.padding_side,
                    tokenizer=(tokenizer.backend_tokenizer.to_str() if hasattr(tokenizer, 'backend_tokenizer')
                               else dict(vocab=tokenizer.get_vocab(), init=tokenizer.init_kwargs)),
                    image_processor=getattr(processor, 'image_processor', None).to_dict()
                    if getattr(processor, 'image_processor', None) is not None else None,
                    versions={p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'tokenizers', 'Pillow')},
                    code={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                          for name in ('training_cache.py', 'client.py', 'subtype_utils.py', 'methylation_utils.py')})
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode()).hexdigest()


class CachedCollator:
    def __init__(self, collator, directory, fingerprint, max_gb=16.0):
        if not max_gb > 0:
            raise ValueError('Cache budget must be positive')
        self.collator = collator
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.max_bytes = int(max_gb * 1024**3)
        self.stats = dict(hits=0, misses=0, skipped_writes=0, corrupt_entries=0, seconds=0.0)

    def __call__(self, examples):
        import torch
        start = time.perf_counter()
        h = hashlib.sha256(self.fingerprint.encode())
        _hash_value(h, examples)
        target = self.directory / (h.hexdigest() + '.pt')
        if target.is_file():
            try:
                batch = torch.load(target, map_location='cpu', weights_only=True)
                if not isinstance(batch, dict) or not all(isinstance(v, torch.Tensor) for v in batch.values()):
                    raise ValueError('Invalid cached batch')
                self.stats['hits'] += 1
                self.stats['seconds'] += time.perf_counter() - start
                return batch
            except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError):
                self.stats['corrupt_entries'] += 1
                target.unlink(missing_ok=True)
        self.stats['misses'] += 1
        batch = self.collator(examples)
        if not all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' for v in batch.values()):
            raise ValueError('Preprocessing cache requires CPU tensor batches')
        # Serialize under a filesystem lock so concurrent clients/ranks cannot
        # exceed the shared budget. Existing entries are immutable.
        with (self.directory / '.cache.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                used = sum(p.stat().st_size for p in self.directory.glob('*.pt'))
                estimate = sum(v.numel()*v.element_size() for v in batch.values()) + 65536
                if used + estimate > self.max_bytes:
                    self.stats['skipped_writes'] += 1
                else:
                    fd, name = tempfile.mkstemp(prefix='.batch-', dir=self.directory)
                    os.close(fd)
                    try:
                        with open(name, 'wb') as handle:
                            torch.save(dict(batch), handle)
                        if used + Path(name).stat().st_size <= self.max_bytes:
                            os.replace(name, target)
                        else:
                            self.stats['skipped_writes'] += 1
                    finally:
                        Path(name).unlink(missing_ok=True)
        self.stats['seconds'] += time.perf_counter() - start
        return batch


def maybe_cache_collator(collator, processor, args):
    if not getattr(args, 'preprocess_cache_dir', None):
        return collator
    return CachedCollator(collator, args.preprocess_cache_dir,
                          processor_fingerprint(processor, args.task, args.max_seq_length),
                          args.preprocess_cache_max_gb)
