"""Serialize GPU work in external FLARE clients sharing one physical GPU."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time


@contextmanager
def client_gpu_slot(path, site, phase):
    """Hold a process lock during model initialization or one local round.

    Simulator thread limits do not serialize persistent external clients. Use
    the SAME absolute lock path in all clients sharing a GPU. Never unlink a
    live lock file: waiters must keep locking the same inode. The OS releases
    the lock if a client exits unexpectedly.
    """
    if not path:
        yield
        return
    target = Path(path)
    if not target.is_absolute():
        raise ValueError('Client GPU lock must have an absolute shared path')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a') as handle:
        start = time.monotonic()
        print(f'GPU_SLOT_WAIT site={site} phase={phase} pid={os.getpid()}', flush=True)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        print(f'GPU_SLOT_ACQUIRED site={site} phase={phase} pid={os.getpid()} '
              f'wait_sec={time.monotonic()-start:.2f}', flush=True)
        try:
            yield
        finally:
            print(f'GPU_SLOT_RELEASED site={site} phase={phase} pid={os.getpid()}', flush=True)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
