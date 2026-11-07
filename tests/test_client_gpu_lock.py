import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest

from client_gpu_lock import client_gpu_slot


def competing_client(path, active, peak, events, barrier):
    barrier.wait(timeout=15)
    for phase in ('initialize', 'round-0', 'round-1'):
        with client_gpu_slot(path, mp.current_process().name, phase):
            with active.get_lock():
                active.value += 1
                peak.value = max(peak.value, active.value)
            time.sleep(.03)
            with active.get_lock():
                active.value -= 1
            events.put(phase)


def interrupted_client(path, acquired):
    with client_gpu_slot(path, 'interrupted', 'round-0'):
        acquired.set()
        time.sleep(30)


class ClientGPULockTests(unittest.TestCase):
    def test_three_external_processes_never_overlap_across_rounds(self):
        ctx = mp.get_context('spawn')
        with tempfile.TemporaryDirectory() as td:
            active, peak = ctx.Value('i', 0), ctx.Value('i', 0)
            events, barrier = ctx.Queue(), ctx.Barrier(3)
            clients = [ctx.Process(target=competing_client,
                       args=(str(Path(td)/'gpu.lock'), active, peak, events, barrier)) for _ in range(3)]
            for client in clients:
                client.start()
            try:
                for client in clients:
                    client.join(timeout=20)
                    self.assertEqual(client.exitcode, 0)
                phases = [events.get(timeout=2) for _ in range(9)]
                self.assertEqual(peak.value, 1)
                self.assertEqual(active.value, 0)
                for phase in ('initialize', 'round-0', 'round-1'):
                    self.assertEqual(phases.count(phase), 3)
            finally:
                for client in clients:
                    if client.is_alive():
                        client.terminate()
                    client.join(timeout=5)

    def test_process_exit_releases_lock(self):
        ctx = mp.get_context('spawn')
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td)/'gpu.lock')
            acquired = ctx.Event()
            child = ctx.Process(target=interrupted_client, args=(path, acquired))
            child.start()
            try:
                self.assertTrue(acquired.wait(timeout=15))
            finally:
                child.terminate()
                child.join(timeout=5)
            with client_gpu_slot(path, 'replacement', 'round-0'):
                pass

    def test_disabled_and_invalid_path(self):
        with client_gpu_slot(None, 'separate-gpu', 'round-0'):
            pass
        with self.assertRaisesRegex(ValueError, 'absolute'):
            with client_gpu_slot('relative.lock', 'site-1', 'round-0'):
                pass


if __name__ == '__main__':
    unittest.main()
