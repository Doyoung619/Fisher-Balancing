"""Bounded persistent CPU-worker versus 4-GPU-worker complete batch throughput.

This is a timing-only E-greedy suite. It never contributes experimental runs.
"""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
import scipy
from .common import atomic_json

_CACHE = {}
_BACKEND = 'cpu'
_DEVICE = None


def initialize(root, backend):
    global _BACKEND, _DEVICE
    _BACKEND = backend
    if backend == 'cuda':
        import torch
        torch.set_num_threads(1)
        ident = mp.current_process()._identity[-1]
        _DEVICE = f'cuda:{(ident - 1) % 4}'
        torch.zeros(1, device=_DEVICE)
    for ds in ['obd', 'movielens', 'lastfm']:
        with np.load(Path(root) / 'data' / 'prepared' / f'{ds}.npz') as z:
            _CACHE[ds] = {k: z[k] for k in ['X_pool', 'X_test', 'y_pool', 'y_test']}


def probe(task):
    from .numerics import fit_glm
    from .evaluation import evaluate_checkpoint
    from .selectors import select_batch, shared_candidates
    ds, d, repeat = task
    family = {'obd': 'bernoulli', 'movielens': 'gaussian', 'lastfm': 'poisson'}[ds]
    z = _CACHE[ds]; X, y, Xt, yt = z['X_pool'][:, :d], z['y_pool'], z['X_test'][:, :d], z['y_test']
    n, m, K = (3880, 200, 512) if ds == 'obd' else (460, 20, 256)
    rng = np.random.default_rng(530 + repeat)
    ids = rng.choice(len(X), n, replace=False)
    candidates = shared_candidates(len(X), ids, ds, repeat, 20, K)
    backend = _DEVICE if _BACKEND == 'cuda' else ('batched' if d == 5 or (d == 10 and K == 256) else 'secular')
    t0 = time.perf_counter()
    theta, fit = fit_glm(X[ids], y[ids], family)
    # These timing-only references exercise the same evaluator computations. They
    # are not interpreted as correct scientific references or written to runs.
    timing_ref = theta.copy()
    selected, audit = select_batch('e_greedy', X, ids, candidates, theta, family, m, 1., rng,
                                    {'e_backend': backend, 'dataset': ds, 'seed': repeat, 'acquisition_floor': 1e-12}, batch_index=20)
    ids = np.r_[ids, selected]
    theta, fit = fit_glm(X[ids], y[ids], family, theta0=theta)
    metrics, arrays = evaluate_checkpoint(X, y, Xt, yt, ids, theta, family, 1., timing_ref, timing_ref, fit)
    with tempfile.TemporaryFile() as f:
        np.savez_compressed(f, **arrays); f.flush()
    return {'pid': os.getpid(), 'device': _DEVICE, 'seconds': time.perf_counter() - t0,
            'converged': fit['fit_converged'], 'selected_hash': __import__('hashlib').sha256(selected.tobytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--backend', choices=['cpu','cuda'], required=True); p.add_argument('--workers', type=int, required=True)
    p.add_argument('--tasks', type=int, default=108); a = p.parse_args()
    base = [(ds, d) for ds in ['obd','movielens','lastfm'] for d in [5,10,20]]
    tasks = [(base[i % len(base)][0], base[i % len(base)][1], i // len(base)) for i in range(a.tasks)]
    # CPU forks before any torch load; CUDA requires spawn to avoid inherited contexts.
    context = mp.get_context('spawn' if a.backend == 'cuda' else 'fork')
    start = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=context, initializer=initialize,
                                initargs=(str(a.root),a.backend)) as ex:
        warm = list(ex.map(probe, [tasks[i % len(tasks)] for i in range(a.workers)]))
        warm_seconds = time.perf_counter() - start
        t = time.perf_counter(); results = list(ex.map(probe, tasks)); measured = time.perf_counter() - t
    out = {'backend':a.backend,'requested_workers':a.workers,'actual_pids': sorted(set(v['pid'] for v in results)),
           'devices': sorted(set(str(v['device']) for v in results)), 'task_count':len(tasks),
           'persistent_wall_seconds':measured, 'tasks_per_second':len(tasks)/measured,
           'startup_plus_warmup_seconds':warm_seconds,'end_to_end_with_startup_seconds':time.perf_counter()-start,
           'median_task_seconds':float(np.median([v['seconds'] for v in results])),
           'all_fits_converged':all(v['converged'] for v in results),'selected_hashes':[v['selected_hash'] for v in results],
           'python':sys.executable,'numpy_version':np.__version__,'scipy_version':scipy.__version__,
           'cpu_kernel_policy':'batched for d5 and d10 K256; secular otherwise',
           'scope':'late-state E-greedy, all3 datasets x d5/10/20, complete selection/refit/evaluation/local NPZ I/O; persistent workers',
           'limitations':['E-greedy throughput only; separate benchmark covers other CPU classes and SDP',
                          'Timing-only evaluator references are prior-state theta, not scientific references',
                          'CPU and CUDA environments have different installed NumPy versions']}
    atomic_json(a.root/f'concurrency_benchmark_{a.backend}.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['selected_hashes','actual_pids']}),flush=True)


if __name__ == '__main__':main()
