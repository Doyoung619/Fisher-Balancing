"""Optional isolated CUDA timing/parity audit; does not write scientific runs.

Use the existing torch environment; no system dependencies or CUDA changes.
Lane measurements are a score-kernel prototype, explicitly separate from the
production complete selection/fit/evaluation/storage measurements.
"""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy
import torch
from .common import atomic_json
from .numerics import fit_glm
from .evaluation import evaluate_checkpoint
from .selectors import select_batch, shared_candidates, shared_initial_ids, curvature, e_scores_secular, e_scores_batched

FAMILIES = {'obd': 'bernoulli', 'movielens': 'gaussian', 'lastfm': 'poisson'}


def gpu_query():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader'], text=True, timeout=10)
    except Exception as exc:
        return repr(exc)


def complete_batch(ds, X, y, Xt, yt, ids, candidates, m, family, reference, matched, backend):
    start = time.perf_counter()
    theta, audit = fit_glm(X[ids], y[ids], family)
    t = time.perf_counter()
    picked, trace = select_batch('e_greedy', X, ids, candidates, theta, family, m, 1., np.random.default_rng(123),
                                 {'e_backend': backend, 'dataset': ds, 'seed': 912, 'acquisition_floor': 1e-12}, batch_index=10)
    acquisition = time.perf_counter() - t
    selected = np.r_[ids, picked]
    theta, audit = fit_glm(X[selected], y[selected], family, theta0=theta)
    metrics, arrays = evaluate_checkpoint(X, y, Xt, yt, selected, theta, family, 1., reference, matched, audit)
    t = time.perf_counter()
    # Include serialization and actual local disk writes; these are discarded timing artifacts.
    with tempfile.TemporaryFile() as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        bytes_written = f.tell()
    return {'end_to_end_seconds': time.perf_counter() - start, 'acquisition_seconds': acquisition,
            'evaluation_seconds': metrics['evaluation_seconds'], 'io_seconds': time.perf_counter() - t,
            'storage_bytes': bytes_written, 'fit_converged': audit['fit_converged'],
            'picked_ids': picked.tolist(), 'final_theta': theta.tolist(),
            'max_selector_direct_check_error': max((v['direct_check_error'] for v in trace['slot_trace']), default=0.)}


def lane_probe(W, U, lanes, device):
    # Distinct permutations create independent lane inputs from the same real curvature state.
    Ws = np.stack([W.copy() for _ in range(lanes)])
    Us = np.stack([np.roll(U, i * 17, axis=0) for i in range(lanes)])
    def compute():
        wt = torch.as_tensor(Ws, dtype=torch.float64, device=device)
        ut = torch.as_tensor(Us, dtype=torch.float64, device=device)
        mats = wt[:, None] + ut[:, :, :, None] * ut[:, :, None, :]
        scores = torch.linalg.eigvalsh(mats)[..., 0]
        return scores.cpu().numpy()
    compute()
    times = []
    for _ in range(3):
        t = time.perf_counter(); values = compute(); times.append(time.perf_counter() - t)
    t = time.perf_counter()
    exact = np.stack([e_scores_secular(Ws[i], Us[i]) for i in range(lanes)])
    cpu = time.perf_counter() - t
    t = time.perf_counter()
    direct = np.stack([e_scores_batched(Ws[i], Us[i]) for i in range(lanes)])
    direct_cpu = time.perf_counter() - t
    return {'lanes': lanes, 'device': device, 'gpu_seconds': float(np.median(times)),
            'cpu_secular_seconds': cpu, 'cpu_batched_seconds': direct_cpu,
            'best_cpu_over_gpu_speed_ratio': min(cpu, direct_cpu) / float(np.median(times)), 'cpu_over_gpu_speed_ratio': cpu / float(np.median(times)),
            'max_score_abs_error': float(np.max(np.abs(values - exact))),
            'scope': 'one candidate scoring step across lanes; gather/host-to-device/device-to-host included; no scheduling/refit/evaluation',
            'production_integration': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--max-seconds', type=float, default=180.)
    args = parser.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    torch.set_num_threads(1)
    started = time.perf_counter(); deadline = started + args.max_seconds
    result = {'status': 'running', 'interpreter': sys.executable, 'torch_version': torch.__version__, 'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
              'cuda_available': torch.cuda.is_available(), 'dtype': 'float64', 'gpu_before': gpu_query(),
              'allowed_gpu_ids': [0, 1, 2, 3], 'actual_devices_measured': [], 'complete_batch_cases': [], 'lane_score_cases': [],
              'measurement_limit_seconds': args.max_seconds,
              'scope': 'actual real-data E-greedy complete batch with exact CPU secular/batched vs CUDA eigenscoring; all fit/evaluation/NPZ I/O costs included',
              'known_gaps': ['Other selectors and SDP remain implemented on CPU; no unsupported CUDA solver claim',
                             'Multi-lane kernels are prototypes, not production scheduling implementations',
                             'Timing-only labeled prefixes after initialization are uniformly chosen without reference labels and never enter scientific runs',
                             'This is a bounded pilot, not an exhaustive search over all heterogeneous schedules',
                             'CUDA environment NumPy may differ from primary interpreter; CPU concurrency benchmark is measured separately in primary environment']}
    if not torch.cuda.is_available():
        result['status'] = 'unavailable'; atomic_json(args.root / 'cuda_benchmark.json', result); return
    devices = [f'cuda:{i}' for i in range(min(4, torch.cuda.device_count()))]
    for device in devices:
        e_scores_batched(np.eye(5), np.zeros((2, 5)), backend=device)
        torch.cuda.reset_peak_memory_stats(device)
    for ds in FAMILIES:
        with np.load(args.root / 'data' / 'prepared' / f'{ds}.npz') as z:
            raw = {key: z[key] for key in ['X_pool', 'y_pool', 'X_test', 'y_test']}
        initial = shared_initial_ids(raw['X_pool'], ds, 912, 80)
        for d in [5, 10, 20]:
            X, y, Xt, yt = raw['X_pool'][:, :d].copy(), raw['y_pool'], raw['X_test'][:, :d].copy(), raw['y_test']
            m, K, late = (200, 512, 3880) if ds == 'obd' else (20, 256, 460)
            reference, _ = fit_glm(X, y, FAMILIES[ds], reference=True)
            for n in [80, late]:
                if time.perf_counter() > deadline:
                    result['status'] = 'bounded_budget_reached'; break
                rest = np.setdiff1d(np.arange(len(X)), initial)
                rng = np.random.default_rng(225)
                ids = np.r_[initial, rng.choice(rest, n - 80, replace=False)]
                candidates = shared_candidates(len(X), ids, ds, 912, 10, K)
                matched, _ = fit_glm(X, y, FAMILIES[ds], rho=len(X) / (n + m), reference=True)
                row = {'dataset': ds, 'd': d, 'n_labeled': n, 'm': m, 'K': K, 'backends': {}}
                for backend in ['secular', 'batched'] + devices:
                    runs = [complete_batch(ds, X, y, Xt, yt, ids, candidates, m, FAMILIES[ds], reference, matched, backend) for _ in range(2)]
                    row['backends'][backend] = {'median_end_to_end_seconds': float(np.median([r['end_to_end_seconds'] for r in runs])),
                                               'median_acquisition_seconds': float(np.median([r['acquisition_seconds'] for r in runs])),
                                               'measurements': runs}
                cpu = dict(row['backends']['secular']['measurements'][0])
                for backend, measurement in row['backends'].items():
                    actual = measurement['measurements'][0]
                    measurement['identical_selected_ids_to_cpu'] = actual['picked_ids'] == cpu['picked_ids']
                    measurement['max_final_theta_abs_difference_to_cpu'] = float(np.max(np.abs(np.asarray(actual['final_theta']) - cpu['final_theta'])))
                    measurement['cpu_over_backend_end_to_end_speed_ratio'] = row['backends']['secular']['median_end_to_end_seconds'] / measurement['median_end_to_end_seconds']
                    for r in measurement['measurements']:
                        r.pop('picked_ids'); r.pop('final_theta')
                result['complete_batch_cases'].append(row)
                print(json.dumps({'dataset': ds, 'd': d, 'n': n, 'timings': {k: v['median_end_to_end_seconds'] for k,v in row['backends'].items()}}), flush=True)
                if ds == 'obd' and n == late and d in [5, 20]:
                    theta, _ = fit_glm(X[ids], y[ids], FAMILIES[ds])
                    wc = curvature(X[candidates] @ theta, FAMILIES[ds]); wl = curvature(X[ids] @ theta, FAMILIES[ds])
                    W = np.eye(d) + (X[ids].T * wl) @ X[ids]; U = X[candidates] * np.sqrt(wc)[:, None]
                    for device in devices:
                        for lanes in [1, 4, 8, 16]:
                            result['lane_score_cases'].append({'dataset': ds, 'd': d, 'n_labeled': n, **lane_probe(W, U, lanes, device)})
                atomic_json(args.root / 'cuda_benchmark.json', result)
    result['status'] = 'complete' if len(result['complete_batch_cases']) == 18 else 'bounded_budget_reached'
    result['actual_devices_measured'] = list(range(len(devices)))
    result['all_complete_selector_parity_passed'] = all(b['identical_selected_ids_to_cpu'] for r in result['complete_batch_cases'] for b in r['backends'].values())
    cpu_total = sum(r['backends']['secular']['median_end_to_end_seconds'] for r in result['complete_batch_cases'])
    result['cpu_case_time_sum_seconds'] = cpu_total
    result['best_cpu_case_time_sum_seconds'] = sum(min(r['backends'][b]['median_end_to_end_seconds'] for b in ['secular', 'batched']) for r in result['complete_batch_cases'])
    result['best_cpu_by_dimension'] = {str(d): min(['secular','batched'], key=lambda b: sum(r['backends'][b]['median_end_to_end_seconds'] for r in result['complete_batch_cases'] if r['d'] == d)) for d in [5,10,20]}
    result['cuda_case_time_sums_seconds'] = {dev: sum(r['backends'][dev]['median_end_to_end_seconds'] for r in result['complete_batch_cases']) for dev in devices}
    result['recommended_supported_backend'] = min({'secular': cpu_total, **result['cuda_case_time_sums_seconds']}, key=lambda k: {'secular': cpu_total, **result['cuda_case_time_sums_seconds']}[k])
    result['recommendation_scope'] = 'E-greedy representative batch suite only; CPU worker throughput and SDP evidence belong in runtime_plan'
    result['gpu_peak_allocated_bytes'] = {dev: torch.cuda.max_memory_allocated(dev) for dev in devices}
    result['gpu_peak_reserved_bytes'] = {dev: torch.cuda.max_memory_reserved(dev) for dev in devices}
    result['gpu_after'] = gpu_query(); result['wall_seconds'] = time.perf_counter() - started
    atomic_json(args.root / 'cuda_benchmark.json', result)
    print(json.dumps({k: result[k] for k in ['status','wall_seconds','all_complete_selector_parity_passed','recommended_supported_backend','cpu_case_time_sum_seconds','cuda_case_time_sums_seconds']}), flush=True)


if __name__ == '__main__':
    main()
