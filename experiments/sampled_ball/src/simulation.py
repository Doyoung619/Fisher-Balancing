"""Vectorized response replicates, exact batch feedback, and atomic recovery."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from .estimators import fit_gaussian, fit_bernoulli

INFO_METHODS = {'FB', 'AOpt', 'DOpt', 'EGreedy'}
_STOP = False


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:8], 'little') % (2**63-1)


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('w') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('wb') as f:
        np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def checkpoint_counts(N_run, m):
    counts = np.rint(np.geomspace(m, N_run, 60) / m).astype(int) * m
    required = [m, N_run] + [n for n in (100, 200, 500, 1000, 2000, 5000, 10000) if n <= N_run and n % m == 0]
    return sorted(set(counts.tolist() + required))


def _condition(job, config):
    c = job['condition']
    if isinstance(c, dict):
        return c.copy()
    conditions = config['conditions']
    if isinstance(conditions, dict):
        return dict(conditions[c], id=c)
    return next(x.copy() for x in conditions if x.get('id', x.get('name')) == c)


def _numpy(t):
    return t.detach().cpu().numpy().copy() if torch.is_tensor(t) else t


def _stop(signum, frame):
    global _STOP
    _STOP = True


def _record(state, theta_star):
    n = state['N']
    if state['records'] and state['records'][-1]['N'] == n:
        return
    Vstar, Vhat = state['Vstar'], state['Vhat']
    record = dict(N=n, theta=_numpy(state['theta']), error=_numpy(state['theta'] - theta_star),
                  Vstar=_numpy(Vstar), Vhat=_numpy(Vhat),
                  true_min=_numpy(torch.linalg.eigvalsh(Vstar)[:, 0] / n),
                  plugin_min=_numpy(torch.linalg.eigvalsh(Vhat)[:, 0] / n),
                  tracking_op=_numpy(torch.linalg.eigvalsh(Vhat - Vstar).abs().amax(-1) / n))
    state['records'].append(record)


def _save_result(run, job, state, condition, replicate_ids, config):
    if not state['records']:
        return
    arrays = {key: np.asarray([r[key] for r in state['records']]) for key in state['records'][0]}
    R = condition['L']**2 * (1 if job['family'] == 'gaussian' else .25)
    tau0 = R * math.sqrt(condition['m'] / (8 * math.log(condition['d']))) / config.get('N_cal', 400)**.75
    arrays.update(replicate_ids=np.asarray(replicate_ids), tau=tau0 * arrays['N']**.75,
                  tau0=np.asarray(tau0), family=np.asarray(job['family']),
                  condition=np.asarray(condition.get('id', condition.get('name', 'central'))),
                  method=np.asarray(job['method']), complete=np.asarray(state['N'] == config['N_run']))
    atomic_npz(run/'results'/f"{job['id']}.npz", **arrays)


def _save_checkpoint(path, state, generator, shared):
    n = state['N']
    payload = dict(state)
    if state.get('X') is not None:
        payload['X'] = state['X'][:, :n].contiguous()
    # Labels are not needed for continuation; cumulative X'y is sufficient.
    payload['generator_state'] = generator.get_state()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('wb') as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def run_job(job, config, run_dir, device='cpu'):
    """Run/resume a job containing one method and a vectorized replicate chunk.

    Noise innovations depend only on family/condition/replicate/observation,
    pairing every method on the common response-replicate subset. A job's
    acquisition generator state is saved after completed feedback boundaries.
    """
    from .oracles import candidate_cloud, select_batch
    global _STOP
    _STOP = False
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)
    torch.set_num_threads(int(config.get('torch_threads', 1)))
    device = torch.device(device)
    condition = _condition(job, config)
    name = condition.get('id', condition.get('name', 'central'))
    condition['name'] = name; condition['id'] = name
    family, method = job['family'], job['method']
    d, m, N_run = condition['d'], condition['m'], int(config['N_run'])
    if N_run % m:
        raise ValueError('N_run must be a multiple of every batch size')
    R = int(job.get('replicates', job.get('R', 1)))
    rep_start = int(job.get('replicate_start', 0))
    replicate_ids = list(range(rep_start, rep_start + R))
    shared = family == 'gaussian' and method in INFO_METHODS
    dtype = torch.float64
    eye = torch.eye(d, dtype=dtype, device=device)
    rho = float(config.get('rho', 1.))
    run = Path(run_dir)
    for dirname in ('results', 'checkpoints', 'progress', 'oracle_states'):
        (run/dirname).mkdir(parents=True, exist_ok=True)
    checkpoint_path = run/'checkpoints'/f"{job['id']}.pt"
    direction = config.get('directions', {}).get(str(d))
    if direction is None:
        direction = np.random.default_rng(stable_seed('theta_direction', config.get('direction_seed', 9173), d)).normal(size=d)
        direction /= np.linalg.norm(direction)
    theta_star = torch.as_tensor(direction, dtype=dtype, device=device) * (condition['kappa'] / condition['L'])
    signature_keys = ('N_run', 'K', 'Q', 'full_scan', 'rho', 'parameter_radius', 'N_cal',
                      'global_cloud_seed', 'global_noise_seed', 'global_policy_seed',
                      'directions', 'direction_seed', 'ucb_delta', 'ts_covariance_scale',
                      'estimator_normalized_kkt_tolerance')
    signature = stable_seed(job, {k:config.get(k) for k in signature_keys})
    source_digest = hashlib.sha256(b''.join((Path(__file__).parent / name).read_bytes()
                                          for name in ('simulation.py', 'estimators.py', 'oracles.py'))).hexdigest()
    generator = torch.Generator(device=device)
    generator.manual_seed(stable_seed('policy', config.get('global_policy_seed', 20261005), family, name, method, rep_start))
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state['signature'] != signature:
            raise RuntimeError(f'Checkpoint protocol mismatch: {checkpoint_path}')
        if state.get('source_digest') != source_digest:
            raise RuntimeError(f'Checkpoint science-source mismatch: {checkpoint_path}')
        generator.set_state(state['generator_state'].cpu())
        if family == 'bernoulli':
            oldX = state['X']; state['X'] = torch.empty((R, N_run, d), dtype=dtype, device=device)
            state['X'][:, :state['N']] = oldX
    else:
        state = dict(N=0, batch_index=0, theta=torch.zeros((R, d), dtype=dtype, device=device),
                     Vhat=rho * eye.repeat(R, 1, 1), Vstar=rho * eye.repeat(R, 1, 1),
                     C=torch.zeros((1 if shared else R, d, d), dtype=dtype, device=device),
                     b=torch.zeros((R, d), dtype=dtype, device=device),
                     X=torch.empty((R, N_run, d), dtype=dtype, device=device) if family == 'bernoulli' else None,
                     method_state={}, records=[], signature=signature, source_digest=source_digest, active_seconds=0.,
                     acquisition_seconds=0., refit_seconds=0., checkpoint_seconds=0.,
                     max_kkt=0., fit_iterations=0, boundary_fits=0, saved_oracle_states=[])
    # Independent per-replicate streams preserve pairing regardless of job chunks.
    innovations = np.stack([getattr(np.random.default_rng(stable_seed('response', config.get('global_noise_seed', 712367), family, name, rep)),
                                   'standard_normal' if family == 'gaussian' else 'random')(N_run) for rep in replicate_ids])
    noise = torch.as_tensor(innovations, dtype=dtype, device=device)
    raw_checkpoints = config.get('checkpoint_counts')
    if isinstance(raw_checkpoints, dict):
        raw_checkpoints = raw_checkpoints.get(name)
    checkset = set(raw_checkpoints or checkpoint_counts(N_run, m))
    checkset = {int(n) for n in checkset if 0 < n <= N_run and n % m == 0} | {m, N_run}
    oracle_targets = {'early': min(N_run, max(m, (100//m)*m)),
                      'middle': max(m, (N_run//(2*m))*m), 'late': N_run}
    started = time.monotonic(); initial_N = state['N']; last_saved = started; last_progress = 0.
    timing = time.monotonic()
    status = 'running'

    def progress(status_):
        elapsed = state['active_seconds']
        diag = state.get('last_oracle_diag', {})
        payload = dict(job=job, job_id=job['id'], family=family, condition=name, method=method,
                       device=str(device), pid=os.getpid(), N=state['N'], batch_index=state['batch_index'],
                       replicates=R, completed_responses=state['N'] * R, total_responses=N_run * R,
                       completed_trajectories=R if state['N'] == N_run else 0,
                       updated_at=time.time(), active_seconds=elapsed, elapsed_seconds=elapsed,
                       acquisition_seconds=state['acquisition_seconds'], refit_seconds=state['refit_seconds'],
                       checkpoint_seconds=state['checkpoint_seconds'], max_kkt=state['max_kkt'],
                       fit_iterations=state['fit_iterations'], boundary_fits=state['boundary_fits'],
                       status=status_, checkpoint_path=str(checkpoint_path),
                       oracle_diagnostics=diag, method_state=state['method_state'], shared_gaussian_acquisition=shared)
        atomic_json(run/'progress'/f"{job['id']}.json", payload)
        return payload

    progress(status)
    while state['N'] < N_run:
        if _STOP or (run/'STOP_REQUESTED').exists():
            status = 'stopped'; break
        n = state['N']; batch = state['batch_index'] + 1
        if device.type == 'cuda': torch.cuda.synchronize(device)
        t0 = time.monotonic()
        cloud = None if method == 'Random' else candidate_cloud(family, condition, batch, config.get('K', 4096), config.get('global_cloud_seed', 20261005), device)
        if shared:
            Xb0, method_state, diag = select_batch(method, family, state['theta'][:1], state['Vhat'][:1],
                                                 state['C'] + rho * eye, condition, batch, cloud=cloud,
                                                 generator=generator, state=state['method_state'], config=config)
            Xb = Xb0.expand(R, -1, -1)
        else:
            Xb, method_state, diag = select_batch(method, family, state['theta'], state['Vhat'],
                                                state['C'] + rho * eye, condition, batch, cloud=cloud,
                                                generator=generator, state=state['method_state'], config=config)
        if Xb.shape != (R, m, d):
            raise RuntimeError(f'Invalid acquisition shape {Xb.shape}')
        if device.type == 'cuda': torch.cuda.synchronize(device)
        t1 = time.monotonic()
        predictor = torch.matmul(Xb, theta_star)
        yb = predictor + noise[:, n:n+m] if family == 'gaussian' else (noise[:, n:n+m] < torch.sigmoid(predictor)).to(dtype)
        state['b'] += torch.matmul(Xb.transpose(-1, -2), yb.unsqueeze(-1)).squeeze(-1)
        Cnew = torch.matmul(Xb[:1].transpose(-1, -2), Xb[:1]) if shared else torch.matmul(Xb.transpose(-1, -2), Xb)
        state['C'] += Cnew
        if family == 'gaussian':
            state['Vstar'] = (state['C'] + rho * eye).expand(R, -1, -1)
            theta, Vhat, fit = fit_gaussian(state['C'], state['b'], rho, config.get('parameter_radius', 4.), n+m)
        else:
            p = torch.sigmoid(predictor); w = p * (1-p)
            state['Vstar'] += torch.matmul(Xb.transpose(-1, -2), w.unsqueeze(-1) * Xb)
            state['X'][:, n:n+m] = Xb
            theta, Vhat, fit = fit_bernoulli(state['X'][:, :n+m], initial=state['theta'], rho=rho,
                                           radius=config.get('parameter_radius', 4.),
                                           tol=config.get('estimator_normalized_kkt_tolerance', 1e-8), b=state['b'])
        if not fit['success']:
            atomic_json(run/'progress'/f"{job['id']}.failure.json", dict(N=n+m, fit=fit, job=job))
            raise RuntimeError(f'Cumulative constrained MLE did not converge at N={n+m}: {fit}')
        if device.type == 'cuda': torch.cuda.synchronize(device)
        t2 = time.monotonic()
        state.update(N=n+m, batch_index=batch, theta=theta, Vhat=Vhat, method_state=method_state, last_oracle_diag=diag)
        state['acquisition_seconds'] += t1-t0; state['refit_seconds'] += t2-t1
        state['fit_iterations'] += fit['iterations']; state['boundary_fits'] += fit['boundary']
        state['max_kkt'] = max(state['max_kkt'], fit['kkt'])
        if state['N'] in checkset:
            _record(state, theta_star)
            _save_result(run, job, state, condition, replicate_ids, config)
        if method == 'FB' and name == 'central' and rep_start == 0:
            for phase, target in oracle_targets.items():
                if state['N'] >= target and phase not in state['saved_oracle_states']:
                    information_bound = condition['L']**2 * (1 if family == 'gaussian' else .25)
                    tau0 = information_bound * math.sqrt(m/(8*math.log(d))) / config.get('N_cal', 400)**.75
                    atomic_npz(run/'oracle_states'/f"{job['id']}_{phase}.npz", theta=_numpy(theta[0]),
                               Vhat=_numpy(Vhat[0]), W=_numpy(Vhat[0]), H=_numpy(state['C'][0]+rho*eye),
                               tau=np.asarray(tau0 * (m*(batch+1))**.75), N=np.asarray(state['N']),
                               batch_index=np.asarray(batch+1), family=np.asarray(family),
                               condition=np.asarray(json.dumps(condition)), phase=np.asarray(phase))
                    state['saved_oracle_states'].append(phase)
        now = time.monotonic()
        state['active_seconds'] += now-timing; timing=now
        if now-last_saved >= config.get('checkpoint_interval_seconds', 60):
            st = time.monotonic(); _save_checkpoint(checkpoint_path, state, generator, shared)
            state['checkpoint_seconds'] += time.monotonic()-st; last_saved=time.monotonic()
        if now-last_progress >= config.get('progress_interval_seconds', 5):
            progress(status); last_progress=now
    state['active_seconds'] += time.monotonic()-timing
    _save_checkpoint(checkpoint_path, state, generator, shared)
    _save_result(run, job, state, condition, replicate_ids, config)
    if state['N'] == N_run:
        missing = checkset - {r['N'] for r in state['records']}
        if missing: raise RuntimeError(f'Missing metric checkpoints: {sorted(missing)}')
        status = 'complete'
    result = progress(status)
    result.update(invocation_seconds=time.monotonic()-started, invocation_responses=(state['N']-initial_N)*R)
    return result
