"""Small CPU workflows, generated data only; outputs are not research results."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def continuous(out):
    from src.common import resolved_config, code_hash, json_hash
    from src.simulation import run_task
    import numpy as np
    config = resolved_config(smoke=True)
    config.update(N_run=40, checkpoint_counts=[0, 20, 40])
    for family in config['families']:
        run = out / family
        run.mkdir(parents=True)
        for name in ['checkpoints', 'trajectories', 'summaries', 'workers']:
            (run / name).mkdir()
        (run / 'config.json').write_text(json.dumps(config, indent=2))
        (run / 'manifest.json').write_text(json.dumps(
            dict(config_hash=json_hash(config), code_hash=code_hash(), demo_only=True)))
        task = dict(id=f'{family}__center__Ours__r00000', family=family,
                    condition='center', method='Ours', replicate=0)
        run_task(run, task, 40)
        with np.load(run / 'trajectories' / f"{task['id']}.npz") as data:
            assert data['N'][-1] == 40
    print('Continuous: 2 generated trajectories completed', flush=True)


def sampled(out):
    import csv
    import numpy as np
    from src.control import make_config, make_jobs
    from src.simulation import run_job
    from src.plotting import risk_metrics
    config = make_config('preferred_full_scan', N=40)
    config.update(K=64, Q=64, conditions=[dict(id='central', d=5, L=1., m=5, kappa=1.5)],
                  fb_replicates_central=4, comparator_replicates_central=4)
    out.mkdir(parents=True)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    rows = []
    for job in make_jobs(config):
        result = run_job(job, config, out, device='cpu')
        assert result['status'] == 'complete' and result['N'] == 40
        with np.load(out / 'results' / f"{job['id']}.npz") as data:
            # Only raw Q risks are exported; the unused scaling arguments are
            # placeholders, not theoretical reference values for this demo.
            metrics = risk_metrics(data['error'], data['N'], gamma=1., trace_inv=1.)
            rows.append(dict(family=job['family'], method=job['method'], N=40,
                             replicates=4, R2_squared=float(metrics['R2_squared'][-1]),
                             Rdir_squared=float(metrics['Rdir_squared'][-1])))
    with (out / 'demo_endpoints.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Sampled: 14 CPU jobs completed, raw uncentered risks exported', flush=True)


def real(out):
    import csv
    import numpy as np
    import yaml
    from real_response import runner
    from real_response.evaluation import prepare_references
    from real_response.reporting import summarize
    from real_response.artifacts import verify
    config = yaml.safe_load((ROOT / 'experiments/real_response/configs/smoke.yaml').read_text())
    config.update(dimensions=[5], methods=['fb_range', 'a_opt', 'random'],
                  run_seeds=[0, 1], n0=40, profile='SYNTHETIC_DEMO_ONLY')
    config['uncertainty']['bootstrap_draws'] = 50
    out.mkdir(parents=True)
    workspace = out / 'synthetic_workspace'
    data_root = workspace / 'data/prepared'
    data_root.mkdir(parents=True)
    # Keep all generated fixtures separate from the real-data cache. Dataset
    # keys select existing GLM families; these rows are explicitly artificial.
    with (ROOT / 'examples/toy_pool.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    X = np.array([[float(row[f'x{i}']) for i in range(5)] for row in rows])
    for ds, family in [('obd', 'bernoulli'), ('movielens', 'gaussian'), ('lastfm', 'poisson')]:
        config['datasets'][ds] = dict(additional_labels=20, batch_size=5, candidate_cap=32,
                                      source='SYNTHETIC_DEMO_ONLY')
        y = np.array([float(row[f'y_{family}']) for row in rows])
        np.savez_compressed(data_root / f'{ds}.npz', X_pool=X[:120], y_pool=y[:120],
                            X_test=X[120:], y_test=y[120:])
        (data_root / f'{ds}.json').write_text(json.dumps(dict(subset_version='SYNTHETIC_DEMO_ONLY')))
    prepare_references(workspace, config)
    runner.ROOT = workspace
    runner.prepared.cache_clear()
    runner.references.cache_clear()
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    from real_response.common import tasks_for
    for task in tasks_for(config):
        result = runner.run_one(task, config, out)
        assert result['state'] == 'EVALUATION_COMPLETE', result
    summarize(config, out)
    verification = verify(config, out)
    assert verification['failed_or_incomplete'] == 0, verification
    assert verification['aggregate_complete'], verification
    assert not verification['MAIN_COMPLETE']  # A demo cannot be a full research run.
    print('Real-response engine: 18 artificial-data trajectories, reporting and verification passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New directory; default is a temporary directory.')
    parser.add_argument('--part', choices=['continuous', 'sampled', 'real'])
    parser.add_argument('--skip-sampled', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    if args.part:
        path = args.output.resolve()
        sys.path.insert(0, str(ROOT / 'experiments' / dict(
            continuous='continuous_ofat', sampled='sampled_ball', real='real_response')[args.part]))
        globals()[args.part](path)
        return
    def execute(path):
        for part in ['continuous', 'sampled', 'real']:
            if part == 'sampled' and args.skip_sampled:
                continue
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--part', part,
                            '--output', str(path / part)], check=True)
    if args.output:
        path = args.output.resolve()
        path.mkdir(parents=True, exist_ok=False)
        execute(path)
        print(f'Demo artifacts: {path}')
    else:
        with tempfile.TemporaryDirectory(prefix='fisher-smoke-') as folder:
            execute(Path(folder))


if __name__ == '__main__':
    main()
