"""Run a portable profile without historical campaign directories or timing pilots."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import yaml
    from real_response import cli, runtime
    from real_response.artifacts import metadata, verify
    from real_response.common import output_for, tasks_for
    from real_response.evaluation import prepare_references
    from real_response.reporting import summarize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/smoke.yaml')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--prepare', action='store_true', help='Download and prepare raw real data first.')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--data-root', type=Path, help='Optional raw-data directory.')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    config = yaml.safe_load(args.config.read_text())
    config['offline'] = args.offline
    if args.data_root:
        config['data_root'] = str(args.data_root.resolve())
    if args.prepare:
        cli.prepare(config)
    else:
        # Add matched budgets for this profile to the local reference cache.
        # No raw download or data preprocessing occurs on this path.
        prepare_references(ROOT, config)
    out = args.output.resolve() if args.output else output_for(config)
    out.mkdir(parents=True, exist_ok=True)
    metadata(config, out)
    plan = {'cpu_workers': args.workers, 'sdp_queue_limit': min(2, args.workers)}
    print(f'Running {len(tasks_for(config))} trajectories -> {out}', flush=True)
    result = runtime.run_grid(config, out, plan)
    if result['failed']:
        raise RuntimeError(f'Trajectories failed; inspect {out}')
    summarize(config, out)
    verification = verify(config, out)
    if verification['failed_or_incomplete'] or not verification['aggregate_complete']:
        raise RuntimeError(f'Incomplete verification; inspect {out}/verification.json')
    print(verification, flush=True)


if __name__ == '__main__':
    main()
