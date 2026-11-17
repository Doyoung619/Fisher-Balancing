"""Run each experiment in a separate interpreter to isolate its `src` package."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-data', action='store_true',
                        help='Run external-data tests too; prepare all datasets first.')
    parser.add_argument('--skip-sampled', action='store_true', help='For environments without PyTorch.')
    args = parser.parse_args()
    names = ['continuous_ofat', 'real_response']
    if not args.skip_sampled:
        names.append('sampled_ball')
    failed = False
    for name in names:
        print(f'\n== {name} ==', flush=True)
        command = [sys.executable, '-m', 'pytest', 'tests', '-q', '-ra']
        if name == 'real_response' and not args.include_data:
            command += ['-m', 'not requires_data']
        result = subprocess.run(command, cwd=ROOT / 'experiments' / name)
        failed |= result.returncode != 0
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
