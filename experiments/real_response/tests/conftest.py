"""External-data tests are explicit and never download data during pytest."""
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_runtest_setup(item):
    if item.get_closest_marker('requires_data'):
        required = [ROOT / 'data' / 'prepared' / f'{ds}.npz'
                    for ds in ('obd', 'movielens', 'lastfm')]
        required += [ROOT / 'data' / 'references' / f'{ds}_d{d}.npz'
                     for ds in ('obd', 'movielens', 'lastfm') for d in (5, 10, 20)]
        if not all(path.is_file() for path in required):
            pytest.skip('Prepare real datasets and references first; see experiments/real_response/README.md')
