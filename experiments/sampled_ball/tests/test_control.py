"""CPU-only scheduling must not strand Bernoulli information-method jobs."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from src.control import make_config, make_jobs


def test_cpu_only_supervisor_completes_bernoulli_fb(tmp_path):
    config = make_config(N=20)
    config.update(K=32, Q=32, families=['bernoulli'], methods=['FB'],
                  conditions=[dict(id='central', d=5, L=1., m=5, kappa=1.5)],
                  fb_replicates_central=2)
    jobs = make_jobs(config)
    (tmp_path / 'config.json').write_text(json.dumps(config))
    (tmp_path / 'jobs.json').write_text(json.dumps(jobs))
    # This reduced one-condition fixture exercises scheduling/worker completion.
    # Full-grid diagnostic/plot subprocesses are outside this test's scope.
    script = ('from pathlib import Path; import sys; from src import control; '
              'control.subprocess.call=lambda *a,**k: 0; '
              'sys.exit(control.supervise(Path(sys.argv[1]),cpu_workers=1,gpu_workers=0))')
    subprocess.run([sys.executable, '-c', script, str(tmp_path)], check=True,
                   cwd=Path(__file__).resolve().parents[1], timeout=30,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['status'] == 'COMPLETE' and status['completed_jobs'] == 1
    done = json.loads((tmp_path / 'done' / f"{jobs[0]['id']}.json").read_text())
    assert done['device'] == 'cpu'
    with np.load(tmp_path / 'results' / f"{jobs[0]['id']}.npz") as result:
        assert result['N'][-1] == 20 and result['error'].shape[-2:] == (2, 5)
