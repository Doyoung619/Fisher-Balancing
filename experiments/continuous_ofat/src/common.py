import hashlib
import json
import os
import subprocess
import platform
import sys
import importlib.metadata
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['Ours','A-opt','D-opt','E-greedy','UCB','TS','Random']
GAUSSIAN_IDENTICAL = ['Ours','A-opt','D-opt','E-greedy']


@lru_cache(None)
def environment_versions():
    packages=['numpy','scipy','cvxpy','clarabel','matplotlib','PyYAML','mpmath','threadpoolctl']
    return dict(python=platform.python_version(),**{p:importlib.metadata.version(p) for p in packages})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def code_hash():
    files = sorted([*ROOT.glob('src/*.py'),*ROOT.glob('scripts/*'),*ROOT.glob('configs/*')])
    return json_hash({str(f.relative_to(ROOT)):digest(f) for f in files if f.is_file()})


def seed(master, *parts):
    words = np.frombuffer(hashlib.sha256(json.dumps(parts).encode()).digest(), dtype='<u4')
    return np.random.SeedSequence([master,*map(int,words)])


def policy_rngs(config, task):
    return {name:np.random.default_rng(seed(config['master_seed'],'policy',task['family'],task['condition'],task['method'],task['replicate'],name))
            for name in ['solver','tie','rounding','ts','random']}


def noise_rng(config, task, d):
    return np.random.default_rng(seed(config['master_seed'],'response',task['family'],d,task['replicate']))


def task_id(family, condition, method, rep):
    return f'{family}__{condition}__{method}__r{rep:05d}'


def resolved_config(smoke=False):
    from .methods import METHOD_IDS
    c = json.loads((ROOT/'sources/OFAT_SPEC.json').read_text())
    c.pop('specification_only',None)
    c['methods']=METHODS.copy()
    c['method_ids']={method:METHOD_IDS[method] for method in METHODS}
    c['excluded_methods']=['E-relax']
    c['specification_amendment']='sources/USER_AMENDMENT_20261005.md'
    c['expected_family_condition_method_combinations']=98
    c['expected_logical_trajectories']=28672
    c.update(oracle_scalar_grid=17,oracle_starts=4,oracle_maxiter=150,
             e_relax_max_rounds=256,oracle_audit_interval_batches=200,
             estimator_max_iter=80,workers=64,blas_threads=1,backend='cpu',
             gaussian_exact_deduplication=True,plot_interval_seconds=3600,
             checkpoint_counts=sorted(set([0,20,400,2000,20000]+(20*np.rint(np.geomspace(1,1000,80))).astype(int).tolist())))
    c['directions']={}
    for d in [5,10,20]:
        u=np.random.default_rng(seed(c['master_seed'],'environment',d)).normal(size=d)
        c['directions'][str(d)]=(u/np.linalg.norm(u)).tolist()
    if smoke:
        c.update(N_run=400,preview_N=40,preview_replicates=1,replicates_center=1,replicates_noncentral=1,
                 checkpoint_counts=[0,20,40,80,160,400],workers=8)
    return c


def validate_config(c, production=True):
    assert len(c['conditions'])==7 and c['methods']==METHODS and len(c['families'])==2
    center=next(row for row in c['conditions'] if row['id']=='center')
    for row in c['conditions']:
        assert sum(row[k]!=center[k] for k in ['d','L','m','kappa'])==(row['id']!='center')
        assert row['kappa']/row['L']<c['parameter_radius']
        assert c['N_run']%row['m']==0
    if production:
        assert c['N_run']==20000 and c['replicates_center']==512 and c['replicates_noncentral']==256
    return True


def jobs(c):
    result=[]
    for rep in range(max(c['replicates_center'],c['replicates_noncentral'])):
        for row in c['conditions']:
            if rep >= c['replicates_center' if row['id']=='center' else 'replicates_noncentral']:
                continue
            for family in c['families']:
                for method in c['methods']:
                    item=dict(id=task_id(family,row['id'],method,rep),family=family,condition=row['id'],method=method,replicate=rep)
                    if family=='gaussian' and method in GAUSSIAN_IDENTICAL[1:] and c.get('gaussian_exact_deduplication'):
                        item['alias_of']=task_id(family,row['id'],'Ours',rep)
                    result.append(item)
    return result


def validate_run(run_dir):
    run=Path(run_dir)
    manifest=json.loads((run/'manifest.json').read_text())
    c=json.loads((run/'config.json').read_text())
    if manifest['config_hash']!=json_hash(c):
        raise RuntimeError('Immutable config hash mismatch; create a new run')
    if manifest['code_hash']!=code_hash():
        raise RuntimeError('Code hash mismatch; incompatible resume refused')
    for name,h in manifest['source_hashes'].items():
        if digest(ROOT/'sources'/name)!=h:
            raise RuntimeError('Source hash mismatch')
    if manifest.get('jobs_hash') and digest(run/'jobs.json')!=manifest['jobs_hash']:
        raise RuntimeError('Immutable queue hash mismatch')
    if manifest.get('environment_freeze_sha256') and digest(ROOT/'environment.freeze.txt')!=manifest['environment_freeze_sha256']:
        raise RuntimeError('Pinned environment manifest changed')
    if manifest.get('environment_versions') and environment_versions()!=manifest['environment_versions']:
        raise RuntimeError('Installed runtime dependency versions changed')
    return c


def prepare_run(output=None, smoke=False, profile=None):
    from .reference import build_references
    c=resolved_config(smoke)
    if profile:
        p=json.loads(Path(profile).read_text())
        c.update(workers=p['workers'],blas_threads=p.get('blas_threads',1),backend='cpu')
    validate_config(c,not smoke)
    run=Path(output) if output else ROOT/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run.mkdir(parents=True,exist_ok=False)
    for sub in ['checkpoints','trajectories','summaries','figures','logs','workers','task_results']:
        (run/sub).mkdir()
    (run/'config.json').write_text(json.dumps(c,indent=2))
    (run/'jobs.json').write_text(json.dumps(jobs(c),indent=2))
    (run/'references.json').write_text(json.dumps(build_references(c),indent=2))
    if profile:
        (run/'profile.json').write_bytes(Path(profile).read_bytes())
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),code_hash=code_hash(),config_hash=json_hash(c),
                  source_hashes={p.name:digest(p) for p in (ROOT/'sources').iterdir() if p.is_file()},
                  logical_trajectories=len(jobs(c)),physical_trajectories=sum('alias_of' not in j for j in jobs(c)),
                  gaussian_shared_methods=GAUSSIAN_IDENTICAL,gaussian_alias_temperature_trace='Shared Ours audit trace; A/D/E comparator acquisitions do not use temperature',
                  method_ids=c['method_ids'],jobs_hash=digest(run/'jobs.json'),environment_freeze_sha256=digest(ROOT/'environment.freeze.txt'),environment_versions=environment_versions(),
                  python=sys.executable,smoke=smoke)
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return run


def prepare_continuation(parent,output=None,N=None,extra_repetitions=0):
    """Explicit opt-in only: fork immutable provenance, retain every RNG state."""
    from .checkpoint import load_pickle,atomic_pickle
    import shutil
    parent=Path(parent).resolve();c=validate_run(parent)
    old_manifest=json.loads((parent/'manifest.json').read_text())
    c=dict(c);old_N=c['N_run']
    if N is not None:
        if N<old_N or N%20:raise ValueError('Continuation N must be >= current N and divisible by20')
        c['N_run']=N
        c['checkpoint_counts']=sorted(set(c['checkpoint_counts']+[int(n) for n in 20*np.rint(np.geomspace(old_N/20,N/20,30))]))
    if extra_repetitions<0:raise ValueError('extra_repetitions must be nonnegative')
    for key in ['replicates_center','replicates_noncentral']:c[key]+=extra_repetitions
    if c['N_run']==old_N and extra_repetitions==0:raise ValueError('No continuation requested')
    c['production_extensions_enabled']=True
    validate_config(c,False)
    run=Path(output) if output else ROOT/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-continuation')
    run.mkdir(parents=True,exist_ok=False)
    for sub in ['checkpoints','trajectories','summaries','figures','logs','workers','task_results']:(run/sub).mkdir()
    (run/'config.json').write_text(json.dumps(c,indent=2));(run/'jobs.json').write_text(json.dumps(jobs(c),indent=2))
    for name in ['references.json','profile.json']:
        if (parent/name).exists():shutil.copy2(parent/name,run/name)
    manifest=dict(old_manifest,created_utc=datetime.now(timezone.utc).isoformat(),config_hash=json_hash(c),code_hash=code_hash(),
                  logical_trajectories=len(jobs(c)),physical_trajectories=sum('alias_of' not in t for t in jobs(c)),
                  jobs_hash=digest(run/'jobs.json'),continuation_of=str(parent),parent_manifest_sha256=digest(parent/'manifest.json'))
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for p in (parent/'checkpoints').glob('*.pkl'):
        s=load_pickle(p);s['config_hash']=manifest['config_hash'];s['code_hash']=manifest['code_hash'];atomic_pickle(run/'checkpoints'/p.name,s)
    for folder,pattern in [('checkpoints','*.status.json'),('trajectories','*.npz'),('summaries','*.json')]:
        for p in (parent/folder).glob(pattern):
            # Immutable files replace atomically in the new run, never mutate parent.
            os.link(p,run/folder/p.name)
    return run


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--smoke',action='store_true');a.add_argument('--profile');a.add_argument('--output');a.add_argument('--write-configs',action='store_true');a.add_argument('--parent');a.add_argument('--N',type=int);a.add_argument('--extra-repetitions',type=int,default=0);args=a.parse_args()
    if args.write_configs:
        for name,smoke in [('ofat',False),('smoke',True)]:
            (ROOT/'configs'/f'{name}.yaml').write_text(yaml.safe_dump(resolved_config(smoke),sort_keys=False))
    if args.parent:
        print(prepare_continuation(args.parent,args.output,args.N,args.extra_repetitions))
    elif args.prepare:
        print(prepare_run(args.output,args.smoke,args.profile))
