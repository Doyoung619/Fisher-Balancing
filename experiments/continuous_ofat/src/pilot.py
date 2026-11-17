"""Actual-policy long-history pilots, separate deterministic IDs from production."""
import argparse,json,time,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import numpy as np
from .common import ROOT,resolved_config,code_hash,json_hash,task_id,digest
from .checkpoint import load_pickle,atomic_json
from .simulation import run_task
from .reference import build_references


def main(output,workers=12):
    c=resolved_config();run=Path(output);run.mkdir(exist_ok=False,parents=True)
    for directory in ['checkpoints','trajectories','summaries','figures','logs','workers','task_results']:(run/directory).mkdir()
    tasks=[]
    for family,condition,methods in [('bernoulli','center',c['methods']),('bernoulli','dim20',['Ours','E-greedy','E-relax']),('gaussian','center',['Ours','UCB','TS','Random'])]:
        for method in methods:
            tasks.append(dict(id=task_id(family,condition,method,100000),family=family,condition=condition,method=method,replicate=100000))
    atomic_json(run/'config.json',c);atomic_json(run/'jobs.json',tasks);atomic_json(run/'references.json',build_references(c))
    atomic_json(run/'manifest.json',dict(config_hash=json_hash(c),code_hash=code_hash(),pilot=True,
                purpose='Actual same-policy full N20000 histories; timing only, independent reserved replicate100000',
                source_hashes={p.name:digest(p) for p in (ROOT/'sources').iterdir() if p.is_file()}))
    start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(run_task,str(run),task,c['N_run']):task for task in tasks}
        for future in as_completed(futures):
            task=futures[future];result=future.result();state=load_pickle(run/'checkpoints'/f"{task['id']}.pkl")
            condition=next(r for r in c['conditions'] if r['id']==task['condition'])
            for n,lo,hi in [(100,20,200),(2000,1800,2200),(19000,18000,20000)]:
                records=[r for r in state['batch_diagnostics'] if lo<=r['N']<=hi]
                if not records:continue
                values=[r['acquisition_seconds']+r['refit_seconds'] for r in records]
                rows.append(dict(family=task['family'],condition=task['condition'],method=task['method'],d=condition['d'],L=condition['L'],m=condition['m'],backend='cpu',history_N=n,
                    batch_seconds=float(np.mean(values)),median_seconds=float(np.median(values)),min_seconds=min(values),max_seconds=max(values),samples=len(values),
                    acquisition_seconds=float(np.mean([r['acquisition_seconds'] for r in records])),refit_seconds=float(np.mean([r['refit_seconds'] for r in records])),
                    actual_same_policy_history=True,pilot_task=task['id']))
            atomic_json(run/'actual_history_profile.json',dict(rows=rows,elapsed_seconds=time.perf_counter()-start,completed=sum(r['history_N']==19000 for r in rows),expected=len(tasks),workers=workers))
            print(f"pilot complete {task['id']} N={result['N']} seconds={result['elapsed_seconds']:.3f}",flush=True)
    print(f'All {len(tasks)} actual long-history pilots completed in {time.perf_counter()-start:.2f}s',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=12);a=p.parse_args();main(a.output,a.workers)
