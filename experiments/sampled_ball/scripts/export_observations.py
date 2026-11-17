#!/usr/bin/env python3
"""Export original synthetic observations, verifying against the frozen run.

Bernoulli X is retained in the original checkpoint; labels are reconstructed
from the original independent response seeds. Gaussian query trajectories are
replayed with read-only instrumentation of the frozen simulation, and every
saved scientific checkpoint must agree bit for bit before export is accepted.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[variable]='1'

def export_one(run_string, destination_string, job):
    import numpy as np
    import torch
    run,destination=Path(run_string),Path(destination_string)
    sys.path.insert(0,str(run/'source_snapshot'))
    from src import simulation,oracles
    torch.set_num_threads(1)
    config=json.loads((run/'config.json').read_text())
    c=dict(job['condition']);c['name']=c['id']
    R,N,d,m=job['replicates'],config['N_run'],c['d'],c['m']
    family,method=job['family'],job['method']
    ids=np.arange(job.get('replicate_start',0),job.get('replicate_start',0)+R)
    theta_star=torch.tensor(config['directions'][str(d)],dtype=torch.float64)*(c['kappa']/c['L'])
    innovations=np.stack([getattr(np.random.default_rng(simulation.stable_seed(
        'response',config['global_noise_seed'],family,c['id'],int(r))),
        'standard_normal' if family=='gaussian' else 'random')(N) for r in ids])
    noise=torch.from_numpy(innovations)
    original=torch.load(run/'checkpoints'/f"{job['id']}.pt",map_location='cpu',weights_only=False)
    report=dict(job_id=job['id'],family=family,method=method,condition=c['id'],replicates=R,N=N,d=d,
                source_checkpoint=f"run/checkpoints/{job['id']}.pt",response_seed=config['global_noise_seed'])
    started=time.monotonic()
    if family=='bernoulli':
        X=original['X'][:,:N].numpy()
        y=np.empty((R,N),dtype=np.uint8)
        b=torch.zeros((R,d),dtype=torch.float64)
        minimum_margin=float('inf')
        for n in range(0,N,m):
            xb=original['X'][:,n:n+m]
            probability=torch.sigmoid(torch.matmul(xb,theta_star))
            minimum_margin=min(minimum_margin,float((noise[:,n:n+m]-probability).abs().min()))
            yb=(noise[:,n:n+m]<probability).to(torch.float64)
            y[:,n:n+m]=yb.numpy().astype(np.uint8)
            b+=torch.matmul(xb.transpose(-1,-2),yb.unsqueeze(-1)).squeeze(-1)
        error=float((b-original['b']).abs().max())
        if not torch.allclose(b,original['b'],rtol=2e-12,atol=2e-10):
            raise AssertionError(f"Bernoulli sufficient-statistic mismatch {job['id']}: {error}")
        if minimum_margin<=1e-12:
            raise AssertionError(f"Bernoulli label reconstruction too near a threshold: {minimum_margin}")
        report.update(export_origin='stored X; labels reconstructed using original per-replicate uniforms',
                      b_max_absolute_difference=error,minimum_label_decision_margin=minimum_margin,
                      shared_design=False,validated=True)
    else:
        shared=method in simulation.INFO_METHODS
        X=np.empty((1 if shared else R,N,d),dtype=np.float64)
        y=np.empty((R,N),dtype=np.float64)
        target=dict(np.load(run/'results'/f"{job['id']}.npz",allow_pickle=False))
        positions={int(n):i for i,n in enumerate(target['N'])}
        real_select,real_record=oracles.select_batch,simulation._record
        real_save_checkpoint,real_save_result=simulation._save_checkpoint,simulation._save_result
        checked=[]
        def capture(*args,**kwargs):
            xb,state,diag=real_select(*args,**kwargs)
            n=(int(args[6])-1)*m
            X[:,n:n+m]=xb.numpy()
            xb_all=xb.expand(R,-1,-1) if shared else xb
            y[:,n:n+m]=(torch.matmul(xb_all,theta_star)+noise[:,n:n+m]).numpy()
            return xb,state,diag
        def verify_record(state,truth):
            real_record(state,truth)
            record=state['records'][-1];i=positions[state['N']]
            for key in ['theta','error','Vstar','Vhat','true_min','plugin_min','tracking_op']:
                if not np.array_equal(record[key],target[key][i]):
                    delta=float(np.max(np.abs(record[key]-target[key][i])))
                    raise AssertionError(f"Gaussian replay mismatch {job['id']}/{state['N']}/{key}: {delta}")
            checked.append(state['N'])
        def verify_checkpoint(path,state,generator,shared):
            if state['N']==N:
                for key in ['theta','C','b','Vhat','Vstar']:
                    if not torch.equal(state[key],original[key]):
                        raise AssertionError(f"Terminal Gaussian state mismatch: {job['id']}/{key}")
        oracles.select_batch=capture
        simulation._record=verify_record
        simulation._save_checkpoint=verify_checkpoint
        simulation._save_result=lambda *args,**kwargs:None
        try:
            with tempfile.TemporaryDirectory(prefix='gaussian_replay_') as scratch:
                simulation.run_job(job,config,scratch,'cpu')
        finally:
            oracles.select_batch=real_select;simulation._record=real_record
            simulation._save_checkpoint=real_save_checkpoint;simulation._save_result=real_save_result
        if checked!=target['N'].tolist():raise AssertionError('Checkpoint coverage mismatch')
        report.update(export_origin='deterministic replay of frozen source and original seeds',
                      checked_checkpoints=len(checked),all_saved_states_bit_exact=True,
                      terminal_sufficient_statistics_bit_exact=True,shared_design=shared,validated=True)
    destination.mkdir(parents=True,exist_ok=True)
    path=destination/f"{job['id']}.npz"
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez(f,X=X,y=y,replicate_ids=ids,theta_star=theta_star.numpy(),
                 N=np.asarray(N),m=np.asarray(m),family=np.asarray(family),method=np.asarray(method),
                 condition=np.asarray(c['id']),shared_design=np.asarray(report['shared_design']),
                 provenance=np.asarray(report['export_origin']))
    os.replace(temp,path)
    report.update(file=f"observations/{path.name}",X_shape=list(X.shape),y_shape=list(y.shape),
                  file_bytes=path.stat().st_size,seconds=time.monotonic()-started)
    (destination/(job['id']+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    run=Path(a.run).resolve();out=Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=True)
    jobs=json.loads((run/'jobs.json').read_text());rows=[];started=time.monotonic()
    manifest=json.loads((run/'manifest.json').read_text())
    for name in ['simulation.py','oracles.py','estimators.py']:
        path=run/'source_snapshot/src'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['source_sha256']['src/'+name]
    # The longest replay jobs start first. Independent labels are reconstructed
    # concurrently without changing or extending the original experiment.
    jobs.sort(key=lambda j:(j['family']!='gaussian',j['method']!='EGreedy',-j['condition']['d']))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(export_one,str(run),str(out),job):job for job in jobs}
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            result=dict(status='complete' if len(rows)==len(jobs) else 'running',completed_jobs=len(rows),
                        total_jobs=len(jobs),elapsed_seconds=time.monotonic()-started,rows=rows)
            temp=out/'export_manifest.tmp';temp.write_text(json.dumps(result,indent=2)+'\n')
            os.replace(temp,out/'export_manifest.json')
            print(f"{len(rows)}/{len(jobs)} {row['job_id']} {row['seconds']:.1f}s",flush=True)
    assert sum(r['replicates']*r['N'] for r in rows)==51200000

if __name__=='__main__':main()
