"""Measured production kernels at early/middle/late full continuous histories."""
import argparse,json,time,csv,copy,resource,os
from pathlib import Path
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
from .common import ROOT,resolved_config,policy_rngs
from .estimators import fit_bernoulli,fit_gaussian
from .models import information,responses


def profile(output,condition_id=None):
    from .methods import select_batch
    c=resolved_config();rng=np.random.default_rng(819203);rows=[]
    path=Path(output);path.parent.mkdir(exist_ok=True,parents=True)
    # Histories are real volume-uniform continuous queries with independently
    # generated responses. Every timed refit updates the complete old history.
    # These label-free pilot designs avoid basing routing on any algorithm wins.
    for condition in c['conditions']:
        if condition_id is not None and condition['id']!=condition_id:continue
        d,L,m=condition['d'],condition['L'],condition['m'];theta=np.array(c['directions'][str(d)])*condition['kappa']/L
        X=rng.normal(size=(20020,d));X*=L*rng.random(20020)[:,None]**(1/d)/np.linalg.norm(X,axis=1)[:,None]
        for family in c['families']:
            y=responses(family,X,theta,rng.normal(size=len(X)) if family=='gaussian' else rng.random(len(X)))
            for n in [20,2000,18000,20000-m]:
                n=(n//m)*m
                if family=='bernoulli':th,Vh,_=fit_bernoulli(X[:n],y[:n])
                else:th,Vh,_=fit_gaussian(X[:n].T@X[:n],X[:n].T@y[:n],N=n)
                H=np.eye(d)+X[:n].T@X[:n]
                for method in c['methods']:
                    task=dict(family=family,condition=condition['id'],method=method,replicate=0)
                    samples=[];acqs=[];fits=[];diags=[];state={}
                    for repeat in range(2):
                        starts=policy_rngs(c,dict(task,replicate=repeat))
                        t0=time.perf_counter()
                        xb,state,diag=select_batch(method,family,th,Vh,H,condition,n//m+1,starts,state,c)
                        t1=time.perf_counter()
                        yb=responses(family,xb,theta,rng.normal(size=m) if family=='gaussian' else rng.random(m))
                        if family=='bernoulli':_,_,fit=fit_bernoulli(np.vstack([X[:n],xb]),np.r_[y[:n],yb],th)
                        else:_,_,fit=fit_gaussian(H-np.eye(d)+xb.T@xb,X[:n].T@y[:n]+xb.T@yb,N=n+m)
                        t2=time.perf_counter()
                        if not fit['success']:raise RuntimeError(fit)
                        samples.append(t2-t0);acqs.append(t1-t0);fits.append(t2-t1);diags.append(diag)
                    row=dict(family=family,method=method,d=d,L=L,m=m,condition=condition['id'],backend='cpu',history_N=n,
                             batch_seconds=float(np.median(samples)),acquisition_seconds=float(np.median(acqs)),refit_seconds=float(np.median(fits)),
                             min_seconds=min(samples),max_seconds=max(samples),samples=samples,oracle_diagnostics=diags)
                    rows.append(row)
                print(f"profile {condition['id']} {family} N={n}: last batch {rows[-1]['batch_seconds']:.6f}s",flush=True)
                payload=dict(rows=rows,workers=c['workers'],blas_threads=1,backend='cpu',rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                             history_construction='Real independent continuous volume-uniform queries and generated responses; each policy acquired/refitted new batches at early, middle, and late full-history lengths. Off-policy pilot history; ongoing actual-policy timings update ETA.')
                path.write_text(json.dumps(payload,indent=2,default=lambda o:o.tolist() if isinstance(o,np.ndarray) else o.item()))
    fields=['family','condition','method','d','L','m','backend','history_N','batch_seconds','acquisition_seconds','refit_seconds','min_seconds','max_seconds']
    with path.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return payload


def parallel_profile(output,workers):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing as mp
    from .common import code_hash,environment_versions
    c=resolved_config();path=Path(output);parts=path.parent/(path.stem+'_parts');parts.mkdir(parents=True,exist_ok=True)
    rows=[]
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(profile,str(parts/(cond['id']+'.json')),cond['id']) for cond in c['conditions']]
        for future in as_completed(futures):
            result=future.result();rows.extend(result['rows'])
            payload=dict(rows=rows,workers=c['workers'],blas_threads=1,backend='cpu',profile_workers=workers,
                         code_hash=code_hash(),environment_versions=environment_versions(),
                         completed_conditions=len(rows)//(2*len(c['methods'])*4),expected_conditions=7,
                         history_construction=result['history_construction'])
            path.write_text(json.dumps(payload,indent=2,default=lambda o:o.tolist() if isinstance(o,np.ndarray) else o.item()))
    assert len(rows)==len(c['families'])*len(c['conditions'])*len(c['methods'])*4
    fields=['family','condition','method','d','L','m','backend','history_N','batch_seconds','acquisition_seconds','refit_seconds','min_seconds','max_seconds']
    with path.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'audits/profile.json'));p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    with threadpool_limits(1):
        if a.workers>1:parallel_profile(a.output,a.workers)
        else:profile(a.output)
