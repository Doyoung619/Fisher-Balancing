"""Measured early/late acquisition, cumulative-refit, and disk-write timings."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from .oracles import candidate_cloud, select_batch
from .estimators import fit_gaussian, fit_bernoulli
from .simulation import atomic_json, atomic_npz


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)


def benchmark(output, quick=False):
    torch.set_num_threads(1)
    devices=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
    rows=[]; start=time.time()
    cases=[('gaussian',20,10),('bernoulli',10,10),('bernoulli',20,10),('bernoulli',10,5),('bernoulli',10,20)]
    # Full-history synthetic late states are used only for profiling and clearly
    # identified; benchmark responses are never reused in production trajectories.
    for device in devices:
        for K in (2048,4096):
            for family,d,m in cases:
                condition=dict(name='benchmark',d=d,L=1.,m=m,kappa=1.5)
                sync(device);cloud_start=time.perf_counter()
                candidate_cloud(family,condition,1,K,7,device)
                sync(device)
                rows.append(dict(component='cloud',device=device,family=family,d=d,m=m,R=1,K=K,seconds=time.perf_counter()-cloud_start))
                methods=('FB','AOpt','DOpt','EGreedy','UCB','TS','Random')
                for method in methods:
                    R=1 if family=='gaussian' and method in ('FB','AOpt','DOpt','EGreedy') else (128 if method=='FB' else 32)
                    if quick and method not in ('FB','EGreedy','TS'): continue
                    eye=torch.eye(d,dtype=torch.float64,device=device)
                    theta=torch.zeros((R,d),dtype=torch.float64,device=device); theta[:,0]=1.5
                    for phase,n in [('early',0),('late',10000)]:
                        W=(1+n*(1 if family=='gaussian' else .22)/d)*eye.repeat(R,1,1)
                        fitted=theta if n else torch.zeros_like(theta)
                        generator=torch.Generator(device=device).manual_seed(2)
                        cloud=candidate_cloud(family,condition,max(1,n//m),K,7,device)
                        cfg=dict(K=K,Q=96 if K==2048 else 128)
                        # One warmup handles CUDA kernels and secular JIT compilation.
                        if family=='gaussian' and d==20 and method=='FB' and phase=='early':
                            select_batch(method,family,fitted,W,W,condition,max(1,n//m),cloud=cloud,generator=generator,config=cfg)
                        sync(device);t=time.perf_counter()
                        X,_,diag=select_batch(method,family,fitted,W,W,condition,max(1,n//m),cloud=cloud,generator=generator,config=cfg)
                        sync(device);elapsed=time.perf_counter()-t
                        row=dict(component='acquisition',device=device,family=family,d=d,m=m,R=R,K=K,Q=cfg['Q'],method=method,phase=phase,N=n,seconds=elapsed)
                        rows.append(row)
                    atomic_json(output,dict(status='running',elapsed_seconds=time.time()-start,rows=rows))
                if K!=2048: continue
                for R in ((32,128,256) if family=='bernoulli' and d==10 and m==10 else (32,128) if family=='bernoulli' else (64,256)):
                    for n in (100,5000,10000):
                        if quick and R==256: continue
                        rng=torch.Generator(device=device).manual_seed(71)
                        X=torch.randn((R,n+m,d),dtype=torch.float64,device=device,generator=rng)/np.sqrt(d)
                        true=torch.zeros((R,d),dtype=torch.float64,device=device);true[:,0]=1.5
                        y=torch.randn((R,n+m),dtype=torch.float64,device=device,generator=rng)+(X*true[:,None,:]).sum(-1) if family=='gaussian' else (torch.rand((R,n+m),dtype=torch.float64,device=device,generator=rng)<torch.sigmoid((X*true[:,None,:]).sum(-1))).double()
                        b=(X.transpose(-1,-2)@y[:,:,None]).squeeze(-1)
                        if family=='bernoulli':
                            fitted,_,_=fit_bernoulli(X[:,:n],y[:,:n],initial=true)
                        else:
                            C=X.transpose(-1,-2)@X
                        sync(device); t=time.perf_counter()
                        if family=='gaussian':
                            _,_,diag=fit_gaussian(C,b,N=n+m)
                        else:
                            _,_,diag=fit_bernoulli(X,initial=fitted,b=b)
                        sync(device);elapsed=time.perf_counter()-t
                        rows.append(dict(component='refit',device=device,family=family,d=d,m=m,R=R,K=K,N=n,phase='early' if n==100 else 'late',seconds=elapsed,iterations=diag['iterations'],success=diag['success'],synthetic_history=True))
                        if not diag['success']: raise RuntimeError('Benchmark MLE convergence failed')
                        if n==10000 and R==128 and family=='bernoulli' and m==10:
                            with tempfile.TemporaryDirectory(dir=str(Path(output).parent)) as tmp:
                                sync(device);t=time.perf_counter()
                                payload=dict(N=n,X=X[:,:n].cpu(),b=b.cpu(),theta=fitted.cpu())
                                target=Path(tmp)/'checkpoint.pt'
                                with target.open('wb') as f:
                                    torch.save(payload,f);f.flush()
                                    import os;os.fsync(f.fileno())
                                rows.append(dict(component='checkpoint',device=device,family=family,d=d,m=m,R=R,N=n,seconds=time.perf_counter()-t,bytes=target.stat().st_size))
                        del X,y,b
                    atomic_json(output,dict(status='running',elapsed_seconds=time.time()-start,rows=rows))
                print(json.dumps(dict(device=device,K=K,family=family,d=d,m=m,rows=len(rows),elapsed_seconds=time.time()-start)),flush=True)
    result=dict(status='complete',elapsed_seconds=time.time()-start,rows=rows,
                notes='Single-thread CPU versus CUDA; synthetic complete histories, previous-MLE warm start with one new batch. Acquisition uses prescribed sampled cloud and exact virtual-update protocol.')
    atomic_json(output,result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--quick',action='store_true')
    args=p.parse_args();Path(args.output).parent.mkdir(parents=True,exist_ok=True);benchmark(args.output,args.quick)
