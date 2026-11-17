"""Complete acquisition plus full-history refit throughput, persistent workers."""
import os,json,time,sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def workload(args):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from src.common import resolved_config,policy_rngs
    from src.methods import select_batch
    from src.estimators import fit_bernoulli
    from src.models import responses
    route,index,repeats=args
    cfg=resolved_config();cfg['oracle_audit_interval_batches']=0
    row=cfg['conditions'][index%len(cfg['conditions'])];d=row['d'];m=row['m'];n=18000
    rng=np.random.default_rng(901+index);X=rng.normal(size=(n,d));X*=row['L']*rng.random(n)[:,None]**(1/d)/np.linalg.norm(X,axis=1)[:,None]
    truth=np.asarray(cfg['directions'][str(d)])*row['kappa']/row['L'];y=responses('bernoulli',X,truth,rng.random(n))
    with threadpool_limits(1):th,Vh,_=fit_bernoulli(X,y)
    H=np.eye(d)+X.T@X
    if route=='mixed':
        from gpu_profile import fit_mixed
        import torch
        device=f'cuda:{index%4}'
    values=[];tstart=time.perf_counter()
    with threadpool_limits(1):
        for j in range(repeats):
            for method in cfg['methods']:
                task=dict(family='bernoulli',condition=row['id'],method=method,replicate=index)
                a=time.perf_counter();xb,_,diag=select_batch(method,'bernoulli',th,Vh,H,row,n//m+1+j,policy_rngs(cfg,task),{},cfg)
                yb=responses('bernoulli',xb,truth,rng.random(m));xx=np.vstack([X,xb]);yy=np.r_[y,yb]
                if route=='mixed':new=fit_mixed(xx,yy,th,device=device)
                else:new,_,_=fit_bernoulli(xx,yy,th)
                values.append(time.perf_counter()-a)
    import resource
    return dict(seconds=time.perf_counter()-tstart,batches=len(values),rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,route=route)


def main(output):
    rows=[]
    for route in ['cpu','mixed']:
        for workers in [8,16,24,32,48,64]:
            # Spawn is production's process model. Warm workers before measuring.
            with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn')) as pool:
                warm=list(pool.map(workload,[(route,i,0) for i in range(workers)]))
                t=time.perf_counter();results=list(pool.map(workload,[(route,i,2) for i in range(workers)]));elapsed=time.perf_counter()-t
            row=dict(route=route,workers=workers,wall_seconds=elapsed,batches=sum(r['batches'] for r in results),
                     batches_per_second=sum(r['batches'] for r in results)/elapsed,rss_per_worker_mb=max(r['rss_mb'] for r in results))
            rows.append(row);print(row,flush=True)
            Path(output).write_text(json.dumps(dict(rows=rows),indent=2))

if __name__=='__main__':main(sys.argv[1])
