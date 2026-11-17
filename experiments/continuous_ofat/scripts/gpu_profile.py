"""Standalone CPU/GPU/mixed float64 full-history numerical workload benchmark.

Run with the preinstalled torch interpreter; production uses pinned scipy.
Includes full Newton likelihood/gradient/Hessian recomputation, CPU constrained
quadratic solves and host-device transfer, and grouped resident GPU kernels.
"""
import json,time,sys,resource
import numpy as np
import torch
from scipy.special import expit
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

torch.set_num_threads(1)
DTYPE=torch.float64

def quad(A,b):
    e,U=np.linalg.eigh(A);c=U.T@b
    if np.linalg.norm(c/e)<=4:return U@(c/e)
    h=max(1,np.linalg.norm(b)/4)
    while np.linalg.norm(c/(e+h))>4:h*=2
    eta=brentq(lambda a:np.linalg.norm(c/(e+a))-4,0,h)
    return U@(c/(e+eta))

def fit_cpu(X,y,initial):
    t=initial.copy();I=np.eye(len(t))
    for _ in range(12):
        z=X@t;p=expit(z);g=X.T@(p-y)+t;H=X.T@((p*(1-p))[:,None]*X)+I
        if np.linalg.norm(g)/len(y)<1e-8:return t
        t=quad(H,H@t-g)
    return t

def fit_mixed(X,y,initial,device='cuda:0',resident=None):
    xx,yy=resident if resident else (torch.as_tensor(X,device=device),torch.as_tensor(y,device=device))
    t=torch.as_tensor(initial.copy(),device=device);I=torch.eye(len(t),device=device,dtype=DTYPE)
    for _ in range(12):
        z=xx@t;p=torch.sigmoid(z);g=xx.T@(p-yy)+t;H=xx.T@((p*(1-p))[:,None]*xx)+I
        gg=g.cpu().numpy();HH=H.cpu().numpy();tt=t.cpu().numpy()
        if np.linalg.norm(gg)/len(y)<1e-8:return tt
        t=torch.as_tensor(quad(HH,HH@tt-gg),device=device)
    return t.cpu().numpy()

def timing(fn,repeats=6,gpu=False):
    fn()
    if gpu:torch.cuda.synchronize()
    values=[]
    for _ in range(repeats):
        a=time.perf_counter();fn()
        if gpu:torch.cuda.synchronize()
        values.append(time.perf_counter()-a)
    return dict(median=float(np.median(values)),min=min(values),max=max(values),samples=values)

def main(out):
    rows=[];rng=np.random.default_rng(20261005)
    for d in [5,10,20]:
        for n in [100,2000,20000]:
            X=rng.normal(size=(n,d));X/=np.linalg.norm(X,axis=1)[:,None];X*=rng.uniform(size=n)[:,None]**(1/d)
            truth=rng.normal(size=d);truth*=1.5/np.linalg.norm(truth);y=(rng.random(n)<expit(X@truth)).astype(float)
            # Real warm-start change from the previous completed batch.
            with threadpool_limits(1):initial=fit_cpu(X[:-10],y[:-10],np.zeros(d))
            for threads in [1,2,4]:
                with threadpool_limits(threads):
                    tm=timing(lambda:fit_cpu(X,y,initial))
                rows.append(dict(d=d,history_N=n,backend='cpu',threads=threads,**tm))
            resident=(torch.as_tensor(X,device='cuda:0'),torch.as_tensor(y,device='cuda:0'))
            for tag,res in [('mixed_transfer',None),('mixed_resident',resident)]:
                with threadpool_limits(1):tm=timing(lambda:fit_mixed(X,y,initial,resident=res),gpu=True)
                rows.append(dict(d=d,history_N=n,backend=tag,threads=1,**tm))
            for group in [4,16,32]:
                # Independent compatible trajectories with their own predictors.
                xx=torch.as_tensor(np.stack([X+rng.normal(0,1e-5,X.shape) for _ in range(group)]),device='cuda:0')
                tt=torch.as_tensor(np.stack([initial+rng.normal(0,.01,d) for _ in range(group)]),device='cuda:0')
                yy=torch.as_tensor(np.tile(y,(group,1)),device='cuda:0')
                def kernel():
                    z=torch.bmm(xx,tt[:,:,None])[:,:,0];p=torch.sigmoid(z)
                    g=torch.bmm(xx.transpose(1,2),(p-yy)[:,:,None])[:,:,0]+tt
                    H=torch.bmm(xx.transpose(1,2),p[:,:,None]*(1-p[:,:,None])*xx)
                    return g.cpu().numpy(),H.cpu().numpy()
                tm=timing(kernel,gpu=True)
                rows.append(dict(d=d,history_N=n,backend='gpu_grouped_one_newton_kernel',group=group,per_trajectory=tm['median']/group,**tm))
    data=dict(rows=rows,torch=torch.__version__,device=torch.cuda.get_device_name(0),gpu_uuid='see resources.json',
              qualifications='Grouped GPU measures a full history derivative/Hessian sweep, not a whole constrained refit; mixed includes CPU constrained solves. CPU acquisition/SDP timings are measured separately. No short-history extrapolation.',
              rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    with open(out,'w') as f:json.dump(data,f,indent=2)
    for row in rows:
        if row['d']==20:print({k:v for k,v in row.items() if k not in ['samples','max','min']},flush=True)

if __name__=='__main__':main(sys.argv[1])
