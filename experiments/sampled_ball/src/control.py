"""Persistent, resumable sampled-ball job queue and measured progress reporting."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import os
import statistics
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['FB','AOpt','DOpt','EGreedy','UCB','TS','Random']
CONDITIONS = [dict(id=name,d=d,L=L,m=m,kappa=1.5) for name,d,L,m in [
    ('central',10,1.,10),('dim5',5,1.,10),('dim20',20,1.,10),
    ('radius05',10,.5,10),('radius2',10,2.,10),('batch5',10,1.,5),('batch20',10,1.,20)]]

def now(): return datetime.now(timezone.utc).isoformat()

def read(path, default=None):
    try: return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError): return default

def atomic(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(data,f,indent=2,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def source_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['src','scripts','sources'] for p in sorted((ROOT/folder).rglob('*'))
            if p.is_file() and '__pycache__' not in str(p) and p.suffix in ['.py','.sh','.tex','.txt','.md']}

def make_config(preset='preferred_full_scan', N=10000):
    import numpy as np
    fast=preset.startswith('fast')
    directions={}
    for d in [5,10,20]:
        u=np.random.default_rng(np.random.SeedSequence([20261005,d])).normal(size=d)
        directions[str(d)]=(u/np.linalg.norm(u)).tolist()
    full_scan='full_scan' in preset
    return dict(project='FisherBalancing-sampled-ball',preset=preset,N_run=N,
        K=2048 if fast else 4096,Q=(2048 if fast else 4096) if full_scan else (96 if fast else 128),full_scan=full_scan,
        families=['gaussian','bernoulli'],methods=METHODS,conditions=CONDITIONS,directions=directions,
        rho=1.,parameter_radius=4.,N_cal=400,ucb_delta=.05,ts_covariance_scale=1.,
        fb_replicates_central=128 if fast else 256,fb_replicates_noncentral=64 if fast else 128,
        comparator_replicates_central=32 if fast else 64,comparator_replicates_noncentral=16 if fast else 32,
        global_cloud_seed=20261006,global_noise_seed=20261007,global_policy_seed=20261008,
        checkpoint_interval_seconds=45,progress_interval_seconds=10,bootstrap_resamples=200,
        bernoulli_radial_levels=[.25,.5,.75,1.],float_precision='float64',
        query_domain='continuous Euclidean ball',oracle='fresh Monte-Carlo approximation of the continuous acquisition oracle',
        feedback='after_entire_batch',initial_theta='zero',forced_exploration=False,
        temperature='tau0 = R sqrt(m/(8 log(d))) / 400**0.75; tau_t = tau0*(m*t)**0.75; no clipping',
        user_budget_amendment='User explicitly permits N_run=5000 when measured ETA exceeds two hours; supersedes attachment minimum 10000.',
        shortlist_qualification=('Shortlist disabled: full K scan at every virtual position. Independent audit of the prescribed Q128->256 acceleration found material unaudited score losses; correctness fallback preserves K/N/replicates.' if full_scan else 'Prescribed audited shortlist, subject to independent validation.'),
        ucb_convention='continuous_ofat/src/methods.py: design-covariance ellipsoid, delta .05, radius4, no within-batch covariance update',
        ts_convention='continuous_ofat/src/methods.py: independent per-query N(theta,Vhat^-1), scale1',
        cpu_workers=28,gpu_workers=4,backend_assignment='bernoulli_light_cpu',
        exact_oracle_theorem_applies=False)

def job_backend(job,config):
    if config.get('gpu_workers',4)==0:
        return 'cpu'
    if config.get('backend_assignment')=='bernoulli_light_cpu' and job['method'] in ['UCB','TS','Random']:
        return 'cpu'
    return 'cpu' if job['family']=='gaussian' else 'cuda'

def stratified_eta(run,jobs,progresses,completed,active,config):
    projection=read(run/'job_projection.json',{})
    baseline=projection.get('jobs',{})
    if not baseline:return None
    done={j['id'] for j in completed};ratios={};fraction={}
    for j in jobs:
        jid=j['id'];p=progresses[jid];f=min(1.,p.get('N',0)/config['N_run'])
        # Approximate cumulative-history work grows quadratically for logistic
        # refits and linearly for acquisition. Live scaling corrects the model.
        frac=f if j['family']=='gaussian' else .5*f+.5*f*f
        fraction[jid]=frac
        base=baseline.get(jid,{}).get('estimated_seconds',0)
        if p.get('N',0)>=100 and p.get('active_seconds',0)>1 and base*frac>0:
            value=p['active_seconds']/(base*frac)
            key=job_backend(j,config)
            ratios.setdefault(key,[]).append(value)
    scales={key:statistics.median(values) for key,values in ratios.items()}
    costs={}
    for j in jobs:
        jid=j['id'];base=baseline.get(jid,{}).get('estimated_seconds',0)
        if not base:return None
        p=progresses[jid];frac=fraction[jid];device=job_backend(j,config)
        factor=scales.get(device,2.5)
        if p.get('N',0)>=100 and p.get('active_seconds',0)>1 and frac>0:
            factor=p['active_seconds']/(base*frac)
        costs[jid]=0 if jid in done else base*factor*(1-frac)
    slots={'cpu':[0.]*config.get('cpu_workers',28),'cuda':[0.]*config.get('gpu_workers',4)}
    active_ids=set()
    for p,j,device,log in active.values():
        group='cpu' if device=='cpu' else 'cuda';loads=slots[group]
        if loads:loads[loads.index(min(loads))]+=costs[j['id']]
        active_ids.add(j['id'])
    for j in jobs:
        if j['id'] in done or j['id'] in active_ids:continue
        loads=slots[job_backend(j,config)]
        if loads:loads[loads.index(min(loads))]+=costs[j['id']]
    remaining=max([max(v,default=0) for v in slots.values()],default=0)
    return dict(seconds=remaining+(180 if len(done)<len(jobs) else 0),live_scaling=scales,
                method='measured component profile scaled by actual CPU/GPU job times; logistic cumulative-history work included; 180s analysis allowance')

def make_jobs(config):
    jobs=[]
    for family in config['families']:
        for c in config['conditions']:
            for method in config['methods']:
                central=c['id']=='central'
                key=('fb_' if method=='FB' else 'comparator_')+'replicates_'+('central' if central else 'noncentral')
                jobs.append(dict(id=f"{family}_{c['id']}_{method}",family=family,condition=c,
                    method=method,replicate_start=0,replicates=config[key]))
    # Complete central panels first; then balance OFAT sweeps.
    return sorted(jobs,key=lambda j:(j['condition']['id']!='central',j['method']!='FB',
                                    j['condition']['id'],METHODS.index(j['method'])))

def initialize(run,preset,N,benchmark=None):
    run=Path(run).resolve();run.mkdir(parents=True,exist_ok=True)
    if (run/'config.json').exists(): raise RuntimeError('Run already initialized; use resume')
    config=make_config(preset,N)
    atomic(run/'config.json',config);atomic(run/'jobs.json',make_jobs(config))
    atomic(run/'manifest.json',dict(created_at=now(),source_directory=str(ROOT),output_directory=str(run),
        manuscript_path=str(ROOT/'sources/FisherBalancing_source.tex'),git_commit=None,
        source_sha256=source_hashes(),selected_preset=preset,N_run=N,K=config['K'],Q=config['Q'],
        benchmark_evidence=read(benchmark,{}) if benchmark else {},
        hardware=dict(platform=__import__('platform').platform(),logical_cpus=os.cpu_count()),config=config))
    (ROOT/'CURRENT_RUN').write_text(str(run)+'\n')
    return config

def status(run):
    s=read(Path(run)/'status.json',{'status':'INITIALIZED','ETA':'calibrating'})
    print(json.dumps(s,indent=2));return s

def worker(run,job_id,device):
    (run/'locks').mkdir(exist_ok=True)
    job_lock=(run/'locks'/f'{job_id}.lock').open('w')
    # A surviving worker from a crashed supervisor owns the lock until its
    # atomic checkpoint is safe. Never run two copies of one trajectory group.
    fcntl.flock(job_lock,fcntl.LOCK_EX)
    from .simulation import run_job
    config=read(run/'config.json'); jobs=read(run/'jobs.json')
    job=next(j for j in jobs if j['id']==job_id)
    result=run_job(job,config,str(run),device=device)
    atomic(run/'summaries'/f'{job_id}.json',result)

def progress_status(run,jobs,active,started,samples,attempts,config):
    progresses={j['id']:read(run/'progress'/f"{j['id']}.json",{}) for j in jobs}
    completed=[j for j in jobs if (run/'done'/f"{j['id']}.json").exists()]
    total_responses=sum(j['replicates']*config['N_run'] for j in jobs)
    responses=sum(j['replicates']*(config['N_run'] if j in completed else
                  int(progresses[j['id']].get('N',0))) for j in jobs)
    total=sum(j['replicates'] for j in jobs);done=sum(j['replicates'] for j in completed)
    clock=time.time(); samples.append((clock,responses,done))
    while len(samples)>2 and clock-samples[0][0]>300:samples.popleft()
    dt=clock-samples[0][0]
    rate=(responses-samples[0][1])/dt if dt>5 else 0
    traj_rate=60*(done-samples[0][2])/dt if dt>5 else 0
    eta=(total_responses-responses)/rate if rate>0 else None
    # This is recent measured response throughput, explicitly not a guarantee.
    stratified=stratified_eta(run,jobs,progresses,completed,active,config)
    throughput_eta=eta
    if stratified:eta=stratified['seconds']
    finish=datetime.now(timezone.utc)+timedelta(seconds=eta) if eta is not None else None
    diagnostics=[p.get('method_state',{}) for p in progresses.values()]
    audits=sum(float(d.get('audit_count',0)) for d in diagnostics)
    misses=sum(float(d.get('miss_count',0)) for d in diagnostics)
    active_rows=[dict(job=j['id'],family=j['family'],condition=j['condition']['id'],method=j['method'],
                    device=device,pid=p.pid,N=progresses[j['id']].get('N',0)) for p,j,device,log in active.values()]
    result=dict(timestamp=now(),timestamp_local=datetime.now(ZoneInfo('America/New_York')).isoformat(),
        status='STOPPING' if (run/'STOP_REQUESTED').exists() else 'RUNNING',stage='production',
        preset=config['preset'],N_run=config['N_run'],K=config['K'],Q_initial=config['Q'],
        completed_trajectories=done,total_trajectories=total,completed_jobs=len(completed),total_jobs=len(jobs),
        completed_responses=responses,total_responses=total_responses,
        active_cpu_workers=sum(a['device']=='cpu' for a in active_rows),
        active_gpus=sorted(set(a['device'] for a in active_rows if a['device']!='cpu')),
        active=active_rows,recent_trajectories_per_minute=traj_rate,recent_responses_per_second=rate,
        elapsed_seconds=clock-started,estimated_remaining_seconds=eta,ETA='calibrating' if eta is None else f'{eta/60:.1f} minutes',
        estimated_finish_utc=finish.isoformat() if finish else None,
        estimated_finish_local=finish.astimezone(ZoneInfo('America/New_York')).isoformat() if finish else None,
        eta_method=stratified['method'] if stratified else 'rolling five-minute actual response throughput; varies with remaining methods/history length',
        eta_live_scaling=stratified['live_scaling'] if stratified else {},throughput_only_eta_seconds=throughput_eta,
        failed_attempts=sum(max(0,n-1) for n in attempts.values()),
        failed_jobs=len(list((run/'failed').glob('*.json'))),
        shortlist_miss_rate=misses/audits if audits else None,
        shortlist_audits=audits,shortlist_misses=misses,
        shortlist_max_relative_score_loss=max([d.get('max_relative_score_loss',0) for d in diagnostics],default=0),
        adaptive_Q_by_job={key:p.get('method_state',{}).get('Q',config['Q']) for key,p in progresses.items() if p},
        checkpoint_path=str(run/'checkpoints'),run_directory=str(run))
    atomic(run/'status.json',result)
    with (run/'progress.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    print(f"{result['timestamp']} {result['status']} {config['preset']} jobs={len(completed)}/{len(jobs)} trajectories={done}/{total} responses={responses}/{total_responses} rate={rate:.1f}/s ETA={result['ETA']} failures={result['failed_jobs']}",flush=True)
    return result

def supervise(run,cpu_workers=28,gpu_workers=4):
    config=read(run/'config.json');jobs=read(run/'jobs.json');started=time.time()
    if cpu_workers<1 or gpu_workers<0:
        raise ValueError('cpu_workers must be positive and gpu_workers nonnegative')
    config={**config,'cpu_workers':cpu_workers,'gpu_workers':gpu_workers}
    if gpu_workers:
        import torch
        if torch.cuda.device_count()<gpu_workers:
            raise ValueError('Requested GPU workers exceed visible CUDA devices; use --gpu-workers 0 for CPU')
    for p in ['logs','done','failed','progress','summaries']: (run/p).mkdir(exist_ok=True)
    lock=(run/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic(run/'supervisor.json',dict(pid=os.getpid(),started=now(),cpu_workers=cpu_workers,gpu_workers=gpu_workers))
    attempts=read(run/'attempts.json',{}); active={};samples=deque();last_status=0.;last_plot=0.;plotter=None
    slots=['cpu']*cpu_workers+[f'cuda:{i}' for i in range(gpu_workers)]
    while True:
        for slot,(p,j,device,log) in list(active.items()):
            if p.poll() is None:continue
            log.close();del active[slot]
            progress=read(run/'progress'/f"{j['id']}.json",{})
            if p.returncode==0 and progress.get('N',0)>=config['N_run']:
                atomic(run/'done'/f"{j['id']}.json",dict(completed_at=now(),device=device))
            elif not (run/'STOP_REQUESTED').exists():
                with (run/'failures.jsonl').open('a') as f:
                    f.write(json.dumps(dict(timestamp=now(),job=j['id'],returncode=p.returncode,attempt=attempts[j['id']]))+'\n')
                if attempts[j['id']]>=3:atomic(run/'failed'/f"{j['id']}.json",dict(returncode=p.returncode,attempts=attempts[j['id']]))
        running={j['id'] for p,j,d,l in active.values()}
        pending=[j for j in jobs if j['id'] not in running and not (run/'done'/f"{j['id']}.json").exists()
                 and not (run/'failed'/f"{j['id']}.json").exists()]
        if not (run/'STOP_REQUESTED').exists():
            for slot,device in enumerate(slots):
                if slot in active:continue
                preferred='cpu' if device=='cpu' else 'cuda'
                eligible=[j for j in pending if job_backend(j,config)==preferred]
                if not eligible:continue
                j=eligible[0];pending.remove(j);attempts[j['id']]=attempts.get(j['id'],0)+1
                atomic(run/'attempts.json',attempts)
                log=(run/'logs'/f"{j['id']}.log").open('a')
                p=subprocess.Popen([sys.executable,'-m','src.control','worker','--run-dir',str(run),
                    '--job-id',j['id'],'--device',device],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                active[slot]=(p,j,device,log)
        if time.time()-last_status>=15:
            report=progress_status(run,jobs,active,started,samples,attempts,config);last_status=time.time()
        finished=all((run/'done'/f"{j['id']}.json").exists() for j in jobs)
        stopping=(run/'STOP_REQUESTED').exists()
        if not active and (finished or stopping or not pending):break
        if time.time()-last_plot>300 and list((run/'results').glob('*.npz')) and (plotter is None or plotter.poll() is not None):
            log=(run/'logs'/'plot.log').open('a')
            plotter=subprocess.Popen([sys.executable,'-m','src.plotting','--run-dir',str(run),'--preview'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            log.close();last_plot=time.time()
        time.sleep(2)
    report=progress_status(run,jobs,active,started,samples,attempts,config)
    report['status']='COMPLETE' if finished else ('STOPPED' if stopping else 'FAILED')
    report['stage']='postprocessing' if finished else 'production';atomic(run/'status.json',report)
    if finished:
        if plotter is not None:plotter.wait()
        for module in ['diagnostics','plotting']:
            with (run/'logs'/'postprocessing.log').open('a') as log:
                ret=subprocess.call([sys.executable,'-m',f'src.{module}','--run-dir',str(run)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if ret:report['postprocessing_error']=module;report['status']='POSTPROCESSING_FAILED';break
        report['stage']='finished';report['completed_at']=now();atomic(run/'status.json',report)
    return 0 if report['status'] in ['COMPLETE','STOPPED'] else 1

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','supervise','watch','worker','status','stop','resume'])
    p.add_argument('--run-dir',required=True);p.add_argument('--preset',default='preferred_full_scan');p.add_argument('--N',type=int,default=10000)
    p.add_argument('--benchmark');p.add_argument('--cpu-workers',type=int,default=28);p.add_argument('--gpu-workers',type=int,default=4)
    p.add_argument('--job-id');p.add_argument('--device',default='cpu');a=p.parse_args();run=Path(a.run_dir).resolve()
    if a.command=='init':initialize(run,a.preset,a.N,a.benchmark)
    elif a.command=='status':status(run)
    elif a.command=='stop':(run/'STOP_REQUESTED').write_text(now()+'\n');print('Graceful stop requested; checkpoints at batch boundaries.')
    elif a.command=='resume':(run/'STOP_REQUESTED').unlink(missing_ok=True)
    elif a.command=='worker':worker(run,a.job_id,a.device)
    elif a.command=='supervise':sys.exit(supervise(run,a.cpu_workers,a.gpu_workers))
    elif a.command=='watch':
        lock=(run/'watchdog.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (run/'STOP_REQUESTED').exists():
            ret=subprocess.call([sys.executable,'-m','src.control','supervise','--run-dir',str(run),
                '--cpu-workers',str(a.cpu_workers),'--gpu-workers',str(a.gpu_workers)],cwd=ROOT)
            state=read(run/'status.json',{}).get('status')
            if ret==0 or state in ['COMPLETE','FAILED','POSTPROCESSING_FAILED']:break
            print('Supervisor failed; restarting in 10 seconds',flush=True);time.sleep(10)

if __name__=='__main__':main()
