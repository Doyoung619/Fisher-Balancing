"""One isolated trajectory, checkpointed at completed feedback boundaries."""
import json
import os
import pickle
import signal
import time
from pathlib import Path
import numpy as np
from .common import policy_rngs,noise_rng,task_id,GAUSSIAN_IDENTICAL,json_hash
from .models import responses,information,weight
from .estimators import fit_gaussian,fit_bernoulli
from .checkpoint import atomic_pickle,atomic_json,load_pickle

_STOP=False


def _signal_stop(signum,frame):
    global _STOP
    _STOP=True


def _npz_atomic(path, **arrays):
    path=Path(path);tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with open(tmp,'wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def _initial(config,task,condition):
    d=condition['d'];N=config['N_run'];rho=config['rho']
    rngs=policy_rngs(config,task)
    response_rng=noise_rng(config,task,d)
    # Innovations indexed solely by family/d/rep/global observation index.
    # Saved generator is advanced only when labels are revealed, below.
    return dict(N=0,batch_index=0,theta=np.zeros(d),Vhat=rho*np.eye(d),Vstar=rho*np.eye(d),
                C=np.zeros((d,d)),b=np.zeros(d),X=np.empty((N,d)),y=np.empty(N),
                rng_states={k:v.bit_generator.state for k,v in rngs.items()},response_rng_state=response_rng.bit_generator.state,
                method_state={},records=[],batch_diagnostics=[],tau=[],active_seconds=0.,
                config_hash=json_hash(config),code_hash=None,task=task)


def _record(state,theta_star):
    if state['records'] and state['records'][-1]['N']==state['N']:
        return
    state['records'].append(dict(N=state['N'],theta=state['theta'].copy(),error=state['theta']-theta_star,
                                 Vstar=state['Vstar'].copy(),Vhat=state['Vhat'].copy()))


def _save(run,task,state,rngs,response_rng,manifest,config):
    if state['N']==config['N_run']:
        found={r['N'] for r in state['records']}
        if not set(config['checkpoint_counts']).issubset(found):
            raise RuntimeError('Missing requested checkpoints; completion not committed')
    state['rng_states']={k:v.bit_generator.state for k,v in rngs.items()}
    state['response_rng_state']=response_rng.bit_generator.state
    state['code_hash']=manifest['code_hash']
    # Do not serialize unwritten np.empty tail; no changing accumulated history.
    payload=dict(state,X=state['X'][:state['N']].copy(),y=state['y'][:state['N']].copy())
    atomic_pickle(run/'checkpoints'/f"{task['id']}.pkl",payload)
    if state['records']:
        arrays={key:np.asarray([r[key] for r in state['records']]) for key in ['N','theta','error','Vstar','Vhat']}
        _npz_atomic(run/'trajectories'/f"{task['id']}.npz",**arrays,tau=np.asarray(state['tau']))
        if task['family']=='gaussian' and task['method']=='Ours':
            for method in GAUSSIAN_IDENTICAL[1:]:
                alias=task_id('gaussian',task['condition'],method,task['replicate'])
                dest=run/'trajectories'/f'{alias}.npz'
                temp=dest.with_name(dest.name+f'.tmp.{os.getpid()}')
                if temp.exists():temp.unlink()
                os.link(run/'trajectories'/f"{task['id']}.npz",temp)
                os.replace(temp,dest)
    diag=state['batch_diagnostics']
    summary=dict(task=task,N=state['N'],batches=len(diag),active_seconds=state['active_seconds'],
                 max_kkt=max((d['fit']['kkt'] for d in diag),default=0),
                 boundary_fits=sum(d['fit']['boundary'] for d in diag),
                 failed_fits=sum(not d['fit']['success'] for d in diag),
                 acquisition_seconds=sum(d['acquisition_seconds'] for d in diag),
                 refit_seconds=sum(d['refit_seconds'] for d in diag),
                 diagnostics=diag,gaussian_aliases=GAUSSIAN_IDENTICAL if task['family']=='gaussian' and task['method']=='Ours' else [])
    atomic_json(run/'summaries'/f"{task['id']}.json",summary)
    # Sidecar is the artifact commit marker. A crash after checkpoint but before
    # exports causes resume to replay _save from the new checkpoint, not skip it.
    directory_fd=os.open(run/'trajectories',os.O_DIRECTORY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)
    atomic_json(run/'checkpoints'/f"{task['id']}.status.json",dict(N=state['N'],batch_index=state['batch_index'],updated_at=time.time(),active_seconds=state['active_seconds']))


def run_task(run_dir,task,target_N):
    from .methods import select_batch
    global _STOP
    _STOP=False
    signal.signal(signal.SIGTERM,_signal_stop)
    signal.signal(signal.SIGINT,_signal_stop)
    run=Path(run_dir);config=json.loads((run/'config.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    condition=next(c for c in config['conditions'] if c['id']==task['condition']);d=condition['d'];m=condition['m'];rho=config['rho']
    acquisition_config={k:v for k,v in config.items() if k not in ['directions','conditions','target_norm_rule','fixed_signal_strength']}
    theta_star=np.asarray(config['directions'][str(d)])*(condition['kappa']/condition['L'])
    checkpoint=run/'checkpoints'/f"{task['id']}.pkl"
    if checkpoint.exists():
        state=load_pickle(checkpoint)
        if state['config_hash']!=manifest['config_hash'] or state['code_hash']!=manifest['code_hash']:
            raise RuntimeError('Trajectory provenance mismatch')
        X=np.empty((config['N_run'],d));X[:state['N']]=state['X'];state['X']=X
        y=np.empty(config['N_run']);y[:state['N']]=state['y'];state['y']=y
    else:
        state=_initial(config,task,condition)
        _record(state,theta_star)
    rngs=policy_rngs(config,task)
    for k,rng in rngs.items():rng.bit_generator.state=state['rng_states'][k]
    response_rng=noise_rng(config,task,d);response_rng.bit_generator.state=state['response_rng_state']
    start=time.monotonic();start_N=state['N'];last_saved=start;last_heartbeat=0;last_compute=start
    checkset=set(config['checkpoint_counts'])
    while state['N']<target_N:
        if _STOP or (run/'STOP_REQUESTED').exists():break
        n=state['N'];batch_index=state['batch_index']+1
        t0=time.monotonic()
        acquisition_condition={k:v for k,v in condition.items() if k!='kappa'}
        if task['family']=='gaussian' and task['method']=='Ours' and config.get('gaussian_exact_deduplication'):
            from .design_cache import gaussian_design
            from .methods import temperature,METHOD_IDS
            design,key=gaussian_design(d,condition['L'],config['N_run'])
            Xb=design[n:n+m];new_method_state=state['method_state']
            tau,tau0,tau_range=temperature('gaussian',d,condition['L'],m,batch_index,config['N_cal'])
            diag=dict(status='analytic',method_id=METHOD_IDS['Ours'],frozen_estimate=True,batch_index=batch_index,
                      exact_shared_design_cache=list(key),tau=tau,tau0=tau0,tau_range=tau_range)
        else:
            Xb,new_method_state,diag=select_batch(task['method'],task['family'],state['theta'],state['Vhat'],state['C']+rho*np.eye(d),acquisition_condition,batch_index,rngs,state['method_state'],acquisition_config)
        Xb=np.asarray(Xb,dtype=np.float64)
        if Xb.shape!=(m,d) or not np.isfinite(Xb).all() or np.max(np.linalg.norm(Xb,axis=1))>condition['L']+1e-9*max(1,condition['L']):
            raise RuntimeError('Acquisition returned invalid or infeasible queries')
        t1=time.monotonic()
        innovation=response_rng.standard_normal(m) if task['family']=='gaussian' else response_rng.random(m)
        yb=responses(task['family'],Xb,theta_star,innovation)
        state['X'][n:n+m]=Xb;state['y'][n:n+m]=yb
        state['C']+=Xb.T@Xb;state['b']+=Xb.T@yb
        state['Vstar']+=Xb.T@(weight(task['family'],Xb@theta_star)[:,None]*Xb)
        if task['family']=='gaussian':
            theta,Vhat,fit=fit_gaussian(state['C'],state['b'],rho,config['parameter_radius'],n+m)
        else:
            theta,Vhat,fit=fit_bernoulli(state['X'][:n+m],state['y'][:n+m],state['theta'],rho,config['parameter_radius'],config['estimator_normalized_kkt_tolerance'])
        t2=time.monotonic()
        if not fit['success']:
            # Preserve diagnostic failure evidence without accepting a failed estimate.
            atomic_json(run/'summaries'/f"{task['id']}.failure.json",dict(task=task,N=n+m,fit=fit))
            raise RuntimeError(f'Constrained MLE failed KKT at N={n+m}: {fit}')
        state.update(N=n+m,batch_index=batch_index,theta=theta,Vhat=Vhat,method_state=new_method_state)
        if task['method']=='Ours':
            R=condition['L']**2*(1 if task['family']=='gaussian' else .25)
            tau=R*np.sqrt(m/(8*np.log(d)))*((n+m)/config['N_cal'])**.75
            state['tau'].append(tau)
        state['batch_diagnostics'].append(dict(N=n+m,acquisition_seconds=t1-t0,refit_seconds=t2-t1,fit=fit,oracle=diag))
        state['active_seconds']+=time.monotonic()-last_compute
        now=time.monotonic()
        if state['N'] in checkset:_record(state,theta_star)
        if state['N'] in checkset or now-last_saved>=config['checkpoint_interval_seconds'] or state['N']==target_N:
            _save(run,task,state,rngs,response_rng,manifest,config);last_saved=time.monotonic()
        if now-last_heartbeat>=min(30,config['status_interval_seconds']) or state['N']==target_N:
            import resource
            atomic_json(run/'workers'/f'{os.getpid()}.json',dict(pid=os.getpid(),task_id=task['id'],N=state['N'],target_N=target_N,
                        updated_at=time.time(),rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,backend='cpu'))
            last_heartbeat=now
        last_compute=time.monotonic()
    _save(run,task,state,rngs,response_rng,manifest,config)
    if state['N']==config['N_run']:
        required=set(config['checkpoint_counts']);found={r['N'] for r in state['records']}
        if not required.issubset(found):raise RuntimeError('Missing requested checkpoints')
        if not checkpoint.exists():raise RuntimeError('Missing durable trajectory')
    return dict(N=state['N'],elapsed_seconds=time.monotonic()-start,batches=(state['N']-start_N)//m,responses=state['N']-start_N)
