import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import signal
import time

import numpy as np
import pytest

from src.checkpoint import atomic_json, atomic_pickle, file_lock, load_json, load_pickle
from src.supervisor import checkpoint_progress, estimate_eta, integrate_curve, ordered_jobs, physical_jobs


def _lock_probe(path, pipe):
    try:
        with file_lock(path):
            pipe.send('acquired')
    except RuntimeError:
        pipe.send('locked')


def test_atomic_checkpoint_preserves_previous_value_on_failed_serialization(tmp_path):
    path=tmp_path/'state.json'
    atomic_json(path,{'N':20})
    with pytest.raises(ValueError):
        atomic_json(path,{'N':float('nan')})
    assert load_json(path)=={'N':20}
    assert not list(tmp_path.glob('*.tmp'))
    atomic_pickle(tmp_path/'state.pkl',{'N':40,'rng':(1,2,3)})
    assert load_pickle(tmp_path/'state.pkl')=={'N':40,'rng':(1,2,3)}


def test_run_lock_blocks_duplicate_supervisor(tmp_path):
    ctx=multiprocessing.get_context('spawn')
    recv,send=ctx.Pipe(duplex=False)
    with file_lock(tmp_path/'run.lock'):
        process=ctx.Process(target=_lock_probe,args=(str(tmp_path/'run.lock'),send))
        process.start()
        assert recv.recv()=='locked'
        process.join(timeout=10)
        assert process.exitcode==0
    with file_lock(tmp_path/'run.lock'):
        pass


def test_alias_progress_is_shared_and_resume_prefers_latest_durable_state(tmp_path):
    jobs=[{'id':'master'},{'id':'alias','alias_of':'master'}]
    atomic_json(tmp_path/'checkpoints'/'master.status.json',{'N':400})
    atomic_json(tmp_path/'task_results'/'master.json',{'N':200})
    assert len(physical_jobs(jobs))==1
    assert checkpoint_progress(tmp_path,jobs)=={'master':400,'alias':400}


def test_balanced_waves_and_exact_queue_alias_counts():
    from src.common import resolved_config, jobs as make_jobs
    config=resolved_config()
    jobs=make_jobs(config)
    if isinstance(jobs,dict):
        jobs=jobs['jobs']
    ordered=ordered_jobs(jobs,config)
    assert len(jobs)==28672
    assert len(ordered)==22528
    assert all(task['replicate']==0 for task in ordered[:77])
    assert {task['method'] for task in ordered[:77]}=={'Ours','A-opt','D-opt','E-greedy','UCB','TS','Random'}


def test_eta_integrates_actual_early_and_late_history_and_capacity():
    config={'N_run':20000,'preview_replicates':32,'conditions':[{'id':'center','d':10,'L':1.,'m':10}]}
    task={'id':'one','family':'bernoulli','condition':'center','method':'Ours','replicate':0}
    rows=[dict(family='bernoulli',method='Ours',d=10,L=1.,m=10,history_N=N,batch_seconds=s) for N,s in [(0,1.),(20000,3.)]]
    assert integrate_curve([(0,1.),(20000,3.)],0,20000,10)==4000
    assert integrate_curve([(0,1.),(20000,3.)],10000,20000,10)==2500
    duplicate={**task,'id':'two'}
    eta=estimate_eta([task,duplicate],config,{},rows,2,20000)
    assert eta['seconds']==4000
    assert estimate_eta([task],config,{},rows[:1],2,20000)['seconds'] is None
    assert estimate_eta([task],config,{'one':20000},[],2,20000)['seconds']==0


def _simulation_fixture(run,N=80):
    from src.common import resolved_config,json_hash,code_hash
    config=resolved_config(True)
    config.update(N_run=N,checkpoint_counts=list(range(0,N+1,20)))
    run.mkdir()
    for name in ['checkpoints','trajectories','summaries','workers','task_results']:
        (run/name).mkdir()
    atomic_json(run/'config.json',config)
    atomic_json(run/'manifest.json',{'config_hash':json_hash(config),'code_hash':code_hash()})
    return [{'id':f'bernoulli__center__{method}__r00000','family':'bernoulli','condition':'center','method':method,'replicate':0} for method in ['TS','Random']]


def _assert_same_trajectory(first,second,task,tolerance=1e-8):
    a=load_pickle(first/'checkpoints'/f"{task['id']}.pkl")
    b=load_pickle(second/'checkpoints'/f"{task['id']}.pkl")
    for key in ['X','y','theta','Vstar','Vhat']:
        discrepancy=np.linalg.norm(a[key]-b[key])/max(1.,np.linalg.norm(a[key]))
        assert discrepancy<=tolerance,(key,discrepancy)
    assert a['N']==b['N']
    assert a['rng_states']==b['rng_states']
    assert a['response_rng_state']==b['response_rng_state']


def test_real_spawn_worker_count_and_fresh_process_resume(tmp_path,monkeypatch):
    """Independent processes restore all policy and response RNG states."""
    from src.supervisor import execute_task,worker_init
    for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        monkeypatch.setenv(variable,'1')
    one,two=tmp_path/'one',tmp_path/'two'
    tasks=_simulation_fixture(one)
    _simulation_fixture(two)
    ctx=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=1,mp_context=ctx,initializer=worker_init) as pool:
        for result in [pool.submit(execute_task,str(one),task,80) for task in tasks]:
            assert result.result(timeout=180)['N']==80
    with ProcessPoolExecutor(max_workers=2,mp_context=ctx,initializer=worker_init) as pool:
        for result in [pool.submit(execute_task,str(two),task,40) for task in tasks]:
            assert result.result(timeout=180)['N']==40
    # A fresh process pool forces estimator and RNG restoration from disk.
    with ProcessPoolExecutor(max_workers=2,mp_context=ctx,initializer=worker_init) as pool:
        for result in [pool.submit(execute_task,str(two),task,80) for task in reversed(tasks)]:
            assert result.result(timeout=180)['N']==80
    for task in tasks:
        _assert_same_trajectory(one,two,task)


def test_sigterm_checkpoints_completed_batch_and_resumes(tmp_path,monkeypatch):
    from src.supervisor import execute_task,worker_init
    for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        monkeypatch.setenv(variable,'1')
    straight,interrupted=tmp_path/'straight',tmp_path/'interrupted'
    horizon=2000  # Allow the parent to deliver SIGTERM before this fast policy finishes.
    tasks=_simulation_fixture(straight,N=horizon)
    _simulation_fixture(interrupted,N=horizon)
    task=tasks[0]
    ctx=multiprocessing.get_context('spawn')
    child=ctx.Process(target=execute_task,args=(str(interrupted),task,horizon))
    child.start()
    status=interrupted/'checkpoints'/f"{task['id']}.status.json"
    deadline=time.monotonic()+90
    while time.monotonic()<deadline and child.is_alive():
        if load_json(status,{}).get('N',0)>=20:
            os.kill(child.pid,signal.SIGTERM)
            break
        time.sleep(.01)
    else:
        if child.is_alive():
            child.terminate()
        child.join(timeout=15)
        pytest.fail('Child did not expose a completed batch checkpoint before exit/deadline')
    child.join(timeout=90)
    assert child.exitcode==0
    prefix=load_pickle(interrupted/'checkpoints'/f"{task['id']}.pkl")
    assert 20<=prefix['N']<horizon and prefix['N']%10==0
    with ProcessPoolExecutor(max_workers=2,mp_context=ctx,initializer=worker_init) as pool:
        results=[pool.submit(execute_task,str(run),task,horizon) for run in [straight,interrupted]]
        assert all(result.result(timeout=180)['N']==horizon for result in results)
    _assert_same_trajectory(straight,interrupted,task)


def test_eta_live_correction_remains_method_stratified():
    from src.supervisor import timing_stratum
    config={'N_run':20000,'conditions':[{'id':'center','d':10,'L':1.,'m':10}]}
    a={'id':'one','family':'bernoulli','condition':'center','method':'Ours','replicate':0}
    b={**a,'id':'two','method':'E-relax'}
    rows=[dict(family='bernoulli',method=method,d=10,L=1.,m=10,history_N=n,batch_seconds=1) for method in ['Ours','E-relax'] for n in [0,20000]]
    ratios={timing_stratum(a,config):[.5]*10,timing_stratum(b,config):[3.]*10}
    estimate=estimate_eta([a,b],config,{},rows,1,20000,ratios=ratios)
    assert estimate['seconds']==7000
    assert estimate['empirical_range_seconds']==[7000,7000]


def test_eta_respects_preview_barrier_and_preserves_unknown_strata():
    config={'N_run':20000,'preview_N':2000,'preview_replicates':32,'conditions':[{'id':'center','d':10,'L':1.,'m':10}]}
    a={'id':'one','family':'bernoulli','condition':'center','method':'Ours','replicate':0}
    b={**a,'id':'two','method':'E-relax'}
    rows=[]
    for method,times in [('Ours',[100,100,.1,.1]),('E-relax',[.1,.1,100,100])]:
        rows.extend(dict(family='bernoulli',method=method,d=10,L=1.,m=10,history_N=n,batch_seconds=t) for n,t in zip([0,2000,2020,20000],times))
    estimate=estimate_eta([a,b],config,{},rows,2,20000)
    flat=estimate_eta([a,b],config,{},rows,2,20000,stage_aware=False)
    assert estimate['seconds']==pytest.approx(199900.1)
    assert flat['seconds']==pytest.approx(179920.1)
    assert estimate['preview_barrier']['preview_seconds']==20000
    assert estimate['preview_barrier']['production_seconds']==pytest.approx(179900.1)
    missing=estimate_eta([a,b],config,{},rows[:4],2,20000)
    assert missing['seconds'] is None and missing['unprofiled_tasks']==1


def test_stale_heartbeat_is_explicit_but_live_slow_worker_is_retained(tmp_path):
    from src.supervisor import current_workers
    atomic_json(tmp_path/'workers'/f'{os.getpid()}.json',{'pid':os.getpid(),'task_id':'slow','N':20,'updated_at':time.time()-100})
    workers=current_workers(tmp_path,{'slow'})
    assert len(workers)==1 and workers[0]['process_alive']
    assert workers[0]['heartbeat_stale'] and workers[0]['heartbeat_age_seconds']>=100
    assert current_workers(tmp_path,{'different'})==[]


def test_completed_active_future_does_not_make_negative_pending(tmp_path):
    from src.supervisor import write_status
    (tmp_path/'logs').mkdir()
    config={'N_run':20,'preview_N':20,'preview_replicates':1,'conditions':[{'id':'center','d':10,'L':1.,'m':10}]}
    task={'id':'one','family':'bernoulli','condition':'center','method':'Random','replicate':0}
    status=write_status(tmp_path,[task],config,'production','RUNNING',{'one'},{},time.time(),1,progress_cache={'one':20})
    assert status['logical']['completed']==1
    assert status['logical']['running']==0
    assert status['logical']['pending']==0
