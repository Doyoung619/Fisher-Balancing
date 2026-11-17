import copy
import json
import inspect
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,check_grad
from src.common import resolved_config,validate_config,jobs,policy_rngs,noise_rng,json_hash,code_hash
from src.estimators import quadratic_ball,fit_gaussian,fit_bernoulli,logistic_objective,kkt
from src.models import information,responses


def test_exact_grid_and_independent_streams():
    c=resolved_config();assert validate_config(c)
    js=jobs(c);assert len(js)==28672
    assert len({(t['family'],t['condition'],t['method']) for t in js})==98
    assert sum('alias_of' not in t for t in js)==22528
    assert all(t['method']!='E-relax' for t in js)
    a=dict(family='gaussian',condition='center',method='Random',replicate=8)
    b=dict(a,method='TS',condition='batch5')
    assert np.array_equal(noise_rng(c,a,10).normal(size=40),noise_rng(c,b,10).normal(size=40))
    r=policy_rngs(c,a);r['ts'].normal(size=1000)
    assert np.array_equal(noise_rng(c,a,10).normal(size=40),noise_rng(c,b,10).normal(size=40))
    assert not np.array_equal(noise_rng(c,a,10).normal(size=40),noise_rng(c,dict(a,replicate=9),10).normal(size=40))
    for row in c['conditions']:
        truth=np.array(c['directions'][str(row['d'])])*row['kappa']/row['L']
        assert np.isclose(np.linalg.norm(truth)*row['L'],1.5)


def test_gaussian_constraint_not_radial_clip():
    A=np.diag([1.,5.,40.]);b=np.array([10.,20.,30.]);theta,eta=quadratic_ball(A,b,4.)
    ref=minimize(lambda t:.5*t@A@t-b@t,np.zeros(3),jac=lambda t:A@t-b,method='SLSQP',
                 constraints={'type':'ineq','fun':lambda t:16-t@t,'jac':lambda t:-2*t},options={'ftol':1e-12,'maxiter':500})
    assert np.linalg.norm(theta-ref.x)<1e-6
    assert np.linalg.norm(A@theta-b+eta*theta)<1e-10
    clip=np.linalg.solve(A,b);clip*=4/np.linalg.norm(clip)
    assert .5*theta@A@theta-b@theta < .5*clip@A@clip-b@clip - .1


def test_logistic_derivatives_and_interior_boundary_fits():
    rng=np.random.default_rng(5);X=rng.normal(size=(120,5));y=rng.integers(0,2,120).astype(float);t=rng.normal(size=5)*.3
    f=lambda a:logistic_objective(a,X,y)[0];g=lambda a:logistic_objective(a,X,y)[1]
    assert check_grad(f,g,t)<1e-5
    H=logistic_objective(t,X,y)[2];eps=1e-5
    finite=np.column_stack([(g(t+eps*np.eye(5)[i])-g(t-eps*np.eye(5)[i]))/(2*eps) for i in range(5)])
    assert np.max(np.abs(H-finite))<1e-7
    for ytest,radius in [(y,4.),((X[:,0]>0).astype(float),.2)]:
        theta,H,diag=fit_bernoulli(X,ytest,radius=radius)
        assert diag['success'] and diag['kkt']<=1e-8
        ref=minimize(lambda a:logistic_objective(a,X,ytest)[0],np.zeros(5),jac=lambda a:logistic_objective(a,X,ytest)[1],
                     method='SLSQP',constraints={'type':'ineq','fun':lambda a:radius**2-a@a,'jac':lambda a:-2*a},options={'ftol':1e-12})
        assert np.linalg.norm(theta-ref.x)<2e-6
        assert np.allclose(H,information('bernoulli',X,theta),atol=1e-12)


def make_test_run(tmp_path,method='Random',family='bernoulli',N=80):
    c=resolved_config(True);c.update(N_run=N,checkpoint_counts=list(range(0,N+1,20)))
    run=tmp_path;run.mkdir(exist_ok=True)
    for folder in ['checkpoints','trajectories','summaries','workers']:(run/folder).mkdir(exist_ok=True)
    (run/'config.json').write_text(json.dumps(c))
    (run/'manifest.json').write_text(json.dumps(dict(config_hash=json_hash(c),code_hash=code_hash())))
    task=dict(id=f'{family}__center__{method}__r00000',family=family,condition='center',method=method,replicate=0)
    return run,task


def test_resume_exact_full_history_and_feedback(tmp_path):
    from src.simulation import run_task
    from src.checkpoint import load_pickle
    one,t=make_test_run(tmp_path/'one');two,_=make_test_run(tmp_path/'two')
    run_task(one,t,80);run_task(two,t,40);run_task(two,t,80)
    a=load_pickle(one/'checkpoints'/f"{t['id']}.pkl");b=load_pickle(two/'checkpoints'/f"{t['id']}.pkl")
    for key in ['X','y','theta','Vhat','Vstar']:assert np.array_equal(a[key],b[key])
    assert a['rng_states']==b['rng_states'] and a['response_rng_state']==b['response_rng_state']
    assert np.allclose(a['Vhat'],information('bernoulli',a['X'],a['theta']),atol=1e-12)
    stale=information('bernoulli',a['X'],np.zeros(10));assert np.linalg.norm(a['Vhat']-stale)>1e-4


def test_acquisition_signature_has_no_truth_and_frozen_ucb():
    from src.methods import select_batch
    assert 'theta_star' not in inspect.signature(select_batch).parameters
    c=resolved_config();row=c['conditions'][0];task=dict(family='bernoulli',condition='center',method='UCB',replicate=1)
    a=select_batch('UCB','bernoulli',np.zeros(10),np.eye(10),np.eye(10),row,1,policy_rngs(c,task),{},c)[0]
    c2=copy.deepcopy(c);c2['directions']['10']=(-np.array(c['directions']['10'])).tolist()
    b=select_batch('UCB','bernoulli',np.zeros(10),np.eye(10),np.eye(10),row,1,policy_rngs(c2,task),{},c2)[0]
    assert np.array_equal(a,b) and np.max(np.linalg.norm(a-a[0],axis=1))<1e-10


def test_crash_between_checkpoint_and_exports_recovers(tmp_path,monkeypatch):
    import pytest
    from src import simulation
    run,task=make_test_run(tmp_path/'crash')
    original=simulation._npz_atomic
    def broken(*a,**kw):raise OSError('Injected export failure after durable checkpoint')
    monkeypatch.setattr(simulation,'_npz_atomic',broken)
    with pytest.raises(OSError):simulation.run_task(run,task,80)
    assert (run/'checkpoints'/f"{task['id']}.pkl").exists()
    assert not (run/'checkpoints'/f"{task['id']}.status.json").exists()
    monkeypatch.setattr(simulation,'_npz_atomic',original)
    simulation.run_task(run,task,80)
    assert json.loads((run/'checkpoints'/f"{task['id']}.status.json").read_text())['N']==80
    with np.load(run/'trajectories'/f"{task['id']}.npz") as data:assert data['N'][-1]==80


def test_exact_gaussian_cache_all_rows_methods_and_batch_sizes():
    from src.design_cache import gaussian_design
    from src.methods import select_batch
    c=resolved_config()
    for row in c['conditions']:
        d,m,L=row['d'],row['m'],row['L'];expected,key=gaussian_design(d,L,80)
        for method in ['Ours','A-opt','D-opt','E-greedy']:
            W=np.eye(d);task=dict(family='gaussian',condition=row['id'],method=method,replicate=5)
            for n in range(0,80,m):
                X,_,diag=select_batch(method,'gaussian',np.arange(d)/d,W,W,row,n//m+1,policy_rngs(c,task),{},c)
                assert np.array_equal(X,expected[n:n+m])
                W+=X.T@X
