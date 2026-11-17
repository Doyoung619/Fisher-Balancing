import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from src.estimators import fit_gaussian, fit_bernoulli
from src.simulation import run_job


def test_constrained_gaussian_matches_scipy():
    torch.set_num_threads(1); torch.manual_seed(31)
    X = torch.randn(3, 18, 4, dtype=torch.float64); C = X.transpose(-1,-2) @ X
    b = torch.randn(3, 4, dtype=torch.float64)*90
    theta, H, diag = fit_gaussian(C, b, N=18)
    assert diag['success']
    for r in range(3):
        A, v = H[r].numpy(), b[r].numpy()
        opt = minimize(lambda t:(.5*t@A@t-v@t,A@t-v),np.zeros(4),jac=True,
                       constraints={'type':'ineq','fun':lambda t:16-t@t,'jac':lambda t:-2*t},
                       method='SLSQP',options={'ftol':1e-11,'maxiter':100})
        assert np.linalg.norm(theta[r].numpy()-opt.x) < 1e-6


def test_logistic_full_history_terminal_hessian():
    torch.set_num_threads(1); torch.manual_seed(4)
    X = torch.randn(5, 150, 6, dtype=torch.float64)
    y = (torch.rand(5, 150) < torch.sigmoid(X[:,:,0])).double()
    theta, H, diag = fit_bernoulli(X, y)
    assert diag['success'] and diag['kkt'] <= 1e-8
    p = torch.sigmoid((X*theta[:,None,:]).sum(-1))
    direct = X.transpose(-1,-2) @ ((p*(1-p))[:,:,None]*X) + torch.eye(6)
    assert torch.allclose(H, direct, atol=1e-11, rtol=1e-11)
    e = theta-torch.tensor([1.,0,0,0,0,0]); Q = e.T@e/len(e)
    assert torch.allclose(torch.trace(Q), e.square().sum(-1).mean())
    for r in range(5):
        XX, yy = X[r].numpy(), y[r].numpy()
        def objective(t):
            z=XX@t;p=1/(1+np.exp(-z))
            return np.logaddexp(0,z).sum()-yy@z+.5*t@t,XX.T@(p-yy)+t
        opt=minimize(objective,np.zeros(6),jac=True,method='BFGS',options={'gtol':1e-7})
        assert np.linalg.norm(theta[r].numpy()-opt.x)<1e-6


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_ball_boundary_and_shared_gaussian(device):
    if device.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    C=torch.eye(4,dtype=torch.float64,device=device)[None]*3
    b=torch.tensor([[1.,2.,3.,4.],[30.,20.,10.,15.]],dtype=torch.float64,device=device)
    shared,_,s=fit_gaussian(C,b,N=10)
    separate,_,_=fit_gaussian(C.expand(2,-1,-1),b,N=10)
    assert s['success'] and torch.allclose(shared,separate,atol=1e-12)
    X=torch.ones((2,100,1),dtype=torch.float64,device=device)
    theta,_,diag=fit_bernoulli(X,torch.ones((2,100),dtype=torch.float64,device=device),radius=2.)
    assert diag['success'] and diag['boundary']==2
    assert torch.allclose(theta,torch.ones_like(theta)*2,atol=1e-10)


def _config():
    return dict(N_run=40, K=64, Q=16, conditions=[dict(id='central',d=5,L=1.,m=5,kappa=1.5)],
                checkpoint_interval_seconds=100,progress_interval_seconds=100)


@pytest.mark.parametrize('family',['gaussian','bernoulli'])
@pytest.mark.parametrize('method',['FB','AOpt','DOpt','EGreedy','UCB','TS','Random'])
@pytest.mark.parametrize('full_scan', [False, True])
def test_small_job(tmp_path,family,method,full_scan):
    job=dict(id='test',family=family,condition='central',method=method,replicates=4)
    config=_config();config['full_scan']=full_scan
    if full_scan: config['Q']=config['K']
    result=run_job(job,config,tmp_path)
    assert result['status']=='complete' and result['N']==40
    z=np.load(tmp_path/'results/test.npz')
    assert z['error'].shape[-2:]==(4,5)
    assert np.isfinite(z['plugin_min']).all()
    assert np.max(np.linalg.norm(z['theta'],axis=-1))<=4+1e-8


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_exact_resume(tmp_path, monkeypatch, device):
    if device.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    import src.oracles as oracles
    job=dict(id='test',family='bernoulli',condition='central',method='TS',replicates=4)
    config=_config();config['full_scan']=True;config['Q']=config['K']
    expected=tmp_path/'expected';resumed=tmp_path/'resumed'
    run_job(job,config,expected,device); original=oracles.select_batch
    def wrapped(*args,**kwargs):
        answer=original(*args,**kwargs)
        if args[6]==4: (resumed/'STOP_REQUESTED').touch()
        return answer
    monkeypatch.setattr(oracles,'select_batch',wrapped)
    partial=run_job(job,config,resumed,device)
    assert partial['N']==20
    (resumed/'STOP_REQUESTED').unlink()
    monkeypatch.setattr(oracles,'select_batch',original)
    run_job(job,config,resumed,device)
    a=np.load(expected/'results/test.npz'); b=np.load(resumed/'results/test.npz')
    for key in ('N','error','Vhat','Vstar'): np.testing.assert_array_equal(a[key],b[key])
