"""Independent objective checks; no archived finite-pool utilities imported."""
import inspect
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit
from src.oracles import (quadratic_trust, logistic_quadratic, curvature, a_opt,
    e_greedy, ucb, rank_one_minimum, audit_quadratic)
from src.methods import select_batch,temperature,e_relax,waterfill


def rngs(seed=41):
    return {k:np.random.default_rng(seed+i) for i,k in enumerate(('solver','tie','rounding','ts','random'))}


def cloud(rng,d,n=12000,L=1.):
    X=rng.normal(size=(n,d));X/=np.linalg.norm(X,axis=1)[:,None]
    # Include interior radii as well as the entire sphere.
    X[n//2:]*=rng.uniform(size=n-n//2)[:,None]**(1/d)
    return L*X


def independent_max(fun,d,rng,nstarts=12):
    best=-np.inf
    for v in cloud(rng,d,nstarts):
        sol=minimize(lambda z:-fun(z),v,method='SLSQP',
            constraints=[{'type':'ineq','fun':lambda z:1-z@z}],
            options={'ftol':2e-13,'maxiter':250})
        z=sol.x/max(1.,np.linalg.norm(sol.x))
        best=max(best,fun(z))
    return best


def test_quadratic_trust_hard_and_interior_cases():
    A=np.diag([1.,3.]);b=np.array([1.,0.])
    y,diag=quadratic_trust(A,b,2.)
    assert diag['hard_case']
    assert np.isclose(np.linalg.norm(y),2.)
    assert np.isclose(y[0],.5,atol=1e-12)
    assert np.linalg.norm((diag['multiplier']*np.eye(2)-A)@y-b)<1e-10
    y,diag=quadratic_trust(-np.eye(3),np.array([.2,0,0]),1.)
    assert np.allclose(y,[.2,0,0])
    assert diag['multiplier']==0


@pytest.mark.parametrize('seed',range(4))
def test_trust_region_global_against_independent_multistart(seed):
    r=np.random.default_rng(seed);d=5;A=r.normal(size=(d,d));A=(A+A.T)/2;b=r.normal(size=d)
    y,_=quadratic_trust(A,b,1.)
    score=y@A@y+2*b@y
    independent=independent_max(lambda z:z@A@z+2*b@z,d,r)
    assert score>=independent-2e-8


@pytest.mark.parametrize('seed',range(4))
def test_logistic_psd_reduction_against_direct_coordinates(seed):
    r=np.random.default_rng(seed+12);d=5;Q=r.normal(size=(d,d));Q=Q.T@Q
    theta=r.normal(size=d);theta*=3/np.linalg.norm(theta)
    x,diag=logistic_quadratic(Q,theta,1.)
    xa,strict=audit_quadratic(Q,theta,1.,r,n_starts=16,grid_size=129)
    assert np.linalg.norm(x)<=1+1e-10
    assert abs(diag['score']-strict['score'])/strict['score']<1e-6
    assert diag['score']<=diag['global_upper_bound']+1e-10
    X=cloud(r,d)
    scores=curvature(theta,X)*np.einsum('ij,jk,ik->i',X,Q,X)
    assert diag['score']>=np.max(scores)-1e-7


def test_a_opt_exact_rational_objective():
    r=np.random.default_rng(77);d=5;W=np.diag([1.,1.6,2.,4.,7.]);theta=r.normal(size=d)
    x,diag=a_opt(W,theta,1.,r,n_random=5)
    inverse=np.linalg.inv(W)
    def f(z):
        w=float(curvature(theta,z))
        return w*(z@inverse@inverse@z)/(1+w*z@inverse@z)
    assert np.isclose(diag['score'],f(x))
    actual=np.trace(inverse)-np.trace(np.linalg.inv(W+curvature(theta,x)*np.outer(x,x)))
    assert abs(actual-f(x))<1e-12
    assert f(x)>=independent_max(f,d,r,20)-1e-8


def test_rank_one_secular_matches_dense_eigenvalues():
    r=np.random.default_rng(53)
    for _ in range(100):
        a=np.sort(r.uniform(.3,6.,size=5));x=r.normal(size=5);w=r.uniform()
        independent=np.linalg.eigvalsh(np.diag(a)+w*np.outer(x,x))[0]
        assert abs(rank_one_minimum(a,x,w)-independent)<1e-10
    assert rank_one_minimum(np.array([1.,1.,2.]),np.ones(3),.25)==1.


def test_e_primary_and_repeated_eigenvalue_secondary():
    r=np.random.default_rng(211);d=5;theta=np.array([.8,-1.,.4,.2,.2])
    W=np.diag([1.,1.,2.,3.,4.])
    x,diag=e_greedy(W,theta,1.,r)
    xd,dd=logistic_quadratic(np.linalg.inv(W),theta,1.)
    assert diag['primary_tie']
    assert abs(np.linalg.eigvalsh(W+curvature(theta,x)*np.outer(x,x))[0]-1)<1e-12
    assert np.allclose(x,xd)
    W=np.diag([1.,1.4,2.,3.,4.])
    x,diag=e_greedy(W,theta,1.,r,n_random=5)
    def f(z):return np.linalg.eigvalsh(W+curvature(theta,z)*np.outer(z,z))[0]
    actual=f(x)
    assert abs(actual-diag['score'])<1e-11
    assert actual>=independent_max(f,d,r,20)-2e-7
    assert diag['secular_discrepancy']<1e-10


def test_ucb_global_reduction():
    r=np.random.default_rng(14);d=5;A=r.normal(size=(d,d));W=np.eye(d)+A.T@A
    theta=r.normal(size=d);beta=3.2
    x,diag=ucb(W,theta,1.,beta);B=np.linalg.inv(W)
    f=lambda z:theta@z+beta*np.sqrt(z@B@z)
    assert diag['status']=='analytic'
    assert f(x)>=independent_max(f,d,r,20)-1e-9


def test_all_gaussian_sequential_designs_identical_and_optimal():
    c=dict(d=5,L=.5,m=10);W=np.eye(5);theta=np.arange(5)/5
    batches=[]
    for method in ('Ours','A-opt','D-opt','E-greedy'):
        X,_,diag=select_batch(method,'gaussian',theta,W,W,c,1,rngs(),{},{})
        batches.append(X)
        assert diag['status']=='analytic'
    for X in batches[1:]:assert np.array_equal(X,batches[0])
    assert np.allclose(batches[0].T@batches[0],.5*np.eye(5))


def test_relax_rounding_does_not_replace_actual_with_expected_information():
    d=20;m=5;W=np.eye(d)
    X,state,diag=select_batch('E-relax','gaussian',np.zeros(d),W,W,dict(d=d,L=1,m=m),1,rngs(),{},{})
    assert X.shape==(m,d)
    assert np.isclose(np.sum(state['weights']),1.)
    assert np.min(state['weights'])>=0
    assert np.isclose(diag['relaxed_objective'],1+m/d)
    assert np.isclose(diag['rounded_objective'],1.)
    assert diag['rounding_loss']>.2


def test_bernoulli_relax_continuous_support_and_feasible_weights():
    d=3;theta=np.array([1.1,.3,-.2]);W=np.diag([1.,1.12,1.23]);m=5
    X,state,diag=e_relax(W,theta,1.,m,np.random.default_rng(12),{}, {'e_relax_max_generations':32})
    assert X.shape==(m,d)
    assert np.max(np.linalg.norm(X,axis=1))<1+1e-9
    assert abs(state['weights'].sum()-1)<1e-12
    assert np.all(state['weights']>=0)
    assert len(state['support'])>d
    G=np.einsum('i,ij,ik->jk',state['weights']*curvature(theta,state['support']),state['support'],state['support'])
    assert abs(np.linalg.eigvalsh(W+m*G)[0]-diag['relaxed_objective'])<1e-10
    assert diag['global_upper_bound']>=diag['relaxed_objective']-1e-10


def test_no_reference_information_in_signature_and_ucb_frozen():
    sig=inspect.signature(select_batch)
    assert 'theta_star' not in sig.parameters
    assert 'reference' not in sig.parameters
    d=5;W=np.eye(d);theta=np.arange(d)/5;c=dict(d=d,L=1.,m=10)
    X,_,_=select_batch('UCB','bernoulli',theta,W,W,c,1,rngs(),{},{})
    assert np.all(X==X[0])


def test_temperature_uses_fixed_calibration_and_no_clipping():
    tau,tau0,tr=temperature('bernoulli',10,1.,10,40)
    assert np.isclose(tau,tr)
    assert temperature('bernoulli',10,1.,10,2000)[0]>tr
    W=np.eye(5);c=dict(d=5,L=1.,m=5)
    a,_,d1=select_batch('Ours','gaussian',np.zeros(5),W,W,c,3,rngs(),{},{'N_run':2000})
    b,_,d2=select_batch('Ours','gaussian',np.zeros(5),W,W,c,3,rngs(),{},{'N_run':20000})
    assert np.array_equal(a,b)
    assert d1['tau']==d2['tau']


def test_random_radius_volume_distribution():
    d=5;W=np.eye(d)
    X,_,_=select_batch('Random','gaussian',np.zeros(d),W,W,dict(d=d,L=2,m=20000),1,rngs(),{},{})
    radial=(np.linalg.norm(X,axis=1)/2)**d
    assert abs(radial.mean()-.5)<.01
    assert radial.min()>=0 and radial.max()<=1


def test_relax_support_weights_stay_aligned_through_actual_400_response_history():
    from src.estimators import fit_bernoulli
    from src.models import responses
    d=10;m=10;theta=np.zeros(d);W=np.eye(d);H=np.eye(d)
    r=np.random.default_rng(20261005);truth=r.normal(size=d);truth*=1.5/np.linalg.norm(truth)
    policy=rngs();state={};history=np.empty((0,d));labels=np.empty(0)
    config={'oracle_audit_interval_batches':200}
    for batch in range(1,41):
        X,state,diag=select_batch('E-relax','bernoulli',theta,W,H,dict(d=d,L=1,m=m),batch,policy,state,config)
        assert len(state['support'])==len(state['weights'])
        y=responses('bernoulli',X,truth,r.uniform(size=m))
        history=np.vstack([history,X]);labels=np.r_[labels,y]
        theta,W,fit=fit_bernoulli(history,labels,theta)
        assert fit['success']
        H+=X.T@X
    assert len(labels)==400


def test_relax_capped_generation_preserves_last_solved_measure():
    d=5;W=np.diag([1.,1.1,1.2,1.3,1.4]);theta=np.arange(1,d+1)/d
    settings={'e_relax_support_capacity':7,'e_relax_max_generations':2,'e_relax_relative_tolerance':1e-12}
    state={}
    for _ in range(4):
        X,state,diag=e_relax(W,theta,1.,5,np.random.default_rng(2),state,settings)
        assert len(state['support'])==len(state['weights'])
        assert np.isclose(state['weights'].sum(),1.)


def test_scalar_near_zero_maximum_regression():
    # A coarse objective-value grid missed a narrow positive-z maximum here;
    # the one-sided envelope derivative must cause continuous refinement.
    r=np.random.default_rng(208);d=5
    A=r.normal(size=(d,d));W=np.eye(d)+A.T@A
    theta=r.normal(size=d);theta*=3.8/np.linalg.norm(theta)
    Q=np.linalg.inv(W)
    x,diag=logistic_quadratic(Q,theta,1.,grid_size=17)
    xa,strict=audit_quadratic(Q,theta,1.,r,n_starts=20,grid_size=129)
    assert abs(diag['score']-strict['score'])/strict['score']<1e-8


@pytest.mark.parametrize('d,extra',[(5,0),(10,8)])
def test_affine_support_exchange_preserves_current_information(d,extra):
    from src.methods import affine_support_exchange
    r=np.random.default_rng(504+d)
    n=d*(d+1)//2+1+extra
    X=cloud(r,d,n);weights=r.uniform(size=n);weights/=weights.sum()
    theta=r.normal(size=d);point=r.normal(size=d);point/=np.linalg.norm(point)
    actual_before=np.einsum('i,ij,ik->jk',weights*curvature(theta,X),X,X)
    Y,p,diagnostic=affine_support_exchange(X,weights,point,theta,1.)
    actual_after=np.einsum('i,ij,ik->jk',p*curvature(theta,Y),Y,Y)
    assert len(Y)==n
    assert np.allclose(Y[-1],point)
    assert np.min(p)>=0 and abs(p.sum()-1)<1e-12
    assert np.linalg.norm(actual_after-actual_before,ord=2)<1e-12
    assert diagnostic<1e-12


def test_affine_exchange_handles_redundant_old_support():
    from src.methods import affine_support_exchange
    d=5;n=d*(d+1)//2+1
    X=np.eye(d)[np.arange(n)%d];weights=np.full(n,1/n)
    theta=np.arange(d)/d;point=np.ones(d)/np.sqrt(d)
    before=np.einsum('i,ij,ik->jk',weights*curvature(theta,X),X,X)
    Y,p,residual=affine_support_exchange(X,weights,point,theta,1.)
    after=np.einsum('i,ij,ik->jk',p*curvature(theta,Y),Y,Y)
    assert len(Y)==n and np.allclose(Y[-1],point)
    assert np.linalg.norm(after-before)<1e-12
