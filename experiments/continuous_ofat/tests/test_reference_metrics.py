import json
import numpy as np
from scipy.special import expit
from src.reference import compute_reference,build_references
from src.metrics import compute_metrics,mc_directional_stability
from src.common import resolved_config


def test_independent_full_ball_reference_and_unique_matrix():
    rng=np.random.default_rng(4319)
    for d in [5,10,20]:
        u=rng.normal(size=d);u/=np.linalg.norm(u)
        for family in ['gaussian','bernoulli']:
            for L in [.5,1.,2.]:
                ref=compute_reference(family,d,L,1.5,u)
                gamma=ref['gamma_star']
                assert ref['gamma_lower']<=gamma*(1+1e-15)
                assert ref['gamma_upper']>=gamma*(1-1e-15)
                assert (ref['gamma_upper']-ref['gamma_lower'])/gamma<1e-7
                assert ref['uniqueness_status']=='analytically_established'
                P=np.array(ref['dual_P']);M=np.array(ref['M_star'])
                assert np.linalg.eigvalsh(P)[0]>.03
                assert np.allclose(M,gamma*np.eye(d),atol=1e-14)
                assert np.isclose(np.trace(np.linalg.inv(M)),ref['trace_inv_star'])
                X=np.array(ref['support']);p=np.array(ref['weights'])
                assert np.all(np.linalg.norm(X,axis=1)<=L+1e-12)
                assert np.all(p>=0) and abs(p.sum()-1)<1e-12
                w=np.ones(len(X)) if family=='gaussian' else expit(X@(1.5/L*u))*expit(-X@(1.5/L*u))
                assert np.allclose(X.T@((w*p)[:,None]*X),M)
                if family=='bernoulli':
                    lp=ref['independent_adaptive_lp']
                    assert lp['lower']<=ref['gamma_upper']*(1+1e-10)
                    assert (ref['gamma_upper']-lp['lower'])/gamma<1e-7
                    # Independent dense global scalar audit of the analytic dual.
                    t=np.linspace(-1,1,200001)
                    q=ref['dual_q']
                    values=L**2*expit(1.5*t)*expit(-1.5*t)*(q*t*t+(1-q)*(1-t*t)/(d-1))
                    assert values.max()<=ref['gamma_upper']+1e-14
                    x=rng.normal(size=(2000,d));x*=L/np.linalg.norm(x,axis=1)[:,None]
                    xx=np.einsum('ni,ij,nj->n',x,P,x)
                    ww=expit(x@(1.5/L*u))*expit(-x@(1.5/L*u))
                    assert np.max(ww*xx)<=ref['gamma_upper']+1e-13


def test_reference_caching_reuses_center_across_batch_conditions():
    config=resolved_config()
    refs=build_references(config)
    assert len(refs)==14
    for family in config['families']:
        for condition in ['batch5','batch20']:
            for field in ['gamma_star','gamma_lower','gamma_upper','M_star','trace_inv_star']:
                assert refs[family+'/center'][field]==refs[family+'/'+condition][field]
        assert np.isclose(refs[family+'/radius05']['gamma_star'],refs[family+'/center']['gamma_star']/4)
        assert np.isclose(refs[family+'/radius2']['gamma_star'],refs[family+'/center']['gamma_star']*4)


def test_uncentered_risk_and_order_of_eigenvalue_operations():
    errors=np.array([[[2.,0.]],[[0.,2.]],[[2.,0.]],[[0.,2.]]])
    V=np.array([[[[2.,0.],[0.,8.]]],[[[8.,0.],[0.,2.]]],[[[2.,0.],[0.,8.]]],[[[8.,0.],[0.,2.]]]])
    result,stability=compute_metrics(errors,V,V,[20],.1,20.,bootstrap=20)
    assert np.array_equal(result['Q_hat'][0],2*np.eye(2))
    assert result['R2_squared'][0]==4
    assert result['Rdir_squared'][0]==2
    assert result['true_information'][0]==.1
    assert result['plugin_information'][0]==.1
    assert result['directional_ratio'][0]==4
    assert result['scaled_R2'][0]==80
    assert result['tracking_op_per_N'][0]==0
    assert result['per_replicate_squared_error'].mean()==np.trace(result['Q_hat'][0])
    assert result['Rdir_squared'][0]!=np.mean(np.sum(errors**2,axis=-1))
    assert np.linalg.norm(result['mean_error'][0])>0
    assert not np.allclose(result['Q_hat'][0],np.cov(errors[:,0,:],rowvar=False,bias=True))
    assert 'not guaranteed' in stability['limitation']


def test_stability_recomputes_covariance_without_centering():
    errors=np.tile([2.,1.],(32,1))
    data=mc_directional_stability(errors,.5,10,bootstrap=20)
    assert all(np.isclose(row['prefix_ratio'],25) for row in data['blocks'])
    assert np.allclose(data['descriptive_bootstrap_quantiles_025_975'],[25,25])


def test_plotter_only_uses_complete_consistent_repetition_groups(tmp_path):
    from src.plotting import collect_groups
    (tmp_path/'trajectories').mkdir()
    config=resolved_config(True)
    config['preview_N']=20
    for method,end in [('Ours',20),('Random',0)]:
        N=np.array([0,20]) if end else np.array([0])
        np.savez(tmp_path/'trajectories'/f'gaussian__center__{method}__r00000.npz',N=N,
                 error=np.zeros((len(N),10)),Vstar=np.tile(np.eye(10),(len(N),1,1)),Vhat=np.tile(np.eye(10),(len(N),1,1)))
    groups,files,skipped=collect_groups(tmp_path,config,True)
    assert len(groups)==1 and ('gaussian','center','Ours') in groups
    assert not skipped and len(files)==1
