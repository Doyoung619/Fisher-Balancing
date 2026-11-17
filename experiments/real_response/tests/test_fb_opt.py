"""Independent FB-opt scale/Gibbs checks and frozen old-selector regression."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.linalg import expm
from real_response.selectors import fb_opt_scale, select_batch, curvature
from real_response.common import ROOT


@pytest.mark.parametrize('family,wbar', [('gaussian',1.),('bernoulli',.25)])
@pytest.mark.parametrize('m',[1,4])
def test_scale_and_exact_soft_gibbs(family,wbar,m):
    rng=np.random.default_rng(184)
    X=rng.normal(size=(35,5))*2
    X[-1]*=4 # Not in the shortlist: full-pool maximum must still be used.
    original=X.copy();old=np.arange(8);cand=np.arange(8,30);theta=rng.normal(size=5)/4
    Tmax=23;config={'fb_opt_Tmax':Tmax}
    scale=fb_opt_scale(X,family,m,Tmax)
    L=np.linalg.norm(X,axis=1).max()
    assert scale['L']==pytest.approx(L)
    assert scale['B']==pytest.approx(wbar*L**2)
    from math import fsum,log,sqrt
    S_T=fsum((m*t)**(-.75) for t in range(1,Tmax+1))
    expected=wbar*L**2*sqrt(m*S_T/((m*Tmax)**.75*log(5)))
    assert scale['S_T']==pytest.approx(S_T,rel=1e-14)
    assert scale['tau0_opt']==pytest.approx(expected,rel=1e-14)
    assert scale['tau0_opt']<scale['tau0_previous']
    A=(m*Tmax)**.75*log(5);C=(wbar*L**2)**2*m*S_T
    objective=lambda tau:A*tau+C/tau
    assert objective(expected)<objective(expected*.9)
    assert objective(expected)<objective(expected*1.1)
    ids,a=select_batch('fb_opt',X,old,cand,theta,family,m,1.,rng,config)
    W=np.eye(5)+(X[old].T*curvature(X[old]@theta,family))@X[old]
    tau=scale['tau0_opt']*m**.75
    remaining=list(cand)
    for ix,row in zip(ids,a['slot_trace']):
        P=expm(-W/tau);P/=np.trace(P)
        weights=curvature(X[remaining]@theta,family)
        scores=weights*np.einsum('ij,ij->i',X[remaining]@P,X[remaining])
        assert ix==remaining[int(np.argmax(scores))]
        assert row['score']==pytest.approx(scores.max(),rel=1e-11)
        assert row['temperature']==tau
        assert row['spectral_entropy']>0 # m=1 remains soft.
        W+=curvature(X[ix:ix+1]@theta,family)[0]*np.outer(X[ix],X[ix])
        remaining.remove(ix)
    np.testing.assert_allclose(a['W_end'],W)
    np.testing.assert_array_equal(X,original)
    # Same L/tau0 even when the maximum-radius point has already been selected.
    _,b=select_batch('fb_opt',X,np.r_[old,34],cand,theta,family,m,1.,rng,config)
    assert a['fb_opt']==b['fb_opt']
    assert b['temperature']==a['temperature'] # Warm observations never enter N_t.
    _,later=select_batch('fb_opt',X,old,cand,theta,family,m,1.,rng,config,batch_index=2)
    assert later['temperature']==pytest.approx(scale['tau0_opt']*(2*m)**.75)
    assert later['temperature']>tau
    assert len(set(ids))==m and not set(ids)&set(old)
    # Forbidden evaluation metadata cannot affect temperature/acquisition.
    alt,b=select_batch('fb_opt',X,old,cand,theta,family,m,1.,rng,
                      {**config,'theta_ref':np.ones(5)*1000,'future_responses':np.ones(35)*100,
                       'fb_tau0':10000,'fb_range_multiplier':10000})
    np.testing.assert_array_equal(alt,ids)
    assert b['temperature']==tau


@pytest.mark.parametrize('family',['gaussian','bernoulli','poisson'])
@pytest.mark.parametrize('method',['fb_sample','fb_range'])
def test_old_fb_unchanged(family,method):
    path=ROOT/'outputs/fb_opt_20261005/original_source/selectors.py'
    if not path.is_file():
        pytest.skip('Historical selector snapshot is not distributed; current formula tests run independently')
    spec=importlib.util.spec_from_file_location('frozen_selectors',path)
    oldmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmod)
    rng=np.random.default_rng(48);X=rng.normal(size=(40,5))/3;theta=rng.normal(size=5)
    args=(method,X,np.arange(8),np.arange(8,35),theta,family,5,1.)
    before,a=oldmod.select_batch(*args,np.random.default_rng(89))
    after,b=select_batch(*args,np.random.default_rng(89))
    np.testing.assert_array_equal(before,after)
    for k in ['W_start','W_end','slot_trace','selected_weights_raw']:
        assert a[k]==b[k]


def test_unsupported_family():
    with pytest.raises(ValueError):fb_opt_scale(np.eye(3),'poisson',1,10)


def test_bad_horizon_or_dimension():
    with pytest.raises(ValueError,match='d > 1'):fb_opt_scale(np.ones((5,1)),'gaussian',1,10)
    with pytest.raises(ValueError,match='positive integer'):fb_opt_scale(np.eye(3),'gaussian',2,0)
    with pytest.raises(ValueError,match='positive integer'):fb_opt_scale(np.eye(3),'gaussian',1,1.5)
    with pytest.raises(ValueError,match='planned finite horizon'):
        select_batch('fb_opt',np.eye(3),np.array([0]),np.array([1,2]),np.zeros(3),'gaussian',1,1.,np.random.default_rng(3))


@pytest.mark.parametrize('dataset',['movielens','obd'])
@pytest.mark.requires_data
def test_reference_change_cannot_change_selections(tmp_path,monkeypatch,dataset):
    import yaml
    from real_response import runner
    c=yaml.safe_load((ROOT/'configs/fb_opt_m20.yaml').read_text())
    task=(dataset,5,0,'fb_opt')
    a=runner.run_one(task,c,tmp_path/'first',round_limit=2)
    b=runner.run_one(task,c,tmp_path/'repeat',round_limit=2)
    assert a['state']==b['state']=='CHECKPOINTED'
    original=runner.references
    def altered(*args,**kwargs):
        ref={k:v.copy() for k,v in original(*args,**kwargs).items()}
        ref['theta_full']=ref['theta_full']+.5
        return ref
    monkeypatch.setattr(runner,'references',altered)
    changed=runner.run_one(task,c,tmp_path/'changed_ref',round_limit=2)
    assert changed['state']=='CHECKPOINTED',changed
    paths=[runner.run_path(tmp_path/name,task)/'checkpoints/002.npz' for name in ['first','repeat','changed_ref']]
    z=[np.load(p) for p in paths]
    for other in z[1:]:
        np.testing.assert_array_equal(z[0]['selected_ids'],other['selected_ids'])
        np.testing.assert_array_equal(z[0]['theta_hat'],other['theta_hat'])
    assert not np.array_equal(z[0]['error_ref'],z[2]['error_ref'])
