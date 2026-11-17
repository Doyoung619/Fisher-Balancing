"""Range anchor, soft selection, endpoint equivalence, leakage and reproducibility."""
import json
import numpy as np
import pytest
from scipy.linalg import expm
from real_response.common import ROOT
from real_response.selectors import select_batch,fb_opt2_scale,fb_opt2_temperature,curvature


@pytest.mark.parametrize('family',['gaussian','bernoulli'])
@pytest.mark.parametrize('m',[1,4])
def test_exact_range_anchor_and_soft_acquisition(family,m):
    rng=np.random.default_rng(426);X=rng.normal(size=(45,5))/4;theta=rng.normal(size=5)
    old=np.arange(8);cand=np.arange(8,40);Tmax=17
    config={'fb_opt2_Tmax':Tmax,'dataset':'unit','seed':0}
    scale=fb_opt2_scale(family,5,m,Tmax,config)
    old_ids,old_a=select_batch('fb_range',X,old,cand,theta,family,m,1.,np.random.default_rng(51),config,batch_index=Tmax)
    assert scale['tau_range']==old_a['slot_trace'][0]['temperature']
    assert scale['tau0_opt2']==scale['tau_range']/(m*Tmax)**.75
    temperatures=[fb_opt2_temperature(scale,t) for t in range(1,Tmax+1)]
    assert np.all(np.diff(temperatures)>0) and temperatures[-1]==scale['tau_range']
    np.testing.assert_allclose(temperatures,[scale['tau0_opt2']*(m*t)**.75 for t in range(1,Tmax+1)],rtol=1e-15)
    last,a=select_batch('fb_opt2',X,old,cand,theta,family,m,1.,np.random.default_rng(51),config,batch_index=Tmax)
    np.testing.assert_array_equal(last,old_ids)
    assert a['slot_trace']==old_a['slot_trace']
    first,a=select_batch('fb_opt2',X,old,cand,theta,family,m,1.,np.random.default_rng(51),config)
    W=np.eye(5)+(X[old].T*curvature(X[old]@theta,family))@X[old]
    remaining=list(cand);tau=temperatures[0]
    for chosen,row in zip(first,a['slot_trace']):
        P=expm(-(W-np.linalg.eigvalsh(W)[0]*np.eye(5))/tau);P/=np.trace(P)
        scores=curvature(X[remaining]@theta,family)*np.einsum('ij,ij->i',X[remaining]@P,X[remaining])
        assert chosen==remaining[int(np.argmax(scores))]
        assert row['score']==pytest.approx(scores.max(),rel=1e-10)
        assert row['spectral_entropy']>0 and row['temperature']==tau
        remaining.remove(chosen);W+=curvature(X[chosen:chosen+1]@theta,family)[0]*np.outer(X[chosen],X[chosen])
    other,b=select_batch('fb_opt2',X,old,cand,theta,family,m,1.,np.random.default_rng(51),
        {**config,'theta_ref':np.ones(5)*123,'future_labels':np.zeros(45)})
    np.testing.assert_array_equal(first,other)
    _,b=select_batch('fb_opt2',X,np.r_[old,44],cand,theta,family,m,1.,np.random.default_rng(51),config)
    assert b['temperature']==a['temperature'] # n0 never affects the schedule.


@pytest.mark.parametrize('dataset',['movielens','obd'])
@pytest.mark.requires_data
def test_real_reference_mutation_and_repeat(tmp_path,monkeypatch,dataset):
    import yaml
    from real_response import runner
    c=yaml.safe_load((ROOT/'configs/fb_opt2_m20.yaml').read_text());task=(dataset,5,0,'fb_opt2')
    for name in ['first','repeat']:
        assert runner.run_one(task,c,tmp_path/name,round_limit=2)['state']=='CHECKPOINTED'
    original=runner.references
    def modified(*args,**kwargs):
        ref={k:v.copy() for k,v in original(*args,**kwargs).items()};ref['theta_full']+=.5;return ref
    monkeypatch.setattr(runner,'references',modified)
    assert runner.run_one(task,c,tmp_path/'changed_reference',round_limit=2)['state']=='CHECKPOINTED'
    z=[np.load(runner.run_path(tmp_path/name,task)/'checkpoints/002.npz') for name in ['first','repeat','changed_reference']]
    for other in z[1:]:
        np.testing.assert_array_equal(z[0]['selected_ids'],other['selected_ids'])
        np.testing.assert_array_equal(z[0]['theta_hat'],other['theta_hat'])
    assert not np.array_equal(z[0]['error_ref'],z[2]['error_ref'])


def test_unsupported_configuration():
    with pytest.raises(ValueError):fb_opt2_scale('poisson',5,20,10)
    with pytest.raises(ValueError):fb_opt2_scale('gaussian',5,20,0)
