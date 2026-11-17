import copy,json
from pathlib import Path
import numpy as np
import pytest,yaml
from real_response.common import ROOT,scientific_config,digest
from real_response.runner import run_one,checkpoint_files,run_path,candidates_for,prepared
from real_response.selectors import select_batch,shared_initial_ids

@pytest.mark.parametrize('method',['fb_sample','bait_glm','e_sdp','badge'])
@pytest.mark.requires_data
def test_resume_same_revealed_rows_and_matrices(tmp_path,method):
    c=yaml.safe_load((ROOT/'configs/smoke.yaml').read_text());task=('obd',5,0,method)
    a=tmp_path/'full';b=tmp_path/'resumed'
    r=run_one(task,c,a);assert r['state']=='EVALUATION_COMPLETE',r
    first=run_one(task,c,b,round_limit=1);assert first['completed_rounds']==1,first
    r=run_one(task,c,b);assert r['state']=='EVALUATION_COMPLETE',r
    pa,pb=run_path(a,task),run_path(b,task)
    assert np.array_equal(np.load(pa/'selected_ids.npy'),np.load(pb/'selected_ids.npy'))
    assert np.allclose(np.load(pa/'theta_checkpoints.npy'),np.load(pb/'theta_checkpoints.npy'),atol=1e-12)
    za,zb=np.load(pa/'information_matrices.npz'),np.load(pb/'information_matrices.npz')
    for key in za.files:assert np.allclose(za[key],zb[key],atol=1e-12)
    assert run_one(task,c,b)['reused']


@pytest.mark.requires_data
def test_corrupt_checkpoint_tail_recovers(tmp_path):
    c=yaml.safe_load((ROOT/'configs/smoke.yaml').read_text());task=('movielens',5,0,'random')
    run_one(task,c,tmp_path,round_limit=1);p=run_path(tmp_path,task)
    (p/'checkpoints/001.npz').write_bytes(b'incomplete write')
    assert len(checkpoint_files(p))==1
    r=run_one(task,c,tmp_path);assert r['state']=='EVALUATION_COMPLETE',r
    assert len(checkpoint_files(p))==3


@pytest.mark.requires_data
def test_no_hidden_labels_or_evaluator_passed_to_selector():
    import inspect
    args=inspect.signature(select_batch).parameters
    assert not any(x in args for x in ['y_pool','theta_full','y_test','evaluator'])
    X,y,_,_=prepared('obd',5);ids=np.arange(80);cand=np.arange(80,208);theta=np.zeros(5)
    before=y.copy();a,_=select_batch('fb_sample',X,ids,cand,theta,'bernoulli',20,1,np.random.default_rng(0),{})
    hidden=y.copy();hidden[80:]=1-hidden[80:]
    b,_=select_batch('fb_sample',X,ids,cand,theta,'bernoulli',20,1,np.random.default_rng(0),{})
    assert np.array_equal(a,b) and np.array_equal(y,before)


def test_candidates_method_independent_and_common_rows_priority():
    a=candidates_for('obd',0,1,1000,np.arange(80),200)
    b=candidates_for('obd',0,1,1000,np.arange(90),200)
    common=set(a)&set(b)
    assert [i for i in a if i in common]==[i for i in b if i in common]
    assert not set(a)&set(range(80))


def test_scientific_hash_excludes_device_and_requested_seed_count():
    c=yaml.safe_load((ROOT/'configs/server_main.yaml').read_text());d=copy.deepcopy(c)
    d['run_seeds']=[0,1];d['runtime']['allowed_gpu_ids']=[2];d['profile']='server_preview'
    assert digest(scientific_config(c))==digest(scientific_config(d))
    d['fb_tau0']=.03
    assert digest(scientific_config(c))!=digest(scientific_config(d))
