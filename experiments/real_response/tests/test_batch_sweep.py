import copy,json,shutil
import numpy as np
import pytest,yaml
from real_response.common import ROOT,scientific_config,tasks_for,run_path
from real_response.runner import run_one,references
from real_response import reporting

@pytest.mark.parametrize('profile,count,rounds',[('batch_m40',600,10),('batch_m80',600,5),('obd_batch_m20',330,200),('obd_batch_m40',330,100),('obd_batch_m80',330,50)])
def test_batch_configs_change_only_scope_and_batch_size(profile,count,rounds):
    base=yaml.safe_load((ROOT/'configs/server_main.yaml').read_text());c=yaml.safe_load((ROOT/f'configs/{profile}.yaml').read_text())
    assert len(tasks_for(c))==count
    assert c['run_seeds']==list(range(10)) and c['dimensions']==[5,10,20]
    for ds,dc in c['datasets'].items():
        assert dc['additional_labels']//dc['batch_size']==rounds
        assert {k:v for k,v in dc.items() if k!='batch_size'}=={k:v for k,v in base['datasets'][ds].items() if k!='batch_size'}
    c['datasets']=base['datasets']
    assert scientific_config(c)==scientific_config(base)

@pytest.mark.parametrize('m',[40,80])
@pytest.mark.requires_data
def test_changed_batch_exact_budget_and_isolated_reference_cache(tmp_path,m):
    c=yaml.safe_load((ROOT/f'configs/batch_m{m}.yaml').read_text());refdir=tmp_path/'cache';refdir.mkdir()
    for ext in ['npz','json']:shutil.copy2(ROOT/f'data/references/movielens_d5.{ext}',refdir/f'movielens_d5.{ext}')
    c['reference_cache_dir']=str(refdir);c['datasets']['movielens']['additional_labels']=80
    task=('movielens',5,0,'fb_range');result=run_one(task,c,tmp_path/'out')
    assert result['state']=='EVALUATION_COMPLETE',result
    p=run_path(tmp_path/'out',task);frame=reporting.read_csv_safe(p/'trajectory.csv')
    assert frame.n_additional.tolist()==list(range(0,81,m))
    assert np.load(p/'selected_ids.npy').size==160
    np.testing.assert_array_equal(references('movielens',5)['theta_full'],references('movielens',5,str(refdir))['theta_full'])


def test_export_uses_configured_batch_size_and_dataset_scope(tmp_path):
    c=yaml.safe_load((ROOT/'configs/batch_m80.yaml').read_text())
    reporting.export_paper(c,tmp_path)
    text=(tmp_path/'paper/experiments_real_response.tex').read_text()
    assert 'batches of 80' in text and '5 completed rounds' in text
    assert 'Open Bandit' not in text and 'batches of 20' not in text
