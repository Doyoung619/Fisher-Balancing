"""Read-only independent reconstruction of prepared real responses and ranks."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import unicodedata
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT / 'outputs/data_audit')
parser.add_argument('--data-root', type=Path, default=ROOT / 'data/raw', help='Raw source archive root.')
args = parser.parse_args()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
RAW = args.data_root.resolve()
META = {name: json.loads((ROOT / f'data/prepared/{name}.json').read_text()) for name in ['obd','movielens','lastfm']}
CHECKS=[]

def check(name, condition, **details):
    row={'check':name,'passed':bool(condition),**details};CHECKS.append(row)
    if not condition:raise AssertionError(row)

def normalize(v):
    return '' if pd.isna(v) else ' '.join(unicodedata.normalize('NFKC',str(v)).casefold().split())

# Reconstruct Last.fm qualifying users, selected artists and counts from raw logs.
L=pd.read_parquet(RAW/'lastfm/lastfm-dataset-50.snappy.parquet')
start=pd.Timestamp('2008-05-05T00:00:00Z');hstart=start-pd.Timedelta(days=180);end=start+pd.Timedelta(weeks=52)
L['timestamp']=pd.to_datetime(L.timestamp,utc=True)
key=L.artist_id.fillna('').astype(str).str.strip().str.lower()
L['key']=np.where(key.ne(''),'mbid:'+key,'name:'+L.artist_name.map(normalize))
H=L[(L.timestamp>=hstart)&(L.timestamp<start)].copy()
H['week']=((H.timestamp-hstart).dt.total_seconds()//(7*86400)).astype(int)
qualification=H.groupby('user_id').agg(n=('timestamp','size'),weeks=('week','nunique'))
users=sorted(qualification.index[(qualification.n>=100)&(qualification.weeks>=4)])
H=H[H.user_id.isin(users)]
artist_count=H[H.key!='name:'].groupby('key').size()
artists=sorted(artist_count.index,key=lambda k:(-int(artist_count[k]),k))[:20]
check('lastfm_history_only_user_eligibility',users==META['lastfm']['eligible_users'],eligible_users=len(users))
check('lastfm_history_only_artist_selection',artists==META['lastfm']['selected_artists'],artist_count=len(artists))
E=L[(L.timestamp>=start)&(L.timestamp<end)&L.user_id.isin(users)&L.key.isin(artists)].copy()
E['week']=((E.timestamp-start).dt.total_seconds()//(7*86400)).astype(int)
counts=E.groupby(['user_id','key','week']).size()
check('lastfm_weekly_count_conservation',int(counts.sum())==len(E),archived_events=len(E),count_sum=int(counts.sum()),grid_cells=len(users)*20*52,nonzero_grid_cells=len(counts))
check('lastfm_week_boundary_utc',int(((start+pd.Timedelta(days=7)-start).total_seconds())//(7*86400))==1 and int(((end-pd.Timedelta(microseconds=1)-start).total_seconds())//(7*86400))==51)

O=pd.read_csv(RAW/'obd/all.csv',index_col=0)
I=pd.read_csv(RAW/'obd/item_context.csv',index_col=0)
check('obd_original_record_ids_unique',O.index.is_unique)
check('obd_schema_real_clicks_and_item_foreign_key',len(O)==10000 and O.click.isin([0,1]).all() and O.item_id.isin(I.item_id).all() and I.item_id.is_unique)
with zipfile.ZipFile(RAW/'movielens/ml-100k.zip') as z:
    M=pd.read_csv(io.BytesIO(z.read('ml-100k/u.data')),sep='\t',header=None,names=['user','movie','rating','time'])
    MU=pd.read_csv(io.BytesIO(z.read('ml-100k/u.user')),sep='|',header=None)
    MI=pd.read_csv(io.BytesIO(z.read('ml-100k/u.item')),sep='|',header=None,encoding='latin-1')
check('movielens_foreign_keys_and_real_rating_scale',len(M)==100000 and M.rating.between(1,5).all() and M.user.isin(MU[0]).all() and M.movie.isin(MI[0]).all() and MU[0].is_unique and MI[0].is_unique)

summary=[];ranks=[]
for ds in META:
    meta=META[ds]
    with np.load(ROOT/f'data/prepared/{ds}.npz',allow_pickle=False) as file:Z={k:file[k] for k in file.files}
    check(ds+'_partition_ids_unique_disjoint',len(np.unique(Z['row_ids_pool']))==len(Z['row_ids_pool']) and len(np.unique(Z['row_ids_test']))==len(Z['row_ids_test']) and not len(np.intersect1d(Z['row_ids_pool'],Z['row_ids_test'])))
    check(ds+'_finite_features_bounded_pool',np.isfinite(Z['X_pool']).all() and np.isfinite(Z['X_test']).all() and np.linalg.norm(Z['X_pool'],axis=1).max()<=1+1e-12)
    check(ds+'_scaled_intercept',np.allclose(Z['X_pool'][:,0],1/meta['common_scale'],rtol=0,atol=1e-15))
    for split in ['pool','test']:
        ids=Z['row_ids_'+split];y=Z['y_'+split]
        if ds=='obd':
            index=[int(s.split(':')[1]) for s in ids];expected=O.loc[index,'click'].to_numpy()
            check(ds+'_'+split+'_chronological',bool(np.all(pd.to_datetime(O.loc[index,'timestamp'],utc=True).astype('int64') < int(meta['split_boundary']))) if split=='pool' else bool(np.all(pd.to_datetime(O.loc[index,'timestamp'],utc=True).astype('int64')>=int(meta['split_boundary']))))
        elif ds=='movielens':
            index=[int(s.split(':')[1]) for s in ids];expected=M.loc[index,'rating'].to_numpy()-3
            check(ds+'_'+split+'_chronological',bool(np.all(M.loc[index,'time']<int(meta['split_boundary']))) if split=='pool' else bool(np.all(M.loc[index,'time']>=int(meta['split_boundary']))))
            check(ds+'_'+split+'_raw_rating_preserved',np.array_equal(Z['original_rating_'+split],M.loc[index,'rating'].to_numpy()))
        else:
            cells=[s[len('lastfm:'):].split('|') for s in ids]
            expected=np.array([counts.get((u,a,int(w)),0) for u,a,w in cells])
            check(ds+'_'+split+'_week_partition',all(int(w)<42 for _,_,w in cells) if split=='pool' else all(42<=int(w)<52 for _,_,w in cells))
            check(ds+'_'+split+'_integer_nonnegative_counts',np.all(y>=0) and np.all(y==np.floor(y)))
        check(ds+'_'+split+'_responses_equal_raw_archive',np.array_equal(y,expected),rows=len(y))
    for d in [5,10,20]:
        X=Z['X_pool'][:,:d];singular=np.linalg.svd(X,compute_uv=False);rank=np.linalg.matrix_rank(X)
        check(ds+f'_d{d}_rank',rank==d,rank=int(rank))
        ranks.append({'dataset':ds,'d':d,'rank':int(rank),'rows':len(X),'smallest_singular_value':singular[-1],'largest_singular_value':singular[0],'condition_number':singular[0]/singular[-1],'warning_codes':';'.join(meta['dimensions'][str(d)]['warning_codes']),'feature_hash':meta['feature_hash']})
    summary.append({'dataset':ds,'family':meta['family'],'pool_rows':meta['pool_rows'],'test_rows':meta['test_rows'],'pool_positive_count':meta['pool_target']['positive_count'],'test_positive_count':meta['test_target']['positive_count'],'pool_zero_fraction':meta['pool_target']['zero_fraction'],'test_zero_fraction':meta['test_target']['zero_fraction'],'pool_mean':meta['pool_target']['mean'],'pool_variance':meta['pool_target']['variance'],'pool_max':meta['pool_target']['max'],'eligible_users':meta.get('eligible_user_count'),'selected_artists':meta.get('selected_artist_count'),'raw_feature_rank':meta['raw_nonconstant_rank'],'pca_components':19,'common_scale':meta['common_scale'],'max_pool_norm':meta['pool_norm_range'][1],'max_test_norm':meta['test_norm_range'][1],'feature_hash':meta['feature_hash'],'prepared_artifact_sha256':meta['artifact_sha256'],'response_hash':meta['response_hash']})
for filename,rows in [('prepared_data_summary.csv',summary),('feature_rank_audit.csv',ranks)]:
    with (OUT/filename).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(OUT/'data_integrity_audit.json').write_text(json.dumps({'status':'PASS','check_count':len(CHECKS),'checks':CHECKS,'source':'scripts/audit_prepared_data.py','scope':'independent raw-label/count/split/foreign-key/rank reconstruction; no training or target alteration'},indent=2)+'\n')
print(json.dumps({'output':str(OUT),'checks':len(CHECKS),'all_passed':all(c['passed'] for c in CHECKS)}))
