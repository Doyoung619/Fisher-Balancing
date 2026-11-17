from pathlib import Path
import os, json, hashlib, tempfile, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FAMILIES={'obd':'bernoulli','movielens':'gaussian','lastfm':'poisson'}
def serial(v):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,Path): return str(v)
    raise TypeError(type(v).__name__)
def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,default=serial,separators=(',',':')).encode()).hexdigest()
def filehash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def atomic_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(obj,indent=2,default=serial,allow_nan=True));os.replace(tmp,p)
def atomic_npz(p,**arr):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with open(tmp,'wb') as f: np.savez_compressed(f,**arr)
    os.replace(tmp,p)
def read_json(p,default=None):
    try:return json.loads(Path(p).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default

def stream_seed(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')
def scientific_config(config):
    return {k:v for k,v in config.items() if k not in ['run_seeds','run_seeds_preview','profile','runtime','uncertainty','metrics','output_dir','offline','data_root','stop_after','max_wall_minutes','reference_cache_dir']}
def protocol_identity(config):
    c=scientific_config(config)
    data={k:filehash(ROOT/f'data/prepared/{k}.npz') for k in config['datasets']}
    source={k:filehash(ROOT/'real_response'/k) for k in ['data.py','selectors.py','family_selectors.py','numerics.py','evaluation.py','runner.py']}
    return {'config':c,'data_hashes':data,'implementation_hashes':source,'schema_version':2}
def output_for(config):
    return ROOT/'outputs'/ 'real_response'/digest(protocol_identity(config))[:20]
def tasks_for(config):
    return [(ds,d,s,m) for ds in config['datasets'] for d in config['dimensions'] for s in config['run_seeds'] for m in config['methods'] if m!='badge' or ds=='obd']
def run_path(out,task):
    ds,d,s,m=task;return Path(out)/'runs'/ds/str(d)/str(s)/m
