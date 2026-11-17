"""Committed batch checkpoints, deterministic hidden-response trajectories."""
import os,json,time,traceback,copy,functools
from pathlib import Path
import numpy as np
import pandas as pd
from .common import *
from .numerics import fit_glm
from .evaluation import evaluate_checkpoint
from .selectors import select_batch,shared_initial_ids,shared_candidates,fb_opt_scale,fb_opt2_scale,fb_opt2_temperature
from .family_selectors import FAMILY_METHODS, select_family_batch

@functools.lru_cache(maxsize=9)
def prepared(ds,d):
    z=np.load(ROOT/f'data/prepared/{ds}.npz',allow_pickle=False)
    return np.ascontiguousarray(z['X_pool'][:,:d]),z['y_pool'],np.ascontiguousarray(z['X_test'][:,:d]),z['y_test']
@functools.lru_cache(maxsize=9)
def references(ds,d,reference_cache_dir=None):
    reference_dir=Path(reference_cache_dir) if reference_cache_dir else ROOT/'data/references'
    z=np.load(reference_dir/f'{ds}_d{d}.npz',allow_pickle=False)
    return {k:z[k] for k in z.files}
def initial_ids(ds,seed,n0):
    p=ROOT/f'data/initial/{ds}_{seed}_{n0}.npy'
    if p.exists():return np.load(p)
    X=prepared(ds,20)[0]
    return shared_initial_ids(X,ds,seed,n0=n0)
def candidates_for(ds,seed,round_id,N,selected,K):
    # Random-key priorities use a cryptographically keyed independent PCG stream;
    # the same row has the same priority across every method.
    return shared_candidates(N,np.asarray(selected,dtype=int),ds,seed,round_id,K)
def checkpoint_files(p):
    valid=[]
    for j in sorted((p/'checkpoints').glob('*.json')):
        a=j.with_suffix('.npz')
        if not a.exists():continue
        meta=read_json(j)
        try:
            with np.load(a,allow_pickle=False) as z:
                if 'theta_hat' not in z.files or 'selected_ids' not in z.files:continue
                if not np.isfinite(z['theta_hat']).all():continue
            if meta and meta.get('committed'):valid.append((j,a,meta))
        except Exception:continue
    # Broken tail never becomes scientific state.
    contiguous=[]
    for i,x in enumerate(valid):
        if x[2]['metrics']['round']!=i:break
        contiguous.append(x)
    return contiguous

def finalize_run(p):
    files=checkpoint_files(p)
    rows=[x[2]['metrics'] for x in files]
    arrays=[dict(np.load(x[1],allow_pickle=False)) for x in files]
    frame=pd.DataFrame(rows);tmp=p/'trajectory.csv.tmp';frame.to_csv(tmp,index=False);os.replace(tmp,p/'trajectory.csv')
    mapping={'theta_checkpoints':'theta_hat','error_ref_checkpoints':'error_ref','error_matched_checkpoints':'error_matched'}
    for name,key in mapping.items():np.save(p/(name+'.npy'),np.stack([a[key] for a in arrays]))
    np.save(p/'selected_ids.npy',arrays[-1]['selected_ids'])
    np.save(p/'batch_offsets.npy',np.array([len(a['selected_ids']) for a in arrays]))
    atomic_npz(p/'information_matrices.npz',**{k:np.stack([a[k] for a in arrays]) for k in ['V_ref','V_hat','V_acq','H_design']})
    atomic_npz(p/'information_spectra.npz',**{k:np.linalg.eigvalsh(np.stack([a[k] for a in arrays])) for k in ['V_ref','V_hat','V_acq','H_design']})
    # Per-batch JSON audit is embedded losslessly; slot arrays remain replayable.
    audit=[x[2]['acquisition_audit'] for x in files]
    atomic_npz(p/'batch_audit.npz',audit_json=np.array(json.dumps(audit,default=serial)))
    atomic_npz(p/'slot_trace.npz',audit_json=np.array(json.dumps([a.get('slot_trace',[]) for a in audit],default=serial)))
    atomic_npz(p/'checkpoint.npz',**arrays[-1],rng_state_json=np.array(json.dumps(files[-1][2]['rng_state'])))
    return len(files)

def run_one(task,config,out,round_limit=None,deadline=None):
    ds,d,seed,method=task;p=run_path(out,task);p.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();cpu_start=time.process_time()
    try:
        return _run(task,config,out,p,started,cpu_start,round_limit,deadline)
    except Exception as e:
        error={'state':'FAILED','task':task,'error':repr(e),'traceback':traceback.format_exc(),'at':time.time()}
        atomic_json(p/'status.json',error)
        return error

def _run(task,config,out,p,started,cpu_start,round_limit,deadline):
    ds,d,seed,method=task;dc=config['datasets'][ds];n0=config['n0'];rho=config['rho'];family=FAMILIES[ds]
    rounds=int(np.ceil(dc['additional_labels']/dc['batch_size']))
    prior=read_json(p/'status.json',{})
    if prior.get('state')=='EVALUATION_COMPLETE' and prior.get('completed_rounds')==rounds and len(checkpoint_files(p))==rounds+1:
        return {**prior,'reused':True}
    X,y,Xt,yt=prepared(ds,d);ref=references(ds,d,config.get('reference_cache_dir'))
    if method in ('fb_opt','fb_opt2') and dc['additional_labels'] % dc['batch_size']:
        raise ValueError('Finite-horizon FB-opt requires complete equal-size adaptive batches')
    opt_scale=fb_opt_scale(X,family,dc['batch_size'],rounds) if method=='fb_opt' else None
    if method=='fb_opt2':opt_scale=fb_opt2_scale(family,d,dc['batch_size'],rounds,config)
    opt_metadata_path=p/(method+'_metadata.json')
    if opt_scale:
        atomic_json(opt_metadata_path,{**opt_scale,'dataset':ds,'family':family,'d':d,'seed':seed,
            'm':dc['batch_size'],'n0':n0,'additional_budget':dc['additional_labels'],'rho':rho,
            'state':'RUNNING','temperature_by_round':[{'round':r,'N_t':r*dc['batch_size'],
                'tau_t':fb_opt2_temperature(opt_scale,r) if method=='fb_opt2' else opt_scale['tau0_opt']*(r*dc['batch_size'])**.75} for r in range(1,rounds+1)]})
    reference_dir=Path(config['reference_cache_dir']) if config.get('reference_cache_dir') else ROOT/'data/references'
    feature_hash=filehash(ROOT/f'data/prepared/{ds}.npz');reference_hash=filehash(reference_dir/f'{ds}_d{d}.npz')
    sci=digest({'task':task,'config':scientific_config(config),'feature':feature_hash,'reference':reference_hash})
    execution_hash=read_json(Path(out)/'execution_manifest.json',{}).get('execution_hash','manual')
    if (p/'config.json').exists():
        old=read_json(p/'config.json')
        if old['scientific_hash']!=sci:raise RuntimeError('Scientific identity changed; refusing overwrite')
    else:atomic_json(p/'config.json',{'task':task,'scientific_hash':sci,'execution_hash':execution_hash,'feature_hash':feature_hash,'reference_hash':reference_hash,'config':scientific_config(config)})
    rng=np.random.default_rng(stream_seed('selector',ds,d,seed,method))
    files=checkpoint_files(p);warm_ref=warm_hat=None
    if files:
        last=np.load(files[-1][1],allow_pickle=False);theta=last['theta_hat'];selected=last['selected_ids'].tolist()
        rng.bit_generator.state=files[-1][2]['rng_state'];next_round=files[-1][2]['metrics']['round']+1
        warm_ref=files[0][2]['metrics']['lambda_min_reference'];warm_hat=files[0][2]['metrics']['lambda_min_plugin']
        previous_elapsed=files[-1][2]['metrics']['cumulative_elapsed_seconds']
    else:selected=initial_ids(ds,seed,n0).tolist();theta=None;next_round=0;previous_elapsed=0.
    for r in range(next_round,rounds+1):
        if (round_limit is not None and r>round_limit) or (deadline and time.time()>=deadline):
            result={'state':'CHECKPOINTED','task':task,'completed_rounds':r-1,'wall_seconds':time.monotonic()-started}
            atomic_json(p/'status.json',result);return result
        acquisition_seconds=0.;candidate_count=0;batch_size=0;audit={'sdp_status':'NOT_APPLICABLE_WARM_START' if r==0 else 'NOT_APPLICABLE_METHOD','fallback_used':False}
        if r:
            batch_size=min(dc['batch_size'],n0+dc['additional_labels']-len(selected))
            cand=candidates_for(ds,seed,r,len(X),selected,dc['candidate_cap']);candidate_count=len(cand)
            rng_before=copy.deepcopy(rng.bit_generator.state);frozen_theta=theta.copy();t=time.monotonic()
            selector_config={**config,'dataset':ds,'seed':seed,'acquisition_floor':1e-12 if family!='gaussian' else 0.}
            if opt_scale:selector_config['_'+method+'_fixed_scale']=opt_scale
            selector_config['e_backend']=config.get('e_backend_by_dimension',{}).get(d,'secular')
            if method in FAMILY_METHODS:
                # B-EMCM alone needs the observed responses, never the full y.
                chosen,audit=select_family_batch(method,X,np.asarray(selected),cand,theta,family,batch_size,rho,rng,
                    selector_config,batch_index=r,observed_y=y[selected].copy() if method=='b_emcm' else None)
            else:
                chosen,audit=select_batch(method,X,np.asarray(selected),cand,theta,family,batch_size,rho,rng,selector_config,batch_index=r)
            acquisition_seconds=time.monotonic()-t
            chosen=np.asarray(chosen,dtype=int)
            assert len(chosen)==batch_size and len(set(chosen))==batch_size and not set(chosen).intersection(selected)
            assert set(chosen).issubset(set(cand))
            audit.update(candidate_ids=cand,selected_ids=chosen,pre_batch_theta=frozen_theta,rng_state_before=rng_before,rng_state_after=copy.deepcopy(rng.bit_generator.state))
            selected.extend(chosen.tolist())
        t=time.monotonic();theta,fit=fit_glm(X[selected],y[selected],family,rho=rho,theta0=theta);fit_seconds=time.monotonic()-t
        if not fit['fit_converged']:raise RuntimeError(f'Unconverged fit at round {r}: {fit}')
        n=len(selected);match_idx=np.where(ref['matched_ns']==n)[0]
        if not len(match_idx):raise RuntimeError(f'Missing matched reference n={n}')
        matched=ref['theta_matched'][match_idx[0]]
        t=time.monotonic();metrics,arrays=evaluate_checkpoint(X,y,Xt,yt,selected,theta,family,rho,ref['theta_full'],matched,fit,extra={'n0':n0,'warm_reference_lambda':warm_ref,'warm_plugin_lambda':warm_hat})
        evaluation_seconds=time.monotonic()-t
        if r==0:warm_ref=metrics['lambda_min_reference'];warm_hat=metrics['lambda_min_plugin']
        metadata=read_json(ROOT/f'data/prepared/{ds}.json',{})
        if not metrics['metrics_complete']:
            raise RuntimeError(f'Evaluation quality gate failed: {metrics["warning_codes"]}')
        metrics.update(dataset=ds,subset_version=metadata.get('subset_version','v2_fixed_real_pool'),family=family,d=d,seed=seed,method=method,scientific_hash=sci,execution_hash=execution_hash,feature_hash=feature_hash,reference_hash=reference_hash,round=r,batch_size=batch_size,n0=n0,n_additional=n-n0,n_total=n,n_pool=len(X),candidate_count=candidate_count,fit_seconds=fit_seconds,acquisition_seconds=acquisition_seconds,evaluation_seconds=evaluation_seconds,io_seconds=0.,process_cpu_seconds=time.process_time()-cpu_start,cumulative_elapsed_seconds=previous_elapsed+time.monotonic()-started,backend='numpy_cpu_float64',gpu_uuid=None)
        if opt_scale:
            fields=['tau_range','tau0_opt2','range_R','Tmax','Nmax','method_version'] if method=='fb_opt2' else ['L','w_bar','B','Tmax','Nmax','S_T','tau0_opt','tau0_previous','tau0_ratio_new_previous','method_version']
            metrics.update({k:opt_scale[k] for k in fields})
            metrics['temperature']=audit.get('temperature') if r else None
        sdp_fields=['sdp_status','sdp_solve_iterations','sdp_primal_residual','sdp_dual_residual','sdp_psd_residual','sdp_inclusion_residual','sdp_ub','sdp_lb','sdp_gap','fallback_used','fallback_reason']
        for key in sdp_fields:metrics[key]=audit.get(key,False if key=='fallback_used' else None)
        if r==0:metrics['sdp_status']='NOT_APPLICABLE_WARM_START'
        elif method!='e_sdp':metrics['sdp_status']='NOT_APPLICABLE_METHOD'
        arrays.update(theta_hat=theta,selected_ids=np.asarray(selected,dtype=np.int64))
        io_start=time.monotonic();base=p/'checkpoints'/f'{r:03d}'
        atomic_npz(base.with_suffix('.npz'),**arrays)
        metrics['io_seconds']=time.monotonic()-io_start
        atomic_json(base.with_suffix('.json'),{'committed':True,'metrics':metrics,'rng_state':rng.bit_generator.state,'acquisition_audit':audit})
        atomic_json(p/'status.json',{'state':'RUNNING','task':task,'completed_rounds':r,'total_rounds':rounds,'pid':os.getpid(),'updated_at':time.time(),'elapsed_seconds':previous_elapsed+time.monotonic()-started})
    records=finalize_run(p)
    result={'state':'EVALUATION_COMPLETE','task':task,'completed_rounds':rounds,'checkpoint_records':records,'wall_seconds':time.monotonic()-started,'total_elapsed_seconds':previous_elapsed+time.monotonic()-started,'scientific_hash':sci,'finished_at':time.time()}
    if opt_scale:
        opt_metadata=read_json(opt_metadata_path)
        opt_metadata.update(state='EVALUATION_COMPLETE',completed_rounds=rounds,
            actual_temperature_by_round=[{'round':r,'tau_t':obj['acquisition_audit']['temperature']}
                for _,_,obj in checkpoint_files(p) if (r:=obj['metrics']['round'])>0])
        atomic_json(opt_metadata_path,opt_metadata)
    atomic_json(p/'status.json',result);return result

def reevaluate_one(task,config,out,missing_only=True):
    p=run_path(out,task);ds,d,seed,method=task;files=checkpoint_files(p)
    X,y,Xt,yt=prepared(ds,d);ref=references(ds,d,config.get('reference_cache_dir'));warm_ref=warm_hat=None
    for j,a,obj in files:
        z=dict(np.load(a,allow_pickle=False));n=len(z['selected_ids']);old=obj['metrics'];matched=ref['theta_matched'][np.where(ref['matched_ns']==n)[0][0]]
        if missing_only and old.get('metrics_complete') and all(k in z for k in ['V_ref','V_hat','error_ref','error_matched']):
            if old['round']==0:warm_ref=old['lambda_min_reference'];warm_hat=old['lambda_min_plugin']
            continue
        # Fits are never re-run: loss/gradient audits are recomputed at stored theta.
        from .numerics import fit_audit_at_theta
        fit=fit_audit_at_theta(z['theta_hat'],X[z['selected_ids']],y[z['selected_ids']],FAMILIES[ds],config['rho'])
        m,arr=evaluate_checkpoint(X,y,Xt,yt,z['selected_ids'],z['theta_hat'],FAMILIES[ds],config['rho'],ref['theta_full'],matched,fit,extra={'n0':config['n0'],'warm_reference_lambda':warm_ref,'warm_plugin_lambda':warm_hat})
        obj['metrics'].update(m);z.update(arr);atomic_npz(a,**z);atomic_json(j,obj)
        if old['round']==0:warm_ref=m['lambda_min_reference'];warm_hat=m['lambda_min_plugin']
    if files:finalize_run(p)
    return len(files)
