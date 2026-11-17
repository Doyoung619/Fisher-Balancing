from pathlib import Path
import re,json,time
import numpy as np
import pandas as pd
from .common import *
from .reporting import read_csv_safe

def required_columns():
    return json.loads((ROOT/'configs/required_scalar_columns.json').read_text())

def verify(config,out):
    out=Path(out);records=[];complete=0;checkpoints=0;failures=[]
    required=required_columns()
    for task in tasks_for(config):
        ds,d,seed,method=task;p=run_path(out,task);expected=1+int(np.ceil(config['datasets'][ds]['additional_labels']/config['datasets'][ds]['batch_size']));issues=[];n=0;hashes={}
        try:
            status=read_json(p/'status.json',{})
            if status.get('state')!='EVALUATION_COMPLETE':issues.append('RUN_NOT_EVALUATION_COMPLETE')
            f=read_csv_safe(p/'trajectory.csv');n=len(f)
            if n!=expected:issues.append(f'RECORD_COUNT_{n}_EXPECTED_{expected}')
            missing=sorted(set(required)-set(f.columns))
            if missing:issues.append('MISSING_SCALARS:'+','.join(missing))
            if not f['metrics_complete'].all():issues.append('INCOMPLETE_METRICS_FLAG')
            if not f['fit_converged'].all():issues.append('UNCONVERGED_FIT')
            if f['fit_grad_relative'].max()>1.0001e-8:issues.append('STATIONARITY_FAILED')
            for key in ['theta_checkpoints','error_ref_checkpoints','error_matched_checkpoints']:
                a=np.load(p/(key+'.npy'))
                if a.shape!=(expected,d) or not np.isfinite(a).all():issues.append('INVALID_ARRAY:'+key)
                hashes[key]=filehash(p/(key+'.npy'))
            mats=np.load(p/'information_matrices.npz');spec=np.load(p/'information_spectra.npz')
            for k in ['V_ref','V_hat','V_acq','H_design']:
                a=mats[k]
                if a.shape!=(expected,d,d) or not np.isfinite(a).all():issues.append('INVALID_MATRIX:'+k)
                if not np.allclose(np.linalg.eigvalsh(a),spec[k],rtol=1e-9,atol=1e-10):issues.append('SPECTRUM_MISMATCH:'+k)
            hashes['information_matrices']=filehash(p/'information_matrices.npz')
            if not np.allclose(f.lambda_min_reference,spec['V_ref'][:,0]):issues.append('REFERENCE_EIG_MISMATCH')
            if not np.allclose(f.lambda_min_plugin,spec['V_hat'][:,0]):issues.append('PLUGIN_EIG_MISMATCH')
            if not np.allclose(f.parameter_l2_to_full,np.linalg.norm(np.load(p/'error_ref_checkpoints.npy'),axis=1)):issues.append('L2_MISMATCH')
            selected=np.load(p/'selected_ids.npy')
            if len(selected)!=config['n0']+config['datasets'][ds]['additional_labels'] or len(np.unique(selected))!=len(selected):issues.append('LABEL_BUDGET_OR_DUPLICATES')
            if (f.lambda_min_reference.diff().dropna() < -1e-9).any():issues.append('REFERENCE_MONOTONICITY')
            if (f.min_eigenvalue_gap.abs()>f.tracking_op+1e-8).any():issues.append('WEYL_BOUND')
            if ds=='movielens' and not np.allclose(mats['V_ref'],mats['V_hat']):issues.append('GAUSSIAN_INFORMATION_MISMATCH')
            if not issues:complete+=1
            checkpoints+=n
        except Exception as exc:issues.append(repr(exc))
        row={'dataset':ds,'d':d,'seed':seed,'method':method,'complete':not issues,'checkpoint_count':n,'expected_checkpoints':expected,'issues':' | '.join(issues),'array_hashes':json.dumps(hashes)};records.append(row)
        if issues:failures.append(row)
    pd.DataFrame(records).to_csv(out/'metric_completeness.csv',index=False);pd.DataFrame(failures,columns=list(records[0])).to_csv(out/'failures.csv',index=False)
    aggregate=read_csv_safe(out/'summary_by_condition.csv') if (out/'summary_by_condition.csv').exists() else pd.DataFrame()
    aggregates_complete=len(aggregate)==len({(t[0],t[1],t[3]) for t in tasks_for(config)}) and ('complete_10_seed' not in aggregate or aggregate.complete_10_seed.all() or len(config['run_seeds'])!=10)
    aggregate_checks=[]
    for ds,d,method in sorted({(t[0],t[1],t[3]) for t in tasks_for(config)}):
        p=out/'aggregate'/ds/str(d)/method;issues=[]
        try:
            table=read_csv_safe(p/'risk_trajectory.csv');q=np.load(p/'Q_ref_checkpoints.npy');qm=np.load(p/'Q_matched_checkpoints.npy')
            expected=1+int(np.ceil(config['datasets'][ds]['additional_labels']/config['datasets'][ds]['batch_size']))
            if q.shape!=(expected,d,d) or qm.shape!=q.shape or len(table)!=expected:issues.append('AGGREGATE_SHAPE')
            if not np.allclose(np.trace(q,axis1=1,axis2=2),table.R2_squared_ref):issues.append('Q_TRACE')
            if not np.allclose(np.linalg.eigvalsh(q)[:,-1],table.Rdir_squared_ref):issues.append('Q_LAMBDA_MAX')
            if not np.allclose(table.Rdir_ref**2,table.Rdir_squared_ref):issues.append('RDIR_SQRT')
            if not np.allclose(table.R2_ref**2,table.R2_squared_ref):issues.append('R2_SQRT')
            if len(config['run_seeds'])==10 and not table.complete_10_seed.all():issues.append('MISSING_TEN_SEEDS')
            if not all(json.loads(s)==config['run_seeds'] for s in table.used_seed_ids):issues.append('SEED_IDS')
            bs=np.load(p/'endpoint_seed_bootstrap.npz');indices=bs['resample_indices']
            if indices.shape!=(config['uncertainty']['bootstrap_draws'],len(config['run_seeds'])):issues.append('BOOTSTRAP_SHAPE')
            if not (out/'figures'/f'{ds}_d{d}_primary.pdf').exists():issues.append('MISSING_PRIMARY_FIGURE')
        except Exception as exc:issues.append(repr(exc))
        aggregate_checks.append({'dataset':ds,'d':d,'method':method,'complete':not issues,'issues':' | '.join(issues)})
    pd.DataFrame(aggregate_checks).to_csv(out/'aggregate_completeness.csv',index=False)
    aggregates_complete=aggregates_complete and all(r['complete'] for r in aggregate_checks)
    result={'expected_trajectories':len(records),'complete_trajectories':complete,'failed_or_incomplete':len(records)-complete,'checkpoint_records':checkpoints,'expected_checkpoint_records':sum(r['expected_checkpoints'] for r in records),'aggregate_endpoint_cells':len(aggregate),'aggregate_checkpoint_records':sum(1+int(np.ceil(config['datasets'][r['dataset']]['additional_labels']/config['datasets'][r['dataset']]['batch_size'])) for r in aggregate_checks if r['complete']),'aggregate_complete':bool(aggregates_complete),'verified_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'MAIN_COMPLETE':complete==len(records) and aggregates_complete and len(config['run_seeds'])==10 and config['dimensions']==[5,10,20]}
    atomic_json(out/'verification.json',result)
    return result

def metadata(config,out):
    out=Path(out);identity=protocol_identity(config)
    previous=read_json(out/'protocol_manifest.json')
    if previous is not None and previous!=identity:
        raise RuntimeError('Output directory belongs to a different scientific protocol; use a new directory')
    atomic_json(out/'protocol_manifest.json',identity)
    execution={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'python_command':__import__('sys').executable,'argv':__import__('sys').argv,'hostname':__import__('socket').gethostname(),'code_hashes':identity['implementation_hashes'],'scalar_schema_sha256':filehash(ROOT/'configs/required_scalar_columns.json')}
    execution['execution_hash']=digest(execution);atomic_json(out/'execution_manifest.json',execution)
    atomic_json(out/'theory_scope_audit.json',{k:{'status':'NOT_IDENTIFIED_IN_REAL_DATA','reason':'Finite logged real-response pool; full-pool ridge estimator is an empirical reference, not an unknown generating parameter or full-ball theorem constant.'} for k in ['theta_star','V_star','response_noise_expectation','gamma_star','g_star','M_star','D_T','theorem_beta_epsilon','T_g_star_Rdir_squared']})
    rows=[]
    for ds in config['datasets']:
        for method in config['methods']:rows.append({'dataset':ds,'method':method,'applicable':method!='badge' or ds=='obd','reason':'Classification-only BADGE' if method=='badge' and ds!='obd' else ''})
    pd.DataFrame(rows).to_csv(out/'method_coverage.csv',index=False)
    atomic_json(out/'input_document_audit.json',{'scalar_schema':'configs/required_scalar_columns.json','manuscript_pdf':'NOT_PROVIDED_IN_WORKSPACE','PAT_feedback':'NOT_PROVIDED_IN_WORKSPACE','v1_reuse':False,'reason':'Real-response protocol was implemented from explicit v2 formulas; the supplied synthetic manuscript does not identify real-data truth.'})
