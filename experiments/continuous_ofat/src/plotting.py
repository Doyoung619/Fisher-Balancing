"""Automatic complete-group plots and auditable, uncentered Monte Carlo summaries."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .metrics import compute_metrics

METHODS=['Ours','A-opt','D-opt','E-greedy','E-relax','UCB','TS','Random']
COLORS=dict(zip(METHODS,['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#7B3294','#555555']))
STYLES=dict(zip(METHODS,['-','--',':','-.','-','--',':','-.']))
METRICS=['true_information','plugin_information','directional_ratio','scaled_R2']
LABELS=[r'$\mathbb{E}\lambda_{\min}(V^\star)/N$',r'$\mathbb{E}\lambda_{\min}(\widehat V)/N$',
        r'$N\gamma^\star\lambda_{\max}(\widehat Q)$',r'$N\,\mathrm{tr}(\widehat Q)$']


def norm(s): return re.sub('[^a-z0-9]','',s.lower())


def canonical_method(s):
    aliases={'fbrangeanchoredcontinuous':'Ours','ucbellipsoidbatched':'UCB','tsgaussianperturbationbatched':'TS'}
    return aliases.get(norm(s),next((m for m in METHODS if norm(m)==norm(s)),s))


def write_csv(path,rows):
    if not rows:
        path.write_text('')
        return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


def finite_or_none(v):
    return float(v) if np.isfinite(v) else None


def collect_groups(run,config,preview=False):
    target=int(config['preview_N'] if preview else config['N_run'])
    cap=int(config['preview_replicates']) if preview else None
    groups={}
    scanned=[]
    skipped=[]
    for path in sorted((run/'trajectories').glob('*.npz')):
        parts=path.stem.split('__')
        if len(parts)!=4 or not re.fullmatch('r[0-9]+',parts[3]):
            skipped.append({'file':str(path),'reason':'unrecognized trajectory filename'})
            continue
        family,condition,method,replicate=parts[0],parts[1],canonical_method(parts[2]),int(parts[3][1:])
        if cap is not None and replicate>=cap: continue
        try:
            with np.load(path,allow_pickle=False) as archive:
                N=np.array(archive['N'],dtype=int)
                if not len(N) or N[-1]<target: continue
                indices=np.flatnonzero(N<=target)
                data={}
                data['N']=N[indices]
                data['replicate']=replicate
                data['file']=str(path)
            groups.setdefault((family,condition,method),[]).append(data)
            stat=path.stat()
            scanned.append({'file':str(path.relative_to(run)),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'replicate':replicate})
        except (OSError,KeyError,ValueError) as error:
            skipped.append({'file':str(path),'reason':str(error)})
    return groups,scanned,skipped


def summarize(run,config,references,out,preview):
    groups,files,skipped=collect_groups(run,config,preview)
    input_by_file={f['file']:f for f in files}
    summaries={}
    rows=[]
    endpoints=[]
    stability_all={}
    completeness=[]
    for family in config['families']:
        for condition in config['conditions']:
            for method in config['methods']:
                key=(family,condition['id'],method)
                records=groups.get(key,[])
                expected=int(config['preview_replicates'] if preview else config['replicates_center'] if condition['id']=='center' else config['replicates_noncentral'])
                completeness.append({'family':family,'condition':condition['id'],'method':method,'eligible_replicates':len(records),'expected_replicates':expected})
                if not records: continue
                records.sort(key=lambda r:r['replicate'])
                if len({r['replicate'] for r in records})!=len(records):
                    raise ValueError('Duplicate replicate IDs in '+str(key))
                common=np.array(sorted(set.intersection(*(set(r['N']) for r in records))),int)
                positions=[np.searchsorted(r['N'],common) for r in records]
                arrays={name:[] for name in ['error','Vstar','Vhat']}
                for record in records:
                    checkpoint_digest=hashlib.sha256(common.tobytes())
                    with np.load(record['file'],allow_pickle=False) as archive:
                        idx=np.searchsorted(archive['N'],common)
                        for name in arrays:
                            value=np.asarray(archive[name])[idx]
                            arrays[name].append(value)
                            checkpoint_digest.update(name.encode());checkpoint_digest.update(value.tobytes())
                    entry=input_by_file[str(Path(record['file']).relative_to(run))]
                    entry['selected_checkpoint_content_sha256']=checkpoint_digest.hexdigest()
                    entry['selected_N']=common.tolist()
                arrays={name:np.stack(values) for name,values in arrays.items()}
                ref=references[f'{family}/{condition["id"]}']
                metric,stability=compute_metrics(arrays['error'],arrays['Vstar'],arrays['Vhat'],common,ref['gamma_star'],ref['trace_inv_star'],config.get('rho',1.))
                safe='__'.join(key)
                reps=np.array([r['replicate'] for r in records])
                np.savez_compressed(out/'data'/f'{safe}.npz',**metric,replicate_ids=reps,errors=arrays['error'])
                dump_json(out/'data'/f'{safe}_mc.json',stability)
                summaries[key]={'metrics':metric,'stability':stability,'ref':ref,'expected':expected,'replicate_ids':reps}
                stability_all[safe]=stability
                for j,n in enumerate(common):
                    row={'family':family,'condition':condition['id'],'method':method,'N':int(n),'replicates':len(records),'expected_replicates':expected}
                    for name,value in metric.items():
                        if isinstance(value,np.ndarray) and value.shape==(len(common),): row[name]=finite_or_none(value[j])
                    row['gamma_star']=ref['gamma_star']
                    row['trace_inv_star']=ref['trace_inv_star']
                    rows.append(row)
                last=rows[-1].copy()
                last.update({'directional_target_discrepancy':last['directional_ratio']-1,
                             'R2_target_relative_discrepancy':last['scaled_R2']/ref['trace_inv_star']-1,
                             'reference_relative_gap':ref['relative_reference_gap'],
                             'reference_uniqueness':ref['uniqueness_status'],
                             'directional_bootstrap_descriptive_low':stability['descriptive_bootstrap_quantiles_025_975'][0],
                             'directional_bootstrap_descriptive_high':stability['descriptive_bootstrap_quantiles_025_975'][1],
                             'mc_qualification':stability['limitation']})
                endpoints.append(last)
    write_csv(out/'metrics.csv',rows)
    write_csv(out/'endpoint_table.csv',endpoints)
    write_csv(out/'completeness.csv',completeness)
    dump_json(out/'mc_precision.json',stability_all)
    # Pair methods by repetition ID. Each direction uses the common fixed truth.
    paired=[]
    for (family,condition,method),data in summaries.items():
        ours=summaries.get((family,condition,'Ours'))
        if method=='Ours' or ours is None: continue
        ids,oi,bi=np.intersect1d(ours['replicate_ids'],data['replicate_ids'],return_indices=True)
        if not len(ids) or ours['metrics']['N'][-1]!=data['metrics']['N'][-1]: continue
        difference=data['metrics']['per_replicate_squared_error'][bi,-1]-ours['metrics']['per_replicate_squared_error'][oi,-1]
        paired.append({'family':family,'condition':condition,'method':method,'reference_method':'Ours','N':int(data['metrics']['N'][-1]),
                       'paired_replicates':len(ids),'MSE_difference_method_minus_Ours':float(difference.mean()),
                       'paired_MC_SEM':float(difference.std(ddof=1)/np.sqrt(len(ids))) if len(ids)>1 else None})
    write_csv(out/'paired_endpoint_MSE.csv',paired)
    return summaries,files,skipped,completeness


def plot_four(axes,summaries,family,condition,normalized=False,ours_only=False,condition_label=None):
    keys=[k for k in summaries if k[:2]==(family,condition) and (not ours_only or k[2]=='Ours')]
    if not keys:
        for ax in np.ravel(axes): ax.text(.5,.5,'Awaiting complete\ntrajectories',ha='center',va='center',transform=ax.transAxes)
        return
    for key in keys:
        method=key[2]
        data=summaries[key]
        metric,ref=data['metrics'],data['ref']
        positive=metric['N']>0
        ns=metric['N'][positive]
        for j,ax in enumerate(np.ravel(axes)):
            metric_name=(['true_information_normalized','plugin_information_normalized','directional_ratio','R2_normalized'] if normalized else METRICS)[j]
            curve=metric[metric_name][positive]
            label=condition_label if condition_label else f'{method} (r={int(metric["replicates"][-1])})'
            kwargs={} if condition_label else {'color':COLORS[method],'ls':STYLES[method]}
            ax.plot(ns,curve,label=label,lw=1.5,marker='.' if len(ns)<3 else None,**kwargs)
            # Genuine SEM bands only for the linear Monte Carlo means.
            sem_name=['true_information_sem','plugin_information_sem',None,'scaled_R2_sem'][j]
            if sem_name and len(keys)<=8 and condition_label is None:
                standard=metric[sem_name][positive]
                if normalized: standard=standard/(ref['gamma_star'] if j<2 else ref['trace_inv_star'])
                ax.fill_between(ns,curve-standard,curve+standard,color=COLORS[method],alpha=.075)
    ref=summaries[keys[0]]['ref']
    targets=[1,1,1,1] if normalized else [ref['gamma_star'],ref['gamma_star'],1,ref['trace_inv_star']]
    for j,ax in enumerate(np.ravel(axes)):
        if condition_label is None:
            ax.axhline(targets[j],color='black',lw=1.,ls='--',label='Reference' if j==0 else None)
            if j==0 and not normalized:
                ax.plot(ns,ref['gamma_star']+1/ns,color='gray',ls=':',lw=.8,label='True finite-N ceiling')
        ax.set_xscale('log')
        ax.set_xlabel('Fresh responses N')
        ax.set_ylabel('Normalized to reference' if normalized and j!=2 else LABELS[j])
        ax.grid(alpha=.2)


def save_figure(fig,out,name,manifest,conditions,preview):
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(out/f'{name}.png',dpi=145,bbox_inches='tight')
    plt.close(fig)
    manifest.append({'name':name,'pdf':f'{name}.pdf','png':f'{name}.png','conditions':conditions,'preview':preview})


def generate_figures(summaries,config,out,preview,complete):
    manifest=[]
    stage=('PREVIEW' if complete else 'PREVIEW — INCOMPLETE COMPARISON') if preview else 'FINAL' if complete else 'INCOMPLETE PRODUCTION'
    note='Shading: ±1 Monte Carlo SEM where defined; directional eigenvalue has finite-repetition bias.'
    for family in config['families']:
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        plot_four(axes,summaries,family,'center')
        if axes[0,0].get_legend_handles_labels()[0]: axes[0,0].legend(fontsize=8,ncol=2)
        fig.suptitle(f'{stage} — {family}: central condition d=10, L=1, m=10, κ=1.5')
        fig.text(.5,.01,note,ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.03,1,.95])
        save_figure(fig,out,f'{family}_center',manifest,['center'],preview)
        for axis,conditions in [('dimension',['dim5','center','dim20']),('radius',['radius05','center','radius2']),('batch',['batch5','center','batch20'])]:
            fig=plt.figure(figsize=(19,10),layout='constrained')
            subfigures=fig.subfigures(1,3,wspace=.09)
            for subfig,condition in zip(subfigures,conditions):
                row=next(c for c in config['conditions'] if c['id']==condition)
                axes=subfig.subplots(2,2)
                plot_four(axes,summaries,family,condition)
                if axes[0,0].get_legend_handles_labels()[0]: axes[0,0].legend(fontsize=6)
                subfig.suptitle(f"{condition}: d={row['d']}, L={row['L']}, m={row['m']}")
            fig.suptitle(f'{stage} — {family}: {axis} OFAT (same center reused)',fontsize=15)
            save_figure(fig,out,f'{family}_{axis}_ofat',manifest,conditions,preview)
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        for row in config['conditions']:
            plot_four(axes,summaries,family,row['id'],normalized=True,ours_only=True,condition_label=row['id'])
        for ax in axes.flat: ax.axhline(1,color='black',ls='--',lw=1)
        if axes[0,0].get_legend_handles_labels()[0]: axes[0,0].legend(fontsize=8,ncol=2)
        fig.suptitle(f'{stage} — {family}: Ours, normalized OFAT; seven unique conditions')
        fig.tight_layout(rect=[0,.03,1,.95])
        save_figure(fig,out,f'{family}_ours_normalized_ofat',manifest,[r['id'] for r in config['conditions']],preview)
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        for method in METHODS:
            data=summaries.get((family,'center',method))
            if data is None: continue
            metric=data['metrics']; positive=metric['N']>0
            axes[0,0].plot(metric['N'][positive],metric['tracking_op_per_N'][positive],label=method,color=COLORS[method])
            axes[0,1].plot(np.arange(1,len(metric['mean_true_information_spectrum'][-1])+1),metric['mean_true_information_spectrum'][-1],label=method,color=COLORS[method])
            blocks=data['stability']['blocks']
            axes[1,0].plot([b['replicates'] for b in blocks],[b['prefix_ratio'] for b in blocks],label=method,color=COLORS[method],marker='.')
            axes[1,1].plot(metric['N'][positive],np.linalg.norm(metric['mean_error'],axis=1)[positive],label=method,color=COLORS[method])
        axes[0,0].set(xscale='log',xlabel='N',ylabel='Mean ‖Vhat−Vstar‖op/N')
        axes[0,1].set(xlabel='Eigenvalue index',ylabel='Endpoint true information spectrum/N')
        axes[1,0].set(xscale='log',xlabel='Prefix repetition count',ylabel='Endpoint directional ratio (full Q recomputed)')
        axes[1,0].axhline(1,color='black',ls='--')
        axes[1,1].set(xscale='log',xlabel='N',ylabel='Norm of mean error')
        if axes[0,0].get_legend_handles_labels()[0]: axes[0,0].legend(fontsize=8,ncol=2)
        for ax in axes.flat: ax.grid(alpha=.2)
        fig.suptitle(f'{stage} — {family}: tracking and Monte Carlo diagnostics')
        fig.tight_layout(rect=[0,.03,1,.95])
        save_figure(fig,out,f'{family}_tracking_mc',manifest,['center'],preview)
    return manifest


def copy_runtime_numerical(run,out,fig_manifest,preview):
    # Preserve actual profile table in the figure artifact; never synthesize timing.
    timing=run/'timing_profile.csv'
    if timing.exists():
        (out/'timing_profile.csv').write_bytes(timing.read_bytes())
        try:
            with timing.open() as stream: rows=list(csv.DictReader(stream))
            numeric_names=['batch_seconds','seconds','wall_seconds','elapsed_seconds','seconds_per_batch','median_seconds']
            field=next((f for f in numeric_names if rows and f in rows[0]),None)
            if field:
                labels=['/'.join(str(r.get(f,'')) for f in ['family','method','d','m','history_N','N','backend']) for r in rows]
                vals=[float(r[field]) for r in rows]
                fig,ax=plt.subplots(figsize=(14,max(4,len(rows)*.13)))
                ax.barh(np.arange(len(rows)),vals)
                ax.set_yticks(np.arange(len(rows)),labels,fontsize=5)
                ax.set_xlabel(field)
                ax.set_title('Measured timing profile; early and late histories kept separate')
                save_figure(fig,out,'runtime_diagnostics',fig_manifest,[],preview)
        except (ValueError,TypeError): pass
    diagnostic=[]
    oracle_rows=[]
    for directory in [run/'summaries',run/'diagnostics']:
        if not directory.exists(): continue
        for path in sorted(directory.glob('*.json')):
            try: item=json.loads(path.read_text())
            except (ValueError,OSError): continue
            # Batch records remain in the original durable artifact. Aggregate one
            # trajectory at a time so the full queue never resides in plot memory.
            row={'file':str(path.relative_to(run))}
            def visit(prefix,value):
                if isinstance(value,dict):
                    for k,v in value.items(): visit(prefix+'.'+k if prefix else k,v)
                elif isinstance(value,(str,int,float,bool)) or value is None: row[prefix]=value
            visit('',{k:v for k,v in item.items() if k!='diagnostics'})
            status_counts={}; numerical={}
            def audit_value(prefix,value):
                if isinstance(value,dict):
                    for k,v in value.items(): audit_value(prefix+'.'+k if prefix else k,v)
                elif isinstance(value,list):
                    for v in value: audit_value(prefix,v)
                elif isinstance(value,str) and ('status' in prefix or 'certificate' in prefix):
                    status_counts[value]=status_counts.get(value,0)+1
                elif isinstance(value,(float,int)) and any(k in prefix for k in ['gap','discrepancy','loss','residual','violation','error']):
                    if np.isfinite(value): numerical[prefix]=max(numerical.get(prefix,-np.inf),float(value))
            for batch in item.get('diagnostics',[]): audit_value('',batch.get('oracle',{}))
            row.update({'oracle_max.'+k:v for k,v in numerical.items()})
            row.update({'oracle_status_count.'+k:v for k,v in status_counts.items()})
            diagnostic.append(row)
            for k,v in numerical.items(): oracle_rows.append({'file':row['file'],'diagnostic':k,'maximum':v})
    write_csv(out/'numerical_diagnostics.csv',diagnostic)
    dump_json(out/'numerical_diagnostics.json',{'trajectory_summaries':diagnostic,'batch_detail_location':str(run/'summaries'),'qualification':'Only certified separation upper bounds are mathematical bounds. Multistart discrepancies are empirical audits.'})
    write_csv(out/'oracle_accuracy.csv',oracle_rows)
    if diagnostic:
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        kkt=[float(r.get('max_kkt',0)) for r in diagnostic if 'max_kkt' in r]
        axes[0].hist(kkt,bins=30)
        axes[0].set(xlabel='Per-trajectory maximum normalized KKT residual',ylabel='Trajectories')
        maxima={}
        for r in oracle_rows: maxima[r['diagnostic']]=max(maxima.get(r['diagnostic'],-np.inf),r['maximum'])
        if maxima:
            labels=list(maxima)
            axes[1].barh(np.arange(len(labels)),[maxima[k] for k in labels])
            axes[1].set_yticks(np.arange(len(labels)),labels,fontsize=6)
            axes[1].set_xlabel('Maximum recorded numerical diagnostic')
        else: axes[1].text(.5,.5,'Oracle diagnostics stored in original batch summaries',ha='center',va='center',transform=axes[1].transAxes)
        fig.suptitle('Numerical and oracle diagnostics (raw maxima; no success filtering)')
        fig.tight_layout()
        save_figure(fig,out,'oracle_numerical_diagnostics',fig_manifest,[],preview)


def make_plots(run,preview=False):
    run=Path(run).resolve()
    config=json.loads((run/'config.json').read_text())
    references=json.loads((run/'references.json').read_text())
    out=run/'figures'/('preview' if preview else 'final')
    (out/'data').mkdir(parents=True,exist_ok=True)
    summaries,files,skipped,completeness=summarize(run,config,references,out,preview)
    complete=bool(completeness) and all(r['eligible_replicates']==r['expected_replicates'] for r in completeness)
    figures=generate_figures(summaries,config,out,preview,complete)
    copy_runtime_numerical(run,out,figures,preview)
    source_manifest=json.loads((run/'manifest.json').read_text()) if (run/'manifest.json').exists() else {}
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    result={'generated_utc':datetime.now(timezone.utc).isoformat(),'stage':'PREVIEW' if preview else 'FINAL' if complete else 'INCOMPLETE_PRODUCTION',
            'comparison_complete':complete,'master_seed':config['master_seed'],'config_sha256':sha(run/'config.json'),
            'references_sha256':sha(run/'references.json'),'source_manifest':source_manifest,
            'figures':figures,'trajectory_inputs':files,'skipped_files':skipped,'groups':len(summaries),
            'central_reused':True,'metrics_order':METRICS,'common_checkpoint_rule':'Within a group use intersection of all eligible complete trajectory checkpoints; preserve repetition IDs.',
            'gaussian_overlap_note':'Verified identical Gaussian policies may overlap exactly; logical metadata retained and no curve displacement.',
            'mc_qualification':'Bands on linear means are ±1 Monte Carlo SEM. Endpoint directional resampling ranges are descriptive, not guaranteed confidence intervals. No shrinkage or debiasing.'}
    dump_json(out/'figure_manifest.json',result)
    print(json.dumps({'figures':str(out),'stage':result['stage'],'eligible_groups':len(summaries),'expected_groups':len(config['families'])*len(config['conditions'])*len(config['methods']),'complete':complete}))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run','--run-dir',dest='run',required=True)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--phase',choices=['preview','final'])
    args=parser.parse_args()
    make_plots(args.run,args.preview or args.phase=='preview')

if __name__=='__main__':main()
