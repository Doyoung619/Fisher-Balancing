"""Offline, uncentered Monte Carlo risk summaries and theorem-reference figures."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .reference import build_references, _atomic_json

METHODS = ['FB','AOpt','DOpt','EGreedy','UCB','TS','Random']
COLORS = dict(zip(METHODS,['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#555555']))
STYLES = dict(zip(METHODS,['-','--',':','-.','--',':','-.']))
METRICS = ['true_information','plugin_information','directional_ratio','scaled_R2']
LABELS = [r'$\mathrm{mean}_r\,\lambda_{\min}(V^\star)/N$',r'$\mathrm{mean}_r\,\lambda_{\min}(\widehat V)/N$',r'$N\gamma^\star\lambda_{\max}(\widehat Q)$',r'$N\,\mathrm{tr}(\widehat Q)$']
SWEEPS = {'dimension':['dim5','central','dim20'], 'radius':['radius05','central','radius2'], 'batch':['batch5','central','batch20']}


def _csv(path, rows):
    path = Path(path)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(rows)


def _sem(values, axis=1):
    if values.shape[axis] < 2:
        return np.full(np.delete(values.shape,axis),np.nan)
    return np.std(values,axis=axis,ddof=1)/np.sqrt(values.shape[axis])


def risk_metrics(errors, N, gamma, trace_inv, bootstrap=0, seed=20261005):
    """errors has shape (checkpoint, independent response replicate, dimension).

    Bootstrap preserves the uncentered second moment and recomputes the maximum
    eigenvalue for each resample; it never substitutes SEM of projected errors.
    """
    errors,N = np.asarray(errors,dtype=float),np.asarray(N,dtype=float)
    if errors.ndim != 3 or errors.shape[0] != len(N):
        raise ValueError('Expected errors[C,R,d] and N[C]')
    C,R,d = errors.shape
    Q = np.einsum('cri,crj->cij',errors,errors,optimize=True)/R
    squared = np.sum(errors*errors,axis=-1)
    trace = np.trace(Q,axis1=-2,axis2=-1)
    if not np.allclose(trace,squared.mean(axis=1),rtol=1e-12,atol=1e-14):
        raise AssertionError('Uncentered MSE trace identity failed')
    result = {'Q':Q,'R2_squared':trace,'Rdir_squared':np.linalg.eigvalsh(Q)[:,-1],
        'per_replicate_squared_error':squared,'N':N.astype(int),
        'replicates':np.full(C,R,dtype=int)}
    result['directional_ratio'] = N*gamma*result['Rdir_squared']
    result['scaled_R2'] = N*trace
    result['scaled_R2_sem'] = N*_sem(squared)
    result['R2_normalized'] = result['scaled_R2']/trace_inv
    if bootstrap:
        if R < 2:
            # No response uncertainty can be estimated from a single response.
            for name in ('directional_ratio','scaled_R2'):
                result[name+'_bootstrap_low'] = np.full(C,np.nan)
                result[name+'_bootstrap_high'] = np.full(C,np.nan)
        else:
            rng = np.random.default_rng(seed)
            # Same resampled response identities at all checkpoints preserves
            # the trajectory dependence and yields pointwise bootstrap ranges.
            counts = rng.multinomial(R,np.full(R,1/R),size=bootstrap).astype(float)
            dir_samples, mse_samples = np.empty((bootstrap,C)),np.empty((bootstrap,C))
            # Vectorized resamples, replicates, and dimensions. Checkpoint chunks
            # bound working memory for d=20 and hundreds of resamples.
            for start in range(0,C,8):
                end = min(C,start+8)
                outer = np.einsum('cri,crj->crij',errors[start:end],errors[start:end],optimize=True)
                qb = np.einsum('br,crij->bcij',counts,outer,optimize=True)/R
                dir_samples[:,start:end] = np.linalg.eigvalsh(qb)[...,-1]*(N[start:end]*gamma)[None,:]
                mse_samples[:,start:end] = np.trace(qb,axis1=-2,axis2=-1)*N[start:end][None,:]
            for name,sample in [('directional_ratio',dir_samples),('scaled_R2',mse_samples)]:
                result[name+'_bootstrap_low'],result[name+'_bootstrap_high'] = np.quantile(sample,[.025,.975],axis=0)
    return result


def collect_groups(run):
    groups, skipped, files = {},[],[]
    for path in sorted((run/'results').glob('*.npz')):
        try:
            with np.load(path,allow_pickle=False) as z:
                family,condition,method = (str(z[k].item()) for k in ['family','condition','method'])
                item = {k:np.asarray(z[k]) for k in ['N','error','replicate_ids','true_min','plugin_min','tracking_op']}
                item['file'] = str(path)
            if not len(item['N']):
                continue
            if item['error'].shape[:2] != (len(item['N']),len(item['replicate_ids'])):
                raise ValueError('Bad checkpoint/replicate shape')
            groups.setdefault((family,condition,method),[]).append(item)
            stat = path.stat()
            files.append({'path':str(path.relative_to(run)),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})
        except (OSError,KeyError,ValueError) as error:
            skipped.append({'path':str(path),'error':str(error)})
    return groups,files,skipped


def summarize(run, out, references, bootstrap=200):
    groups, files, skipped = collect_groups(run)
    summaries, rows, endpoints = {},[],[]
    cache = run/'analysis_cache'
    cache.mkdir(exist_ok=True)
    for key, chunks in sorted(groups.items()):
        # All points in a curve use exactly the same independent response IDs.
        common = np.array(sorted(set.intersection(*(set(v['N'].tolist()) for v in chunks))),dtype=int)
        if not len(common):
            continue
        ids = np.concatenate([v['replicate_ids'] for v in chunks])
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Duplicate response replicate IDs in '+str(key))
        order = np.argsort(ids)
        arrays = {}
        for name in ['error','true_min','plugin_min','tracking_op']:
            arrays[name] = np.concatenate([v[name][np.searchsorted(v['N'],common)] for v in chunks],axis=1)[:,order]
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('Nonfinite trajectory result in '+str(key))
        ref = references['/'.join(key[:2])]
        digest = hashlib.sha256()
        digest.update(common.tobytes());digest.update(arrays['error'].tobytes())
        digest.update(f'{bootstrap}:{ref["gamma_star"]}:{ref["trace_inv_star"]}'.encode())
        cache_path = cache/(digest.hexdigest()+'.npz')
        if cache_path.exists():
            with np.load(cache_path,allow_pickle=False) as z:
                metric = dict(z)
        else:
            metric = risk_metrics(arrays['error'],common,ref['gamma_star'],ref['trace_inv_star'],bootstrap if key[2]=='FB' else 0)
            np.savez_compressed(cache_path,**metric)
        for input_name,output_name in [('true_min','true_information'),('plugin_min','plugin_information'),('tracking_op','tracking')]:
            # Simulator stores these three diagnostics already divided by N.
            values = arrays[input_name]
            metric[output_name] = values.mean(axis=1)
            metric[output_name+'_sem'] = _sem(values)
        metric['true_information_normalized'] = metric['true_information']/ref['gamma_star']
        metric['plugin_information_normalized'] = metric['plugin_information']/ref['gamma_star']
        safe = '__'.join(key)
        np.savez_compressed(out/'data'/f'{safe}.npz',**metric,replicate_ids=ids[order],errors=arrays['error'])
        summaries[key] = {'metrics':metric,'ref':ref,'ids':ids[order]}
        for i,n in enumerate(common):
            row = dict(zip(['family','condition','method'],key))
            row.update(N=int(n),replicates=len(ids),gamma_star=ref['gamma_star'],trace_inv_star=ref['trace_inv_star'])
            for name,value in metric.items():
                if isinstance(value,np.ndarray) and value.shape == (len(common),):
                    v = value[i].item()
                    row[name] = v if np.isfinite(v) else None
            rows.append(row)
        endpoints.append(rows[-1].copy())
    _csv(out/'metrics.csv',rows)
    _csv(out/'endpoint_table.csv',endpoints)
    return summaries,files,skipped


def _four(axes, summaries, family, condition, normalized=False):
    keys = [(family,condition,m) for m in METHODS if (family,condition,m) in summaries]
    for key in keys:
        method = key[2]
        metric,ref = summaries[key]['metrics'],summaries[key]['ref']
        n = metric['N']
        for j,ax in enumerate(axes):
            source = METRICS[j]
            scale = ref['gamma_star'] if normalized and j<2 else ref['trace_inv_star'] if normalized and j==3 else 1.
            curve = metric[source]/scale
            ax.plot(n,curve,label=f"{method} (R={metric['replicates'][0]})",color=COLORS[method],ls=STYLES[method],lw=1.6 if method=='FB' else 1.1)
            if method == 'FB' and j>=2 and source+'_bootstrap_low' in metric:
                lo,hi = metric[source+'_bootstrap_low']/scale,metric[source+'_bootstrap_high']/scale
                ax.fill_between(n,lo,hi,color=COLORS[method],alpha=.18,lw=0)
            elif j != 2 and source+'_sem' in metric:
                sem = metric[source+'_sem']/scale
                ax.fill_between(n,curve-sem,curve+sem,color=COLORS[method],alpha=.055,lw=0)
            target = 1. if normalized or j==2 else ref['gamma_star'] if j<2 else ref['trace_inv_star']
            if method == keys[0][2]:
                ax.axhline(target,color='black',ls='--',lw=1.,label='Continuous-theory reference')
    for j,ax in enumerate(axes):
        ax.set_xscale('log')
        ax.set_xlabel('Responses N')
        ax.set_ylabel('Normalized target = 1' if normalized else LABELS[j])
        ax.grid(alpha=.18)
        if not keys:
            ax.text(.5,.5,'Awaiting saved trajectories',ha='center',va='center',transform=ax.transAxes)


def _save(fig, out, name):
    paths = []
    for suffix in ['png','pdf']:
        path = out/(name+'.'+suffix)
        fig.savefig(path,dpi=145,bbox_inches='tight')
        paths.append(str(path))
    plt.close(fig)
    return paths


def make_figures(summaries, config, out):
    paths = []
    conditions = {c['id']:c for c in config['conditions']}
    for family in config.get('families',['gaussian','bernoulli']):
        for sweep, names in SWEEPS.items():
            fig, axes = plt.subplots(3,4,figsize=(17,10),squeeze=False)
            for row,name in enumerate(names):
                _four(axes[row],summaries,family,name)
                c = conditions.get(name,{})
                axes[row,0].set_title(f"{name}: d={c.get('d','?')}, L={c.get('L','?')}, m={c.get('m','?')}",loc='left',fontsize=10)
            handles,labels = [],[]
            for ax in axes.flat:
                hh,ll = ax.get_legend_handles_labels()
                for h,l in zip(hh,ll):
                    if l not in labels:
                        handles.append(h);labels.append(l)
            fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
            fig.suptitle(f'{family.capitalize()} — {sweep} OFAT | sampled-ball approximate oracle',fontsize=14)
            fig.tight_layout(rect=[0,.07,1,.96])
            paths += _save(fig,out,f'{family}_{sweep}_ofat')
        fig,axes = plt.subplots(2,2,figsize=(12,8))
        palette = plt.cm.tab10(np.linspace(0,1,7))
        for color,(name,c) in zip(palette,conditions.items()):
            item = summaries.get((family,name,'FB'))
            if item is None:
                continue
            metric,ref = item['metrics'],item['ref']
            for j,ax in enumerate(axes.flat):
                source = METRICS[j]
                scale = ref['gamma_star'] if j<2 else ref['trace_inv_star'] if j==3 else 1.
                curve = metric[source]/scale
                ax.plot(metric['N'],curve,label=f'{name} (R={metric["replicates"][0]})',color=color,lw=1.4)
                if j>=2 and source+'_bootstrap_low' in metric:
                    ax.fill_between(metric['N'],metric[source+'_bootstrap_low']/scale,metric[source+'_bootstrap_high']/scale,color=color,alpha=.08,lw=0)
        for j,ax in enumerate(axes.flat):
            ax.axhline(1,color='black',ls='--',lw=1)
            ax.set_xscale('log');ax.set_xlabel('Responses N');ax.grid(alpha=.18)
            ax.set_ylabel([r'$\lambda_{\min}(V^\star)/(N\gamma^\star)$',r'$\lambda_{\min}(\widehat V)/(N\gamma^\star)$',LABELS[2],r'$N\,\mathrm{tr}(\widehat Q)/\mathrm{tr}((M^\star)^{-1})$'][j])
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'{family.capitalize()} FB — normalized OFAT summary | target 1')
        fig.tight_layout(rect=[0,0,1,.96])
        paths += _save(fig,out,f'{family}_FB_normalized_summary')
        fig,ax = plt.subplots(figsize=(8,5))
        for method in METHODS:
            item = summaries.get((family,'central',method))
            if item:
                metric=item['metrics']
                ax.plot(metric['N'],metric['tracking'],label=method,color=COLORS[method])
        ax.set_xscale('log');ax.set_xlabel('Responses N');ax.set_ylabel(r'$\mathrm{mean}_r\,\|\widehat V-V^\star\|_{op}/N$')
        ax.grid(alpha=.18)
        if ax.lines: ax.legend(fontsize=8)
        fig.tight_layout()
        paths += _save(fig,out,f'{family}_central_tracking')
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir','--run',dest='run',required=True)
    parser.add_argument('--config')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--bootstrap',type=int,default=200)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    config_path = Path(args.config) if args.config else run/'config.json'
    config = json.loads(config_path.read_text())
    refs_path = run/'references.json'
    if not refs_path.exists():
        _atomic_json(refs_path,build_references(config))
    refs = json.loads(refs_path.read_text())
    out = run/'figures'/('preview' if args.preview else 'final')
    (out/'data').mkdir(parents=True,exist_ok=True)
    summaries,files,skipped = summarize(run,out,refs,args.bootstrap)
    paths = make_figures(summaries,config,out)
    result = {'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'stage':'preview' if args.preview else 'final','groups':len(summaries),
        'input_files':files,'skipped_files':skipped,'paths':paths,
        'bootstrap_resamples_FB':args.bootstrap,
        'bootstrap_interpretation':'Pointwise descriptive 2.5/97.5 percentile ranges of response-replicate resampling; not guaranteed confidence coverage for the largest eigenvalue, especially at repeated eigenvalues.',
        'risk_definition':'Q=mean(error outer error), without centering. Information eigenvalues precede replicate averaging.',
        'theory_qualification':'Continuous-ball reference targets assume an exact acquisition oracle. This experiment uses fresh Monte-Carlo candidate clouds and does not satisfy exact maximization over the full ball.',
        'all_requested_groups_present':len(summaries)==len(config.get('families',['gaussian','bernoulli']))*len(config['conditions'])*len(config.get('methods',METHODS))}
    _atomic_json(out/'plot_manifest.json',result)
    print(json.dumps({'output':str(out),'groups':len(summaries),'figures':len(paths),'skipped_files':skipped}),flush=True)

if __name__ == '__main__':
    main()
