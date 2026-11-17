"""Sparse, saved-state FB Monte-Carlo oracle-quality checks; no rerun trajectories."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .oracles import candidate_cloud, score_candidates
from .reference import _atomic_json

K_VALUES = (512,1024,2048,4096,8192)


def check_state(path, config):
    with np.load(path,allow_pickle=False) as z:
        theta,W = np.array(z['theta']),np.array(z['W'] if 'W' in z else z['Vhat'])
        family = str(z['family'].item())
        condition = json.loads(str(z['condition'].item()))
        phase = str(z['phase'].item()) if 'phase' in z else path.stem.rsplit('_',1)[-1]
        tau,N,batch = float(z['tau']),int(z['N']),int(z['batch_index'])
    cloud = candidate_cloud(family,condition,batch,K_VALUES[-1],config.get('global_cloud_seed',20261005),'cpu')
    tensor_theta,tensor_W = torch.tensor(theta[None],dtype=torch.float64),torch.tensor(W[None],dtype=torch.float64)
    with torch.no_grad():
        scores = score_candidates('FB',family,tensor_theta,tensor_W,cloud,tau=tau)[0].numpy()
    # Independent direct eigendecomposition score verifies diagnostic consistency.
    vals,vec = np.linalg.eigh(W)
    q = np.exp(-(vals-vals.min())/tau);q /= q.sum()
    P = (vec*q[None,:])@vec.T
    X = cloud.numpy()
    w = np.ones(len(X)) if family == 'gaussian' else expit(X@theta)*expit(-X@theta)
    direct = w*np.sum((X@P)*X,axis=-1)
    discrepancy = float(np.max(np.abs(direct-scores))/max(float(np.max(direct)),1e-300))
    if discrepancy > 1e-10:
        raise AssertionError(f'FB diagnostic score audit failed: {discrepancy}')
    anchor_index = int(np.argmax(scores))
    anchor,anchor_score = X[anchor_index],scores[anchor_index]
    output = []
    for K in K_VALUES:
        # Generation is deterministic and independent of K. Direction-major
        # radial expansion makes each K prefix a full 4-radius Bernoulli cloud.
        index = int(np.argmax(scores[:K]))
        x = X[index]
        cosine = float(np.clip(np.dot(x,anchor)/(np.linalg.norm(x)*np.linalg.norm(anchor)),-1,1))
        row = {'family':family,'condition':condition.get('id',condition.get('name','central')),
            'phase':phase,'N':N,'next_batch_index':batch,'tau':tau,'K':K,
            'score':float(scores[index]),'score_ratio_to_8192':float(scores[index]/anchor_score),
            'selected_index':index,'reference_index_8192':anchor_index,
            'axis_angle_degrees_to_8192':float(np.degrees(np.arccos(abs(cosine)))),
            'signed_angle_degrees_to_8192':float(np.degrees(np.arccos(cosine))),
            'selected_radius_over_L':float(np.linalg.norm(x)/condition['L']),
            'score_validation_max_relative_error':discrepancy,
            'source_state':str(path.resolve()),
            'angular_qualification':'Antipodal directions have identical Fisher increments; axis angle identifies x and -x. Nearly tied scores need not imply angular agreement.'}
        output.append(row)
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir','--run',dest='run',required=True)
    parser.add_argument('--config')
    args=parser.parse_args()
    torch.set_num_threads(1)
    run = Path(args.run).resolve()
    config = json.loads((Path(args.config) if args.config else run/'config.json').read_text())
    out = run/'diagnostics'
    out.mkdir(exist_ok=True)
    rows = []
    for path in sorted((run/'oracle_states').glob('*.npz')):
        rows.extend(check_state(path,config))
    with (out/'oracle_k_sensitivity.csv').open('w',newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    seen = {(row['family'],row['phase']) for row in rows}
    needed = {(family,phase) for family in config.get('families',['gaussian','bernoulli']) for phase in ['early','middle','late']}
    summary = {'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'complete' if needed.issubset(seen) else 'partial' if rows else 'awaiting_states',
        'K_values':list(K_VALUES),'production_K':config['K'],
        'evaluated_states':len(rows)//len(K_VALUES),'missing_states':[list(v) for v in sorted(needed-seen)],
        'scope':'Central FB only; one representative response realization per family and early/middle/late saved state; no full trajectories rerun.',
        'cloud_pairing':'Nested deterministic direction prefixes; Bernoulli uses four radii per direction.',
        'qualification':'Score ratios quantify sampled-oracle sensitivity at saved states. K=8192 is a numerical reference, not an exact continuous maximum or proof of oracle adequacy.',
        'rows':rows}
    _atomic_json(out/'oracle_k_sensitivity.json',summary)
    fig,axes = plt.subplots(2,2,figsize=(11,8),squeeze=False)
    for i,family in enumerate(['gaussian','bernoulli']):
        for phase in ['early','middle','late']:
            chosen = [row for row in rows if row['family']==family and row['phase']==phase]
            if not chosen: continue
            axes[i,0].plot([r['K'] for r in chosen],[r['score_ratio_to_8192'] for r in chosen],marker='o',label=phase)
            axes[i,1].plot([r['K'] for r in chosen],[r['axis_angle_degrees_to_8192'] for r in chosen],marker='o',label=phase)
        for j,ax in enumerate(axes[i]):
            ax.set_xscale('log',base=2);ax.set_xlabel('Candidates K');ax.set_title(family.capitalize());ax.grid(alpha=.2)
            if ax.lines: ax.legend(fontsize=8)
        axes[i,0].axhline(1,color='black',ls='--',lw=.8)
        axes[i,0].set_ylabel(r'$s_K/s_{8192}$')
        axes[i,1].set_ylabel('Selected-axis angle to K=8192 (degrees)')
    fig.suptitle('FB saved-state sampled-ball oracle diagnostic | K=8192 is a numerical reference')
    fig.tight_layout(rect=[0,0,1,.96])
    for ext in ['png','pdf']:
        fig.savefig(out/f'oracle_k_sensitivity.{ext}',dpi=145,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)

if __name__ == '__main__':
    main()
