"""Independent full-ball information references; never imports adaptive oracles.

The Bernoulli reference is a dense one-dimensional approximate-design LP.
Only normalized (L=1) problems are solved; radius changes scale information L².
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.special import expit

GRID_SIZES = (2049, 8193, 32769)


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def normalized_bernoulli_lp(d, kappa, grid_sizes=GRID_SIZES):
    if d < 2:
        raise ValueError('Rotational reference requires d >= 2')
    refinement = []
    final = None
    for count in grid_sizes:
        z = np.linspace(0., 1., int(count))
        w = expit(kappa*z) * expit(-kappa*z)
        a = w*z*z
        b = w*(1-z*z)/(d-1)
        answer = linprog(np.r_[np.zeros(count), -1.],
            A_ub=np.column_stack((-np.vstack((a,b)), np.ones(2))),
            b_ub=np.zeros(2), A_eq=np.r_[np.ones(count), 0.][None],
            b_eq=[1.], bounds=[(0, None)]*count+[(0, None)], method='highs',
            options={'dual_feasibility_tolerance':1e-10,
                     'primal_feasibility_tolerance':1e-10})
        if not answer.success:
            raise RuntimeError('Bernoulli reference LP failed: '+answer.message)
        p = np.maximum(answer.x[:-1], 0.)
        p /= p.sum()
        aa,bb = float(a@p),float(b@p)
        gamma = min(aa,bb)
        active = p > 1e-12
        refinement.append({'grid_size':int(count), 'grid_spacing':1./(count-1),
            'gamma':gamma, 'longitudinal':aa, 'transverse':bb,
            'sum_probability_residual':float(abs(p.sum()-1)),
            'solver_status':int(answer.status),
            'normalized_active_z':z[active].tolist(),
            'active_probability':p[active].tolist()})
        final = {'gamma':gamma, 'a':aa, 'b':bb,
                 'support':z[active].tolist(),'probabilities':p[active].tolist()}
    final['grid_refinement'] = refinement
    final['refinement_relative_change'] = abs(refinement[-1]['gamma']-refinement[-2]['gamma'])/final['gamma']
    # Independent analytic audit valid for these specified kappa/d values. It is
    # not fed to the LP, the grid, or any adaptive acquisition.
    c = kappa*np.tanh(kappa/(2*np.sqrt(d)))/(2*np.sqrt(d))
    if 0 < c < 1:
        exact = float(expit(kappa/np.sqrt(d))*expit(-kappa/np.sqrt(d))/d)
        final['analytic_crosscheck_gamma'] = exact
        final['relative_error_to_analytic_crosscheck'] = abs(final['gamma']-exact)/exact
    if final['refinement_relative_change'] > 1e-6:
        raise RuntimeError('Dense reference grid has not converged to tolerance 1e-6')
    return final


def compute_reference(family, d, L, kappa, direction, normalized_lp=None):
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (d,) or not np.isclose(np.linalg.norm(direction),1.,atol=1e-10):
        raise ValueError('Expected stored unit theta direction')
    if family == 'gaussian':
        gamma = L**2/d
        a = b = gamma
        lp = None
        source = 'Exact full-ball trace bound attained by isotropic boundary design'
    elif family == 'bernoulli':
        lp = normalized_lp or normalized_bernoulli_lp(d,kappa)
        a,b = L**2*lp['a'],L**2*lp['b']
        gamma = min(a,b)
        source = 'Dense 1D rotationally symmetrized approximate-design LP; radius scaling L squared'
    else:
        raise ValueError(family)
    uu = np.outer(direction,direction)
    M = a*uu+b*(np.eye(d)-uu)
    eig = np.linalg.eigvalsh(M)
    return {'family':family,'d':int(d),'L':float(L),'kappa':float(kappa),
        'gamma_star':float(gamma),'M_star':M.tolist(),
        'M_star_eigenvalues':eig.tolist(),'eigenvalues':eig.tolist(),
        'trace_inv_star':float(1/a+(d-1)/b),'a_star':float(a),'b_star':float(b),
        'source':source,'normalized_lp':lp,
        'LP_discretization':None if lp is None else {
            'domain':'z/L in [0,1], using even curvature symmetry',
            'grid_sizes':[r['grid_size'] for r in lp['grid_refinement']],
            'final_grid_spacing':lp['grid_refinement'][-1]['grid_spacing'],
            'solver':'scipy.optimize.linprog(method=highs)',
            'feasibility_tolerance':1e-10},
        'grid_refinement_relative_change':0. if lp is None else lp['refinement_relative_change'],
        'risk_target_qualification':'Continuous exact-oracle FB asymptotic reference. The sampled-ball implementation is a numerical approximation, not covered by an exact-maximization claim.'}


def build_references(config):
    references, cache = {}, {}
    conditions = config['conditions']
    if isinstance(conditions,dict):
        conditions = [dict(c,id=k) for k,c in conditions.items()]
    for family in config.get('families',['gaussian','bernoulli']):
        for condition in conditions:
            d,L,kappa = int(condition['d']),float(condition['L']),float(condition.get('kappa',1.5))
            if 'directions' in config:
                direction = config['directions'][str(d)]
            elif 'direction' in condition:
                direction = condition['direction']
            else:
                raise ValueError('Config must record its deterministic directions')
            key = (d,kappa)
            if family == 'bernoulli' and key not in cache:
                cache[key] = normalized_bernoulli_lp(d,kappa)
            ref = compute_reference(family,d,L,kappa,direction,cache.get(key))
            ref['condition'] = condition.get('id',condition.get('name'))
            references[f"{family}/{ref['condition']}"] = ref
    return references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    path = Path(args.config)
    if path.suffix in ('.yml','.yaml'):
        import yaml
        config = yaml.safe_load(path.read_text())
    else:
        config = json.loads(path.read_text())
    refs = build_references(config)
    _atomic_json(args.output,refs)
    for key,ref in refs.items():
        print(f"{key}: gamma={ref['gamma_star']:.12g}, trace_inv={ref['trace_inv_star']:.12g}, refinement={ref['grid_refinement_relative_change']:.3g}",flush=True)

if __name__ == '__main__':
    main()
