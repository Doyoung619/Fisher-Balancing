"""Independent continuous-ball reference designs. No acquisition code is imported."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog, minimize_scalar
from scipy.special import expit
from mpmath import iv


def curvature(t):
    return expit(t) * expit(-np.asarray(t))


def interval_gamma(d, L, kappa, family):
    """Outward interval arithmetic, followed by outward binary64 conversion."""
    iv.dps = 50
    radius = iv.mpf(str(L))
    dimension = iv.mpf(d)
    if family == 'gaussian':
        value = radius ** 2 / dimension
    else:
        v = iv.mpf(str(kappa)) / iv.sqrt(dimension)
        e = iv.exp(-v)
        value = radius ** 2 * e / ((1 + e) ** 2 * dimension)
    return (float(np.nextafter(float(value.a), -np.inf)),
            float(np.nextafter(float(value.b), np.inf)))


def independent_lp(d, L, kappa, analytic_upper, tolerance=1e-7):
    """Adaptive feasible 1D LP. Its lower bound is independent of the formula.

    Scalar maximization drives refinement but is never presented as a certified
    upper bound: the derivative argument documented in audit_theory.md does that.
    """
    support = np.linspace(0., 1., 129)
    log = []
    for iteration in range(40):
        weight = curvature(kappa * support) * L ** 2
        a = weight * support ** 2
        b = weight * (1-support ** 2) / (d-1)
        objective = np.r_[np.zeros(len(support)), -1.]
        result = linprog(objective, A_ub=np.c_[-np.array([a, b]), np.ones(2)],
                         b_ub=np.zeros(2), A_eq=np.r_[np.ones(len(support)), 0.][None],
                         b_eq=[1.], bounds=[(0, None)]*len(support)+[(None, None)],
                         method='highs', options={'dual_feasibility_tolerance': 1e-10,
                                                 'primal_feasibility_tolerance': 1e-10})
        if not result.success:
            raise RuntimeError('Reference LP failed: '+result.message)
        probabilities = np.maximum(result.x[:-1], 0.)
        probabilities /= probabilities.sum()
        aa, bb = float(a @ probabilities), float(b @ probabilities)
        lower = min(aa, bb)
        q = float(np.clip(-result.ineqlin.marginals[0], 0., 1.))
        log.append({'iteration':iteration, 'support_size':len(support),
                    'feasible_lower':lower, 'certified_upper':analytic_upper,
                    'relative_gap':(analytic_upper-lower)/analytic_upper, 'dual_q':q})
        if analytic_upper-lower <= tolerance*analytic_upper:
            active = probabilities > 1e-12
            return {'lower':lower, 'a':aa, 'b':bb,
                    'z_support':(L*support[active]).tolist(),
                    'weights':probabilities[active].tolist(), 'iterations':log}
        def negative(v):
            w = float(curvature(kappa*v))*L**2
            return -w*(q*v*v+(1-q)*(1-v*v)/(d-1))
        # Independent continuous refinement from an LP dual; multiple intervals.
        peaks = [minimize_scalar(negative,bounds=(j/16,(j+1)/16),
                                 method='bounded',options={'xatol':1e-14}).x
                 for j in range(16)]
        best = min(peaks+[0.,1.],key=negative)
        support = np.unique(np.r_[support, best])
    raise RuntimeError('Independent adaptive reference LP failed relative gap target')


def compute_reference(family, d, L, kappa, direction, audit=True):
    direction = np.asarray(direction,dtype=float)
    if direction.shape != (d,) or abs(np.linalg.norm(direction)-1)>1e-10:
        raise ValueError('Reference requires one stored unit direction of dimension d')
    lower, upper = interval_gamma(d,L,kappa,family)
    if family == 'gaussian':
        gamma = L*L/d
        q = 1/d
        longitudinal = transverse = gamma
        support = (L*np.eye(d)).tolist()
        weights = np.full(d,1/d).tolist()
        lp = None
        description = 'Trace bound, attained by an orthonormal radius-L design.'
    elif family == 'bernoulli':
        t = 1/math.sqrt(d)
        c = kappa*math.tanh(kappa*t/2)/(2*math.sqrt(d))
        if not 0<c<1:
            raise ValueError('Analytic reference proof requires 0<c<1; implement general interval LP for other signals')
        q = (1+(d-1)*c)/d
        gamma = L*L*float(curvature(kappa*t))/d
        longitudinal = transverse = gamma
        U = null_space(direction[None,:])
        x = L*t*direction[None,:] + L*math.sqrt(1-1/d)*U.T
        y = L*t*direction[None,:] - L*math.sqrt(1-1/d)*U.T
        support = np.vstack([x,y]).tolist()
        weights = np.full(2*(d-1),1/(2*(d-1))).tolist()
        lp = independent_lp(d,L,kappa,upper) if audit else None
        description = 'Single symmetric latitude z=L/sqrt(d); analytically global dual maximum.'
    else:
        raise ValueError(family)
    dual = q*np.outer(direction,direction)+(1-q)/(d-1)*(np.eye(d)-np.outer(direction,direction))
    X, p = np.array(support), np.array(weights)
    w = np.ones(len(X)) if family == 'gaussian' else curvature(X @ (kappa/L*direction))
    reconstructed = X.T @ ((p*w)[:,None]*X)
    min_dual = float(np.linalg.eigvalsh(dual)[0])
    residual = float(np.linalg.norm(reconstructed-gamma*np.eye(d),2))
    if min_dual<=0 or residual>1e-12*max(1,L*L):
        raise AssertionError('Reference primal/dual residual failure')
    return {'family':family,'d':d,'L':L,'kappa':kappa,
            'gamma_star':gamma,'gamma_lower':lower,'gamma_upper':upper,
            'relative_reference_gap':(upper-lower)/lower,
            'trace_inv_star':d/gamma,'a_star':longitudinal,'b_star':transverse,
            'M_star':reconstructed.tolist(),'support':support,'weights':weights,
            'z_support':[L/math.sqrt(d)] if family=='bernoulli' else None,
            'dual_q':q,'dual_P':dual.tolist(),'dual_min_eigenvalue':min_dual,
            'primal_matrix_residual_op':residual,
            'certificate':'analytic_global_derivative_and_outward_interval_arithmetic',
            'uniqueness_status':'analytically_established',
            'uniqueness_argument':'For any optimal M, M-gamma I is PSD; tight positive-definite dual P implies tr(P(M-gamma I))=0, hence M=gamma I.',
            'description':description,'independent_adaptive_lp':lp,
            'risk_target_qualification':'FB exact-oracle asymptotic reference; numerical acquisition qualification remains applicable.'}


def build_references(config):
    result, cache = {}, {}
    for family in config['families']:
        for condition in config['conditions']:
            d,L,k = int(condition['d']),float(condition['L']),float(condition['kappa'])
            key = (family,d,L,k)
            if key not in cache:
                cache[key] = compute_reference(family,d,L,k,config['directions'][str(d)])
            result[f"{family}/{condition['id']}"] = dict(cache[key],condition=condition['id'])
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    path=Path(args.config)
    if path.suffix in ('.yaml','.yml'):
        import yaml
        config=yaml.safe_load(path.read_text())
    else:
        config=json.loads(path.read_text())
    result=build_references(config)
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+'\n')
    for key,value in result.items():
        print(f"{key}: gamma={value['gamma_star']:.14g}, trace_inv={value['trace_inv_star']:.14g}, gap={value['relative_reference_gap']:.3g}, unique={value['uniqueness_status']}")

if __name__=='__main__':
    main()
