"""Continuous-ball acquisition oracles; no reference parameter enters this module.

Quadratic inner trust-region and UCB oracles are global analytic/secular
solutions. Logistic outer scalar and constrained multistart searches are
explicitly *uncertified*. See audits/oracle_methods.md for tolerance semantics.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import expit


def curvature(theta, x, family="bernoulli"):
    if family == "gaussian":
        return np.ones(np.asarray(x).shape[:-1])
    s = expit(np.asarray(x) @ theta)
    return s * (1.0 - s)


def canonical_eigenvector(evals, evecs, smallest=True, rtol=1e-11):
    """Project the first possible coordinate axis into the extremal eigenspace."""
    target = evals[0] if smallest else evals[-1]
    tol = rtol * max(1.0, float(np.max(np.abs(evals))))
    basis = evecs[:, np.abs(evals - target) <= tol]
    for j in range(evecs.shape[0]):
        v = basis @ basis[j]
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            return v / norm
    raise ArithmeticError("Empty eigenspace")


def minimum_direction(W, L):
    a, U = eigh(W, check_finite=False)
    return L * canonical_eigenvector(a, U)


def orthogonal_basis(u):
    """Householder orthogonal completion, independent of the true parameter."""
    d = len(u)
    v = np.array(u, copy=True)
    v[0] += 1.0 if u[0] >= 0 else -1.0
    v /= np.linalg.norm(v)
    H = np.eye(d) - 2 * np.outer(v, v)
    return H[:, 1:]


def quadratic_trust_eigen(a, U, b, radius, tol=2e-13):
    """Globally maximize y'Ay+2b'y on ||y||<=radius, including hard cases.

    The multiplier lambda >= max(0,lambda_max(A)) satisfies
    (lambda I-A)y=b. When b has no top-eigenspace component and
    the limiting solution is short, complete it in that eigenspace.
    """
    d = len(a)
    if radius <= 1e-15 or d == 0:
        return np.zeros(d), {"hard_case": False, "multiplier": 0.0}
    c = U.T @ b
    scale = max(float(np.max(np.abs(a))), float(np.linalg.norm(c)/radius), 1e-300)
    # Scaling keeps secular roots accurate even when objective matrices are tiny.
    an = a / scale
    cn = c / scale
    top = max(0.0, float(an[-1]))
    threshold = tol * max(1.0, abs(top))
    singular = np.abs(top-an) <= threshold
    nonsingular = ~singular
    base = np.zeros(d)
    base[nonsingular] = cn[nonsingular] / (top - an[nonsingular])
    cn_top = float(np.linalg.norm(cn[singular]))
    if cn_top <= tol * max(radius, np.linalg.norm(cn)) and np.linalg.norm(base) <= radius*(1+tol):
        if top > 0:
            # The first coordinate projection gives a reproducible hard-case sign.
            dirs = U[:, singular]
            v = None
            for j in range(d):
                p = dirs @ dirs[j]
                if np.linalg.norm(p) > 1e-10:
                    v = p / np.linalg.norm(p)
                    break
            y = U @ base
            y += math.sqrt(max(0.0, radius*radius - float(base @ base))) * v
            return y, {"hard_case": True, "multiplier": top*scale}
        return U @ base, {"hard_case": bool(np.any(singular)), "multiplier": 0.0}
    def secular(lam):
        return float(np.sum((cn/(lam-an))**2) - radius*radius)
    lo = np.nextafter(top, math.inf)
    hi = top + max(1.0, float(np.linalg.norm(cn)/radius))
    while secular(hi) > 0:
        hi = top + 2*(hi-top)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        lam = brentq(secular, lo, hi, xtol=5e-15, rtol=1e-14, maxiter=150)
    y = U @ (cn/(lam-an))
    # Remove roundoff-sized constraint violations; no statistical projection here.
    yn = np.linalg.norm(y)
    if yn > radius:
        y *= radius/yn
    return y, {"hard_case": False, "multiplier": lam*scale}


def quadratic_trust(A, b, radius):
    a, U = eigh((A+A.T)*0.5, check_finite=False)
    return quadratic_trust_eigen(a, U, b, radius)


def logistic_quadratic(Q, theta, L, grid_size=17, scalar_xtol=2e-10):
    """Maximize w(theta'x)x'Qx over the entire closed ball.

    Q may be indefinite (used in secular audits). The conditional quadratic
    maximization is global; a grid plus refinement of every detected scalar
    local maximum is not a global certificate for the outer scalar problem.
    """
    Q = (np.asarray(Q)+np.asarray(Q).T)*0.5
    theta = np.asarray(theta)
    d = len(theta)
    h = np.linalg.norm(theta)
    qa, QU = eigh(Q, check_finite=False)
    bound = 0.25*L*L*max(0.0, float(qa[-1]))
    if h <= 1e-13:
        if qa[-1] <= 0:
            x = np.zeros(d)
        else:
            x = L*canonical_eigenvector(qa, QU, smallest=False)
        return x, {"status": "analytic", "score": float(curvature(theta,x)*(x@Q@x)),
                   "global_upper_bound": bound, "scalar_evaluations": 0, "hard_cases": 0}
    u = theta/h
    U = orthogonal_basis(u)
    A = U.T @ Q @ U
    a, E = eigh(A, check_finite=False)
    b = U.T @ Q @ u
    c = float(u@Q@u)
    evaluations = 0
    hard_cases = 0
    cache={}
    topmask=np.abs(a-a[-1])<=2e-13*max(1.,float(np.max(np.abs(a))))
    btop=float(np.linalg.norm(E[:,topmask].T@b))
    def evaluate(t, return_x=False, derivative=False):
        nonlocal evaluations, hard_cases
        t=float(t)
        if t not in cache:
            z=L*t
            radius=L*math.sqrt(max(0.0,1.0-t*t))
            y,info=quadratic_trust_eigen(a,E,z*b,radius)
            p=float(expit(h*z));w=p*(1-p)
            quadratic=float(y@A@y)+2*z*float(b@y)+z*z*c
            score=w*quadratic
            # Envelope theorem: q'(z)=2b'y+2z(c-lambda). This detects
            # arbitrarily near-endpoint maxima that a value-only grid can miss.
            grad=L*w*(h*(1-2*p)*quadratic+2*float(b@y)+2*z*(c-info["multiplier"]))
            if t==0.:
                grad=.5*L*L*btop if a[-1]>0 else 0.
            elif t==1.:
                if np.linalg.norm(b)>1e-15*max(1.,np.linalg.norm(Q)):
                    grad=-math.inf
                else:
                    grad=L*w*(h*(1-2*p)*quadratic+2*z*(c-max(0.,float(a[-1]))))
            cache[t]=(score,z*u+U@y,grad)
            evaluations+=1
            hard_cases+=int(info["hard_case"])
        score,x,grad=cache[t]
        if return_x:return score,x
        if derivative:return grad
        return score
    ts=np.linspace(0,1,max(5,int(grid_size)))
    scores=np.array([evaluate(t) for t in ts])
    gradients=np.array([evaluate(t,derivative=True) for t in ts])
    candidates=[(float(scores[0]),0.),(float(scores[-1]),1.)]
    covered=set()
    for i in range(len(ts)-1):
        if gradients[i]>0 and gradients[i+1]<0:
            try:
                t=brentq(lambda t:evaluate(t,derivative=True),float(ts[i]),float(ts[i+1]),
                         xtol=min(scalar_xtol,2e-12),rtol=1e-12,maxiter=100)
                candidates.append((evaluate(t),float(t)))
                covered.update((i,i+1))
            except (ValueError,RuntimeError):
                pass
    # Value-based local refinements are a backup for stationary multiplicities
    # and nonsmooth switches where no strict derivative sign bracket appears.
    for i in range(1,len(ts)-1):
        if i not in covered and scores[i]>=scores[i-1] and scores[i]>=scores[i+1]:
            sol=minimize_scalar(lambda t:-evaluate(t),bounds=(ts[i-1],ts[i+1]),
                method="bounded",options={"xatol":scalar_xtol,"maxiter":80})
            candidates.append((-float(sol.fun),float(sol.x)))
    # Include all grid values, including narrow endpoint and hard-case peaks.
    best_grid = int(np.argmax(scores))
    candidates.append((float(scores[best_grid]), float(ts[best_grid])))
    _, t = max(candidates, key=lambda p:p[0])
    score, x = evaluate(t, True)
    if score < 0:
        score, x = 0.0, np.zeros(d)
    return x, {"status":"multistart_uncertified", "score":score,
               "global_upper_bound":bound, "scalar_evaluations":evaluations,
               "hard_cases":hard_cases, "outer_grid_size":len(ts)}


def _starts(theta, W, L, rng, previous=None, n_random=2):
    d = len(theta)
    _, U = eigh(W, check_finite=False)
    out = [U[:,0], U[:,1] if d>1 else -U[:,0]]
    h = np.linalg.norm(theta)
    if h > 1e-12:
        u = theta/h
        v = U[:,0] - u*(u@U[:,0])
        if np.linalg.norm(v)>1e-8:
            out.append(v/np.linalg.norm(v))
    if previous is not None and np.linalg.norm(previous)>1e-12:
        out.append(np.asarray(previous)/max(L,np.linalg.norm(previous)))
    for _ in range(n_random):
        v = rng.normal(size=d)
        out.append(v/np.linalg.norm(v))
    return out


def a_opt(W, theta, L, rng, previous=None, n_random=2, maxiter=120):
    """Continuous multistart SLSQP on the exact rational trace reduction."""
    a, U = eigh(W, check_finite=False)
    B = (U/a)@U.T
    A = (U/(a*a))@U.T
    scale = max(1e-300, .25*L*L/(a[0]*a[0]))
    def value_grad(v):
        x = L*v
        p = expit(theta@x); w=p*(1-p); dw=w*(1-2*p)*theta
        ax = A@x; bx=B@x
        aa=float(x@ax); bb=float(x@bx)
        n=w*aa; den=1+w*bb
        grad=((dw*aa+2*w*ax)*den-n*(dw*bb+2*w*bx))/(den*den)
        return n/den, L*grad
    best = (-math.inf, None)
    failures=0
    for v in _starts(theta,W,L,rng,previous,n_random):
        sol=minimize(lambda z:-value_grad(z)[0]/scale,v,
                     jac=lambda z:-value_grad(z)[1]/scale,method="SLSQP",
                     constraints=[{"type":"ineq","fun":lambda z:1-z@z,"jac":lambda z:-2*z}],
                     options={"ftol":2e-12,"maxiter":maxiter})
        failures+=not sol.success
        z=sol.x/max(1.0,np.linalg.norm(sol.x))
        score=value_grad(z)[0]
        initial_score=value_grad(v)[0]
        if initial_score>score: z,score=v,initial_score
        if score>best[0]: best=(score,L*z)
    return best[1], {"status":"multistart_uncertified","score":best[0],
                     "solver_failures":int(failures),"starts":len(_starts(theta,W,L,np.random.default_rng(0),previous,0))+n_random}


def rank_one_minimum(evals, coords, weight):
    """Independent secular minimum of diag(evals)+weight*coords*coords'."""
    a=np.asarray(evals); v=np.asarray(coords)*math.sqrt(max(0.,weight))
    scale=max(1.,float(np.max(np.abs(a))))
    if len(a)==1: return float(a[0]+v[0]*v[0])
    if a[1]-a[0]<=1e-13*scale or abs(v[0])<=1e-14: return float(a[0])
    lo=np.nextafter(a[0], math.inf); hi=np.nextafter(a[1], -math.inf)
    def f(z): return 1+float(np.sum(v*v/(a-z)))
    if hi<=lo: return float(a[0])
    with np.errstate(divide="ignore",over="ignore",invalid="ignore"):
        if f(lo)>=0: return float(a[0])
        if f(hi)<=0: return float(a[1])
        return float(brentq(f,lo,hi,xtol=2e-13*scale,rtol=1e-14))


def e_greedy(W,theta,L,rng,previous=None,n_random=2,primary_rtol=1e-9,maxiter=120):
    """Secular epigraph primary solve followed by constrained D-score tie solve.

    For lambda1<gamma<lambda2, PSD is equivalent to
    w*x1^2 >= (gamma-lambda1)*(1+w*sum_{i>1} xi^2/(lambda_i-gamma)).
    This smooth scalar constraint avoids differentiating a repeated eigenvalue.
    Global optimality of multistart solves is not claimed.
    """
    a,U=eigh(W,check_finite=False); gap=float(a[1]-a[0]); d=len(theta)
    normscale=max(1.,abs(float(a[0])),abs(float(a[1])))
    tolerance=primary_rtol*normscale
    if gap<=1e-11*normscale:
        Q=(U/a)@U.T
        x,diag=logistic_quadratic(Q,theta,L)
        realized=float(np.linalg.eigvalsh(W+float(curvature(theta,x))*np.outer(x,x))[0])
        diag.update(secondary_score=diag["score"],score=realized,primary=realized,primary_tie=True,primary_tolerance=tolerance,
                    secondary="D-opt",allowed_primary_loss=max(0.,gap),multiplicity_gap=max(0.,gap),secular_discrepancy=0.)
        return x,diag
    th=U.T@theta
    # Keep the epigraph below its exact interlacing cap by much less than tolerance.
    cap=1-min(1e-6,max(1e-12,0.01*tolerance/gap))
    gscale=max(1.,L*L*.25,gap)
    def constraint(vt, target=None):
        v=vt[:d]; x=L*v
        t=float(vt[-1]) if target is None else target
        z=gap*t
        den=a[1:]-a[0]-z
        p=expit(th@x); w=p*(1-p); dw=w*(1-2*p)*L*th
        q=float(np.sum(x[1:]**2/den))
        core=x[0]*x[0]-z*q
        value=(w*core-z)/gscale
        grad=dw*core
        grad[0]+=w*2*L*x[0]
        grad[1:]-=w*2*z*L*x[1:]/den
        if target is None:
            dt=-gap*(1+w*q+z*w*float(np.sum(x[1:]**2/den**2)))
            return value,np.r_[grad,dt]/gscale
        return value,grad/gscale
    initial=_starts(theta,W,L,rng,previous,n_random)
    # A D-opt start improves coverage without changing the primary objective.
    xd,_=logistic_quadratic((U/a)@U.T,theta,L)
    initial.append(xd/L)
    best=(-math.inf,None); failures=0
    # Interlacing gives an exact upper bound lambda2. If a feasible direction
    # reaches it, there is no need to numerically optimize the primary objective.
    for vv in initial:
        coord=L*(U.T@vv)
        val=rank_one_minimum(a,coord,float(curvature(th,coord)))
        if val>best[0]:best=(val,coord)
    d_primary=rank_one_minimum(a,U.T@xd,float(curvature(theta,xd)))
    if d_primary>=a[1]-tolerance:
        return xd,{"status":"multistart_uncertified","score":d_primary,"primary":d_primary,
            "primary_tie":False,"primary_tolerance":tolerance,"allowed_primary_loss":tolerance,
            "actual_secondary_primary_loss":float(a[1]-d_primary),"secondary":"D-opt",
            "primary_interlacing_upper_bound":float(a[1]),"secular_discrepancy":abs(d_primary-float(np.linalg.eigvalsh(W+float(curvature(theta,xd))*np.outer(xd,xd))[0])),
            "solver_failures":0,"primary_certified_by_interlacing":True}
    primary_starts=[] if best[0]>=a[1]-0.01*tolerance else initial
    for vv in primary_starts:
        v=U.T@vv
        true=rank_one_minimum(a,L*v,float(curvature(th,L*v)))
        t=max(0.,min(cap,(true-a[0])/gap))
        z0=np.r_[v,t]
        sol=minimize(lambda z:-z[-1],z0,jac=lambda z:np.r_[np.zeros(d),-1.],
            method="SLSQP",bounds=[(-1,1)]*d+[(0,cap)],
            constraints=[{"type":"ineq","fun":lambda z:1-z[:d]@z[:d],"jac":lambda z:np.r_[-2*z[:d],0.]},
                         {"type":"ineq","fun":lambda z:constraint(z)[0],"jac":lambda z:constraint(z)[1]}],
            options={"ftol":2e-11,"maxiter":maxiter})
        failures+=not sol.success
        for cand in (v,sol.x[:d]/max(1.,np.linalg.norm(sol.x[:d]))):
            x=L*cand
            val=rank_one_minimum(a,x,float(curvature(th,x)))
            if val>best[0]: best=(val,x)
    primary,x=best
    target=min(cap,max(0.,(primary-a[0]-tolerance)/gap))
    primary_before=primary
    secondary_loss=0.
    if target>0:
        def dscore(v):
            x=L*v;p=expit(th@x);w=p*(1-p);q=float(np.sum(x*x/a))
            return w*q,L*(w*(1-2*p)*q*th+2*w*x/a)
        scale=max(1e-300,.25*L*L/a[0])
        candidates=[x/L,U.T@(xd/L)]
        score0=dscore(x/L)[0]
        for v in candidates:
            sol=minimize(lambda z:-dscore(z)[0]/scale,v,jac=lambda z:-dscore(z)[1]/scale,
                method="SLSQP",constraints=[{"type":"ineq","fun":lambda z:1-z@z,"jac":lambda z:-2*z},
                    {"type":"ineq","fun":lambda z:constraint(z,target)[0],"jac":lambda z:constraint(z,target)[1]}],
                options={"ftol":2e-12,"maxiter":maxiter})
            failures+=not sol.success
            cand=L*sol.x/max(1.,np.linalg.norm(sol.x))
            actual=rank_one_minimum(a,cand,float(curvature(th,cand)))
            ds=dscore(cand/L)[0]
            if actual>=primary_before-tolerance*(1+1e-5) and ds>score0:
                x=cand;primary=actual;score0=ds
        secondary_loss=max(0.,primary_before-primary)
    xx=U@x
    eigcheck=float(eigh(W+float(curvature(theta,xx))*np.outer(xx,xx),eigvals_only=True,check_finite=False)[0])
    return xx,{"status":"multistart_uncertified","score":eigcheck,"primary":eigcheck,
        "primary_tie":False,"primary_tolerance":tolerance,"allowed_primary_loss":tolerance,
        "actual_secondary_primary_loss":secondary_loss,"secondary":"D-opt",
        "primary_interlacing_upper_bound":float(a[1]),"secular_discrepancy":abs(eigcheck-primary),
        "solver_failures":int(failures)}


def ucb(W,theta,L,beta):
    """Global UCB via max_z ||theta+beta*W^-1/2 z||, ||z||<=1."""
    a,U=eigh(W,check_finite=False)
    Bhalf=(U/np.sqrt(a))@U.T
    A=(U*(beta*beta/a))@U.T
    b=beta*(Bhalf@theta)
    z,diag=quadratic_trust(A,b,1.)
    direction=theta+beta*(Bhalf@z)
    if np.linalg.norm(direction)==0:
        x=minimum_direction(W,L)
    else:
        x=L*direction/np.linalg.norm(direction)
    bx=U.T@x
    score=float(theta@x+beta*math.sqrt(float(np.sum(bx*bx/a))))
    return x,{"status":"analytic","score":score,"hard_case":diag["hard_case"]}


def audit_quadratic(Q,theta,L,rng,n_starts=12,grid_size=97):
    """Independent direct-coordinate SLSQP plus stricter scalar refinement.

    Sample starts are merely starts, never a fixed candidate action set. This is
    an empirical lower-bound comparison, not a proof of global optimality.
    """
    x,diag=logistic_quadratic(Q,theta,L,grid_size=grid_size,scalar_xtol=2e-12)
    best=float(curvature(theta,x)*(x@Q@x))
    a,U=eigh(Q,check_finite=False)
    scale=max(1e-300,.25*L*L*max(abs(a[0]),abs(a[-1])))
    def vg(v):
        z=L*v;p=expit(theta@z);w=p*(1-p);q=float(z@Q@z)
        grad=L*(w*(1-2*p)*q*theta+2*w*Q@z)
        return w*q,grad
    starts=[x/L,U[:,-1]]
    for _ in range(n_starts):
        v=rng.normal(size=len(theta));starts.append(v/np.linalg.norm(v))
    failures=0
    for v in starts:
        sol=minimize(lambda z:-vg(z)[0]/scale,v,jac=lambda z:-vg(z)[1]/scale,method="SLSQP",
            constraints=[{"type":"ineq","fun":lambda z:1-z@z,"jac":lambda z:-2*z}],
            options={"ftol":5e-14,"maxiter":300})
        failures+=not sol.success
        vv=sol.x/max(1.,np.linalg.norm(sol.x));score=vg(vv)[0]
        if score>best:best=score;x=L*vv
    return x,{"status":"multistart_uncertified","score":best,"audit_starts":len(starts),
              "audit_solver_failures":int(failures)}
