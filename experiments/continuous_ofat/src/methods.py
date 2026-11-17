"""The eight prescribed batched policies, with frozen estimates and no labels."""
from __future__ import annotations
import math
import time
import warnings
import numpy as np
from scipy.linalg import eigh, svd
from scipy.special import expit
from .oracles import (curvature, minimum_direction, logistic_quadratic, a_opt,
                      e_greedy, ucb, canonical_eigenvector, audit_quadratic, orthogonal_basis)

METHOD_IDS = {
    "Ours":"FB-range-anchored-continuous", "A-opt":"A-opt-exact-reduction",
    "D-opt":"D-opt-adaptive-Wynn-batched", "E-greedy":"E-greedy-exact-primary",
    "E-relax":"E-relax-continuous-support-generation", "UCB":"UCB-ellipsoid-batched",
    "TS":"TS-GaussianPerturbation-batched", "Random":"Random-volume-uniform-ball"}
_SDP_CACHE={}


def temperature(family,d,L,m,batch_index,N_cal=400):
    if batch_index<1: raise ValueError("batch_index is one-indexed")
    R=L*L*(1. if family=="gaussian" else .25)
    tau_range=R*math.sqrt(m/(8*math.log(d)))
    tau0=tau_range/N_cal**.75
    return tau0*(m*batch_index)**.75,tau0,tau_range


def systematic_round(support,weights,m,rng):
    """One uniform offset gives exactly m draws from the weighted measure."""
    weights=np.maximum(np.asarray(weights),0.)
    weights=weights/np.sum(weights)
    cs=np.cumsum(weights);cs[-1]=1.
    positions=(np.arange(m)+rng.uniform())/m
    inds=np.searchsorted(cs,positions,side="right")
    return np.asarray(support)[inds],inds


def waterfill(W,L,m,weight=1.):
    a,U=eigh(W,check_finite=False)
    budget=m*weight*L*L
    # The unique water level satisfies sum_i(level-a_i)_+=budget.
    cumulative=np.cumsum(a)
    level=0.
    for k in range(1,len(a)+1):
        level=(budget+cumulative[k-1])/k
        if k==len(a) or level<=a[k]: break
    p=np.maximum(level-a,0.)/budget
    p/=p.sum()
    return L*U.T,p,float(level)


def _sdp_template(d,k):
    key=(d,k)
    if key not in _SDP_CACHE:
        import cvxpy as cp
        p=cp.Variable(k,nonneg=True)
        gamma=cp.Variable()
        Wp=cp.Parameter((d,d),symmetric=True)
        Gp=cp.Parameter((d*d,k))
        inequality=Wp+cp.reshape(Gp@p,(d,d),order="F")-gamma*np.eye(d) >> 0
        problem=cp.Problem(cp.Maximize(gamma),[cp.sum(p)==1,inequality])
        _SDP_CACHE[key]=(problem,p,gamma,Wp,Gp,inequality)
    return _SDP_CACHE[key]


def _solve_support(W,theta,L,m,support,capacity,solver_tol):
    """A finite SDP lower bound; matrices evaluated at the current theta."""
    d=len(theta);n=len(support)
    capacity=min(capacity,int(math.ceil(n/d)*d))
    direct_solver="faer" if capacity<=60 else "qdldl"
    R=.25*L*L
    baseline=float(eigh(W,eigvals_only=True,check_finite=False)[0])
    G=curvature(theta,support)[:,None,None]*support[:,:,None]*support[:,None,:]
    # Pad using an actual existing support point: no hidden zero query is added.
    if n<capacity:
        padded=np.concatenate([G,np.repeat(G[:1],capacity-n,axis=0)])
    else: padded=G
    problem,p,gamma,Wp,Gp,ineq=_sdp_template(d,capacity)
    Wp.value=(W-baseline*np.eye(d))/(m*R)
    Gp.value=np.array([g.reshape(-1,order="F")/R for g in padded]).T
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem.solve(solver="CLARABEL",warm_start=False,verbose=False,
            tol_gap_abs=solver_tol,tol_gap_rel=solver_tol,tol_feas=solver_tol,max_iter=150,max_threads=1,direct_solve_method=direct_solver)
    if p.value is None or ineq.dual_value is None:
        raise RuntimeError(f"E-relax SDP returned no primal/dual solution: {problem.status}")
    raw=np.asarray(p.value).copy()
    if n<capacity: raw[0]+=raw[n:].sum()
    weights=np.maximum(raw[:n],0.)
    weights/=weights.sum()
    # Every returned lower bound is recomputed from an exactly feasible measure.
    M=W+m*np.einsum("i,ijk->jk",weights,G)
    lower=float(eigh(M,eigvals_only=True,check_finite=False)[0])
    P=np.asarray(ineq.dual_value)
    ev,U=eigh((P+P.T)*.5,check_finite=False)
    P=(U*np.maximum(ev,0.))@U.T
    P/=np.trace(P)
    return weights,lower,P,{"sdp_status":problem.status,"sdp_warnings":len(caught),
        "restricted_primal":lower,"raw_weight_minimum":float(np.min(raw)),
        "sdp_iterations":int(problem.solver_stats.num_iters or 0),"sdp_capacity":capacity,"sdp_direct_solver":direct_solver}



def affine_support_exchange(support,weights,new_point,theta,L):
    """Insert a point and delete one old point preserving the current measure.

    There are D=d(d+1)/2 symmetric information coordinates plus unit total
    mass. At n>=D+1 old points, n+1 augmented columns have an affine null
    direction. Move along it until an old probability is zero. This preserves
    actual current-curvature information, not just the SDP objective.
    """
    support=np.asarray(support);weights=np.asarray(weights);d=support.shape[1]
    X=np.vstack([support,new_point]);p=np.r_[weights,0.]
    G=curvature(theta,X)[:,None,None]*X[:,:,None]*X[:,None,:]
    rr,cc=np.triu_indices(d)
    packed=G[:,rr,cc].T/max(.25*L*L,1e-30)
    packed[rr!=cc]*=math.sqrt(2.)
    A=np.vstack([packed,np.ones(len(X))])
    if A.shape[1]<=A.shape[0]:
        raise ValueError("Affine exchange needs support at the Caratheodory bound")
    _,_,Vh=svd(A,full_matrices=True,check_finite=False,lapack_driver="gesvd")
    null=Vh[A.shape[0]:]
    v=null.T@null[:,-1]
    if np.linalg.norm(v)<1e-13:
        v=null[0].copy()
        if v[-1]<0:v=-v
    v/=np.linalg.norm(v)
    negative=np.where(v[:-1]<-1e-13)[0]
    if not len(negative):
        v=-v;negative=np.where(v[:-1]<-1e-13)[0]
    if not len(negative):raise ArithmeticError("No feasible affine support exchange direction")
    ratios=p[negative]/(-v[negative])
    drop=int(negative[int(np.argmin(ratios))])
    step=float(np.min(ratios))
    updated=p+step*v
    updated[drop]=0.
    if np.min(updated)<-1e-10:
        raise ArithmeticError("Affine support exchange violated probabilities")
    updated=np.maximum(updated,0.)
    before=np.einsum("i,ijk->jk",p,G)
    after=np.einsum("i,ijk->jk",updated,G)
    residual=max(float(np.linalg.norm(after-before,ord=2)),abs(float(updated.sum()-1.)))
    if residual>1e-9*max(1.,float(np.linalg.norm(before,ord=2))):
        raise ArithmeticError(f"Affine support exchange residual {residual:g}")
    updated/=updated.sum()
    return np.delete(X,drop,axis=0),np.delete(updated,drop),residual

def e_relax(W,theta,L,m,rng,state,settings):
    """Continuous support generation with restricted SDP and PSD separation."""
    d=len(theta)
    if np.linalg.norm(theta)<=1e-13:
        support,weights,lower=waterfill(W,L,m,.25)
        X,idx=systematic_round(support,weights,m,rng)
        rounded=W+.25*(X.T@X)
        state["support"]=support
        state["weights"]=weights
        return X,state,{"status":"analytic","sdp_status":"analytic_constant_curvature",
            "relaxed_objective":lower,"rounded_objective":float(np.linalg.eigvalsh(rounded)[0]),
            "rounding_loss":lower-float(np.linalg.eigvalsh(rounded)[0]),"weights_sum":float(weights.sum()),
            "weight_minimum":float(weights.min()),"support_size":len(weights),"generation_iterations":0,
            "empirical_gap":0.,"certified_gap":0.,"capped":False}
    tolerance=float(settings.get("e_relax_relative_tolerance",1e-4))
    max_generations=int(settings.get("e_relax_max_generations",settings.get("e_relax_max_rounds",256)))
    capacity=int(settings.get("e_relax_support_capacity",max(4*d+24,d*(d+1)//2+1)))
    solver_tol=float(settings.get("e_relax_sdp_tolerance",1e-8))
    grid_size=int(settings.get("oracle_scalar_grid",17))
    _,U=eigh(W,check_finite=False)
    old=np.asarray(state.get("support",np.empty((0,d))))
    oldweights=np.asarray(state.get("weights",np.ones(len(old))))
    old_count=len(old)
    if len(old):
        order=np.argsort(-oldweights,kind="stable")
        order=order[oldweights[order]>=1e-8]
        old=old[order]
    direction=theta/np.linalg.norm(theta)
    transverse=orthogonal_basis(direction)
    # Current-estimate latitude directions are only initial supports. The
    # objective remains the continuous full-ball SDP with fresh separation.
    cone=np.vstack([L/math.sqrt(d)*direction+L*math.sqrt(1-1/d)*transverse.T,
                    L/math.sqrt(d)*direction-L*math.sqrt(1-1/d)*transverse.T])
    fresh=np.vstack([L*U.T,cone])[:capacity]
    old=old[:max(0,capacity-len(fresh)-1)]
    support=np.concatenate([old,fresh])
    initial_warm_dropped=old_count-len(old)
    # Removing numerical duplicates is only an SDP support simplification;
    # history queries/likelihood are never merged using a distance threshold.
    keep=[]
    for x in support:
        if not any(min(np.linalg.norm(x-y),np.linalg.norm(x+y))<1e-11*max(1.,L) for y in keep):
            keep.append(x)
    support=np.asarray(keep)
    capped=True;gap=math.inf;certgap=math.inf;info={}
    lower=math.nan
    maximum_audit_discrepancy=0.
    audit_refinements=0
    affine_exchanges=0
    max_affine_residual=0.
    for iteration in range(max_generations):
        weights,lower,P,info=_solve_support(W,theta,L,m,support,capacity,solver_tol)
        x,sep=logistic_quadratic(P,theta,L,grid_size=grid_size)
        if settings.get("audit_now",False):
            audit_rng=np.random.default_rng(np.random.SeedSequence([20261005,d,int(settings.get("audit_batch_index",0)),iteration,729]))
            xa,sa=audit_quadratic(P,theta,L,audit_rng,n_starts=6,grid_size=97)
            discrepancy=max(0.,sa["score"]-sep["score"])/max(1e-14,abs(sa["score"]))
            maximum_audit_discrepancy=max(maximum_audit_discrepancy,discrepancy)
            if sa["score"]>sep["score"]:
                x=xa;sep["score"]=sa["score"]
                audit_refinements+=1
            sep["audit_discrepancy_before_refinement"]=discrepancy
        local_upper_like=float(np.trace(P@W)+m*sep["score"])
        # This coarse bound is valid globally even though the tighter residual is empirical.
        true_upper=float(np.trace(P@W)+m*sep["global_upper_bound"])
        gap=max(0.,local_upper_like-lower)
        certgap=max(0.,true_upper-lower)
        scale=max(1.,abs(lower))
        if gap<=tolerance*scale:
            capped=False
            break
        if iteration+1>=max_generations:
            break
        if np.min(np.minimum(np.linalg.norm(support-x,axis=1),np.linalg.norm(support+x,axis=1)))<1e-11*max(1.,L):
            break
        if len(support)>=capacity:
            if capacity>=d*(d+1)//2+1:
                support,weights,residual=affine_support_exchange(support,weights,x,theta,L)
                affine_exchanges+=1
                max_affine_residual=max(max_affine_residual,residual)
            else:
                # Explicitly tiny audit capacities may be below the ambient
                # Caratheodory bound; only a negligible-weight swap is allowed.
                removable=np.where(weights<1e-8)[0]
                if not len(removable):break
                drop=int(removable[np.argmin(weights[removable])])
                support=np.delete(support,drop,axis=0)
                support=np.vstack([support,x])
        else:
            support=np.vstack([support,x])
    if len(support)!=len(weights):
        raise ArithmeticError("E-relax support/weight continuation mismatch")
    state["support"]=support
    state["weights"]=weights
    X,idx=systematic_round(support,weights,m,rng)
    realized=W+np.einsum("i,ij,ik->jk",curvature(theta,X),X,X)
    rounded=float(eigh(realized,eigvals_only=True,check_finite=False)[0])
    diag={"status":"multistart_uncertified",**info,"relaxed_objective":lower,
        "rounded_objective":rounded,"rounding_loss":lower-rounded,"weights_sum":float(weights.sum()),
        "weight_minimum":float(weights.min()),"support_size":len(support),
        "generation_iterations":iteration+1,"empirical_gap":gap,"certified_gap":certgap,
        "generation_tolerance":tolerance,"capped":capped,
        "global_upper_bound":lower+certgap,"separation_status":sep["status"],
        "audit_discrepancy_before_refinement":maximum_audit_discrepancy,
        "audit_refinements":audit_refinements,"affine_exchanges":affine_exchanges,
        "max_affine_information_residual":max_affine_residual,"support_capacity":capacity,
        "initial_warm_support_dropped":initial_warm_dropped}
    return X,state,diag


def select_batch(method,family,theta,Vhat,H,condition,batch_index,rngs,state=None,config=None):
    """Select a batch with the observed-history estimate frozen throughout.

    batch_index is ONE INDEXED. No theta_star, reference matrix, labels, or
    response RNG are accepted. Returned state contains only policy continuations.
    """
    if method not in METHOD_IDS:
        reverse={v:k for k,v in METHOD_IDS.items()}
        method=reverse.get(method,method)
    if method not in METHOD_IDS: raise ValueError(f"Unknown method {method}")
    config={} if config is None else config
    settings={**config,**config.get("numerical",{}),**config.get("oracles",{})}
    state={} if state is None else state
    interval=int(settings.get("oracle_audit_interval_batches",200))
    audit_now=(batch_index==1 or (interval>0 and batch_index%interval==0))
    settings["audit_now"]=audit_now
    settings["audit_batch_index"]=batch_index
    theta=np.asarray(theta);W=np.array(Vhat,copy=True)
    d=int(condition["d"]);L=float(condition["L"]);m=int(condition["m"])
    diag={"method_id":METHOD_IDS[method],"batch_index":int(batch_index),"frozen_estimate":True}
    if method=="Random":
        Z=rngs["random"].normal(size=(m,d))
        X=L*rngs["random"].uniform(size=m)[:,None]**(1/d)*Z/np.linalg.norm(Z,axis=1)[:,None]
        diag.update(status="analytic",distribution="uniform_ball_volume")
    elif method=="TS":
        a,U=eigh(W,check_finite=False)
        perturb=rngs["ts"].normal(size=(m,d))@(U/np.sqrt(a)).T
        draws=theta+perturb
        norms=np.linalg.norm(draws,axis=1)
        for i in np.where(norms==0)[0]:
            draws[i]=rngs["tie"].normal(size=d);norms[i]=np.linalg.norm(draws[i])
        X=L*draws/norms[:,None]
        diag.update(status="analytic",frozen_covariance=True)
    elif method=="UCB":
        rho=float(settings.get("rho",1.));delta=float(settings.get("ucb_delta",.05))
        sigma=1. if family=="gaussian" else .5
        a0=1. if family=="gaussian" else float(expit(4*L)*expit(-4*L))
        sign,ld=np.linalg.slogdet(H)
        if sign<=0: raise ArithmeticError("UCB H not positive definite")
        beta=(sigma*math.sqrt(max(0.,ld-d*math.log(rho)+2*math.log(2/delta)))+math.sqrt(rho)*4)/a0
        x,detail=ucb(H,theta,L,beta)
        X=np.repeat(x[None,:],m,axis=0)
        diag.update(detail,beta=beta,identical_queries=m)
    elif method=="E-relax":
        if family=="gaussian":
            support,weights,lower=waterfill(W,L,m,1.)
            X,_=systematic_round(support,weights,m,rngs["rounding"])
            rounded=float(eigh(W+X.T@X,eigvals_only=True,check_finite=False)[0])
            state.update(support=support,weights=weights)
            diag.update(status="analytic",relaxed_objective=lower,rounded_objective=rounded,
                rounding_loss=lower-rounded,weights_sum=float(weights.sum()),weight_minimum=float(weights.min()),
                support_size=d,generation_iterations=0,empirical_gap=0.,certified_gap=0.,capped=False)
        else:
            X,state,details=e_relax(W,theta,L,m,rngs["rounding"],state,settings)
            diag.update(details)
    else:
        if method=="Ours":
            tau,tau0,trange=temperature(family,d,L,m,batch_index,int(settings.get("N_cal",400)))
            diag.update(tau=tau,tau0=tau0,tau_range=trange)
        X=np.empty((m,d));details=[]
        for j in range(m):
            if family=="gaussian" or np.linalg.norm(theta)<=1e-13:
                x=minimum_direction(W,L)
                item={"status":"analytic","common_minimum_eigenvector":True}
                if method=="E-greedy":
                    vals=eigh(W,eigvals_only=True,check_finite=False)
                    item["primary_tie"]=bool(vals[1]-vals[0]<=1e-11*max(1.,abs(vals[0])))
            elif method in ("Ours","D-opt"):
                a,U=eigh(W,check_finite=False)
                if method=="Ours":
                    q=np.exp(-(a-a[0])/tau);q/=q.sum()
                else:q=1/a
                Q=(U*q)@U.T
                x,item=logistic_quadratic(Q,theta,L,grid_size=int(settings.get("oracle_scalar_grid",17)))
            elif method=="A-opt":
                x,item=a_opt(W,theta,L,rngs["solver"],state.get("last_x"),
                    n_random=int(settings.get("oracle_random_starts",max(0,int(settings.get("oracle_starts",4))-2))),maxiter=int(settings.get("oracle_maxiter",120)))
            else:
                x,item=e_greedy(W,theta,L,rngs["solver"],state.get("last_x"),
                    n_random=int(settings.get("oracle_random_starts",max(0,int(settings.get("oracle_starts",4))-2))),
                    primary_rtol=float(settings.get("e_greedy_primary_rtol",1e-9)),
                    maxiter=int(settings.get("oracle_maxiter",120)))
            if audit_now and family=="bernoulli" and item["status"]!="analytic":
                audit_rng=np.random.default_rng(np.random.SeedSequence([20261005,d,batch_index,j,729]))
                original=float(item.get("score",item.get("primary",0.)))
                if method in ("Ours","D-opt"):
                    xa,ia=audit_quadratic(Q,theta,L,audit_rng,n_starts=8,grid_size=97)
                elif method=="A-opt":
                    xa,ia=a_opt(W,theta,L,audit_rng,x,n_random=10,maxiter=300)
                elif item.get("primary_tie",False):
                    wa,wU=eigh(W,check_finite=False)
                    tieQ=(wU/wa)@wU.T
                    xa,ia=audit_quadratic(tieQ,theta,L,audit_rng,n_starts=8,grid_size=97)
                    original=float(item.get("secondary_score",curvature(theta,x)*(x@tieQ@x)))
                    item["audited_objective"]="D-opt_secondary_on_flat_primary"
                else:
                    xa,ia=e_greedy(W,theta,L,audit_rng,x,n_random=10,primary_rtol=2e-10,maxiter=300)
                audited=float(ia.get("score",ia.get("primary",original)))
                discrepancy=max(0.,audited-original)/max(1e-14,abs(audited))
                item["audit_discrepancy_before_refinement"]=discrepancy
                item["audit_refined"]=bool(audited>original)
                if audited>original:
                    x=xa
                    item["score"]=audited
                item["audit_status"]="empirical_multistart_no_global_certificate"
            X[j]=x
            W+=float(curvature(theta,x,family))*np.outer(x,x)
            state["last_x"]=x
            details.append(item)
        diag.update(status="analytic" if all(q["status"]=="analytic" for q in details) else "multistart_uncertified",
            oracle_calls=m,solver_failures=sum(q.get("solver_failures",0) for q in details),
            primary_ties=sum(q.get("primary_tie",False) for q in details),
            max_secular_discrepancy=max(q.get("secular_discrepancy",0.) for q in details),
            max_allowed_primary_loss=max(q.get("allowed_primary_loss",0.) for q in details),
            scalar_evaluations=sum(q.get("scalar_evaluations",0) for q in details),
            audit_performed=bool(audit_now and family=="bernoulli"),
            audit_max_discrepancy_before_refinement=max(q.get("audit_discrepancy_before_refinement",0.) for q in details),
            audit_refinements=sum(q.get("audit_refined",False) for q in details))
        if settings.get("record_full_oracle_diagnostics",False):diag["queries"]=details
    violation=float(np.max(np.linalg.norm(X,axis=1))-L)
    diag["feasibility_error"]=max(0.,violation)
    if violation>1e-9*max(1.,L):
        raise ArithmeticError(f"Acquisition infeasibility {violation:g}")
    return X,state,diag
