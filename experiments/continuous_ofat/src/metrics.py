"""Monte Carlo manuscript metrics; covariance is intentionally NOT centered."""
from __future__ import annotations
import numpy as np


def sem(x,axis=0):
    x=np.asarray(x)
    if x.shape[axis]<2:
        return np.full(np.delete(x.shape,axis),np.nan)
    return np.std(x,axis=axis,ddof=1)/np.sqrt(x.shape[axis])


def mc_directional_stability(errors, gamma, N, bootstrap=128, seed=8675309):
    """Endpoint blocks and descriptive full-statistic resampling, no coverage claim."""
    errors=np.asarray(errors,float)
    R,d=errors.shape
    rows=[]
    for size in [s for s in [8,16,32,64,128,256,512,1024] if s<=R]:
        vals=[]
        for start in range(0,R-size+1,size):
            block=errors[start:start+size]
            vals.append(float(N*gamma*np.linalg.eigvalsh(block.T@block/size)[-1]))
        prefix=errors[:size]
        rows.append({'replicates':size,'disjoint_blocks':len(vals),
                     'prefix_ratio':float(N*gamma*np.linalg.eigvalsh(prefix.T@prefix/size)[-1]),
                     'block_ratios':vals,'block_mean':float(np.mean(vals)),
                     'block_min':float(min(vals)),'block_max':float(max(vals))})
    rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(bootstrap if R>=2 else 0):
        resample=errors[rng.integers(0,R,R)]
        vals.append(float(N*gamma*np.linalg.eigvalsh(resample.T@resample/R)[-1]))
    return {'replicates':R,'dimension':d,'N':int(N),'blocks':rows,
            'descriptive_bootstrap_quantiles_025_975':np.quantile(vals,[.025,.975]).tolist() if vals else [None,None],
            'bootstrap_repetitions':len(vals),
            'limitation':'Directional maximum eigenvalue is nonlinear and commonly upward biased near tied population eigenspaces. Bands are descriptive Monte Carlo resampling bands, not guaranteed 95% confidence intervals. Larger N alone does not remove finite-repetition error.'}


def compute_metrics(errors,Vstar,Vhat,N,gamma_star,trace_inv_star,rho=1.,bootstrap=128):
    errors,Vstar,Vhat=np.asarray(errors,float),np.asarray(Vstar,float),np.asarray(Vhat,float)
    N=np.asarray(N,int)
    if errors.ndim!=3 or Vstar.shape!=Vhat.shape or Vstar.shape[:3]!=errors.shape:
        raise ValueError('Expected errors [rep,checkpoint,d], Vstar/Vhat [rep,checkpoint,d,d]')
    if len(N)!=errors.shape[1] or not len(N):
        raise ValueError('Checkpoint counts mismatch')
    R,K,d=errors.shape
    Q=np.einsum('rki,rkj->kij',errors,errors)/R
    eigenQ=np.linalg.eigvalsh(Q)
    spectra_star=np.linalg.eigvalsh(Vstar)
    spectra_hat=np.linalg.eigvalsh(Vhat)
    lstar,lhat=spectra_star[...,0],spectra_hat[...,0]
    mse=np.einsum('rki,rki->rk',errors,errors)
    positive=np.where(N>0,N,np.nan)
    rdir2=eigenQ[:,-1]
    rtwo2=np.trace(Q,axis1=-2,axis2=-1)
    tracking=np.max(np.abs(np.linalg.eigvalsh(Vhat-Vstar)),axis=-1)/positive
    results={'N':N,'replicates':np.full(K,R),'Q_hat':Q,'mean_error':errors.mean(0),
             'Rdir_squared':rdir2,'R2_squared':rtwo2,'Rdir':np.sqrt(rdir2),'R2':np.sqrt(rtwo2),
             'true_information':lstar.mean(0)/positive,'plugin_information':lhat.mean(0)/positive,
             'true_information_sem':sem(lstar)/positive,'plugin_information_sem':sem(lhat)/positive,
             'directional_ratio':N*gamma_star*rdir2,'scaled_R2':N*rtwo2,
             'scaled_R2_sem':N*sem(mse),'R2_squared_sem':sem(mse),
             'true_information_normalized':lstar.mean(0)/positive/gamma_star,
             'plugin_information_normalized':lhat.mean(0)/positive/gamma_star,
             'R2_normalized':N*rtwo2/trace_inv_star,
             'true_information_ridge_removed':(lstar.mean(0)-rho)/positive,
             'plugin_information_ridge_removed':(lhat.mean(0)-rho)/positive,
             'true_information_ceiling':gamma_star+rho/positive,
             'tracking_op_per_N':tracking.mean(0),'tracking_op_per_N_sem':sem(tracking),
             'mean_true_information_spectrum':spectra_star.mean(0)/positive[:,None],
             'mean_plugin_information_spectrum':spectra_hat.mean(0)/positive[:,None],
             'per_replicate_squared_error':mse,
             'per_replicate_true_lambda_min':lstar,'per_replicate_plugin_lambda_min':lhat}
    stability=mc_directional_stability(errors[:,-1],gamma_star,N[-1],bootstrap=bootstrap)
    return results,stability
