"""Exact full-ball Gaussian extremizers, shared across repetitions and m.

Starting from rho I, each canonical minimum-eigenspace tie selects the first
least-used coordinate axis. The four sequential design objectives therefore
produce the same cyclic radius-L design for every positive temperature and m.
This is an analytic continuous-ball solution, never a restricted candidate pool.
"""
import numpy as np

TIE_RULE='first-coordinate-projection-into-minimum-eigenspace-v1'
_CACHE={}


def gaussian_design(d,L,N):
    key=(int(d),float(L),TIE_RULE,'no-policy-randomness')
    if key not in _CACHE or len(_CACHE[key])<N:
        X=L*np.eye(d)[np.arange(N)%d]
        X.flags.writeable=False
        _CACHE[key]=X
    return _CACHE[key],key
