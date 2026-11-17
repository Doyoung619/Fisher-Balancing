"""Canonical models; evaluator truth is never an acquisition input."""
import numpy as np
from scipy.special import expit


def weight(family, eta):
    if family == 'gaussian':
        return np.ones_like(eta, dtype=np.float64)
    p = expit(eta)
    return p * (1 - p)


def information(family, X, theta, rho=1.):
    return rho * np.eye(len(theta)) + X.T @ (weight(family, X @ theta)[:, None] * X)


def responses(family, X, theta_star, innovations):
    eta = X @ theta_star
    return eta + innovations if family == 'gaussian' else (innovations < expit(eta)).astype(float)
