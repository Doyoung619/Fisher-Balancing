"""Constrained SUM-loss ridge fits with normalized KKT stopping.

Newton minimizes the quadratic model on the actual radius-four ball, then
backtracks on the actual full-history likelihood. This is not radial clipping.
The KKT norm is divided by max(1,N), including complementarity and feasibility.
"""
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq, minimize


def quadratic_ball(A, b, radius=4.):
    """min 1/2 x'Ax-b'x over ||x||<=radius, SPD A."""
    ev, U = np.linalg.eigh(A)
    c = U.T @ b
    x = U @ (c / ev)
    if np.linalg.norm(x) <= radius:
        return x, 0.
    f = lambda eta: np.linalg.norm(c / (ev + eta)) - radius
    hi = max(1., np.linalg.norm(b) / radius)
    while f(hi) > 0:
        hi *= 2
    eta = brentq(f, 0., hi, xtol=1e-13, rtol=1e-14)
    return U @ (c / (ev + eta)), eta


def kkt(theta, grad, N, radius=4.):
    norm = np.linalg.norm(theta)
    eta = max(0., -float(theta @ grad) / max(norm ** 2, 1e-30)) if norm >= radius - 1e-7 else 0.
    residual = max(np.linalg.norm(grad + eta * theta) / max(1, N),
                   max(0., norm - radius), eta * abs(norm-radius) / max(1, N))
    return float(residual), float(eta)


def fit_gaussian(C, b, rho=1., radius=4., N=1):
    A = C + rho * np.eye(len(b))
    theta, eta = quadratic_ball(A, b, radius)
    residual, _ = kkt(theta, A @ theta - b, N, radius)
    return theta, A, dict(kkt=residual, iterations=1, boundary=bool(eta > 0), success=residual <= 1e-8)


def logistic_objective(theta, X, y, rho=1., hessian=True):
    z = X @ theta
    p = expit(z)
    value = np.logaddexp(0., z).sum() - y @ z + .5 * rho * (theta @ theta)
    grad = X.T @ (p-y) + rho * theta
    H = X.T @ ((p*(1-p))[:, None] * X) + rho * np.eye(X.shape[1]) if hessian else None
    return float(value), grad, H


def fit_bernoulli(X, y, initial=None, rho=1., radius=4., tol=1e-8, max_iter=80):
    theta = np.zeros(X.shape[1]) if initial is None else np.array(initial, copy=True)
    if len(y) == 0:
        return theta * 0, rho*np.eye(len(theta)), dict(kkt=0.,iterations=0,boundary=False,success=True)
    fallback = False
    for iteration in range(max_iter):
        value, grad, H = logistic_objective(theta,X,y,rho)
        residual, eta = kkt(theta,grad,len(y),radius)
        if residual <= tol:
            return theta,H,dict(kkt=residual,iterations=iteration,boundary=bool(eta>0),success=True,fallback=fallback)
        candidate, _ = quadratic_ball(H, H @ theta - grad, radius)
        step = candidate-theta
        descent = float(grad @ step)
        alpha = 1.
        accepted = False
        for _ in range(45):
            trial = theta + alpha*step
            trial_value, _, _ = logistic_objective(trial,X,y,rho,False)
            # Absolute floating point allowance only; stopping still requires KKT.
            if trial_value <= value + 1e-4*alpha*descent + 2e-14*max(1,abs(value)):
                theta = trial
                accepted = True
                break
            alpha *= .5
        if not accepted or np.linalg.norm(alpha*step) < 1e-14:
            break
    # Independent constraint-aware fallback, followed by KKT validation.
    fallback = True
    opt = minimize(lambda t: logistic_objective(t,X,y,rho,False)[:2],theta,jac=True,
                   method='SLSQP',constraints={'type':'ineq','fun':lambda t:radius**2-t@t,'jac':lambda t:-2*t},
                   options={'ftol':1e-12,'maxiter':300})
    theta = opt.x
    value, grad, H = logistic_objective(theta,X,y,rho)
    residual, eta = kkt(theta,grad,len(y),radius)
    # Small stationarity refinement via Newton from SLSQP solution.
    for j in range(12):
        if residual <= tol:
            break
        theta,_ = quadratic_ball(H,H@theta-grad,radius)
        value,grad,H = logistic_objective(theta,X,y,rho)
        residual,eta = kkt(theta,grad,len(y),radius)
    return theta,H,dict(kkt=residual,iterations=iteration+1,boundary=bool(eta>0),success=residual<=tol,fallback=fallback)
