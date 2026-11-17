"""Batched float64 cumulative ridge MLEs on the radius-four parameter ball.

The constrained quadratic subproblem is solved through its secular multiplier;
radial clipping is never substituted for a constrained optimum. All logistic
Hessians use the terminal parameter on the entire cumulative query history.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def quadratic_ball(A, b, radius=4.0):
    """Batched SPD quadratic minimizer and nonnegative ball multiplier."""
    solution = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    boundary = torch.linalg.vector_norm(solution, dim=-1) > radius
    eta = torch.zeros_like(boundary, dtype=A.dtype)
    if not bool(boundary.any()):
        return solution, eta
    ev, U = torch.linalg.eigh(A[boundary])
    c = torch.matmul(U.transpose(-1, -2), b[boundary].unsqueeze(-1)).squeeze(-1)
    lo = torch.zeros_like(ev[:, 0])
    hi = torch.linalg.vector_norm(c, dim=-1) / radius
    # Forty-eight iterations give substantially better than the KKT tolerance.
    for _ in range(48):
        mid = (lo + hi) * .5
        outside = ((c / (ev + mid[:, None])).square().sum(-1) > radius**2)
        lo = torch.where(outside, mid, lo)
        hi = torch.where(outside, hi, mid)
    eta[boundary] = hi
    solution[boundary] = torch.matmul(U, (c / (ev + hi[:, None])).unsqueeze(-1)).squeeze(-1)
    return solution, eta


def kkt(theta, grad, N, radius=4.):
    norm = torch.linalg.vector_norm(theta, dim=-1)
    eta = torch.clamp(-(theta * grad).sum(-1) / norm.square().clamp_min(1e-30), min=0)
    eta = torch.where(norm >= radius - 1e-7, eta, torch.zeros_like(eta))
    stationarity = torch.linalg.vector_norm(grad + eta[:, None] * theta, dim=-1) / max(1, N)
    feasibility = (norm - radius).clamp_min(0)
    complementarity = eta * (norm - radius).abs() / max(1, N)
    return torch.maximum(stationarity, torch.maximum(feasibility, complementarity)), eta


def fit_gaussian(C, b, rho=1., radius=4., N=1):
    """C may have one shared matrix or one matrix per response replicate."""
    eye = torch.eye(b.shape[-1], dtype=b.dtype, device=b.device)
    A = C + rho * eye
    if A.shape[0] == 1 and b.shape[0] > 1:
        # Factor the common Gaussian design once for all response RHS columns.
        theta = torch.linalg.solve(A[0], b.T).T
        outside = torch.linalg.vector_norm(theta, dim=-1) > radius
        if bool(outside.any()):
            theta[outside], _ = quadratic_ball(A.expand(int(outside.sum()), -1, -1), b[outside], radius)
        A = A.expand(b.shape[0], -1, -1)
    else:
        theta, _ = quadratic_ball(A, b, radius)
    grad = torch.matmul(A, theta.unsqueeze(-1)).squeeze(-1) - b
    residual, eta = kkt(theta, grad, N, radius)
    worst = float(residual.max())
    return theta, A, dict(kkt=worst, iterations=1, boundary=int((eta > 0).sum()), success=worst <= 1e-8)


def logistic_objective(theta, X, b, rho=1., hessian=True):
    z = torch.matmul(X, theta.unsqueeze(-1)).squeeze(-1)
    p = torch.sigmoid(z)
    value = F.softplus(z).sum(-1) - (b * theta).sum(-1) + .5 * rho * theta.square().sum(-1)
    grad = torch.matmul(X.transpose(-1, -2), p.unsqueeze(-1)).squeeze(-1) - b + rho * theta
    H = None
    if hessian:
        w = p * (1 - p)
        H = torch.matmul(X.transpose(-1, -2), w.unsqueeze(-1) * X)
        H = H + rho * torch.eye(theta.shape[-1], dtype=theta.dtype, device=theta.device)
    return value, grad, H


def fit_bernoulli(X, y=None, initial=None, rho=1., radius=4., tol=1e-8, max_iter=40, b=None):
    """Warm-start damped Newton with exact ball-constrained quadratic steps."""
    R, N, d = X.shape
    theta = torch.zeros((R, d), dtype=X.dtype, device=X.device) if initial is None else initial.clone()
    if b is None:
        b = torch.matmul(X.transpose(-1, -2), y.unsqueeze(-1)).squeeze(-1)
    iterations = 0
    for iteration in range(max_iter + 1):
        value, grad, H = logistic_objective(theta, X, b, rho)
        residual, eta = kkt(theta, grad, N, radius)
        converged = residual <= tol
        if bool(converged.all()):
            return theta, H, dict(kkt=float(residual.max()), iterations=iterations,
                                 boundary=int((eta > 0).sum()), success=True)
        if iteration == max_iter:
            break
        candidate, _ = quadratic_ball(H, torch.matmul(H, theta.unsqueeze(-1)).squeeze(-1) - grad, radius)
        step = candidate - theta
        step[converged] = 0
        descent = (grad * step).sum(-1)
        alpha = torch.ones(R, dtype=X.dtype, device=X.device)
        # Backtracking is independent for each replicate, without Python loops over R.
        for _ in range(30):
            trial = theta + alpha[:, None] * step
            z = torch.matmul(X, trial.unsqueeze(-1)).squeeze(-1)
            trial_value = F.softplus(z).sum(-1) - (b * trial).sum(-1) + .5 * rho * trial.square().sum(-1)
            accepted = trial_value <= value + 1e-4 * alpha * descent + 2e-14 * value.abs().clamp_min(1)
            if bool(accepted.all()):
                break
            alpha = torch.where(accepted, alpha, .5 * alpha)
        theta = theta + alpha[:, None] * step
        iterations += 1
    return theta, H, dict(kkt=float(residual.max()), iterations=iterations,
                         boundary=int((eta > 0).sum()), success=False)
