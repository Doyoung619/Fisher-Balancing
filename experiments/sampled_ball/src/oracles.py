"""Fresh Monte-Carlo approximations of continuous-ball acquisition oracles.

All estimator and information state is float64. Replicates are batched, clouds
are shared, and no continuous acquisition optimization is performed here.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch

METHODS = ("FB", "AOpt", "DOpt", "EGreedy", "UCB", "TS", "Random")
INFORMATION_METHODS = METHODS[:4]


def temperature(family, d, L, m, batch_index, N_cal=400):
    R = L * L * (1.0 if family == "gaussian" else 0.25)
    tau_range = R * math.sqrt(m / (8 * math.log(d)))
    tau0 = tau_range / N_cal**0.75
    return tau0 * (m * batch_index)**0.75, tau0, tau_range


def candidate_cloud(family, condition, batch_index, K=4096,
                    global_cloud_seed=20261005, device="cpu"):
    """Deterministic cloud; seed is independent of K for nested diagnostics.

    CPU generation is intentional: cloud identity does not depend on the GPU or
    CUDA random-number implementation. Bernoulli points are direction-major.
    """
    if family not in ("gaussian", "bernoulli"):
        raise ValueError(f"Unsupported response family: {family}")
    name = condition.get("name", condition.get("condition", condition.get("id", "")))
    identity = f"{family}|{name}|{int(condition['d'])}|{float(condition['L'])}|{int(condition['m'])}|{int(batch_index)}|{int(global_cloud_seed)}"
    seed = int.from_bytes(hashlib.blake2b(identity.encode(), digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed)
    d, L = int(condition["d"]), float(condition["L"])
    if family == "bernoulli" and K % 4:
        raise ValueError("Bernoulli K must be divisible by four")
    ndirections = K if family == "gaussian" else K // 4
    z = rng.standard_normal((ndirections, d))
    z *= L / np.linalg.norm(z, axis=1, keepdims=True)
    if family == "bernoulli":
        z = (z[:, None, :] * np.array([.25, .5, .75, 1.])[None, :, None]).reshape(K, d)
    return torch.as_tensor(z, dtype=torch.float64, device=device)


def curvature(theta, X, family):
    if family == "gaussian":
        return torch.ones((theta.shape[0], X.shape[-2]), dtype=theta.dtype, device=theta.device)
    if X.ndim == 2:
        eta = theta @ X.T
    else:
        eta = (X * theta[:, None, :]).sum(-1)
    # sigmoid(eta)*sigmoid(-eta) remains accurate near the curvature floor.
    return torch.sigmoid(eta) * torch.sigmoid(-eta)


def _quadratic(X, A):
    return (torch.matmul(X, A) * X).sum(-1)


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _secular_kernel(E, Z, OUT, D: tl.constexpr, K: tl.constexpr,
                        BD: tl.constexpr, BC: tl.constexpr, NITER: tl.constexpr):
        r = tl.program_id(0)
        k = tl.program_id(1) * BC + tl.arange(0, BC)
        q = tl.arange(0, BD)
        a = tl.load(E + r * D + q, q < D, other=float("inf"))
        a0 = tl.load(E + r * D)
        a1 = tl.load(E + r * D + 1)
        gap = tl.maximum(a1 - a0, 0.)
        z2 = tl.load(Z + (r * K + k[:, None]) * D + q[None, :],
                     (k[:, None] < K) & (q[None, :] < D), other=0.)
        low = tl.full((BC,), 0., tl.float64)
        high = tl.full((BC,), 0., tl.float64) + gap
        for _ in range(NITER):
            middle = (low + high) * .5
            den = a[None, :] - a0 - middle[:, None]
            # At a repeated minimum eigenvalue the answer is exactly a0.
            den = tl.where(den == 0., 1.e-300, den)
            f = 1. + tl.sum(z2 / den, axis=1)
            low = tl.where(f <= 0., middle, low)
            high = tl.where(f > 0., middle, high)
        out = a0 + (low + high) * .5
        tl.store(OUT + r * K + k, out, k < K)

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def rank_one_min_eigenvalue(eigenvalues, projected_squared, iterations=32):
    """Minimum eigenvalue after PSD rank-one updates, vectorized over R,K.

    projected_squared[r,k,i] = w(x) * (u_i.T @ x)**2. Cauchy
    interlacing brackets the minimum between a0 and a1. Repeated minimum
    eigenvalues therefore yield exactly a0, including rank-deficient starts.
    Absolute bisection error is at most (a1-a0)/2**(iterations+1), apart
    from the eigendecomposition roundoff.
    """
    R, K, d = projected_squared.shape
    if d == 1:
        return eigenvalues[:, :1] + projected_squared[..., 0]
    if projected_squared.is_cuda and _HAS_TRITON:
        out = torch.empty((R, K), dtype=torch.float64, device=projected_squared.device)
        # A worker can hold tensors on cuda:1/2/3 while Torch's default current
        # device is cuda:0. Triton launches on the current device, so explicitly
        # enter the tensor's device context and restore the prior context after.
        with torch.cuda.device(projected_squared.device):
            _secular_kernel[(R, triton.cdiv(K, 32))](
                eigenvalues.contiguous(), projected_squared.contiguous(), out,
                d, K, triton.next_power_of_2(d), 32, iterations, num_warps=4)
        return out
    # NumPy avoids dozens of small Torch launches in the CPU implementation.
    if not projected_squared.is_cuda:
        a = eigenvalues.detach().numpy()
        z2 = projected_squared.detach().numpy()
        gap = np.maximum(a[:, 1] - a[:, 0], 0.)[:, None]
        lo = np.zeros((R, K))
        hi = np.broadcast_to(gap, (R, K)).copy()
        shifted = (a - a[:, :1])[:, None, :]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for _ in range(iterations):
                mid = (lo + hi) * .5
                den = shifted - mid[..., None]
                den = np.where(den == 0., 1.e-300, den)
                f = 1. + np.sum(z2 / den, axis=-1)
                lo = np.where(f <= 0., mid, lo)
                hi = np.where(f > 0., mid, hi)
        return torch.from_numpy(a[:, :1] + (lo + hi) * .5)
    a0 = eigenvalues[:, :1]
    lo = torch.zeros((R, K), dtype=eigenvalues.dtype, device=eigenvalues.device)
    hi = (eigenvalues[:, 1:2] - a0).clamp_min(0).expand(R, K).clone()
    shifted = (eigenvalues - a0)[:, None, :]
    for _ in range(iterations):
        mid = (lo + hi) * .5
        den = shifted - mid[..., None]
        den = torch.where(den == 0, 1.e-300, den)
        f = 1. + (projected_squared / den).sum(-1)
        lo = torch.where(f <= 0., mid, lo)
        hi = torch.where(f > 0., mid, hi)
    return a0 + (lo + hi) * .5


def _score(method, W, X, weights, tau=None, Winv=None):
    if method == "FB":
        a, U = torch.linalg.eigh(W)
        p = torch.softmax(-(a - a[:, :1]) / tau, dim=-1)
        P = (U * p[:, None, :]) @ U.transpose(-1, -2)
        return weights * _quadratic(X, P)
    if method in ("AOpt", "DOpt"):
        if Winv is None:
            eye = torch.eye(W.shape[-1], dtype=W.dtype, device=W.device).expand_as(W)
            Winv = torch.linalg.solve(W, eye)
        transformed = torch.matmul(X, Winv)
        leverage = weights * (transformed * X).sum(-1)
        if method == "DOpt":
            return leverage
        return weights * transformed.square().sum(-1) / (1. + leverage)
    if method == "EGreedy":
        a, U = torch.linalg.eigh(W)
        projected = torch.matmul(X, U)
        return rank_one_min_eigenvalue(a, weights[..., None] * projected.square())
    raise ValueError(method)


def score_candidates(method, family, theta, W, cloud, tau=None):
    """Direct acquisition scores for tests and oracle-quality diagnostics."""
    return _score(method, W, cloud, curvature(theta, cloud, family), tau)


def _rank_scores(method, raw_scores, W, X, weights):
    if method != "EGreedy":
        return raw_scores
    # The primary objective is completely flat whenever W has a repeated
    # minimum eigenvalue. Use D-optimality only on that mathematically flat
    # primary objective; otherwise return the exact secular objective.
    a = torch.linalg.eigvalsh(W)
    flat = (a[:, 1] - a[:, 0]) <= 1.e-12 * a[:, 0].abs().clamp_min(1.)
    if not bool(flat.any()):
        return raw_scores
    secondary = _score("DOpt", W, X, weights)
    return torch.where(flat[:, None], secondary, raw_scores)


@torch.no_grad()
def select_batch(method, family, theta, Vhat, H, condition, batch_index,
                 cloud=None, generator=None, state=None, config=None):
    """Select all queries before any response is observed.

    Inputs theta[R,d], Vhat/H[R,d,d]; output X[R,m,d]. Gaussian
    response-independent methods must be called with R=1 and broadcast by the
    caller. The persistent state stores aggregate shortlist diagnostics.
    """
    if method not in METHODS:
        raise ValueError(method)
    config = {} if config is None else config
    state = {} if state is None else state
    R, d = theta.shape
    m, L = int(condition["m"]), float(condition["L"])
    dtype, device = theta.dtype, theta.device
    if dtype != torch.float64 or Vhat.dtype != torch.float64 or H.dtype != torch.float64:
        raise ValueError("Estimator and information state must be float64")
    diag = {"method": method, "batch_index": int(batch_index),
            "oracle": "fresh Monte-Carlo approximation of the continuous acquisition oracle",
            "frozen_estimate": True, "frozen_covariance": method in ("UCB", "TS"),
            "tie_rule": "first candidate index among numerically equal maxima"}
    if method == "Random":
        z = torch.randn((R, m, d), dtype=dtype, device=device, generator=generator)
        u = torch.rand((R, m, 1), dtype=dtype, device=device, generator=generator)
        X = L * u.pow(1. / d) * z / torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        diag["distribution"] = "volume_uniform_ball"
        return X, state, diag
    if cloud is None:
        cloud = candidate_cloud(family, condition, batch_index,
                                int(config.get("K", 4096)),
                                int(config.get("global_cloud_seed", 20261005)), device)
    K = cloud.shape[0]
    if method == "TS":
        a, U = torch.linalg.eigh(Vhat)
        z = torch.randn((R, m, d), dtype=dtype, device=device, generator=generator)
        draws = theta[:, None, :] + (z / a[:, None, :].sqrt()) @ U.transpose(-1, -2)
        idx = (draws @ cloud.T).argmax(-1)
        diag.update(covariance_scale=1., independent_draw_per_position=True)
        return cloud[idx], state, diag
    if method == "UCB":
        rho, delta = float(config.get("rho", 1.)), float(config.get("ucb_delta", .05))
        sigma = 1. if family == "gaussian" else .5
        a0 = 1. if family == "gaussian" else math.exp(-4*L) / (1. + math.exp(-4*L))**2
        _, ld = torch.linalg.slogdet(H)
        beta = (sigma * (ld - d*math.log(rho) + 2*math.log(2/delta)).clamp_min(0).sqrt()
                + math.sqrt(rho)*4.) / a0
        eye = torch.eye(d, dtype=dtype, device=device).expand_as(H)
        Hinv = torch.linalg.solve(H, eye)
        scores = theta @ cloud.T + beta[:, None] * _quadratic(cloud, Hinv).clamp_min(0).sqrt()
        idx = scores.argmax(-1)
        diag.update(beta=beta.cpu().tolist(), delta=delta, sigma=sigma, curvature_floor=a0,
                    theta_bound=4., rho=rho, identical_queries=m, label_free_covariance_update=False)
        return cloud[idx][:, None, :].expand(-1, m, -1).clone(), state, diag

    W = Vhat.clone()
    Winv = None
    if method in ("AOpt", "DOpt"):
        eye = torch.eye(d, dtype=dtype, device=device).expand_as(W)
        Winv = torch.linalg.solve(W, eye)
    tau, tau0, tau_range = temperature(family, d, L, m, batch_index, int(config.get("N_cal", 400)))
    if method == "FB":
        diag.update(tau=tau, tau0=tau0, tau_range=tau_range)
    if method == "EGreedy":
        diag.update(e_implementation="exact_rank_one_secular_bisection_32_iterations",
                    e_flat_primary_tie="D-optimal secondary if lambda1-lambda0 <= 1e-12*max(1,abs(lambda0))")
    full_scan = bool(config.get("full_scan", False))
    Q = K if full_scan else min(K, int(state.get("Q", config.get("Q", 128))))
    diag.update(full_scan_every_position=full_scan, shortlist_used=not full_scan)
    state.setdefault("audit_count", 0)
    state.setdefault("miss_count", 0)
    state.setdefault("winner_outside_count", 0)
    state.setdefault("max_relative_score_loss", 0.)
    state.setdefault("sum_relative_score_loss", 0.)
    full_weights = curvature(theta, cloud, family)
    Xout = torch.empty((R, m, d), dtype=dtype, device=device)
    rows = torch.arange(R, device=device)
    audits = {0, m-1}
    if m >= 10:
        audits.add((m-1)//2)
    if full_scan:
        audits = set(range(m))
    local_audits = local_misses = local_outside = 0
    local_max_loss = local_sum_loss = 0.
    shortlist = None
    for j in range(m):
        if j in audits:
            raw = _score(method, W, cloud, full_weights, tau, Winv)
            rank = _rank_scores(method, raw, W, cloud, full_weights)
            best_idx = rank.argmax(-1)
            if full_scan:
                # Correctness fallback: every position scans all K, with no
                # shortlist sorting, gathering, stale scores, or missed maxima.
                pass
            elif shortlist is not None:
                best_short = raw.gather(1, shortlist).max(-1).values
                full_best = raw.gather(1, best_idx[:, None]).squeeze(1)
                loss = ((full_best - best_short) / full_best.abs().clamp_min(1.e-15)).clamp_min(0.)
                outside = ~(shortlist == best_idx[:, None]).any(-1)
                missed = outside & (loss > 1.e-4)
                nmiss, noutside = int(missed.sum()), int(outside.sum())
                local_audits += R
                local_misses += nmiss
                local_outside += noutside
                local_max_loss = max(local_max_loss, float(loss.max()))
                local_sum_loss += float(loss.sum())
                # Audit positions select the true full-cloud winner. Refresh
                # all rows at a miss: paired rows remain easy to vectorize.
                cumulative_rate = (state["miss_count"] + local_misses) / (state["audit_count"] + local_audits)
                if cumulative_rate > .01 and Q < min(256, K):
                    Q = min(256, K)
                    diag["shortlist_auto_increased"] = True
                    shortlist = torch.topk(rank, Q, dim=-1).indices
                elif nmiss:
                    shortlist = torch.topk(rank, Q, dim=-1).indices
            else:
                shortlist = torch.topk(rank, Q, dim=-1).indices
        else:
            Xsmall = cloud[shortlist]
            weights = full_weights.gather(1, shortlist)
            raw = _score(method, W, Xsmall, weights, tau, Winv)
            rank = _rank_scores(method, raw, W, Xsmall, weights)
            # Sorting the shortlist by global index makes equal-score choices
            # agree with the first-index rule used by a full scan.
            maximum = rank.max(-1, keepdim=True).values
            tied = torch.where(rank == maximum, shortlist, K)
            best_idx = tied.min(-1).values
        x = cloud[best_idx]
        weight = full_weights[rows, best_idx]
        Xout[:, j, :] = x
        W.add_(weight[:, None, None] * x[:, :, None] * x[:, None, :])
        if Winv is not None:
            vx = (Winv @ x[:, :, None]).squeeze(-1)
            denominator = 1. + weight * (x * vx).sum(-1)
            Winv.sub_((weight / denominator)[:, None, None] * vx[:, :, None] * vx[:, None, :])
    state.update(Q=Q, audit_count=state["audit_count"]+local_audits,
                 miss_count=state["miss_count"]+local_misses,
                 winner_outside_count=state["winner_outside_count"]+local_outside,
                 max_relative_score_loss=max(state["max_relative_score_loss"], local_max_loss),
                 sum_relative_score_loss=state["sum_relative_score_loss"]+local_sum_loss)
    if full_scan:
        state["full_scan_count"] = state.get("full_scan_count", 0) + R*m
    diag.update(K=K, Q=Q, audit_count=local_audits, miss_count=local_misses,
                winner_outside_count=local_outside, max_relative_score_loss=local_max_loss,
                sum_relative_score_loss=local_sum_loss,
                cumulative_miss_rate=state["miss_count"]/max(1,state["audit_count"]),
                full_scan_count=R*m if full_scan else R*len(audits))
    return Xout, state, diag
