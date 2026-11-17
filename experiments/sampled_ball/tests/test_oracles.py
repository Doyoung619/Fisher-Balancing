import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oracles import (METHODS, candidate_cloud, curvature, rank_one_min_eigenvalue,
                     score_candidates, select_batch, temperature)

torch.set_num_threads(1)
DTYPE = torch.float64


def make_state(R=3, d=5, device="cpu"):
    gen = torch.Generator(device=device).manual_seed(420)
    B = torch.randn(R, d, d, dtype=DTYPE, device=device, generator=gen)
    W = B @ B.transpose(-1, -2) + torch.eye(d, device=device, dtype=DTYPE)
    theta = torch.randn(R, d, dtype=DTYPE, device=device, generator=gen)
    return theta, W


@pytest.mark.parametrize("family", ["gaussian", "bernoulli"])
def test_cloud_reproducible_nested_and_feasible(family):
    c = dict(name="central", d=10, L=2., m=10)
    X = candidate_cloud(family, c, 3, K=256)
    assert torch.equal(X, candidate_cloud(family, c, 3, K=512)[:256])
    assert not torch.equal(X, candidate_cloud(family, c, 4, K=256))
    radii = X.norm(dim=-1)
    assert float(radii.max()) <= 2. + 1.e-14
    if family == "gaussian":
        assert torch.allclose(radii, torch.full_like(radii, 2.))
    else:
        assert torch.allclose(radii.reshape(-1, 4), torch.tensor([.5, 1., 1.5, 2.], dtype=DTYPE).expand(64, 4))


@pytest.mark.parametrize("family", ["gaussian", "bernoulli"])
def test_scores_against_direct_linear_algebra(family):
    theta, W = make_state()
    c = dict(name="central", d=5, L=1., m=10)
    X = candidate_cloud(family, c, 3, K=32)
    weights = curvature(theta, X, family)
    updated = W[:, None] + weights[:, :, None, None] * X[None, :, :, None] * X[None, :, None, :]
    inv = torch.linalg.inv(W)
    exp = torch.matrix_exp(-W / 2.)
    P = exp / exp.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
    fb = weights * ((X @ P) * X).sum(-1)
    assert torch.allclose(score_candidates("FB", family, theta, W, X, tau=2.), fb, atol=1.e-12, rtol=1.e-10)
    a = inv.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None] - torch.linalg.inv(updated).diagonal(dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(score_candidates("AOpt", family, theta, W, X), a, atol=1.e-12, rtol=1.e-10)
    d = torch.linalg.slogdet(updated).logabsdet - torch.linalg.slogdet(W).logabsdet[:, None]
    assert torch.allclose(torch.log1p(score_candidates("DOpt", family, theta, W, X)), d, atol=1.e-12, rtol=1.e-10)
    e = torch.linalg.eigvalsh(updated)[..., 0]
    assert torch.allclose(score_candidates("EGreedy", family, theta, W, X), e, atol=2.e-9, rtol=1.e-10)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_secular_random_and_degenerate(device):
    theta, W = make_state(device=device)
    z = torch.randn(3, 64, 5, dtype=DTYPE, device=device)
    a, U = torch.linalg.eigh(W)
    scores = rank_one_min_eigenvalue(a, (z @ U).square())
    direct = torch.linalg.eigvalsh(W[:, None] + z[..., :, None] * z[..., None, :])[..., 0]
    assert float((scores-direct).abs().max()) < 2.e-9
    a[:] = 1.
    assert torch.equal(rank_one_min_eigenvalue(a, z.square()), torch.ones((3, 64), dtype=DTYPE, device=device))
    # Zero projection on the minimum eigenvector leaves its eigenvalue intact.
    a = torch.tensor([[1., 2., 3., 4., 5.]], dtype=DTYPE, device=device)
    z2 = torch.ones(1, 4, 5, dtype=DTYPE, device=device)
    z2[..., 0] = 0.
    assert torch.allclose(rank_one_min_eigenvalue(a, z2), torch.ones((1, 4), dtype=DTYPE, device=device), atol=1.e-9)


@pytest.mark.parametrize("method", METHODS)
def test_full_scan_selection_and_feasibility(method):
    theta, W = make_state()
    c = dict(name="test", d=5, L=.5, m=5)
    cloud = candidate_cloud("bernoulli", c, 4, K=64)
    gen = torch.Generator().manual_seed(44)
    X, state, diag = select_batch(method, "bernoulli", theta, W, W, c, 4,
                                  cloud=cloud, generator=gen, config={"Q":64})
    assert X.shape == (3, 5, 5)
    assert torch.isfinite(X).all()
    assert float(X.norm(dim=-1).max()) <= .5 + 1.e-14
    if method in ("FB", "AOpt", "DOpt", "EGreedy"):
        virtual = W.clone()
        tau = temperature("bernoulli", 5, .5, 5, 4)[0]
        for j in range(5):
            direct = score_candidates(method, "bernoulli", theta, virtual, cloud, tau)
            best = direct.argmax(-1)
            assert torch.equal(X[:, j], cloud[best])
            w = curvature(theta, X[:, j:j+1], "bernoulli")[:, 0]
            virtual += w[:, None, None] * X[:, j, :, None] * X[:, j, None, :]


def test_e_initial_ties_are_informative_without_changing_primary_objective():
    theta = torch.zeros((1, 5), dtype=DTYPE)
    W = torch.eye(5, dtype=DTYPE)[None]
    c = dict(name="test", d=5, L=1., m=5)
    cloud = candidate_cloud("gaussian", c, 1, K=256)
    initial = score_candidates("EGreedy", "gaussian", theta, W, cloud)
    assert torch.equal(initial, torch.ones_like(initial))
    X, _, _ = select_batch("EGreedy", "gaussian", theta, W, W, c, 1, cloud=cloud)
    assert int(torch.linalg.matrix_rank(X[0])) == 5


def test_temperature_keeps_calibration_coefficient():
    a, a0, arange = temperature("bernoulli", 10, 1., 10, 40)
    b, b0, _ = temperature("bernoulli", 10, 1., 10, 1000)
    assert a0 == b0
    assert a == pytest.approx(arange)
    assert b > arange


@pytest.mark.parametrize("method", ["FB", "AOpt", "DOpt", "EGreedy"])
def test_full_scan_every_position_matches_direct_objective(method):
    theta, W = make_state()
    c = dict(name="fullscan", d=5, L=1., m=10)
    cloud = candidate_cloud("bernoulli", c, 17, K=128)
    state = {}
    X, state, diag = select_batch(method, "bernoulli", theta, W, W, c, 17,
                                  cloud=cloud, state=state,
                                  config={"Q":4, "full_scan":True})
    virtual = W.clone()
    tau = temperature("bernoulli", 5, 1., 10, 17)[0]
    for j in range(10):
        raw = score_candidates(method, "bernoulli", theta, virtual, cloud, tau)
        assert torch.equal(X[:, j], cloud[raw.argmax(-1)])
        w = curvature(theta, X[:, j:j+1], "bernoulli")[:, 0]
        virtual += w[:, None, None] * X[:, j, :, None] * X[:, j, None, :]
    assert state["full_scan_count"] == 30
    assert state["Q"] == 128
    assert not diag["shortlist_used"]
    assert diag["full_scan_every_position"]
    assert state["audit_count"] == 0


@pytest.mark.parametrize("device_index", list(range(torch.cuda.device_count())))
def test_egreedy_all_gpus_with_default_device_zero(device_index):
    torch.cuda.set_device(0)
    device = f"cuda:{device_index}"
    theta, W = make_state(R=2, d=5, device=device)
    c = dict(name="multigpu", d=5, L=1., m=5)
    cloud = candidate_cloud("bernoulli", c, 17, K=128, device=device)
    X, state, diag = select_batch("EGreedy", "bernoulli", theta, W, W, c, 17,
                                  cloud=cloud, config={"full_scan":True})
    torch.cuda.synchronize(device)
    assert torch.cuda.current_device() == 0
    virtual = W.clone()
    for j in range(c["m"]):
        weight = curvature(theta, cloud, "bernoulli")
        direct = torch.linalg.eigvalsh(virtual[:, None] + weight[:, :, None, None]
                                      * cloud[None, :, :, None] * cloud[None, :, None, :])[..., 0]
        assert torch.equal(X[:, j], cloud[direct.argmax(-1)])
        w = curvature(theta, X[:, j:j+1], "bernoulli")[:, 0]
        virtual += w[:, None, None] * X[:, j, :, None] * X[:, j, None, :]
