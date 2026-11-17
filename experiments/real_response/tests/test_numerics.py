import numpy as np
import pytest

from real_response.numerics import derivatives, fit_audit_at_theta, fit_glm, loss_values, quantities
from real_response.evaluation import evaluate_checkpoint


def toy(family, n=113, d=5):
    rng = np.random.default_rng(489)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, d - 1))]) / np.sqrt(d)
    theta = rng.normal(size=d) * 0.4
    mu, _ = quantities(X @ theta, family)
    if family == "bernoulli":
        y = rng.binomial(1, mu)
    elif family == "gaussian":
        y = mu + rng.normal(size=n)
    else:
        y = rng.poisson(mu)
    return X, y, theta


@pytest.mark.parametrize("family", ["bernoulli", "gaussian", "poisson"])
def test_loss_gradient_hessian_finite_differences(family):
    X, y, theta = toy(family)
    f, g, H = derivatives(theta, X, y, family, 1.3)
    h = 1e-5
    numerical_gradient, numerical_hessian = np.zeros(5), np.zeros((5, 5))
    for j in range(5):
        unit = np.eye(5)[j] * h
        fp, gp, _ = derivatives(theta + unit, X, y, family, 1.3)
        fm, gm, _ = derivatives(theta - unit, X, y, family, 1.3)
        numerical_gradient[j] = (fp - fm) / (2 * h)
        numerical_hessian[:, j] = (gp - gm) / (2 * h)
    np.testing.assert_allclose(g, numerical_gradient, atol=3e-8, rtol=2e-7)
    np.testing.assert_allclose(H, numerical_hessian, atol=3e-8, rtol=2e-7)
    # Sum objective exactly equals n times correctly normalized mean objective.
    mean_objective = np.mean(loss_values(X @ theta, y, family)) + 1.3 / len(y) * (theta @ theta) / 2
    np.testing.assert_allclose(f, len(y) * mean_objective)


@pytest.mark.parametrize("family", ["bernoulli", "gaussian", "poisson"])
def test_solver_stationarity_and_reproducible_audit(family):
    X, y, _ = toy(family)
    theta, audit = fit_glm(X, y, family, reference=True)
    assert audit["fit_converged"]
    assert audit["fit_grad_relative"] <= 1e-9
    assert audit["intercept_penalized"]
    np.testing.assert_allclose(audit["train_penalized_objective_sum"], audit["train_loss_sum"] + audit["ridge_penalty"])
    saved = fit_audit_at_theta(theta, X, y, family)
    np.testing.assert_allclose(saved["fit_grad_relative"], audit["fit_grad_relative"])
    if family == "gaussian":
        expected = np.linalg.solve(X.T @ X + np.eye(5), X.T @ y)
        np.testing.assert_allclose(theta, expected, atol=1e-12)


def test_single_class_logistic_and_unclipped_poisson():
    X, y, _ = toy("bernoulli")
    theta, audit = fit_glm(X, np.zeros_like(y), "bernoulli", reference=True)
    assert audit["fit_converged"]
    assert audit["fit_grad_relative"] <= 1e-9
    mu, w = quantities(np.array([1000.0, -1000.0]), "poisson")
    assert np.isinf(mu[0]) and w[1] == 0
    assert np.isinf(loss_values(np.array([1000.0]), np.array([1.0]), "poisson")[0])


@pytest.mark.parametrize("family", ["bernoulli", "gaussian", "poisson"])
def test_checkpoint_information_and_objective_metrics(family):
    X, y, _ = toy(family, n=213)
    theta_full, _ = fit_glm(X, y, family, reference=True)
    selected = np.arange(80)
    theta, audit = fit_glm(X[selected], y[selected], family)
    matched, _ = fit_glm(X, y, family, rho=len(X) / len(selected), reference=True)
    metrics, arrays = evaluate_checkpoint(X, y, X[100:], y[100:], selected, theta,
                                          family, 1.0, theta_full, matched, audit)
    assert metrics["metrics_complete"]
    np.testing.assert_allclose(arrays["V_hat"], derivatives(theta, X[selected], y[selected], family)[2])
    np.testing.assert_allclose(arrays["error_ref"], theta - theta_full)
    assert abs(metrics["min_eigenvalue_gap"]) <= metrics["tracking_op"] + 1e-10
    assert metrics["full_objective_gap_per_row"] >= -1e-10
    assert metrics["matched_full_objective_gap_per_row"] >= -1e-10
    assert metrics["selected_reference_objective_gap"] >= -1e-9
    if family == "gaussian":
        for key in ["V_ref", "V_hat", "V_acq", "H_design"]:
            np.testing.assert_allclose(arrays[key], arrays["H_design"])
        assert metrics["tracking_op"] == 0


def test_full_pool_matched_equals_fixed_and_information_is_additive():
    X, y, _ = toy("bernoulli")
    theta, audit = fit_glm(X, y, "bernoulli", reference=True)
    matched, _ = fit_glm(X, y, "bernoulli", rho=len(X) / len(X), reference=True)
    np.testing.assert_allclose(theta, matched)
    _, weights = quantities(X @ theta, "bernoulli")
    first = np.eye(5) + (X[:80].T * weights[:80]) @ X[:80]
    later = np.eye(5) + (X.T * weights) @ X
    assert np.linalg.eigvalsh(later - first)[0] > -1e-10


@pytest.mark.parametrize("family", ["bernoulli", "gaussian", "poisson"])
def test_predictive_metrics_against_independent_known_arrays(family):
    import math
    X = np.ones((3, 1))
    if family == "bernoulli":
        y, theta = np.array([0., 1., 1.]), np.array([0.])
        expected_nll = math.log(2)
    elif family == "gaussian":
        y, theta = np.array([0., 1., 4.]), np.array([2.])
        expected_nll = 1.5 + math.log(2 * math.pi) / 2
    else:
        y, theta = np.array([0., 2., 4.]), np.array([math.log(2)])
        expected_nll = sum(2 - count * math.log(2) + math.lgamma(count + 1) for count in y) / 3
    audit = fit_audit_at_theta(theta, X, y, family)
    metrics, _ = evaluate_checkpoint(X, y, X, y, [0, 1, 2], theta, family, 1.0,
                                    theta, theta, audit, extra={"n0": 3})
    np.testing.assert_allclose(metrics["test_nll"], expected_nll, atol=1e-12)
    if family == "bernoulli":
        np.testing.assert_allclose(metrics["test_brier"], 0.25)
        np.testing.assert_allclose(metrics["test_auroc"], 0.5)
        np.testing.assert_allclose(metrics["test_ap"], 2 / 3)
        one_class = np.zeros(3)
        metrics, _ = evaluate_checkpoint(X, one_class, X, one_class, [0, 1, 2], theta, family,
                                        1.0, theta, theta, audit, extra={"n0": 3})
        assert metrics["test_auroc"] is None and metrics["metric_na_reasons"]["test_auroc"] == "SINGLE_CLASS_TEST"
        np.testing.assert_allclose(metrics["test_nll"], math.log(2))
    elif family == "gaussian":
        np.testing.assert_allclose(metrics["test_rmse"], math.sqrt(3))
        np.testing.assert_allclose(metrics["test_mae"], 5 / 3)
    else:
        deviance = sum(2 * ((count * math.log(count / 2) if count else 0) - (count - 2)) for count in y) / 3
        np.testing.assert_allclose(metrics["test_poisson_deviance"], deviance)
