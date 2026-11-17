import numpy as np

from real_response.aggregate_risk import bootstrap_risks, risk_statistics


def test_uncentered_second_moment_includes_bias_and_divides_by_S():
    E = np.array([[2.0, 0], [2.0, 0], [2.0, 0]])
    m, a = risk_statistics(E)
    np.testing.assert_allclose(a["Q"], [[4, 0], [0, 0]])
    assert m["Rdir_squared"] == 4
    assert m["R2_squared"] == 4
    assert m["centered_covariance_lambda_max"] == 0
    np.testing.assert_allclose(a["Q"], a["centered_covariance"] + np.outer(a["bias"], a["bias"]))


def test_rms_norm_and_directional_risk_are_distinct():
    m, a = risk_statistics(np.array([[3.0, 0], [0, 4.0]]))
    assert m["mean_L2"] == 3.5
    assert m["R2_squared"] == 12.5
    assert m["Rdir_squared"] == 8
    assert m["R2"] != m["mean_L2"]
    assert m["Rdir_squared_loo_min"] == 9
    assert m["Rdir_squared_loo_max"] == 16


def test_rank_degeneracy_and_paired_bootstrap():
    rng = np.random.default_rng(81)
    E = rng.normal(size=(10, 20))
    m, a = risk_statistics(E)
    assert m["Q_rank"] <= 10
    np.testing.assert_allclose(a["Q"], a["centered_covariance"] + np.outer(a["bias"], a["bias"]), atol=1e-12)
    indices = rng.integers(0, 10, (1000, 10))
    first, second = bootstrap_risks(E, indices), bootstrap_risks(E * 2, indices)
    np.testing.assert_allclose(second["Rdir_squared"], 4 * first["Rdir_squared"])
    _, equal = risk_statistics(np.eye(2))
    np.testing.assert_allclose(equal["top_eigenspace_projector"], np.eye(2))
