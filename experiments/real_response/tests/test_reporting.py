import json
import numpy as np
import pandas as pd
import pytest

from real_response import evaluation, numerics, reporting


def write_saved_runs(tmp_path):
    config = dict(dimensions=[2], methods=["random"], run_seeds=[0, 1], n0=3, rho=1.0,
                  datasets={"movielens": dict(additional_labels=2, batch_size=2)}, uncertainty={"bootstrap_draws": 1000})
    rng = np.random.default_rng(1404)
    X = np.column_stack([np.ones(8), rng.normal(size=8)]) / 2
    y = rng.normal(size=8)
    reference, _ = numerics.fit_glm(X, y, "gaussian", reference=True)
    for seed in config["run_seeds"]:
        p = tmp_path / "runs" / "movielens" / "2" / str(seed) / "random"
        p.mkdir(parents=True)
        rows, states = [], []
        warm_ref = warm_hat = None
        order = rng.permutation(len(X))
        for round_id, n in enumerate([3, 5]):
            selected = order[:n]
            theta, fit = numerics.fit_glm(X[selected], y[selected], "gaussian")
            matched, _ = numerics.fit_glm(X, y, "gaussian", rho=len(X)/n, reference=True)
            metrics, arrays = evaluation.evaluate_checkpoint(X, y, X, y, selected, theta, "gaussian", 1.0,
                reference, matched, fit, extra={"n0": 3, "warm_reference_lambda": warm_ref, "warm_plugin_lambda": warm_hat})
            if round_id == 0:
                warm_ref, warm_hat = metrics["lambda_min_reference"], metrics["lambda_min_plugin"]
            metrics.update(round=round_id, feature_hash="shared_feature", reference_hash="shared_reference", fallback_used=False)
            rows.append(metrics); states.append(arrays)
        pd.DataFrame(rows).to_csv(p / "trajectory.csv", index=False)
        for filename, key in [("error_ref_checkpoints", "error_ref"), ("error_matched_checkpoints", "error_matched")]:
            np.save(p / (filename + ".npy"), np.stack([a[key] for a in states]))
        np.savez_compressed(p / "information_matrices.npz", **{key: np.stack([a[key] for a in states])
                            for key in ["V_ref", "V_hat", "V_acq", "H_design"]})
    return config


def test_saved_output_regeneration_does_not_fit_and_preserves_risk(tmp_path, monkeypatch):
    config = write_saved_runs(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved-output regeneration attempted to refit")
    monkeypatch.setattr(numerics, "fit_glm", forbidden)
    monkeypatch.setattr(evaluation, "fit_glm", forbidden)
    # Mathematical regeneration exports every plotting curve; expensive image
    # rendering is separately exercised on real saved main artifacts.
    monkeypatch.setattr(reporting, "make_figures", lambda *args: None)
    result = reporting.summarize(config, tmp_path)
    assert result["completed_trajectories"] == 2
    assert not result["complete_10_seed"]
    aggregate = tmp_path / "aggregate" / "movielens" / "2" / "random"
    Q = np.load(aggregate / "Q_ref_checkpoints.npy")
    errors = np.stack([np.load(tmp_path / "runs" / "movielens" / "2" / str(s) / "random" / "error_ref_checkpoints.npy") for s in [0, 1]])
    np.testing.assert_allclose(Q, np.einsum("std,ste->tde", errors, errors) / 2)
    curves = pd.read_csv(tmp_path / "plot_data" / "all_risk_trajectories.csv")
    for key in ["mean_lambda_min_reference", "mean_lambda_min_plugin", "mean_l2_ref", "R2_ref", "Rdir_squared_ref",
                "Rdir_squared_matched", "mean_test_nll", "mean_full_objective_gap_per_row"]:
        assert curves[key].notna().all()
    assert (tmp_path / "paper" / "tables_real_response.tex").is_file()
    # Regenerate a second time from the same saved states, without learner calls.
    reporting.summarize(config, tmp_path)
    np.testing.assert_array_equal(Q, np.load(aggregate / "Q_ref_checkpoints.npy"))
    reporting.summarize(config, tmp_path, seeds=[0, 1])
    assert (tmp_path / "preview" / "aggregation_status.json").is_file()


def test_mismatched_feature_hash_rejects_seed_aggregation(tmp_path, monkeypatch):
    config = write_saved_runs(tmp_path)
    p = tmp_path / "runs" / "movielens" / "2" / "1" / "random" / "trajectory.csv"
    frame = pd.read_csv(p)
    frame["feature_hash"] = "different_features"
    frame.to_csv(p, index=False)
    monkeypatch.setattr(reporting, "make_figures", lambda *args: None)
    with pytest.raises(ValueError, match="inconsistent feature_hash"):
        reporting.summarize(config, tmp_path)


def test_safe_csv_reader_preserves_exponent_like_hash_and_types(tmp_path):
    path = tmp_path / "saved.csv"
    scientific_hash = "6e07945374823c9dda15c45b445fe17b12617b2c106af0b47562fd1859e484c2"
    # This real metadata hash triggers a segfault in the installed pandas
    # native numeric-string inference; never send identifiers to that parser.
    path.write_text("scientific_hash,seed,metrics_complete,value,missing,used_seed_ids\n" +
                    scientific_hash + ',6,True,1.25e-6,,"[0, 1]"\n')
    frame = reporting.read_csv_safe(path)
    assert frame.loc[0, "scientific_hash"] == scientific_hash
    assert frame.loc[0, "seed"] == 6
    assert frame.loc[0, "metrics_complete"]
    assert frame.loc[0, "value"] == 1.25e-6
    assert np.isnan(frame.loc[0, "missing"])
    assert frame.loc[0, "used_seed_ids"] == "[0, 1]"
