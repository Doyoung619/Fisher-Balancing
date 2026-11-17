"""Post-main independent scores, persisted failure states and runtime evidence.

Artifact tests inspect already-measured CUDA and production concurrency. They
do not rerun the GPU benchmark or claim a CUDA GLM fitting implementation.
"""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from real_response.common import ROOT, stream_seed
from real_response.runner import run_one, run_path, checkpoint_files
from real_response.selectors import select_batch, shared_candidates


@pytest.mark.parametrize("method", ["a_opt", "d_opt"])
@pytest.mark.parametrize("family", ["bernoulli", "gaussian", "poisson"])
def test_each_reported_design_score_matches_direct_objective_difference(method, family):
    rng = np.random.default_rng(892)
    X = rng.normal(size=(55, 6)) / 3
    theta = rng.normal(size=6)
    old, candidates = np.arange(10), np.arange(10, 40)
    _, audit = select_batch(method, X, old, candidates, theta, family, 12, 1.,
                            np.random.default_rng(9), {})
    W = np.asarray(audit["W_start"])
    U = X[candidates] * np.sqrt(audit["candidate_weights_acquisition"])[:, None]
    available = set(range(len(U)))
    for row in audit["slot_trace"]:
        j = row["candidate_position"]
        if method == "a_opt":
            base = np.trace(np.linalg.inv(W))
            differences = {k: base - np.trace(np.linalg.inv(W + np.outer(U[k], U[k])))
                           for k in available}
        else:
            base = np.linalg.slogdet(W)[1]
            differences = {k: np.linalg.slogdet(W + np.outer(U[k], U[k]))[1] - base
                           for k in available}
        assert row["score"] == pytest.approx(differences[j], rel=1e-9, abs=1e-11)
        assert differences[j] == pytest.approx(max(differences.values()), rel=1e-9, abs=1e-11)
        W = W + np.outer(U[j], U[j])
        available.remove(j)


@pytest.mark.requires_data
def test_forced_sdp_failure_persists_without_claiming_solver_success(tmp_path):
    config = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text())
    config["sdp"]["force_failure"] = True
    task = ("obd", 5, 0, "e_sdp")
    result = run_one(task, config, tmp_path)
    # Trajectory completion is legitimate with explicit fallback; SDP success
    # is a separate recorded state and must never be inferred from completion.
    assert result["state"] == "EVALUATION_COMPLETE"
    p = run_path(tmp_path, task)
    assert json.loads((p / "config.json").read_text())["config"]["sdp"]["force_failure"]
    records = checkpoint_files(p)[1:]
    assert len(records) == 2
    for _, _, saved in records:
        metric, audit = saved["metrics"], saved["acquisition_audit"]
        assert metric["fallback_used"] is True
        assert metric["sdp_status"] == "FAILED_FALLBACK_E_GREEDY"
        assert "failed" in metric["fallback_reason"].lower()
        assert audit["fallback_used"] is True
        assert len(audit["sdp_attempts"]) == 3
        assert all(a["quality_gate_passed"] is False for a in audit["sdp_attempts"])
        assert all("Explicit failure-injection" in a["error"] for a in audit["sdp_attempts"])
        assert metric["sdp_ub"] >= metric["sdp_lb"] - 1e-9
        for field in ("sdp_primal_residual", "sdp_dual_residual", "sdp_psd_residual"):
            assert metric[field] is None
            assert audit["sdp_attempts"][-1]["status"] == "ERROR"
            assert metric["fallback_reason"]  # Explicit cause for unavailable solve residuals.
    with (p / "trajectory.csv").open(newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert all(r["sdp_status"] == "FAILED_FALLBACK_E_GREEDY" and r["fallback_used"] == "True"
               for r in csv_rows[1:])
    with np.load(p / "batch_audit.npz", allow_pickle=False) as z:
        binary_audit = json.loads(str(z["audit_json"]))
    assert sum(bool(a["fallback_used"]) for a in binary_audit) == 2


def test_independent_task_order_and_runtime_device_do_not_change_scientific_rng():
    X = np.random.default_rng(149).normal(size=(75, 5)) / 4
    theta = np.linspace(-.4, .4, 5)
    labeled = np.arange(12)
    methods = ["random", "ts_glm", "badge", "e_sdp", "fb_sample", "d_opt"]
    tasks = [(seed, method) for seed in (0, 3) for method in methods]

    def execute(order, runtime):
        results = {}
        for seed, method in order:
            candidates = shared_candidates(len(X), labeled, "toy", seed, 2, 30)
            rng = np.random.default_rng(stream_seed("selector", "toy", 5, seed, method))
            ids, audit = select_batch(method, X, labeled, candidates, theta, "bernoulli", 6, 1., rng,
                {"dataset": "toy", "seed": seed, "runtime": runtime}, batch_index=2)
            results[(seed, method)] = (ids, audit["rng_state_before"], audit["rng_state_after"],
                                      audit["W_end"])
        return results

    first = execute(tasks, {"allowed_gpu_ids": [0], "cpu_workers": 1, "backend": "cpu"})
    permutation = np.random.default_rng(987).permutation(len(tasks))
    second = execute([tasks[i] for i in permutation],
                     {"allowed_gpu_ids": [3, 2, 1, 0], "cpu_workers": 64, "backend": "auto"})
    for task in tasks:
        a, b = first[task], second[task]
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1] and a[2] == b[2]
        np.testing.assert_allclose(a[3], b[3], rtol=0., atol=0.)


def test_saved_cuda_evidence_covers_every_family_dimension_state_and_device():
    path = ROOT / "cuda_benchmark.json"
    if not path.exists():
        pytest.skip("Measured real-data CUDA artifact is not available")
    audit = json.loads(path.read_text())
    assert audit["status"] == "complete" and audit["dtype"] == "float64"
    assert set(audit["actual_devices_measured"]) == {0, 1, 2, 3}
    expected = {(dataset, d, n)
                for dataset in ("obd", "movielens", "lastfm") for d in (5, 10, 20)
                for n in (80, 3880 if dataset == "obd" else 460)}
    cases = audit["complete_batch_cases"]
    assert {(c["dataset"], c["d"], c["n_labeled"]) for c in cases} == expected
    for case in cases:
        for backend in ["batched", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]:
            result = case["backends"][backend]
            assert result["identical_selected_ids_to_cpu"] is True
            assert result["max_final_theta_abs_difference_to_cpu"] <= 1e-10
            assert all(m["fit_converged"] for m in result["measurements"])
            assert all(m["max_selector_direct_check_error"] <= 1e-8 for m in result["measurements"])
    assert audit["concurrency_comparison"]["identical_selected_hashes"] is True


def _current_main_output():
    pointer = ROOT / "CURRENT_OUTPUT"
    if not pointer.exists():
        pytest.skip("Production execution artifacts are not available")
    output = Path(pointer.read_text().strip())
    statuses = list((output / "runs").glob("*/*/*/*/status.json"))
    if len(statuses) != 930:
        pytest.skip("Complete production run status set is not available")
    return output, statuses


def test_actual_scheduler_concurrency_respects_shared_cpu_and_sdp_limits():
    output, statuses = _current_main_output()
    plan = json.loads((output / "runtime_plan.json").read_text())
    assert plan["cpu_workers"] == 24 and plan["sdp_queue_limit"] == 8
    assert plan["cpu_tokens_shared_across_queues"] is True
    # Independent interval sweep from persisted finish times and actual worker
    # wall durations; do not inspect or copy scheduler dispatch implementation.
    events = []
    for path in statuses:
        status = json.loads(path.read_text())
        assert status["state"] == "EVALUATION_COMPLETE"
        finish = status["finished_at"]
        start = finish - status["wall_seconds"]
        is_sdp = status["task"][3] == "e_sdp"
        events += [(start, 1, is_sdp), (finish, -1, is_sdp)]
    active = sdp_active = peak = sdp_peak = 0
    for _, change, is_sdp in sorted(events, key=lambda e: (e[0], e[1])):
        active += change
        sdp_active += change * is_sdp
        assert 0 <= sdp_active <= active <= plan["cpu_workers"]
        assert sdp_active <= plan["sdp_queue_limit"]
        peak, sdp_peak = max(peak, active), max(sdp_peak, sdp_active)
    assert active == sdp_active == 0
    assert peak == 24 and sdp_peak == 8  # Both budget limits were exercised.
    with (output / "runtime_telemetry.csv").open(newline="") as f:
        telemetry = list(csv.DictReader(f))
    main_rows = [r for r in telemetry if int(r["expected"]) == 930]
    assert main_rows
    assert all(0 <= int(r["inflight"]) <= 24 for r in main_rows)


def test_actual_sdp_missing_fallback_residuals_have_preserved_failure_reasons():
    output, _ = _current_main_output()
    quality = json.loads((output / "sdp_quality_audit.json").read_text())
    assert quality["sdp_batches"] == 1800
    assert quality["invalid_bound_gap_count"] == 0
    assert quality["fallback_count"] == len(quality["fallbacks"])
    for failure in quality["fallbacks"]:
        path = output / failure["run"] / "checkpoints" / f"{failure['round']:03d}.json"
        saved = json.loads(path.read_text())
        audit, metric = saved["acquisition_audit"], saved["metrics"]
        assert metric["fallback_used"] and metric["fallback_reason"]
        assert metric["sdp_status"] == "FAILED_FALLBACK_E_GREEDY"
        assert all(not attempt["quality_gate_passed"] for attempt in audit["sdp_attempts"])
        for field in ("sdp_primal_residual", "sdp_dual_residual", "sdp_psd_residual"):
            if metric[field] is None:
                last = audit["sdp_attempts"][-1]
                assert last["status"] == "ERROR" and last["error"]
                # Earlier numeric attempts remain available, rather than
                # silently erasing useful residual evidence after a retry error.
                assert any(a.get("psd_residual_scaled") is not None for a in audit["sdp_attempts"][:-1])
