"""Evaluation-only reference fits and fully reconstructible checkpoint metrics.

Do not expose full-pool labels, references or outputs of this module to query
selection. References describe this fixed recorded pool, not a true parameter.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.special import xlogy
from sklearn.metrics import average_precision_score, roc_auc_score

from .numerics import derivatives, fit_audit_at_theta, fit_glm, loss_values, objective, quantities

FAMILIES = {"obd": "bernoulli", "movielens": "gaussian", "lastfm": "poisson"}
NUMERICAL_POLICY = {
    "dtype": "float64", "rank_relative_tolerance": 1e-10,
    "rank_absolute_tolerance": 1e-10, "psd_relative_tolerance": 1e-10,
    "near_zero_ratio_denominator": 1e-14, "acquisition_curvature_floor": 1e-12,
    "primary_curvature_floor": 0.0,
}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_npz(path, **arrays):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _hash_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_references(root, config):
    """Cache fixed and every-budget matched reference for each dataset/d.

    Returns ``{dataset: {d: {path, json_path, reference_hash}}}``. Each NPZ
    contains theta_full[d], matched_ns[q], theta_matched[q,d], H_full[d,d],
    full_spectrum[d], and matched_hessians[q,d,d]. Cache identity depends on
    actual prepared data, d, objective/ridge, and numerical policy; never seed,
    method, runtime backend, or profile. Existing matched n's are preserved.
    """
    root = Path(root)
    dest = root / "data" / "references"
    dest.mkdir(parents=True, exist_ok=True)
    output = {}
    for dataset, ds_config in config["datasets"].items():
        prepared = root / "data" / "prepared" / f"{dataset}.npz"
        with np.load(prepared, allow_pickle=False) as data:
            full_X = np.asarray(data["X_pool"], dtype=np.float64)
            y = np.asarray(data["y_pool"], dtype=np.float64)
        family = ds_config.get("family", FAMILIES[dataset])
        rho = float(config.get("rho", 1.0))
        n0 = int(config.get("n0", 80))
        budget = int(ds_config.get("additional_labels", config.get("additional_labels", 400)))
        batch = int(ds_config.get("batch_size", config.get("batch_size", 20)))
        ns = set([n0] + [n0 + min(end, budget) for end in range(batch, budget + batch, batch)])
        # Small correctness runs are independent scientific trajectories but
        # these shared full-pool references may safely be cached in advance.
        ns.update([80, 100, 120])
        if max(ns) > len(y):
            raise ValueError(f"{dataset}: reference schedule exceeds pool rows")
        output[dataset] = {}
        for d in config.get("dimensions", [5, 10, 20]):
            X = np.ascontiguousarray(full_X[:, :int(d)])
            signature = hashlib.sha256(json.dumps({
                "data_sha256": _hash_arrays(X, y), "family": family, "rho": rho,
                "objective": "sum_nll_plus_fixed_ridge_penalized_intercept",
                "reference_tolerance": 1e-9, "numerical_policy": NUMERICAL_POLICY,
            }, sort_keys=True).encode()).hexdigest()
            path = dest / f"{dataset}_d{d}.npz"
            json_path = path.with_suffix(".json")
            with open(path.with_suffix(".lock"), "a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                cached, details = {}, {}
                if path.exists() and json_path.exists():
                    details = json.loads(json_path.read_text())
                    if details.get("reference_hash") == signature:
                        with np.load(path, allow_pickle=False) as data:
                            cached = {k: data[k] for k in data.files}
                if cached and set(map(int, cached["matched_ns"])) >= ns:
                    output[dataset][int(d)] = {"path": str(path), "json_path": str(json_path), "reference_hash": signature}
                    continue
                if cached:
                    theta_full = cached["theta_full"]
                    full_audit = details["full_fit_audit"]
                    matched = {int(n): th for n, th in zip(cached["matched_ns"], cached["theta_matched"])}
                    matched_audits = details["matched_fit_audits"]
                    ns.update(matched)
                else:
                    theta_full, full_audit = fit_glm(X, y, family, rho=rho, reference=True)
                    matched, matched_audits = {}, {}
                if not full_audit["fit_converged"]:
                    raise RuntimeError(f"Reference failed stationarity: {dataset} d={d}: {full_audit}")
                previous = theta_full
                hessians = []
                for n in sorted(ns):
                    rho_matched = rho * len(y) / n
                    if n not in matched:
                        matched[n], audit = fit_glm(X, y, family, rho=rho_matched, theta0=previous, reference=True)
                        if not audit["fit_converged"]:
                            raise RuntimeError(f"Matched reference failed: {dataset} d={d} n={n}: {audit}")
                        matched_audits[str(n)] = audit
                    previous = matched[n]
                    hessians.append(derivatives(previous, X, y, family, rho_matched)[2])
                H_full = derivatives(theta_full, X, y, family, rho)[2]
                sensitivity = details.get("rho_sensitivity", {}) if cached else {}
                sensitivity_thetas, sensitivity_rhos = [], []
                if dataset == "obd":
                    for rr in [0.1, 1.0, 10.0]:
                        th, audit = fit_glm(X, y, family, rho=rr, theta0=theta_full, reference=True)
                        if not audit["fit_converged"]:
                            raise RuntimeError(f"OBD ridge sensitivity reference failed at rho={rr}")
                        sensitivity[str(rr)] = audit
                        sensitivity_rhos.append(rr)
                        sensitivity_thetas.append(th)
                metadata = {
                    "dataset": dataset, "d": int(d), "family": family, "rho": rho,
                    "reference_hash": signature, "n_pool": len(y),
                    "reference_scope": "fixed_pool_all_labels_evaluation_only",
                    "reference_is_true_parameter": False,
                    "matched_reference_note": "matches mean-loss ridge; does not remove adaptive covariate selection bias",
                    "full_fit_audit": full_audit, "matched_fit_audits": matched_audits,
                    "rho_sensitivity": sensitivity,
                    "feature_rank": int(np.linalg.matrix_rank(X)),
                    "full_hessian_spectrum": np.linalg.eigvalsh(H_full).tolist(),
                    "positive_pool": int(np.sum(y > 0)), "negative_pool": int(np.sum(y == 0)),
                    "warning_codes": (["LOW_EVENT_REFERENCE"] if family == "bernoulli" and
                                      min(np.sum(y > 0), np.sum(y == 0)) < 5 * d else []),
                    "numerical_policy": NUMERICAL_POLICY,
                }
                _write_npz(path, theta_full=theta_full, matched_ns=np.asarray(sorted(ns), dtype=np.int64),
                           theta_matched=np.stack([matched[n] for n in sorted(ns)]),
                           H_full=H_full, full_spectrum=np.linalg.eigvalsh(H_full),
                           matched_hessians=np.stack(hessians),
                           rho_sensitivity=np.asarray(sensitivity_rhos),
                           theta_rho_sensitivity=np.asarray(sensitivity_thetas),
                           reference_hash=np.asarray(signature))
                _write_json(json_path, metadata)
                output[dataset][int(d)] = {"path": str(path), "json_path": str(json_path), "reference_hash": signature}
    return output


def _matrix_metrics(matrix, label, rho, n):
    spectrum = np.linalg.eigvalsh((matrix + matrix.T) * 0.5)
    data_ev = spectrum - rho
    tol = max(1.0, float(np.max(np.abs(spectrum)))) * NUMERICAL_POLICY["rank_relative_tolerance"]
    metrics = {
        f"lambda_min_{label}": float(spectrum[0]),
        f"lambda_max_{label}": float(spectrum[-1]),
        f"trace_{label}": float(np.sum(spectrum)),
        f"logdet_{label}": float(np.sum(np.log(spectrum))) if spectrum[0] > 0 else None,
        f"condition_{label}": float(spectrum[-1] / spectrum[0]) if spectrum[0] > 0 else None,
        f"trace_inverse_{label}": float(np.sum(1.0 / spectrum)) if spectrum[0] > 0 else None,
        f"inverse_lambda_min_{label}": float(1.0 / spectrum[0]) if spectrum[0] > 0 else None,
        f"lambda_min_{label}_no_ridge": float(data_ev[0]),
        f"trace_{label}_no_ridge": float(np.sum(data_ev)),
        f"rank_{label}_no_ridge": int(np.sum(data_ev > tol)),
        f"info_rate_{label}_total": float(spectrum[0] / n),
        f"info_rate_{label}_no_ridge_total": float(data_ev[0] / n),
    }
    return metrics, spectrum


def evaluate_checkpoint(X_pool, y_pool, X_test, y_test, selected_ids, theta,
                        family, rho, theta_full, theta_matched, fit_audit, extra=None):
    """Return all mandatory scalar evaluation and raw vector/matrix arrays.

    ``selected_ids`` are zero-based pool positions; caller separately records
    the stable original row IDs. Extra may contain checkpoint metadata, n0,
    warm_reference_lambda, warm_plugin_lambda, and curvature_floor.
    """
    started = time.perf_counter()
    extra = dict(extra or {})
    ids = np.asarray(selected_ids, dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(X_pool)):
        raise ValueError("Selected pool positions must be unique and valid")
    theta, theta_full, theta_matched = map(lambda x: np.asarray(x, dtype=np.float64), [theta, theta_full, theta_matched])
    d, n = len(theta), len(ids)
    X_pool, X_test = np.asarray(X_pool, dtype=np.float64), np.asarray(X_test, dtype=np.float64)
    y_pool, y_test = np.asarray(y_pool, dtype=np.float64), np.asarray(y_test, dtype=np.float64)
    X, y = X_pool[ids], y_pool[ids]
    n0 = int(extra.pop("n0", 80))
    floor = float(extra.pop("curvature_floor", 1e-12)) if family != "gaussian" else 0.0
    warm_ref = extra.pop("warm_reference_lambda", None)
    warm_hat = extra.pop("warm_plugin_lambda", None)
    eta = X @ theta
    _, w_hat = quantities(eta, family)
    _, w_ref = quantities(X @ theta_full, family)
    w_acq = np.maximum(w_hat, floor)
    identity = rho * np.eye(d)
    matrices = {"V_ref": identity + (X.T * w_ref) @ X,
                "V_hat": identity + (X.T * w_hat) @ X,
                "V_acq": identity + (X.T * w_acq) @ X,
                "H_design": identity + X.T @ X}
    for key, matrix in matrices.items():
        matrices[key] = (matrix + matrix.T) * 0.5
    error, matched_error = theta - theta_full, theta - theta_matched
    metrics = {**fit_audit, **extra, "family": family, "d": d, "n_pool": len(y_pool),
               "n_total": n, "n0": n0, "n_additional": n - n0}
    arrays = {**matrices, "theta": theta, "theta_hat": theta, "error_ref": error,
              "error_matched": matched_error, "coordinate_squared_error_ref": error ** 2,
              "coordinate_squared_error_matched": matched_error ** 2,
              "selected_ids": ids}
    warnings = list(metrics.get("warning_codes", []))
    for key, label in [("V_ref", "reference"), ("V_hat", "plugin"),
                       ("V_acq", "acquisition"), ("H_design", "design")]:
        current, spectrum = _matrix_metrics(matrices[key], label, rho, n)
        metrics.update(current)
        arrays[f"spectrum_{key}"] = spectrum
        arrays[f"eigenvalues_{key}"] = spectrum
        arrays[f"spectrum_{key}_no_ridge"] = spectrum - rho
        if spectrum[0] - rho < -1e-10 * max(1, abs(spectrum[-1])):
            warnings.append(f"NON_PSD_DATA_INFORMATION_{key}")
    for label, value in [("reference", warm_ref), ("plugin", warm_hat)]:
        if value is None and n == n0:
            value = metrics[f"lambda_min_{label}"]
        metrics[f"info_gain_{label}_since_warm"] = (
            float(metrics[f"lambda_min_{label}"] - value) if value is not None else None)
        if value is None:
            warnings.append(f"MISSING_WARM_{label.upper()}_EIGENVALUE")
    delta = matrices["V_hat"] - matrices["V_ref"]
    delta_spectrum = np.linalg.eigvalsh(delta)
    tracking = float(np.max(np.abs(delta_spectrum)))
    floor_delta = matrices["V_acq"] - matrices["V_hat"]
    gap = metrics["lambda_min_plugin"] - metrics["lambda_min_reference"]
    if abs(gap) > tracking + 1e-9 * max(1, metrics["lambda_max_reference"]):
        warnings.append("WEYL_TRACKING_INEQUALITY_FAILED")
    metrics.update(
        parameter_l2_to_full=float(np.linalg.norm(error)),
        parameter_l2_squared_to_full=float(error @ error),
        parameter_l2_to_full_matched=float(np.linalg.norm(matched_error)),
        parameter_l2_squared_to_full_matched=float(matched_error @ matched_error),
        theta_norm_l2=float(np.linalg.norm(theta)),
        tracking_op=tracking, tracking_op_per_total=tracking / n,
        tracking_fro=float(np.linalg.norm(delta, "fro")),
        tracking_relative_op=tracking / metrics["lambda_max_reference"],
        min_eigenvalue_gap=float(gap),
        min_eigenvalue_ratio_plugin_reference=metrics["lambda_min_plugin"] / metrics["lambda_min_reference"],
        acquisition_vs_plugin_op=float(np.max(np.abs(np.linalg.eigvalsh(floor_delta)))),
        curvature_floor_count=int(np.sum(w_hat < floor)),
        curvature_floor_fraction=float(np.mean(w_hat < floor)),
        curvature_floor_added_mass=float(np.sum(w_acq - w_hat)),
        selected_positive_count=int(np.sum(y > 0)), selected_nonzero_count=int(np.sum(y != 0)),
        selected_response_mean=float(np.mean(y)), selected_response_variance=float(np.var(y)),
        selected_response_min=float(np.min(y)), selected_response_max=float(np.max(y)),
        current_curvature_min=float(np.min(w_hat)), current_curvature_max=float(np.max(w_hat)),
        current_tiny_curvature_fraction=float(np.mean(w_hat < 1e-12)),
        reference_curvature_min=float(np.min(w_ref)), reference_curvature_max=float(np.max(w_ref)),
        reference_tiny_curvature_fraction=float(np.mean(w_ref < 1e-12)),
    )
    arrays["spectrum_selected_gram"] = np.linalg.eigvalsh(X.T @ X)
    arrays["tracking_delta_spectrum"] = delta_spectrum
    eta_pool = X_pool @ theta
    eta_test = X_test @ theta
    mu_test, _ = quantities(eta_test, family)
    pool_loss = float(np.sum(loss_values(eta_pool, y_pool, family)))
    fixed_pool_objective = pool_loss + 0.5 * rho * float(theta @ theta)
    rho_matched = rho * len(y_pool) / n
    metrics.update(
        pool_loss_mean=pool_loss / len(y_pool),
        pool_nll_mean=float(np.mean(loss_values(eta_pool, y_pool, family, constants=True))),
        pool_penalized_objective_sum=fixed_pool_objective,
        full_objective_gap_per_row=(fixed_pool_objective - objective(theta_full, X_pool, y_pool, family, rho)) / len(y_pool),
        matched_full_objective_gap_per_row=(pool_loss + 0.5 * rho_matched * float(theta @ theta) -
                                           objective(theta_matched, X_pool, y_pool, family, rho_matched)) / len(y_pool),
        selected_reference_objective_gap=objective(theta_full, X, y, family, rho) - objective(theta, X, y, family, rho),
        test_nll=float(np.mean(loss_values(eta_test, y_test, family, constants=True))),
        test_rmse=None, test_mae=None, test_brier=None, test_auroc=None, test_ap=None,
        test_poisson_deviance=None,
    )
    na_reasons = {key: f"NOT_APPLICABLE_{family.upper()}" for key in
                  ["test_rmse", "test_mae", "test_brier", "test_auroc", "test_ap", "test_poisson_deviance"]}
    if family == "gaussian":
        metrics["test_rmse"] = float(np.sqrt(np.mean((mu_test - y_test) ** 2)))
        metrics["test_mae"] = float(np.mean(np.abs(mu_test - y_test)))
    elif family == "bernoulli":
        metrics["test_brier"] = float(np.mean((mu_test - y_test) ** 2))
        if len(np.unique(y_test)) == 2:
            metrics["test_auroc"] = float(roc_auc_score(y_test, eta_test))
            metrics["test_ap"] = float(average_precision_score(y_test, eta_test))
        else:
            na_reasons["test_auroc"] = na_reasons["test_ap"] = "SINGLE_CLASS_TEST"
    else:
        # y log y - y eta - y + exp(eta) avoids log(exp(eta))
        # underflow and defines y=0 exactly; no clipping or count alteration.
        metrics["test_poisson_deviance"] = float(np.mean(2 * (xlogy(y_test, y_test) - y_test * eta_test - y_test + mu_test)))
    for key in list(na_reasons):
        if metrics[key] is not None:
            del na_reasons[key]
    metrics["metric_na_reasons"] = {**metrics.get("metric_na_reasons", {}), **na_reasons}
    for key in ["full_objective_gap_per_row", "matched_full_objective_gap_per_row", "selected_reference_objective_gap"]:
        if metrics[key] < -1e-7 * max(1, abs(metrics["train_penalized_objective_sum"])):
            warnings.append(f"NEGATIVE_OPTIMIZER_GAP_{key}")
    metrics["prediction_overflow_count"] = int(np.sum(~np.isfinite(mu_test)))
    if metrics["prediction_overflow_count"]:
        warnings.append("PREDICTIVE_MEAN_OVERFLOW_UNCLIPPED")
    if family == "bernoulli" and min(np.sum(y_pool > 0), np.sum(y_pool == 0)) < 5 * d:
        warnings.append("LOW_EVENT_REFERENCE")
    metrics["metrics_complete"] = bool(fit_audit["fit_converged"] and not any(
        code.startswith(("MISSING_WARM", "WEYL_TRACKING", "NON_PSD", "NEGATIVE_OPTIMIZER")) for code in warnings))
    metrics["warning_codes"] = sorted(set(warnings))
    metrics["evaluation_seconds"] = time.perf_counter() - started
    return metrics, arrays
