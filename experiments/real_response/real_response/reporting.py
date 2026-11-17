"""Rebuild risk summaries, paired descriptive intervals and publication plots.

All values originate in committed run artifacts. Incomplete seed sets stay
visible and are never promoted to a full ten-seed result.
"""
from __future__ import annotations

import hashlib
import csv
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .aggregate_risk import bootstrap_risks, risk_statistics, scalar_summary


def read_csv_safe(path):
    """Read saved scalar tables without pandas native numeric string inference.

    This installed pandas build segfaulted in lib.maybe_convert_numeric on real
    numeric metadata strings. Python int/float parsing retains the same values
    while identity and structured audit columns stay strings.
    """
    string_fields = {"dataset", "subset_version", "family", "method", "backend", "gpu_uuid",
                     "fit_status", "fit_solver", "sdp_status", "fallback_reason", "warning_codes",
                     "used_seed_ids", "metric_na_reasons", "fit_attempts", "loss_convention",
                     "risk_uncertainty_method", "comparator", "difference_direction", "uncertainty_scope"}
    def scalar(key, value):
        if value is None or value == "":
            return np.nan
        if key in string_fields or "hash" in key or key.endswith("_reason"):
            return value
        if value == "True":
            return True
        if value == "False":
            return False
        try:
            return int(value) if re.fullmatch(r"[+-]?\d+", value) else float(value)
        except (ValueError, OverflowError):
            return value
    with open(path, newline="") as handle:
        rows = [{key: scalar(key, value) for key, value in row.items()} for row in csv.DictReader(handle)]
    return pd.DataFrame(rows)


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
    os.replace(tmp, path)


def _csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _seed_indices(dataset, d, seed_ids, draws):
    # Same paired seed identities -> identical resamples across methods.
    salt = json.dumps(["descriptive_whole_seed_bootstrap_v2", dataset, d, list(seed_ids)])
    seed = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).integers(0, len(seed_ids), (draws, len(seed_ids)))


def _safe_mean(rows, key):
    values = pd.to_numeric(pd.Series([r.get(key) for r in rows]), errors="coerce").to_numpy(dtype=float)
    return scalar_summary(values)


def summarize(config, out, seeds=None):
    source_out = Path(out)
    out = source_out / "preview" if seeds is not None else source_out
    out.mkdir(parents=True, exist_ok=True)
    expected_seeds = list(map(int, seeds if seeds is not None else config.get("run_seeds", list(range(10)))))
    draws = int(config.get("uncertainty", {}).get("bootstrap_draws", 1000))
    groups = {}
    load_failures, run_count, completed_runs = [], 0, 0
    methods = config.get("methods", [])
    dimensions = config.get("dimensions", [5, 10, 20])
    datasets = config["datasets"]
    expected_runs = sum(len([m for m in methods if m != "badge" or ds == "obd"]) *
                        len(dimensions) * len(expected_seeds) for ds in datasets)
    for ds, dc in datasets.items():
        for d in dimensions:
            for method in methods:
                if method == "badge" and ds != "obd":
                    continue
                key = (ds, int(d), method)
                groups[key] = {}
                for seed in expected_seeds:
                    p = source_out / "runs" / ds / str(d) / str(seed) / method
                    if not (p / "trajectory.csv").exists():
                        continue
                    try:
                        # The environment's native pandas CSV parser segfaulted
                        # on a real saved audit string. Python's standards-based
                        # CSV reader preserves the data and avoids that extension.
                        frame = read_csv_safe(p / "trajectory.csv")
                        error = np.load(p / "error_ref_checkpoints.npy", allow_pickle=False)
                        matched = np.load(p / "error_matched_checkpoints.npy", allow_pickle=False)
                        with np.load(p / "information_matrices.npz", allow_pickle=False) as z:
                            information = {k: z[k] for k in ["V_ref", "V_hat", "V_acq", "H_design"]}
                        if len(frame) != len(error) or error.shape != matched.shape:
                            raise ValueError("CSV/error checkpoint shape mismatch")
                        if error.shape[1] != d or not np.all(np.isfinite(error)):
                            raise ValueError("Invalid error vector shape/value")
                        if frame["round"].duplicated().any():
                            raise ValueError("Duplicate checkpoint rounds")
                        groups[key][seed] = dict(frame=frame, error=error, matched=matched, information=information)
                        run_count += 1
                        required = 1 + int(np.ceil(dc["additional_labels"] / dc["batch_size"]))
                        if len(frame) == required and bool(frame["metrics_complete"].all()):
                            completed_runs += 1
                    except Exception as exc:
                        load_failures.append(dict(dataset=ds, d=d, seed=seed, method=method, error=repr(exc)))

    all_rows, endpoints, endpoint_errors = [], [], {}
    for (ds, d, method), runs in groups.items():
        if not runs:
            continue
        p = out / "aggregate" / ds / str(d) / method
        p.mkdir(parents=True, exist_ok=True)
        round_ids = sorted(set(int(r) for run in runs.values() for r in run["frame"]["round"]))
        rows, arrays_by_key = [], {}
        for round_id in round_ids:
            used = sorted(seed for seed, run in runs.items() if round_id in set(run["frame"]["round"]))
            entries, E, M, ref_matrices, hat_matrices = [], [], [], [], []
            for seed in used:
                run = runs[seed]
                ix = int(np.flatnonzero(run["frame"]["round"].to_numpy() == round_id)[0])
                entries.append(run["frame"].iloc[ix].to_dict())
                E.append(run["error"][ix]); M.append(run["matched"][ix])
                ref_matrices.append(run["information"]["V_ref"][ix])
                hat_matrices.append(run["information"]["V_hat"][ix])
            E, M = np.stack(E), np.stack(M)
            fixed, fixed_arrays = risk_statistics(E)
            matched, matched_arrays = risk_statistics(M)
            counts = set((int(e["n_total"]), int(e["n_additional"])) for e in entries)
            if len(counts) != 1:
                raise ValueError(f"Cannot aggregate unmatched label budgets: {ds}/{d}/{method}/{round_id}")
            n_total, n_additional = counts.pop()
            for identity in ["feature_hash", "reference_hash"]:
                identities = {str(e.get(identity)) for e in entries}
                if len(identities) != 1 or identities.intersection({"None", "nan", ""}):
                    raise ValueError(f"Cannot aggregate inconsistent {identity}: {ds}/{d}/{method}/{round_id}: {identities}")
            complete_ten = used == list(range(10))
            warnings = []
            if used != sorted(expected_seeds):
                warnings.append("MISSING_EXPECTED_SEEDS")
            if not complete_ten:
                warnings.append("PRELIMINARY_NOT_COMPLETE_10_SEED")
            if d > len(used):
                warnings.append("RISK_MATRIX_RANK_LIMITED_BY_SEED_COUNT")
            if fixed["top_eigenvalue_degenerate"]:
                warnings.append("TOP_RISK_EIGENVALUE_DEGENERATE")
            row = dict(dataset=ds, d=d, method=method, round=round_id, n_total=n_total,
                       n_additional=n_additional, expected_seed_count=len(expected_seeds),
                       actual_seed_count=len(used), used_seed_ids=json.dumps(used),
                       complete_10_seed=complete_ten,
                       warning_codes=";".join(warnings),
                       risk_uncertainty_method="descriptive_whole_seed_bootstrap_endpoint_and_leave_one_seed_out",
                       bootstrap_draws=draws, bootstrap_endpoint_only=True)
            row["feature_hash"] = entries[0]["feature_hash"]
            row["reference_hash"] = entries[0]["reference_hash"]
            row["n0"] = int(entries[0]["n0"])
            row["planned_additional_budget"] = int(config["datasets"][ds]["additional_labels"])
            row["planned_batch_size"] = int(config["datasets"][ds]["batch_size"])
            row["rho"] = float(config.get("rho", 1.0))
            for suffix, values in [("ref", fixed), ("matched", matched)]:
                for name in ["R2", "R2_squared", "Rdir", "Rdir_squared"]:
                    row[f"{name}_{suffix}"] = values[name]
                for name in ["bias_norm_squared", "centered_covariance_trace", "centered_covariance_lambda_max"]:
                    row[f"{name}_{suffix}"] = values[name]
                row[f"Q_{suffix}_rank"] = values["Q_rank"]
                row[f"Q_{suffix}_top_eigengap"] = values["Q_top_eigengap"]
                row[f"Q_{suffix}_trace"] = values["Q_trace"]
                row[f"top_eigenspace_dimension_{suffix}"] = values["top_eigenspace_dimension"]
                row[f"Rdir_squared_loo_min_{suffix}"] = values["Rdir_squared_loo_min"]
                row[f"Rdir_squared_loo_max_{suffix}"] = values["Rdir_squared_loo_max"]
            for key in ["Rdir_squared_loo_min", "Rdir_squared_loo_max"]:
                row[key] = fixed[key]
            norms = scalar_summary(np.linalg.norm(E, axis=1))
            for key in ["mean", "sd", "sem"]:
                row[f"{key}_l2_ref"] = norms[key]
            row["mean_l2_ref_ci_low"], row["mean_l2_ref_ci_high"] = norms["ci_low"], norms["ci_high"]
            row["mean_l2_matched"] = float(np.linalg.norm(M, axis=1).mean())
            for number, name in [(n_total, "n_total"), (n_additional, "n_additional")]:
                row[f"{name}_Rdir_squared_ref"] = number * fixed["Rdir_squared"]
                row[f"{name}_R2_squared_ref"] = number * fixed["R2_squared"]
            for name in ["lambda_min_reference", "lambda_min_plugin", "tracking_op_per_total", "test_nll",
                         "full_objective_gap_per_row", "matched_full_objective_gap_per_row",
                         "lambda_min_reference_no_ridge", "lambda_min_plugin_no_ridge",
                         "info_rate_reference_total", "info_rate_plugin_total",
                         "info_rate_reference_no_ridge_total", "info_rate_plugin_no_ridge_total",
                         "cumulative_elapsed_seconds", "fit_seconds", "acquisition_seconds", "evaluation_seconds", "io_seconds",
                         "test_rmse", "test_mae", "test_brier", "test_auroc", "test_ap", "test_poisson_deviance"]:
                stats = _safe_mean(entries, name)
                for statistic in ["mean", "sd", "sem", "ci_low", "ci_high"]:
                    row[f"{statistic}_{name}"] = stats[statistic]
            mean_ref, mean_hat = np.mean(ref_matrices, axis=0), np.mean(hat_matrices, axis=0)
            row["lambda_min_of_mean_V_ref"] = float(np.linalg.eigvalsh(mean_ref)[0])
            row["lambda_min_of_mean_V_hat"] = float(np.linalg.eigvalsh(mean_hat)[0])
            raw_arrays = {"mean_V_ref": mean_ref, "mean_V_hat": mean_hat}
            for suffix, data in [("ref", fixed_arrays), ("matched", matched_arrays)]:
                for key, value in data.items():
                    if key == "Rdir_squared_leave_one_out":
                        # Preserve seed identity even with missing round entries.
                        padded = np.full(len(expected_seeds), np.nan)
                        for j, seed in enumerate(used):
                            if len(value):
                                padded[expected_seeds.index(seed)] = value[j]
                        value = padded
                    raw_arrays[f"{key}_{suffix}"] = value
            for key, value in raw_arrays.items():
                arrays_by_key.setdefault(key, []).append(value)
            if round_id == round_ids[-1]:
                required_round = int(np.ceil(config["datasets"][ds]["additional_labels"] / config["datasets"][ds]["batch_size"]))
                row["is_planned_endpoint"] = round_id == required_round
                row["fallback_batch_count"] = int(sum(
                    pd.to_numeric(run["frame"].get("fallback_used", pd.Series(0, index=run["frame"].index)), errors="coerce").fillna(0).sum()
                    for run in runs.values()))
                row["runs_with_fallback"] = int(sum(
                    pd.to_numeric(run["frame"].get("fallback_used", pd.Series(0, index=run["frame"].index)), errors="coerce").fillna(0).any()
                    for run in runs.values()))
                current_trajectory = rows + [row]
                for metric in ["mean_l2_ref", "R2_squared_ref", "Rdir_squared_ref", "mean_lambda_min_reference",
                               "mean_lambda_min_plugin", "mean_tracking_op_per_total", "mean_test_nll",
                               "mean_full_objective_gap_per_row", "Rdir_squared_matched"]:
                    x = np.array([r["n_additional"] for r in current_trajectory], dtype=float)
                    yy = np.array([r[metric] for r in current_trajectory], dtype=float)
                    integral = np.trapezoid(yy, x) if hasattr(np, "trapezoid") else np.trapz(yy, x)
                    row[f"full_budget_normalized_integral_{metric}"] = (
                        float(integral / config["datasets"][ds]["additional_labels"])
                        if row["is_planned_endpoint"] and len(x) == required_round + 1 and x[0] == 0 else None)
                indices = _seed_indices(ds, d, used, draws)
                bootstrap = {"resample_indices": indices, "used_seed_ids": np.asarray(used)}
                for suffix, errors in [("ref", E), ("matched", M)]:
                    bs = bootstrap_risks(errors, indices)
                    for key, values in bs.items():
                        bootstrap[f"{key}_{suffix}"] = values
                        low, high = np.quantile(values, [0.025, 0.975])
                        row[f"{key}_{suffix}_bootstrap_ci_low"] = float(low)
                        row[f"{key}_{suffix}_bootstrap_ci_high"] = float(high)
                np.savez_compressed(p / "endpoint_seed_bootstrap.npz", **bootstrap)
                endpoint_errors[(ds, d, method)] = (used, E, entries, bool(row["is_planned_endpoint"]))
                endpoints.append(row.copy())
            rows.append(row)
            all_rows.append(row)
        _csv(p / "risk_trajectory.csv", rows)
        stacked = {key: np.stack(value) for key, value in arrays_by_key.items()}
        np.save(p / "Q_ref_checkpoints.npy", stacked["Q_ref"])
        np.save(p / "Q_matched_checkpoints.npy", stacked["Q_matched"])
        np.savez_compressed(p / "risk_spectra_projectors.npz", **{k: v for k, v in stacked.items()
                            if any(s in k for s in ["eigenvalue", "eigenvector", "projector"])})
        np.savez_compressed(p / "bias_covariance_checkpoints.npz", **{k: v for k, v in stacked.items()
                            if "bias" in k or "covariance" in k})
        np.savez_compressed(p / "mean_information_checkpoints.npz", mean_V_ref=stacked["mean_V_ref"], mean_V_hat=stacked["mean_V_hat"])
        np.savez_compressed(p / "loo_risk_values.npz", ref=stacked["Rdir_squared_leave_one_out_ref"],
                            matched=stacked["Rdir_squared_leave_one_out_matched"], seed_ids=expected_seeds)
        _csv(p / "loo_risk_ranges.csv", [{k: row[k] for k in ["round", "n_total", "n_additional", "actual_seed_count", "Rdir_squared_loo_min", "Rdir_squared_loo_max"]} for row in rows])

    paired = []
    for (ds, d, method), (used, E, entries, is_endpoint) in endpoint_errors.items():
        random = endpoint_errors.get((ds, d, "random"))
        if method == "random" or random is None or not is_endpoint or not random[3]:
            continue
        common = sorted(set(used) & set(random[0]))
        if not common:
            continue
        ia, ib = [used.index(s) for s in common], [random[0].index(s) for s in common]
        A, B = E[ia], random[1][ib]
        indices = _seed_indices(ds, d, common, draws)
        ba, bb = bootstrap_risks(A, indices), bootstrap_risks(B, indices)
        row = dict(dataset=ds, d=d, method=method, comparator="random", actual_seed_count=len(common), used_seed_ids=json.dumps(common),
                   complete_10_seed=common == list(range(10)), difference_direction="method_minus_random",
                   uncertainty_scope="descriptive_paired_algorithmic_seeds_fixed_pool")
        for metric in ["parameter_l2_to_full", "test_nll", "full_objective_gap_per_row"]:
            delta = [entries[a][metric] - random[2][b][metric] for a, b in zip(ia, ib)]
            for key, value in scalar_summary(delta).items():
                row[f"paired_{metric}_{key}"] = value
        for metric in ["Rdir_squared", "R2_squared"]:
            delta = ba[metric] - bb[metric]
            row[f"paired_{metric}_difference"] = risk_statistics(A)[0][metric] - risk_statistics(B)[0][metric]
            row[f"paired_{metric}_bootstrap_ci_low"], row[f"paired_{metric}_bootstrap_ci_high"] = map(float, np.quantile(delta, [0.025, 0.975]))
        paired.append(row)

    comparison_metrics = ["mean_l2_ref", "R2_squared_ref", "Rdir_squared_ref", "mean_lambda_min_reference",
                          "mean_lambda_min_plugin", "mean_tracking_op_per_total", "mean_test_nll", "mean_full_objective_gap_per_row",
                          "mean_cumulative_elapsed_seconds"]
    for row in endpoints:
        same = [r for r in endpoints if r["dataset"] == row["dataset"] and r["d"] == row["d"]]
        random = next((r for r in same if r["method"] == "random"), None)
        for metric in comparison_metrics:
            value = row.get(metric)
            descending = metric in ["mean_lambda_min_reference", "mean_lambda_min_plugin"]
            peers = [r.get(metric) for r in same if r.get(metric) is not None and np.isfinite(r.get(metric))]
            row[f"within_condition_rank_{metric}"] = (1 + sum(v > value if descending else v < value for v in peers)
                                                       if value is not None and np.isfinite(value) else None)
            common_peers = [r.get(metric) for r in same if r["method"] != "badge" and r.get(metric) is not None and np.isfinite(r.get(metric))]
            row[f"within_condition_common_rank_{metric}"] = (
                1 + sum(v > value if descending else v < value for v in common_peers)
                if row["method"] != "badge" and value is not None and np.isfinite(value) else None)
            if random is None or value is None or random.get(metric) is None:
                row[f"random_relative_ratio_{metric}"] = None
                row[f"random_relative_ratio_reason_{metric}"] = "MISSING_RANDOM_OR_METRIC"
                continue
            baseline = random[metric]
            row[f"random_difference_{metric}"] = value - baseline
            row[f"random_relative_ratio_{metric}"] = value / baseline if abs(baseline) > 1e-14 else None
            row[f"random_relative_ratio_reason_{metric}"] = None if abs(baseline) > 1e-14 else "NEAR_ZERO_RANDOM_DENOMINATOR"
            delta = value - baseline
            tie = abs(delta) <= 1e-10 * max(1.0, abs(value), abs(baseline))
            row[f"random_numerical_win_tie_loss_{metric}"] = "tie" if tie else "win" if (delta > 0 if descending else delta < 0) else "loss"
    _csv(out / "summary_by_condition.csv", endpoints)
    _csv(out / "summary_common_methods.csv", [row for row in endpoints if row["method"] != "badge"])
    _csv(out / "summary_bernoulli_methods.csv", [row for row in endpoints if row["dataset"] == "obd"])
    _csv(out / "paired_comparisons.csv", paired)
    _csv(out / "aggregate_load_failures.csv", load_failures)
    _csv(out / "plot_data" / "all_risk_trajectories.csv", all_rows)
    status = dict(expected_trajectories=expected_runs, loaded_trajectories=run_count,
                  completed_trajectories=completed_runs, incomplete_trajectories=expected_runs-completed_runs,
                  load_failures=len(load_failures), aggregate_checkpoints=len(all_rows),
                  endpoint_conditions=len(endpoints), requested_seeds=expected_seeds,
                  complete_10_seed=expected_seeds == list(range(10)) and completed_runs == expected_runs and not load_failures,
                  report_kind="10-seed full" if expected_seeds == list(range(10)) and completed_runs == expected_runs else "preliminary/incomplete",
                  metrics_scope="fixed_real_pool_reference_not_true_parameter_risk")
    _json(out / "aggregation_status.json", status)
    if seeds is None:
        make_figures(out, pd.DataFrame(all_rows), status)
    export_paper(config, out)
    _write_report(config, out, status, endpoints)
    return status


def make_figures(out, frame, status):
    if frame.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    dest = Path(out) / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    panels = [("mean_l2_ref", "Mean parameter L2 to full-pool reference"),
              ("Rdir_squared_ref", "Uncentered directional risk Rdir squared"),
              ("mean_lambda_min_reference", "Mean minimum eigenvalue: V_ref"),
              ("mean_lambda_min_plugin", "Mean minimum eigenvalue: V_hat"),
              ("mean_tracking_op_per_total", "Information tracking operator norm / total labels"),
              ("mean_test_nll", "Heldout mean full negative log-likelihood")]
    secondary = [("R2_squared_ref", "Uncentered total risk R2 squared"),
                 ("Rdir_squared_matched", "Matched-reference directional risk squared"),
                 ("mean_full_objective_gap_per_row", "Fixed full-pool objective gap / pool rows"),
                 ("mean_matched_full_objective_gap_per_row", "Matched full-pool objective gap / pool rows")]
    information = [("mean_lambda_min_reference_no_ridge", "Mean minimum data-information eigenvalue: reference"),
                   ("mean_lambda_min_plugin_no_ridge", "Mean minimum data-information eigenvalue: plug-in"),
                   ("mean_info_rate_reference_total", "Reference regularized information / total labels"),
                   ("mean_info_rate_plugin_total", "Plug-in regularized information / total labels"),
                   ("mean_info_rate_reference_no_ridge_total", "Reference data information / total labels"),
                   ("mean_info_rate_plugin_no_ridge_total", "Plug-in data information / total labels")]
    roots = [("R2_ref", "Root total risk R2 to full-pool reference"),
             ("Rdir_ref", "Root directional risk Rdir to full-pool reference"),
             ("R2_matched", "Root total risk R2 to matched reference"),
             ("Rdir_matched", "Root directional risk Rdir to matched reference")]
    individual = [
        ("info_reference", "mean_lambda_min_reference", "Mean minimum eigenvalue of V_ref"),
        ("info_plugin", "mean_lambda_min_plugin", "Mean minimum eigenvalue of V_hat"),
        ("l2_reference", "mean_l2_ref", "Mean parameter L2 to full-pool reference"),
        ("risk_dir_squared_reference", "Rdir_squared_ref", "Directional squared risk Rdir squared"),
        ("risk_total_squared_reference", "R2_squared_ref", "Total squared risk R2 squared"),
        ("information_tracking", "mean_tracking_op_per_total", "Information operator-norm discrepancy / total labels"),
        ("test_loss", "mean_test_nll", "Heldout mean full negative log-likelihood"),
    ]
    for (ds, d), condition in frame.groupby(["dataset", "d"], sort=True):
        for suffix, spec, shape in [("primary", panels, (3, 2)), ("diagnostics", secondary, (2, 2)),
                                    ("information_rates", information, (3, 2)), ("root_risks", roots, (2, 2))]:
            fig, axes = plt.subplots(*shape, figsize=(12, 3.25 * shape[0]), squeeze=False)
            for ax, (column, title) in zip(axes.flat, spec):
                for method, data in condition.groupby("method", sort=True):
                    data = data.sort_values("n_additional")
                    ax.plot(data["n_additional"], data[column], label=method, linewidth=1.25,
                            linestyle="--" if method == "ucb_glm" and condition["method"].nunique() > 10 else "-")
                    if column == "mean_l2_ref":
                        ax.fill_between(data["n_additional"].to_numpy(dtype=float),
                                        data["mean_l2_ref_ci_low"].to_numpy(dtype=float),
                                        data["mean_l2_ref_ci_high"].to_numpy(dtype=float), alpha=0.08)
                ax.set_title(title)
                if column in ["mean_lambda_min_reference", "mean_lambda_min_plugin"]:
                    ax.axhline(float(condition["rho"].iloc[0]), color="black", linestyle=":", linewidth=0.9)
                ax.set_xlabel("Additional real responses")
                ax.grid(alpha=0.2)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8)
            seed_counts = sorted(set(condition["actual_seed_count"]))
            qualifier = "10 paired seeds" if seed_counts == [10] else f"PRELIMINARY: actual seeds {seed_counts}"
            fig.suptitle(f"{ds}, d={d} (intercept included); {qualifier}")
            fig.tight_layout(rect=[0, 0.065, 1, 0.97])
            for ext in ["pdf", "png", "svg"]:
                fig.savefig(dest / f"{ds}_d{d}_{suffix}.{ext}", dpi=150)
            plt.close(fig)
        family_extra = [("test_rmse", "mean_test_rmse", "Heldout rating RMSE")] if ds == "movielens" else (
            [("test_poisson_deviance", "mean_test_poisson_deviance", "Heldout mean Poisson deviance")] if ds == "lastfm" else [])
        for figure_key, column, ylabel in individual + family_extra:
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            for method, data in condition.groupby("method", sort=True):
                data = data.sort_values("n_additional")
                ax.plot(data["n_additional"], data[column], label=method, linewidth=1.4,
                        linestyle="--" if method == "ucb_glm" and condition["method"].nunique() > 10 else "-")
                mean_interval = ("mean_l2_ref_ci_low", "mean_l2_ref_ci_high") if column == "mean_l2_ref" else (
                    ("ci_low_" + column[5:], "ci_high_" + column[5:]) if column.startswith("mean_") else None)
                if mean_interval and all(k in data for k in mean_interval):
                    ax.fill_between(data["n_additional"].to_numpy(dtype=float), data[mean_interval[0]].to_numpy(dtype=float),
                                    data[mean_interval[1]].to_numpy(dtype=float), alpha=0.08)
            if figure_key in ["info_reference", "info_plugin"]:
                ax.axhline(float(condition["rho"].iloc[0]), color="black", linestyle=":", linewidth=0.9, label="ridge baseline")
            ax.set_xlabel("Additional real responses")
            ax.set_ylabel(ylabel)
            counts = sorted(set(condition["actual_seed_count"]))
            ax.set_title(f"{ds}, d={d}; " + ("10 paired seeds" if counts == [10] else f"PRELIMINARY actual S={counts}"))
            ax.grid(alpha=0.2)
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
            fig.tight_layout()
            for ext in ["pdf", "png", "svg"]:
                fig.savefig(dest / f"{ds}_d{d}_{figure_key}.{ext}", dpi=150)
            plt.close(fig)
            columns = ["dataset", "d", "method", "round", "n_total", "n_additional", "actual_seed_count", "used_seed_ids",
                       "n0", "planned_additional_budget", "planned_batch_size", "rho", "feature_hash", "reference_hash", column]
            _csv(Path(out) / "plot_data" / f"{ds}_d{d}_{figure_key}.csv", condition[columns])


def export_paper(config, out):
    out = Path(out)
    dest = out / "paper"
    dest.mkdir(parents=True, exist_ok=True)
    summary = out / "summary_by_condition.csv"
    frame = read_csv_safe(summary) if summary.exists() and summary.stat().st_size > 1 else pd.DataFrame()
    complete = bool(not frame.empty and frame["complete_10_seed"].all())
    status = "completed ten-seed cells" if complete else "preliminary completed records; missing seeds remain explicit"
    names = {"obd": "Open Bandit Dataset-small (Bernoulli)", "movielens": "MovieLens-100K (unit-variance Gaussian working model)",
             "lastfm": "a derived 50-user Last.fm subset (Poisson working model)"}
    dataset_text = ", ".join(names.get(ds, ds) for ds in config["datasets"])
    budget_text = "; ".join(f"{ds}: {dc['additional_labels']:,} additional responses in batches of {dc['batch_size']} "
                            f"({int(np.ceil(dc['additional_labels']/dc['batch_size']))} completed rounds)"
                            for ds, dc in config["datasets"].items())
    dimension_text = ",".join(map(str, config.get("dimensions", [5, 10, 20])))
    experiments = (
        "\\paragraph{Real-response protocol.} We evaluate offline active estimation on fixed pools of recorded responses "
        f"from {dataset_text}. "
        f"The parameter dimensions are $d\\in\\{{{dimension_text}\\}}$ including a penalized intercept. "
        f"Pool-only nested unwhitened PCA coordinates, a fixed split, and a {config.get('n0',80)}-label initialization are shared across methods. "
        f"The configured design has {len(config.get('run_seeds', list(range(10))))} paired algorithmic seeds in every dimension: {budget_text}. "
        "The fitting objective is sum negative log-likelihood plus $\\|\\theta\\|^2/2$. "
        f"The accompanying tables contain {status}.\n\n"
        "\\paragraph{Reference metrics.} The full-pool ridge estimator is an evaluation target, not a true parameter. "
        "At every batch boundary we retain signed errors and both reference-evaluated and current-estimate working Hessians. "
        "For $S$ completed paired seeds, $\\widehat Q=S^{-1}\\sum_s e_s e_s^\\top$, "
        "$R_2^2=\\operatorname{tr}(\\widehat Q)$ and $R_{\\mathrm{dir}}^2=\\lambda_{\\max}(\\widehat Q)$. "
        "These uncentered risks include bias and differ from mean L2 and centered covariance. "
        "We also report full-pool references with ridge scaled by pool size divided by the revealed-label count, "
        "heldout prediction loss, and full-pool objective discrepancies.\n"
    )
    appendix = (
        "\\paragraph{Scope and numerical audits.} All GLM coordinates, including the intercept, are penalized. "
        "Primary Hessians use exact curvature, while acquisition-only floors are separately recorded. "
        "Every previously selected row is reweighted at the current fit. BADGE is used only for Bernoulli responses; "
        "BAIT uses a matched-sum Fisher-ratio adaptation and E-SDP records solver/rounding quality. "
        "Student-$t$ scalar intervals and endpoint whole-seed paired bootstrap intervals are descriptive algorithmic-seed summaries. "
        "At $d=20$, the ten-seed second-moment matrix has rank at most ten; leave-one-seed-out ranges diagnose instability. "
        "Matching ridge does not remove adaptive covariate-selection bias. "
        "Finite logged pools, model misspecification, rare clicks, and the Poisson extension do not establish the manuscript's "
        "full-ball or true-parameter asymptotic guarantees. Unknown theoretical constants are not estimated as observed truth.\n"
    )
    (dest / "experiments_real_response.tex").write_text(experiments)
    (dest / "appendix_real_response.tex").write_text(appendix)
    if not frame.empty:
        cols = ["dataset", "d", "method", "actual_seed_count", "mean_l2_ref", "R2_squared_ref", "Rdir_squared_ref", "mean_test_nll"]
        def tex(value):
            if isinstance(value, (float, np.floating)):
                return f"{value:.5g}"
            return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
        lines = ["\\begin{tabular}{lrlrrrrr}", "\\hline",
                 "Dataset & $d$ & Method & Seeds & Mean L2 & $R_2^2$ & $R_{\\rm dir}^2$ & Test NLL \\\\" , "\\hline"]
        lines.extend(" & ".join(tex(v) for v in row) + " \\\\" for row in frame[cols].itertuples(index=False, name=None))
        lines.extend(["\\hline", "\\end{tabular}", ""])
        (dest / "tables_real_response.tex").write_text("\n".join(lines))
    else:
        (dest / "tables_real_response.tex").write_text("% No completed records are available.\n")
    return {"paper_directory": str(dest), "rows": len(frame)}


def _write_report(config, out, status, endpoints):
    lines = ["# Real-response experiment report", "",
             f"Status: **{status['report_kind']}**. Completed trajectories: {status['completed_trajectories']} / {status['expected_trajectories']}. "
             f"Available aggregate checkpoints: {status['aggregate_checkpoints']}.", "",
             "All numbers below are computed from saved signed errors, information matrices, and recorded losses. "
             "The target is the fixed full-pool ridge reference, not an unknown true parameter. "
             "No missing condition is counted as completed.", "",
             "Configured budgets: " + "; ".join(f"{ds}: n0={config.get('n0',80)}, additional={dc['additional_labels']}, "
                f"m={dc['batch_size']}, rounds={int(np.ceil(dc['additional_labels']/dc['batch_size']))}"
                for ds,dc in config['datasets'].items()) + ".", "",
             "| Dataset | d | Method | Actual seeds | Mean L2 | R2 squared | Rdir squared | Test NLL |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in endpoints:
        lines.append(f"| {row['dataset']} | {row['d']} | {row['method']} | {row['actual_seed_count']} | "
                     f"{row['mean_l2_ref']:.6g} | {row['R2_squared_ref']:.6g} | {row['Rdir_squared_ref']:.6g} | {row['mean_test_nll']:.6g} |")
    lines += ["", "The primary errors use a fixed pool reference. Matched references match mean-loss regularization but cannot correct "
              "covariate-selection bias. The uncentered Q uses denominator S and preserves empirical bias. "
              "At d=20 and S=10 its rank is at most ten; top eigengaps and leave-one-seed-out sensitivity are saved. "
              "Bootstrap intervals resample entire paired algorithmic seeds; they are descriptive and do not measure new-dataset uncertainty.", "",
              "The information matrices are exact working-model Hessians. V_ref uses the reference parameter; V_hat reweights all selected rows "
              "at the current fit; V_acq separately records curvature floors. Gaussian response assumptions and Poisson global noise assumptions "
              "are not asserted for real observations. Rare-event flags and ridge sensitivity remain in reference diagnostics.", "",
              "Manuscript mapping: Eq. (4) → exact sum-loss fit audit; Eq. (9) → V_ref/V_hat empirical analogues; "
              "Eq. (15) → uncentered Q/R2 squared/Rdir squared; Eq. (35) → normalized operator-norm tracking analogue. "
              "True-theory constants and certificates are NOT_IDENTIFIED_IN_REAL_DATA. "
              "Only the configured experiment is included here; other optional sensitivity profiles are not implied to be complete.", "",
              "Runtime choices, measured hardware, pilot timings and forecasts are documented separately in environment.json, "
              "runtime_plan.json, benchmark_runtime.json and runtime_telemetry.csv. This report does not replace artifact verification."]
    fallback_total = sum(int(row.get("fallback_batch_count", 0)) for row in endpoints)
    fallback_runs = sum(int(row.get("runs_with_fallback", 0)) for row in endpoints)
    lines.extend(["", f"Recorded selector fallbacks: **{fallback_total} batches in {fallback_runs} trajectories**. "
                  "Fallbacks remain in every all-method comparison; they are not silently counted as successful SDP certificates. "
                  "The per-condition summary includes fallback counts and measured trajectory runtime. "
                  "When present, sdp_quality_audit.json records solver attempts, quality-gate failures and residuals.", "",
                  "Endpoint Random-relative ratios and rankings are calculated within each dataset/d condition only. "
                  "Ratios with denominator magnitude <=1e-14 are NA with an explicit reason and retain raw differences. "
                  "Numerical win/tie/loss uses tolerance 1e-10*max(1,abs(method),abs(Random)); it is not statistical equivalence. "
                  "Trajectory-integrated summaries use the actual full additional-label grid, trapezoid integration, and division by the planned full budget. "
                  "Partial prefixes do not receive a full-budget integrated score."])
    (Path(out) / "report.md").write_text("\n".join(lines) + "\n")
