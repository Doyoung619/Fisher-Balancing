"""Additional post-main scientific checks; no core implementation changes."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pytest
import yaml
from scipy.optimize import minimize

from real_response import numerics, runner
from real_response.common import ROOT
from real_response.selectors import METHODS, curvature, select_batch, shared_candidates


def test_gaussian_matches_independent_generic_optimizer():
    rng = np.random.default_rng(910)
    X = np.column_stack([np.ones(151), rng.normal(size=(151, 6))])
    X[:, 2] += .8 * X[:, 1]
    y = rng.normal(size=151) + 2 * X[:, 0] - .7 * X[:, 3]
    rho = 3.7
    direct, fit = numerics.fit_glm(X, y, 'gaussian', rho=rho)
    def value(theta):
        residual = y - X @ theta
        return .5 * (residual @ residual) + .5 * rho * (theta @ theta)
    def gradient(theta):
        return X.T @ (X @ theta - y) + rho * theta
    result = minimize(value, np.full(X.shape[1], .4), jac=gradient,
                      hess=lambda theta: X.T @ X + rho*np.eye(X.shape[1]),
                      method='trust-exact', options={'gtol':1e-10})
    assert result.success
    np.testing.assert_allclose(direct, result.x, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(fit['train_penalized_objective_sum'], result.fun, rtol=1e-13)
    assert abs(direct[0]) > .1  # The penalized intercept is part of the comparison.


def test_poisson_large_count_armijo_rejects_overflow_without_clipping(monkeypatch):
    rng = np.random.default_rng(911)
    X = np.column_stack([np.ones(123), rng.normal(size=(123, 3)) * .2])
    y = np.ceil(np.exp(X @ np.array([13., .9, -.7, .4])))
    original_y = y.copy()
    accepted_objectives, trial_values = [], []
    objective = numerics.objective
    derivatives = numerics.derivatives
    def record_trial(theta, features, labels, family, rho=1.):
        value = objective(theta, features, labels, family, rho)
        eta = features @ theta
        with np.errstate(over='ignore', invalid='ignore'):
            independent = np.sum(np.exp(eta) - labels * eta) + .5 * rho * np.dot(theta, theta)
        if np.isfinite(independent):
            np.testing.assert_allclose(value, independent, rtol=1e-14)
        else:
            assert np.isinf(value)  # A clipped likelihood would incorrectly make this finite.
        trial_values.append(value)
        return value
    def record_current(theta, features, labels, family, rho=1.):
        answer = derivatives(theta, features, labels, family, rho)
        accepted_objectives.append(answer[0])
        return answer
    monkeypatch.setattr(numerics, 'objective', record_trial)
    monkeypatch.setattr(numerics, 'derivatives', record_current)
    theta, audit = numerics.fit_glm(X, y, 'poisson', reference=True)
    assert audit['fit_converged'] and audit['fit_grad_relative'] <= 1e-9
    assert audit['fit_retry_count'] == 0
    assert audit['fit_line_search_trials'] > audit['fit_iterations']
    assert any(np.isinf(v) for v in trial_values)
    accepted = np.asarray(accepted_objectives)
    assert np.all(np.diff(accepted) <= 1e-10 * (1 + np.abs(accepted[:-1])))
    assert np.array_equal(y, original_y)
    eta = X @ theta
    np.testing.assert_allclose(audit['train_loss_sum'], np.sum(np.exp(eta) - y * eta), rtol=1e-14)
    _, gradient, hessian = derivatives(theta, X, y, 'poisson')
    np.testing.assert_allclose(gradient, X.T @ (np.exp(eta) - y) + theta, atol=1e-10)
    np.testing.assert_allclose(hessian, (X.T * np.exp(eta)) @ X + np.eye(X.shape[1]), rtol=1e-14)


@pytest.mark.requires_data
def test_real_references_fit_each_dimension_instead_of_slicing_coefficients():
    theta = {}
    for d in [5, 10, 20]:
        X, y, _, _ = runner.prepared('obd', d)
        theta[d], fit = numerics.fit_glm(X, y, 'bernoulli', reference=True)
        assert fit['fit_converged']
        saved = runner.references('obd', d)['theta_full']
        np.testing.assert_allclose(theta[d], saved, atol=1e-8, rtol=1e-9)
        assert theta[d].shape == (d,)
    assert np.linalg.norm(theta[5] - theta[20][:5]) > 1e-5
    assert np.linalg.norm(theta[10] - theta[20][:10]) > 1e-5


@pytest.mark.parametrize('dataset,family', [('obd','bernoulli'),('movielens','gaussian'),('lastfm','poisson')])
@pytest.mark.requires_data
def test_whole_real_pool_learner_equals_full_reference(dataset, family):
    X, y, _, _ = runner.prepared(dataset, 5)
    learned, learner = numerics.fit_glm(X, y, family, reference=False)
    full, reference = numerics.fit_glm(X, y, family, reference=True)
    assert learner['fit_converged'] and reference['fit_converged']
    np.testing.assert_allclose(learned, full, rtol=1e-6, atol=1e-6)
    # The only difference between the calls is the stricter reference convergence gate.
    np.testing.assert_allclose(numerics.objective(learned,X,y,family),
                               numerics.objective(full,X,y,family),rtol=1e-12,atol=1e-8)


@dataclass
class RecordedResponseOracle:
    """An instrumented batch oracle around actual recorded OBD labels."""
    labels: np.ndarray
    already_revealed: set
    selecting: bool = False
    reveal_calls: list = field(default_factory=list)
    def reveal(self, ids):
        assert not self.selecting, 'A response was read during frozen-batch selection'
        ids = np.asarray(ids, dtype=int)
        assert len(np.unique(ids)) == len(ids)
        assert self.already_revealed.isdisjoint(ids)
        self.reveal_calls.append(tuple(ids))
        self.already_revealed.update(ids)
        return self.labels[ids].copy()


@dataclass
class ExternalEvaluator:
    theta_full: np.ndarray
    test_labels: np.ndarray
    calls: int = 0
    def record(self, theta):
        self.calls += 1
        # Deliberately arbitrary evaluation-only values; no selection method receives them.
        return {'reference_discrepancy': float(np.linalg.norm(theta-self.theta_full)),
                'heldout_value': float(np.mean(self.test_labels))}


def run_instrumented_real_batches(method, labels, evaluator, rounds=1):
    X, true_y, _, _ = runner.prepared('obd', 5)
    ids = runner.initial_ids('obd', 0, 80).copy()
    oracle = RecordedResponseOracle(np.asarray(labels), set(ids.tolist()))
    observed_y = np.asarray(labels)[ids].copy()
    theta, _ = numerics.fit_glm(X[ids], observed_y, 'bernoulli')
    batches = []
    for round_id in range(1, rounds+1):
        if evaluator is not None:
            evaluator.record(theta)
        candidates = shared_candidates(len(X), ids, 'obd', 0, round_id, 64)
        oracle.selecting = True
        frozen = theta.copy()
        selected, audit = select_batch(method, X, ids, candidates, frozen, 'bernoulli', 8, 1.,
                                       np.random.default_rng(1000+round_id),
                                       {'dataset':'obd','seed':0,'acquisition_floor':1e-12,'fb_opt_Tmax':rounds,'fb_opt2_Tmax':rounds}, batch_index=round_id)
        assert oracle.reveal_calls == [tuple(b) for b in batches]
        oracle.selecting = False
        selected_before_reveal = selected.copy()
        new_y = oracle.reveal(selected)
        assert np.array_equal(selected, selected_before_reveal)
        assert len(oracle.reveal_calls) == round_id
        assert len(selected) == 8
        batches.append(selected.copy())
        ids = np.r_[ids, selected]
        observed_y = np.r_[observed_y, new_y]
        theta, _ = numerics.fit_glm(X[ids], observed_y, 'bernoulli', theta0=frozen)
    return batches, oracle


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.requires_data
def test_hidden_real_labels_and_evaluator_mutations_preserve_next_batch(method):
    _, labels, _, test_labels = runner.prepared('obd', 5)
    initial = runner.initial_ids('obd', 0, 80)
    altered = 1-labels.copy()
    altered[initial] = labels[initial]  # Exact same learner history.
    first, oracle_a = run_instrumented_real_batches(method, labels,
        ExternalEvaluator(np.zeros(5), test_labels.copy()))
    second, oracle_b = run_instrumented_real_batches(method, altered,
        ExternalEvaluator(np.full(5, 1e6), 1-test_labels.copy()))
    np.testing.assert_array_equal(first[0], second[0])
    assert oracle_a.reveal_calls == oracle_b.reveal_calls == [tuple(first[0])]
    # Even flipping the first actual response in this batch cannot alter any of
    # its remaining IDs: every ID was committed before the one atomic reveal.
    just_first_changed = labels.copy(); just_first_changed[first[0][0]] = 1-labels[first[0][0]]
    third, oracle_c = run_instrumented_real_batches(method, just_first_changed, None)
    np.testing.assert_array_equal(first[0], third[0])
    assert len(oracle_c.reveal_calls) == 1


@pytest.mark.parametrize('method', ['fb_sample','bait_glm','badge','e_sdp'])
@pytest.mark.requires_data
def test_evaluator_disabled_or_mutated_preserves_multibatch_scientific_sequence(method):
    _, labels, _, test_labels = runner.prepared('obd', 5)
    baseline, _ = run_instrumented_real_batches(method, labels, None, rounds=3)
    evil = ExternalEvaluator(np.full(5,-1e5), np.full(len(test_labels),1e9))
    changed, oracle = run_instrumented_real_batches(method, labels, evil, rounds=3)
    for a,b in zip(baseline,changed):np.testing.assert_array_equal(a,b)
    assert evil.calls == 3 and len(oracle.reveal_calls) == 3


@pytest.mark.requires_data
def test_real_initialization_ids_and_feature_hash_are_shared_across_all_dimensions_methods(tmp_path):
    config=yaml.safe_load((ROOT/'configs/smoke.yaml').read_text())
    expected=runner.initial_ids('obd',0,80)
    hashes=set()
    for d in [5,10,20]:
        for method in METHODS:
            task=('obd',d,0,method)
            state=runner.run_one(task,config,tmp_path,round_limit=0)
            assert state['state']=='CHECKPOINTED',state
            path=runner.run_path(tmp_path,task)
            checkpoint=runner.checkpoint_files(path)
            assert len(checkpoint)==1
            with np.load(checkpoint[0][1]) as z:np.testing.assert_array_equal(z['selected_ids'],expected)
            hashes.add(checkpoint[0][2]['metrics']['feature_hash'])
        assert np.linalg.matrix_rank(runner.prepared('obd',d)[0][expected])==d
    assert len(hashes)==1


@pytest.mark.parametrize('family',['bernoulli','poisson'])
@pytest.mark.requires_data
def test_all_old_rows_use_current_curvature_and_new_rows_use_frozen_theta(family):
    X, _, _, _ = runner.prepared('obd',5)
    old=runner.initial_ids('obd',0,80)
    candidates=shared_candidates(len(X),old,'obd',0,1,64)
    earlier=np.zeros(5);current=np.array([1.3,-.8,.4,.6,-.5])
    stale=np.eye(5)+(X[old].T*curvature(X[old]@earlier,family))@X[old]
    selected,audit=select_batch('fb_range',X,old,candidates,current,family,8,1.,np.random.default_rng(52),{},batch_index=1)
    exact=np.eye(5)+(X[old].T*curvature(X[old]@current,family))@X[old]
    np.testing.assert_allclose(audit['W_start'],exact,atol=1e-12)
    assert np.linalg.norm(exact-stale)>1e-3
    final=exact+(X[selected].T*curvature(X[selected]@current,family))@X[selected]
    np.testing.assert_allclose(audit['W_end'],final,atol=1e-11)
    np.testing.assert_array_equal(audit['frozen_theta'],current)
