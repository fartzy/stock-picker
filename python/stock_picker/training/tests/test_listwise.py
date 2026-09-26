"""Verifies the hand-derived ListMLE/ListFold gradients and hessians against
finite differences of the loss functions themselves -- a wrong sign or a
dropped term in a custom LightGBM objective doesn't crash anything, it just
silently trains a worse model, so the derivation is checked numerically here
instead of trusted."""

import numpy as np
import pytest

from stock_picker.training.listwise import (
    HESSIAN_FLOOR,
    listfold_day_grad_hess,
    listfold_day_loss,
    listmle_day_grad_hess,
    listmle_day_loss,
    make_listfold_objective,
    make_listmle_objective,
)

RNG = np.random.default_rng(0)
EPS = 1e-5


def _numeric_grad(loss_fn, scores):
    grad = np.zeros_like(scores)
    for i in range(len(scores)):
        bumped_up, bumped_down = scores.copy(), scores.copy()
        bumped_up[i] += EPS
        bumped_down[i] -= EPS
        grad[i] = (loss_fn(bumped_up) - loss_fn(bumped_down)) / (2 * EPS)
    return grad


def _numeric_hess_diagonal(loss_fn, scores):
    hess = np.zeros_like(scores)
    center = loss_fn(scores)
    for i in range(len(scores)):
        bumped_up, bumped_down = scores.copy(), scores.copy()
        bumped_up[i] += EPS
        bumped_down[i] -= EPS
        hess[i] = (loss_fn(bumped_up) - 2 * center + loss_fn(bumped_down)) / EPS**2
    return hess


@pytest.mark.parametrize("n_items", [2, 3, 8, 9])
def test_listmle_gradient_matches_finite_difference(n_items):
    scores = RNG.normal(size=n_items)
    grad, _ = listmle_day_grad_hess(scores)
    np.testing.assert_allclose(grad, _numeric_grad(listmle_day_loss, scores), atol=1e-6)


@pytest.mark.parametrize("n_items", [2, 3, 8, 9])
def test_listmle_hessian_matches_finite_difference(n_items):
    scores = RNG.normal(size=n_items)
    _, hess = listmle_day_grad_hess(scores)
    np.testing.assert_allclose(hess, _numeric_hess_diagonal(listmle_day_loss, scores), atol=1e-4)


@pytest.mark.parametrize("n_items", [2, 3, 8, 9])
def test_listfold_gradient_matches_finite_difference(n_items):
    # Odd sizes exercise the never-selected middle item, whose gradient comes
    # only from the denominators it participates in.
    scores = RNG.normal(size=n_items)
    grad, _ = listfold_day_grad_hess(scores)
    np.testing.assert_allclose(grad, _numeric_grad(listfold_day_loss, scores), atol=1e-6)


@pytest.mark.parametrize("n_items", [2, 3, 8, 9])
def test_listfold_hessian_matches_finite_difference(n_items):
    scores = RNG.normal(size=n_items)
    _, hess = listfold_day_grad_hess(scores)
    np.testing.assert_allclose(hess, _numeric_hess_diagonal(listfold_day_loss, scores), atol=1e-4)


def test_listfold_loss_is_shift_invariant():
    # The paper's selling point: the pairwise-difference structure makes the
    # loss endogenously shift-invariant (Section 3.1).
    scores = RNG.normal(size=10)
    assert listfold_day_loss(scores) == pytest.approx(listfold_day_loss(scores + 123.4))


def test_listfold_true_order_minimizes_over_all_permutations():
    # The paper's own worked example (Section 3.2): scores (5, 4, 1, 0) with
    # the descending assignment as ground truth. Every other assignment of
    # those scores to the true-rank positions must lose.
    from itertools import permutations

    scores = np.array([5.0, 4.0, 1.0, 0.0])
    truth = listfold_day_loss(scores)
    for perm in permutations(scores):
        candidate = np.array(perm)
        if not np.array_equal(candidate, scores):
            assert listfold_day_loss(candidate) > truth


def test_listfold_paper_example_is_not_order_sensitive():
    # Section 3.2's anomaly: from (1, 5, 4, 0), swapping the first two toward
    # the ground truth *increases* the loss -- the loss is deliberately not
    # order sensitive, unlike ListMLE. Guards against "simplifying" the
    # implementation into something order-sensitive later.
    worse_after_swap = listfold_day_loss(np.array([5.0, 1.0, 4.0, 0.0]))
    before_swap = listfold_day_loss(np.array([1.0, 5.0, 4.0, 0.0]))
    assert worse_after_swap > before_swap


def test_listmle_gradient_at_zero_scores_pushes_top_up():
    # At initialization every score is 0; the winner's gradient must be
    # negative (LightGBM descends the gradient, so negative grad raises the
    # score) and the loser's positive.
    grad, _ = listmle_day_grad_hess(np.zeros(4))
    assert grad[0] < 0
    assert grad[-1] > 0


def test_listfold_gradient_at_zero_scores_pushes_ends_apart():
    grad, _ = listfold_day_grad_hess(np.zeros(5))
    assert grad[0] == pytest.approx(-1.0)
    assert grad[-1] == pytest.approx(1.0)
    # Odd-length middle item is never selected; at zero scores its pull is zero.
    assert grad[2] == pytest.approx(0.0)


def test_objective_scatters_by_true_order_across_days():
    # Two days, rows NOT already in true-rank order within each day: the
    # objective must sort by descending label per day, compute, and scatter
    # gradients back to original row positions.
    day_codes = np.array([0, 0, 0, 1, 1, 1])
    labels = np.array([0.01, 0.05, -0.02, 0.03, -0.01, 0.04])
    scores = RNG.normal(size=6)

    objective = make_listmle_objective(day_codes, labels)
    grad, hess = objective(scores, None)

    day_one_order = np.array([1, 0, 2])  # descending label within day 0
    expected_grad, _ = listmle_day_grad_hess(scores[day_one_order])
    np.testing.assert_allclose(grad[day_one_order], expected_grad, atol=1e-12)
    assert np.all(hess >= HESSIAN_FLOOR)


def test_objectives_reject_ungrouped_days():
    with pytest.raises(ValueError):
        make_listfold_objective(np.array([0, 1, 0]), np.array([0.1, 0.2, 0.3]))


def test_listfold_objective_trains_a_lightgbm_booster_that_ranks():
    # End-to-end through LightGBM 4.x's params["objective"] contract: one
    # strongly informative feature, 40 synthetic days -- after training, the
    # booster's scores must rank a fresh day in feature order.
    import lightgbm as lgb
    import pandas as pd

    n_days, per_day = 40, 12
    rows = []
    for day in range(n_days):
        signal = RNG.normal(size=per_day)
        returns = 0.02 * signal + 0.002 * RNG.normal(size=per_day)
        rows.append(pd.DataFrame({"day": day, "signal": signal, "ret": returns}))
    frame = pd.concat(rows, ignore_index=True)

    objective = make_listfold_objective(frame["day"].to_numpy(), frame["ret"].to_numpy())
    dataset = lgb.Dataset(frame[["signal"]], label=frame["ret"])
    booster = lgb.train(
        {"objective": objective, "verbosity": -1, "learning_rate": 0.1, "min_data_in_leaf": 5},
        dataset,
        num_boost_round=30,
    )

    fresh = pd.DataFrame({"signal": np.linspace(-2, 2, 9)})
    scores = booster.predict(fresh)
    assert np.all(np.diff(scores) >= 0)


def test_listfold_trained_booster_survives_pickling():
    # The production Rank model is persisted through ModelStore (pickle). A
    # booster trained with a *callable* objective must not drag the closure
    # into the pickle or fail outright -- this is the gate between "research
    # result" and "promotable to the Rank pickle".
    import pickle

    import lightgbm as lgb
    import pandas as pd

    frame = pd.DataFrame(
        {
            "day": np.repeat(np.arange(10), 8),
            "signal": RNG.normal(size=80),
        }
    )
    frame["ret"] = 0.02 * frame["signal"] + 0.002 * RNG.normal(size=80)

    objective = make_listfold_objective(frame["day"].to_numpy(), frame["ret"].to_numpy())
    dataset = lgb.Dataset(frame[["signal"]], label=frame["ret"])
    booster = lgb.train(
        {"objective": objective, "verbosity": -1, "min_data_in_leaf": 5}, dataset, num_boost_round=10
    )

    revived = pickle.loads(pickle.dumps(booster))
    fresh = pd.DataFrame({"signal": [0.0, 1.0]})
    np.testing.assert_allclose(revived.predict(fresh), booster.predict(fresh))
