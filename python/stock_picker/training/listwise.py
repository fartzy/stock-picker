"""Listwise learn-to-rank losses (ListMLE, ListFold) as LightGBM custom objectives.

Both come from the listwise ranking literature; ListFold specifically from
Zhang, Wu, Chen, "Constructing long-short stock portfolio with a new listwise
learn-to-rank algorithm" (arXiv:2104.12484), which generalizes ListMLE's
top-down Plackett-Luce decomposition into stepwise (top, bottom) pair
selection so the loss cares about both ends of the day's list, not just the
top. We are long-only today, so ListFold's bottom-end emphasis is not
obviously useful here -- it's in the bake-off (see objective_search.py) to
answer that empirically rather than by assumption, and because a short leg
("What if" style) becomes measurable the moment a model orders the bottom
well.

Within a day, the "true order" is the day's tickers sorted by descending
actual day-session return. Scores are LightGBM's raw predictions for that
day's rows.

ListMLE (per day, scores f ordered by true rank, N items):
    L = sum_i [ -f_i + log(sum_{k>=i} exp(f_k)) ]

ListFold with exponential transformation (per day, n = N // 2 steps):
    L = sum_{i<n} [ -(f_i - f_{N-1-i}) + log( A_i * B_i - |S_i| ) ]
    where S_i is positions i..N-1-i, A_i = sum_{S_i} exp(f), B_i =
    sum_{S_i} exp(-f). (sum_{u != v in S} exp(f_u - f_v) factors into
    A*B - |S|, which is what makes the per-step gradients O(1) after two
    cumulative sums.) The middle item of an odd-length day participates in
    every step's denominator but is never itself selected.

LightGBM 4.x custom-objective contract: pass the callable as
`params["objective"]`; it receives (raw_scores, train Dataset) and returns
(grad, hess) aligned to the Dataset's row order. Hessians get a small
positive floor -- ListFold's exact diagonal Hessian is not guaranteed
positive (the paper's own selling point is that the loss is *not* order
sensitive, i.e. non-convex in score space), and LightGBM requires hess > 0.
"""

from __future__ import annotations

import numpy as np

# LightGBM divides by the hessian per leaf; a non-positive value would flip or
# explode leaf outputs, so floor it. Small enough to be inert whenever the
# exact hessian is healthy.
HESSIAN_FLOOR = 1e-6


def _day_true_orders(day_codes: np.ndarray, labels: np.ndarray) -> list[np.ndarray]:
    """Row indices of each day, ordered by descending label within the day.

    `day_codes` must be grouped (all of a day's rows contiguous) -- factorize
    a date column of a frame already sorted by date. Computed once per
    training run; only scores change between boosting rounds, never the true
    order.
    """
    if len(day_codes) and np.any(np.diff(day_codes) < 0):
        raise ValueError("day_codes must be grouped/sorted by day")
    starts = np.unique(day_codes, return_index=True)[1]
    bounds = np.append(starts, len(day_codes))
    orders = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        orders.append(lo + np.argsort(-labels[lo:hi], kind="stable"))
    return orders


def listmle_day_loss(scores_true_order: np.ndarray) -> float:
    """ListMLE negative log-likelihood for one day, scores in true-rank order."""
    f = scores_true_order - scores_true_order.max()
    suffix = np.cumsum(np.exp(f)[::-1])[::-1]
    return float(np.sum(-f + np.log(suffix)))


def listmle_day_grad_hess(scores_true_order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact diagonal gradient/hessian of listmle_day_loss, true-rank order.

    Item at position p appears in the softmax denominator of steps 1..p+1, so
    with Z_i the suffix-exp sum from position i and p_ip = e_p / Z_i:
        grad_p = -1 + sum_{i<=p} p_ip
        hess_p = sum_{i<=p} p_ip * (1 - p_ip)   (always >= 0)
    Both reduce to prefix sums of 1/Z and 1/Z^2.
    """
    f = scores_true_order - scores_true_order.max()
    e = np.exp(f)
    suffix = np.cumsum(e[::-1])[::-1]
    inv_z = np.cumsum(1.0 / suffix)
    inv_z_sq = np.cumsum(1.0 / suffix**2)
    grad = e * inv_z - 1.0
    hess = e * inv_z - e**2 * inv_z_sq
    return grad, hess


def listfold_day_loss(scores_true_order: np.ndarray) -> float:
    """ListFold-exp negative log-likelihood for one day, scores in true-rank order.

    Shift-invariant: A_i scales by exp(-s) and B_i by exp(s) under a shift s,
    so A_i * B_i (and the pair differences) are unchanged -- which is also why
    centering by the mean below is a pure numerical-stability move.
    """
    n_steps = len(scores_true_order) // 2
    if n_steps == 0:
        return 0.0
    a_sums, b_sums, sizes = _listfold_window_sums(scores_true_order)
    f = scores_true_order - scores_true_order.mean()
    top, bottom = f[:n_steps], f[::-1][:n_steps]
    return float(np.sum(-(top - bottom) + np.log(a_sums * b_sums - sizes)))


def _listfold_window_sums(scores_true_order: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step (A_i, B_i, |S_i|) for the shrinking windows i..N-1-i."""
    count = len(scores_true_order)
    n_steps = count // 2
    f = scores_true_order - scores_true_order.mean()
    cum_exp = np.cumsum(np.exp(f))
    cum_exp_neg = np.cumsum(np.exp(-f))
    steps = np.arange(n_steps)
    hi = count - 1 - steps
    prefix = np.where(steps > 0, np.append(0.0, cum_exp)[steps], 0.0)
    prefix_neg = np.where(steps > 0, np.append(0.0, cum_exp_neg)[steps], 0.0)
    a_sums = cum_exp[hi] - prefix
    b_sums = cum_exp_neg[hi] - prefix_neg
    sizes = (count - 2 * steps).astype(float)
    return a_sums, b_sums, sizes


def listfold_day_grad_hess(scores_true_order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact diagonal gradient/hessian of listfold_day_loss, true-rank order.

    With u_p = exp(f_p), v_p = exp(-f_p), D_i = A_i*B_i - |S_i| and item p
    participating in steps 0..m_p-1 (m_p = min(p+1, N-p, n)):
        dL/df_p   = lin_p + u_p * sum(B_i/D_i) - v_p * sum(A_i/D_i)
        d2L/df_p2 = u_p * sum(B_i/D_i) + v_p * sum(A_i/D_i) - 2 * sum(1/D_i)
                    - u_p^2 * sum(B_i^2/D_i^2) - v_p^2 * sum(A_i^2/D_i^2)
                    + 2 * sum(A_i*B_i/D_i^2)
    where lin_p is -1 for a top-half selection, +1 for a bottom-half one, 0
    for an odd day's never-selected middle item, and every sum is a prefix
    sum over steps evaluated at m_p -- so the whole day is O(N log N) (the
    sort) + O(N).
    """
    count = len(scores_true_order)
    n_steps = count // 2
    if n_steps == 0:
        return np.zeros(count), np.zeros(count)

    f = scores_true_order - scores_true_order.mean()
    u = np.exp(f)
    v = np.exp(-f)
    a_sums, b_sums, sizes = _listfold_window_sums(scores_true_order)
    d = a_sums * b_sums - sizes

    b_over_d = np.cumsum(b_sums / d)
    a_over_d = np.cumsum(a_sums / d)
    inv_d = np.cumsum(1.0 / d)
    b_sq = np.cumsum(b_sums**2 / d**2)
    a_sq = np.cumsum(a_sums**2 / d**2)
    ab = np.cumsum(a_sums * b_sums / d**2)

    positions = np.arange(count)
    last_step = np.minimum(np.minimum(positions + 1, count - positions), n_steps) - 1

    lin = np.zeros(count)
    lin[positions < n_steps] = -1.0
    lin[positions >= count - n_steps] = 1.0

    grad = lin + u * b_over_d[last_step] - v * a_over_d[last_step]
    hess = (
        u * b_over_d[last_step]
        + v * a_over_d[last_step]
        - 2.0 * inv_d[last_step]
        - u**2 * b_sq[last_step]
        - v**2 * a_sq[last_step]
        + 2.0 * ab[last_step]
    )
    return grad, hess


def _make_objective(day_codes: np.ndarray, labels: np.ndarray, day_grad_hess) -> callable:
    orders = _day_true_orders(np.asarray(day_codes), np.asarray(labels, dtype=float))

    def objective(scores: np.ndarray, _train_data) -> tuple[np.ndarray, np.ndarray]:
        grad = np.zeros_like(scores, dtype=float)
        hess = np.full_like(scores, HESSIAN_FLOOR, dtype=float)
        for order in orders:
            day_grad, day_hess = day_grad_hess(scores[order])
            grad[order] = day_grad
            hess[order] = np.maximum(day_hess, HESSIAN_FLOOR)
        return grad, hess

    return objective


def make_listmle_objective(day_codes: np.ndarray, labels: np.ndarray) -> callable:
    """LightGBM custom objective (pass as params["objective"]) for ListMLE."""
    return _make_objective(day_codes, labels, listmle_day_grad_hess)


def make_listfold_objective(day_codes: np.ndarray, labels: np.ndarray) -> callable:
    """LightGBM custom objective (pass as params["objective"]) for ListFold-exp."""
    return _make_objective(day_codes, labels, listfold_day_grad_hess)
