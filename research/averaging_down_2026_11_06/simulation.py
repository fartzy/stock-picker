"""Research-only simulations: observe a completed bar, trade at the next open.

No production imports. Dollar limits are decision triggers, not guaranteed fills.
Buy cash includes modeled fees/slippage. Fractional shares isolate sizing effects.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Policy:
    name: str
    max_tranches: int = 1
    initial_tranches: int = 1
    step: float = .05
    loss_limit: float | None = None
    target: float | None = None
    stability: bool = False
    partial_trail: bool = False
    volatility_step: bool = False
    overnight_loss_limit: float | None = None
    overnight_loss_fraction: float | None = None


POLICIES = [
    Policy('single_10k'),
    Policy('full_40k', max_tranches=4, initial_tranches=4),
    Policy('ladder_5', max_tranches=4),
    Policy('ladder_5_stop2k', max_tranches=4, loss_limit=2000),
    Policy('ladder_5_stop2k_be', max_tranches=4, loss_limit=2000, target=0),
    Policy('ladder_5_stop2k_take2', max_tranches=4, loss_limit=2000, target=.02),
    Policy('ladder_5_stop2k_take4', max_tranches=4, loss_limit=2000, target=.04),
    Policy('ladder_5_stop1k_take2', max_tranches=4, loss_limit=1000, target=.02),
    Policy('ladder_5_stop4k_take2', max_tranches=4, loss_limit=4000, target=.02),
    Policy('ladder_stable_stop2k_take2', max_tranches=4, loss_limit=2000, target=.02, stability=True),
    Policy('single_stop500_take2', loss_limit=500, target=.02),
    Policy('ladder_3_stop2k_take2', max_tranches=4, step=.03, loss_limit=2000, target=.02),
    Policy('ladder_7p5_stop2k_take2', max_tranches=4, step=.075, loss_limit=2000, target=.02),
    Policy('ladder_vol_stop2k_take2', max_tranches=4, loss_limit=2000, target=.02, volatility_step=True),
    Policy('ladder_partial2_trail2', max_tranches=4, loss_limit=2000, target=.02, partial_trail=True),
]


def simulate(bars, times, stable, policy, *, cost=.0011, tranche=10000.,
             daily_vol=.03, entry_price=None, keep_path=False, bar_seconds=60,
             overnight_check=None):
    """bars are O,H,L,C,V; times are epoch seconds, stable uses only past bars.

    Daily/one-minute stop and target decisions use closing marks, then next open.
    Risk has priority over profit-taking and adds. At most one add per bar; an
    add is canceled if next open has rebounded above its predeclared threshold.
    A gap straight through the loss threshold cancels an add and exits instead.
    No re-entry after a full exit. Final close is a precommitted time liquidation.
    """
    assert len(bars) == len(times) == len(stable) and len(bars) > 0
    assert np.isfinite(bars[:, :4]).all() and (bars[:, :4] > 0).all()
    if policy.overnight_loss_limit is not None or policy.overnight_loss_fraction is not None:
        assert overnight_check is not None and len(overnight_check) == len(bars)
    step = float(np.clip(1.5 * daily_vol, .02, .10)) if policy.volatility_step else policy.step
    anchor = float(bars[0, 0] if entry_price is None else entry_price)
    spent = tranche * policy.initial_tranches
    qty = spent / (anchor * (1 + cost))
    tranches = policy.initial_tranches
    cash = 0.
    pending = None
    pending_level = 0.
    partial = False
    trail_peak = 0.
    worst = 0.
    peak_mark = 0.
    drawdown = 0.
    adverse = False
    capital_days = 0.
    max_participation = qty / bars[0, 4] if bars[0, 4] > 0 else float('inf')
    path = []
    reason = 'time'
    exit_i = len(bars) - 1

    def mark(price):
        return cash + qty * price * (1 - cost) - spent

    for i, (opened, high, low, close, volume) in enumerate(bars):
        if i:
            capital_days += max(spent - cash, 0) * (times[i] - times[i-1]) / 86400
        if pending:
            action = pending
            pending = None
            if action == 'add':
                # Never average down into a gap that has already exhausted the risk budget.
                if policy.loss_limit is not None and mark(opened) <= -policy.loss_limit:
                    action = 'gap_stop'
                elif opened <= pending_level:
                    new_qty = tranche / (opened * (1 + cost))
                    qty += new_qty
                    spent += tranche
                    tranches += 1
                    participation = new_qty / volume if volume > 0 else float('inf')
                    max_participation = max(max_participation, participation)
            if action == 'half':
                max_participation = max(max_participation, qty/2/volume if volume>0 else float('inf'))
                cash += qty / 2 * opened * (1 - cost)
                qty /= 2
                partial = True
                trail_peak = opened
            elif action in ('stop', 'gap_stop', 'target', 'trail', 'overnight'):
                max_participation = max(max_participation, qty/volume if volume>0 else float('inf'))
                pnl = mark(opened)
                worst = min(worst, pnl)
                drawdown = max(drawdown, peak_mark - pnl)
                cash += qty * opened * (1 - cost)
                qty = 0
                reason = action
                exit_i = i
                if keep_path:
                    path.append({'time':float(times[i]),'price':float(opened),'pnl':float(pnl),'spent':float(spent),'shares':0.,'action':action})
                break
        low_mark = mark(low)
        close_mark = mark(close)
        worst = min(worst, low_mark)
        drawdown = max(drawdown, peak_mark - low_mark)
        peak_mark = max(peak_mark, close_mark)
        adverse = adverse or close <= anchor * .95
        if keep_path:
            path.append({'time':float(times[i]+bar_seconds),'price':float(close),'pnl':float(close_mark),'spent':float(spent),'shares':float(qty),'action':'mark'})
        if i == len(bars) - 1:
            max_participation = max(max_participation, qty/volume if volume>0 else float('inf'))
            capital_days += max(spent-cash,0) * bar_seconds / 86400
            cash += qty * close * (1 - cost)
            qty = 0
            break
        if policy.loss_limit is not None and close_mark <= -policy.loss_limit:
            pending = 'stop'
        elif partial:
            trail_peak = max(trail_peak, close)
            if close <= trail_peak * .98:
                pending = 'trail'
        elif policy.target is not None and close_mark >= spent * policy.target and (policy.target > 0 or adverse):
            pending = 'half' if policy.partial_trail else 'target'
        elif (overnight_check is not None and overnight_check[i] and
              (policy.overnight_loss_limit is not None or policy.overnight_loss_fraction is not None) and
              close_mark < -min(policy.overnight_loss_limit if policy.overnight_loss_limit is not None else float('inf'),
                               spent * policy.overnight_loss_fraction if policy.overnight_loss_fraction is not None else float('inf'))):
            pending = 'overnight'
        elif tranches < policy.max_tranches:
            level = anchor * (1 - step) ** tranches
            if close <= level and (not policy.stability or stable[i]):
                pending = 'add'
                pending_level = level
    result = {'pnl':float(cash-spent), 'max_capital':float(spent),
              'tranches':tranches, 'adds':tranches-policy.initial_tranches,
              'worst_mark':float(worst), 'drawdown':float(drawdown),
              'capital_days':float(capital_days),
              'financing_12pct':float(capital_days*.12/365),
              'duration_days':float((times[exit_i]-times[0]+(bar_seconds if reason=='time' else 0))/86400),
              'exit_reason':reason, 'partial_exit':partial,
              'max_minute_participation':float(max_participation), 'step':step}
    if keep_path:
        result['path'] = path
    return result


def summarize(frame):
    x = frame.pnl.to_numpy()
    n = len(x)
    if not n:
        return {'n':0}
    return {'n':n, 'mean':float(np.mean(x)), 'median':float(np.median(x)),
            'win_pct':float(np.mean(x>0)*100),
            'p05':float(np.percentile(x,5)),
            'expected_shortfall_5':float(np.mean(np.sort(x)[:max(1,int(np.ceil(n*.05)))])),
            'worst':float(np.min(x)), 'best':float(np.max(x)),
            'loss_over_2k_pct':float(np.mean(x < -2000)*100),
            'avg_max_capital':float(frame.max_capital.mean()),
            'avg_adds':float(frame['adds'].mean()),
            'stop_pct':float(frame.exit_reason.isin(['stop','gap_stop']).mean()*100),
            'avg_financing_12pct':float(frame.financing_12pct.mean()),
            'mean_after_financing_12pct':float((frame.pnl-frame.financing_12pct).mean()),
            'worst_mark':float(frame.worst_mark.min()),
            'minute_participation_over10pct':float((frame.max_minute_participation>.1).mean()*100)}


def clustered_mean_ci(frame, value='pnl', repetitions=2000, seed=20261106):
    """Wider envelope of ticker-cluster and entry-date-cluster bootstrap CIs.

    This is a robustness diagnostic, not a formal multi-way coverage guarantee.
    """
    rng = np.random.default_rng(seed)
    intervals = {}
    for group in ('ticker','entry_date'):
        g = frame.groupby(group)[value].agg(['sum','count']).to_numpy()
        if len(g) < 2:
            intervals[group] = [float('nan'),float('nan')]
            continue
        means=[]
        for _ in range(repetitions):
            selected = g[rng.integers(0,len(g),size=len(g))]
            means.append(selected[:,0].sum()/selected[:,1].sum())
        intervals[group] = list(map(float,np.percentile(means,[2.5,97.5])))
    intervals['envelope'] = [min(v[0] for v in intervals.values()),max(v[1] for v in intervals.values())]
    return intervals
