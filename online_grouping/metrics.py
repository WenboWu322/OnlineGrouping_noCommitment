"""Evaluation indicators.

The first block reuses the definitions of Chapter 5 (``tab:metrics``):
number of clusters ``n_b``, setup cost ``C^set``, time-shift penalty
``Phi^pen``, total cost ``C^tot``, and the rescheduling indicators
``S^max``, ``S^avg``, ``r^-``, ``r^+``, ``r^0``.  The second block contains the
indicators needed by an online study: safety outcomes taken from the truth
process, plan stability, the use of emergency work, and the distance to the
clairvoyant reference (regret).

Convention for the shift indicators: the reference date ``tau_n^0`` is the
reference of the last decision taken before the intervention was executed,
which is the convention already used by the Chapter 5 study when a plan is
revised between windows.  Deviations from the *first* announced date are
reported separately as ``plan_changes``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def realized_references(cfg, world, execution_log: list[dict]) -> list[dict]:
    """Truth-based reference date for every serviced and unserviced cycle."""
    from .scenarios import realized_plan
    return realized_plan(cfg, world, execution_log)


def cost_metrics(cfg, strategy_name: str, execution_log: list[dict],
                 truth_log: list[dict], solver_log: list[dict],
                 change_log: list[dict], fallback_log: list[dict],
                 leaked_mass: float, grouping_off: bool,
                 pending_open: int) -> dict:
    n = len(execution_log)
    dispatch_days = sorted({e['executed_day'] for e in execution_log})
    n_b = n if grouping_off else len(dispatch_days)
    c_set = n_b * cfg.scenario.c_set

    shift = 0.0
    shifts: list[int] = []
    ref_errors: list[float] = []
    advanced = delayed = unchanged = 0
    for e in execution_log:
        ref, ex = e.get('reference_real', e.get('reference_hat')), e['executed_day']
        if ref is None:
            continue
        delta = ex - ref
        if e.get('reference_hat') is not None:
            ref_errors.append(abs(e['reference_hat'] - ref))
        shift += (cfg.scenario.eta_minus * (ref - ex) if delta <= 0
                  else cfg.scenario.eta_plus * delta)
        shifts.append(abs(delta))
        advanced += delta < 0
        delayed += delta > 0
        unchanged += delta == 0

    emergency = sum(1 for e in execution_log if e['unscheduled'])
    emergency_cost = emergency * cfg.policy.emergency_extra

    return dict(
        strategy=strategy_name,
        seed=cfg.seed,
        interventions=n,
        dispatches=n_b,
        setup_cost=c_set,
        shift_penalty=float(shift),
        emergency=emergency,
        emergency_cost=emergency_cost,
        total_cost=float(c_set + shift + emergency_cost),
        tasks_per_dispatch=(n / n_b if n_b else 0.0),
        max_shift=int(max(shifts) if shifts else 0),
        mae_reference=float(np.mean(ref_errors)) if ref_errors else float('nan'),
        mean_shift=float(np.mean(shifts)) if shifts else 0.0,
        share_advanced=advanced / n if n else 0.0,
        share_delayed=delayed / n if n else 0.0,
        share_unchanged=unchanged / n if n else 0.0,
        solver_calls=len(solver_log),
        solver_seconds=float(sum(r['seconds'] for r in solver_log)),
        days_without_solve=(cfg.scenario.T + 1 - len(solver_log)),
        plan_changes=len(change_log),
        fallbacks=len(fallback_log),
        below_ref_days=int(sum(r['below_ref'] for r in truth_log)),
        below_min_days=int(sum(r['below_min'] for r in truth_log)),
        min_pressure=float(min((r['pressure'] for r in truth_log), default=float('nan'))),
        leaked_mass=float(leaked_mass),
        open_tasks=pending_open,
        unserviced_alarms=0,
    )


def calibration_metrics(truth_log: list[dict], forecast_log: list[dict],
                        epsilons: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25)) -> dict:
    """Coverage of the predictive pressure distribution (PICP).

    For every recorded forecast the empirical share of realised pressures that
    fall below the ``epsilon``-quantile should equal ``epsilon``; systematic
    deviations show the window bound to be optimistic or conservative.

    Forecasts whose horizon spans a gas-refilling intervention are excluded:
    the pressure step there is a planned structural break, not a prediction
    failure, and mixing it in makes the coverage figures meaningless.
    Coverage is reported per lead time, which is what the calibration figure
    of the chapter plots.
    """
    from scipy.stats import norm
    out: dict[str, float] = {}
    fc = [f for f in forecast_log if f.get('clean', True)]
    if not fc:
        return out
    mean = np.array([f['mean'] for f in fc])
    sd = np.array([f['sd'] for f in fc])
    true = np.array([f['true_pressure'] for f in fc])
    leads = np.array([f['lead'] for f in fc])
    for eps in epsilons:
        q = mean + norm.ppf(eps) * sd
        out[f'picp_{eps:g}'] = float(np.mean(true <= q))
        for lead in sorted(set(leads.tolist())):
            sel = leads == lead
            out[f'picp_{eps:g}_lead{lead}'] = float(np.mean(true[sel] <= q[sel]))
    first = leads == min(leads)
    out['mae_lead1'] = float(np.mean(np.abs(mean[first] - true[first])))
    out['rmse_lead1'] = float(np.sqrt(np.mean((mean[first] - true[first]) ** 2)))
    # sharpness: mean predictive standard deviation
    out['mean_sd'] = float(np.mean(sd))
    out['mean_sd_lead10'] = float(np.mean(sd[leads == 10])) if 10 in set(leads) else None
    return out


def paired_statistics(df, value: str, reference: str, group: str = 'scenario'):
    """Per-seed paired difference against a reference strategy + Wilcoxon test."""
    import pandas as pd
    from scipy.stats import wilcoxon

    wide = df.pivot_table(index=[group, 'seed'], columns='strategy', values=value)
    if reference not in wide.columns:
        raise KeyError(f'reference strategy {reference!r} missing')
    rows = []
    for strat in wide.columns:
        if strat == reference:
            continue
        d = (wide[strat] - wide[reference]).dropna()
        if len(d) < 2:
            p = float('nan')
        else:
            try:
                p = float(wilcoxon(d).pvalue)
            except ValueError:
                p = float('nan')
        rows.append(dict(strategy=strat, reference=reference, metric=value,
                         n=int(len(d)), mean_diff=float(d.mean()),
                         median_diff=float(d.median()),
                         share_worse=float((d > 0).mean()),
                         share_better=float((d < 0).mean()),
                         wilcoxon_p=p))
    return pd.DataFrame(rows)


def wilcoxon_paired(a, b):
    """Paired Wilcoxon signed-rank test on two equal-length per-seed samples."""
    from scipy.stats import wilcoxon
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 2:
        return dict(n=int(min(a.size, b.size)), mean_diff=float('nan'),
                    share_worse=float('nan'), share_better=float('nan'),
                    pvalue=float('nan'))
    d = a - b
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:                        # identical differences
        p = float('nan')
    return dict(n=int(a.size), mean_diff=float(d.mean()),
                share_worse=float((d > 0).mean()), share_better=float((d < 0).mean()),
                pvalue=p)


def regret(costs: dict[str, float], reference: str) -> dict[str, float]:
    base = costs.get(reference)
    if base is None:
        return {}
    return {name: value - base for name, value in costs.items()}
