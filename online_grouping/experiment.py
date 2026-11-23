"""Experiment driver for the Chapter 6 study.

Matrix: scenario variant x random seed x policy.  The environment realisation
depends only on the seed, so every policy sees the same leak rates, shocks,
measurement noise and missing-data pattern; the per-seed differences are
therefore paired and can be tested with the Wilcoxon signed-rank test.

Usage
-----
    python -m online_grouping.experiment --seeds 30 --out results/main
    python -m online_grouping.experiment --configs D_high_cluster --seeds 5 \
        --policies S3_event_ch5win S5_proposed --jobs 4

Outputs ``rows.csv`` (one line per run), ``summary.csv`` / ``summary.json``
(per variant and policy, with paired tests against the Chapter 5 online
baseline) and ``config.json`` (the exact parameter set of the run).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace as dc_replace
from pathlib import Path

import numpy as np

from .config import Config, PolicyConfig, ScenarioConfig
from .controller import OnlineRunner
from .metrics import calibration_metrics, wilcoxon_paired
from .policies import strategies
from .scenarios import World


# Chapter 5's two scenario axes (task density and temporal concentration),
# crossed.  The low-density leak rates are the ones calibrated on the RTE
# monitoring extract; the high-density range reproduces the Chapter 5
# sensitivity setting where consolidation matters most.
VARIANTS: dict[str, ScenarioConfig] = {
    'A_low_uniform': ScenarioConfig(),
    'B_low_cluster': ScenarioConfig(clustered=True),
    'C_high_uniform': ScenarioConfig(alpha_low=0.004, alpha_high=0.008),
    'D_high_cluster': ScenarioConfig(alpha_low=0.004, alpha_high=0.008,
                                     clustered=True),
}

BASELINE = 'S3_event_ch5win'
KEY_METRICS = ('total_cost', 'dispatches', 'interventions', 'setup_cost',
               'shift_penalty', 'emergency_cost', 'below_ref_days',
               'below_min_days', 'unserviced_alarms', 'min_pressure',
               'plan_changes', 'solver_calls', 'solver_seconds',
               'mae_reference', 'tasks_per_dispatch', 'max_shift',
               'share_delayed', 'emergency',
               'picp_0.01', 'picp_0.05', 'picp_0.1', 'picp_0.25',
               'picp_0.05_lead5', 'picp_0.05_lead10', 'picp_0.05_lead20',
               'picp_0.05_lead40', 'mae_lead1', 'mean_sd', 'shocks')


def _one(task) -> dict:
    cfg_dict, policy_name = task
    cfg = _rebuild(cfg_dict)
    world = World(cfg)
    strategy = next(s for s in strategies() if s.name == policy_name)
    start = time.perf_counter()
    run = OnlineRunner(cfg, world, strategy).run()
    row = dict(variant=cfg_dict['variant'], seed=cfg.seed, policy=policy_name,
               wall_seconds=time.perf_counter() - start)
    row.update(run.cost)
    cal = calibration_metrics(run.truth_log, run.forecast_log)
    row.update({k: v for k, v in cal.items()
                if k.startswith('picp') or k in ('mae_lead1', 'rmse_lead1',
                                                 'mean_sd')})
    row['n_forecast'] = len(run.forecast_log)
    for lead in (5, 10, 20, 40):
        key = f'picp_0.05_lead{lead}'
        row[key] = cal.get(key)
    row['events'] = ''
    counts: dict[str, int] = {}
    for f in run.fallback_log:
        counts[f['reason']] = counts.get(f['reason'], 0) + 1
    row['fallbacks_by_reason'] = json.dumps(counts, sort_keys=True)
    row['shocks'] = len(world.shock_log)
    for key in KEY_METRICS:
        row.setdefault(key, None)
    return row


def _rebuild(d: dict) -> Config:
    return Config(scenario=ScenarioConfig(**d['scenario']),
                  policy=PolicyConfig(**d['policy']),
                  seed=d['seed'], days=d['days'])


def run_matrix(variants: dict[str, ScenarioConfig], seeds: range,
               policy_names: list[str], jobs: int = 1,
               policy_cfg: PolicyConfig | None = None) -> list[dict]:
    names = [s.name for s in strategies()
             if not policy_names or s.name in policy_names]
    tasks = []
    for vname, sc in variants.items():
        for seed in seeds:
            cfg = Config(scenario=sc, policy=policy_cfg or PolicyConfig(),
                         seed=seed)
            d = dict(variant=vname, seed=seed,
                     scenario=asdict(cfg.scenario), policy=asdict(cfg.policy),
                     days=cfg.days)
            for pname in names:
                tasks.append((d, pname))
    print(f'{len(tasks)} runs on {len(variants)} variants x {len(seeds)} seeds '
          f'x {len(names)} policies', flush=True)
    rows: list[dict] = []
    t0 = time.perf_counter()
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, row in enumerate(pool.map(_one, tasks, chunksize=4), 1):
                rows.append(row)
                if i % 50 == 0:
                    print(f'  {i}/{len(tasks)} ({time.perf_counter()-t0:.0f}s)',
                          flush=True)
    else:
        for i, (d, pname) in enumerate(tasks, 1):
            rows.append(_one(d, pname))
            if i % 50 == 0:
                print(f'  {i}/{len(tasks)} ({time.perf_counter()-t0:.0f}s)',
                      flush=True)
    print(f'finished in {time.perf_counter()-t0:.0f}s', flush=True)
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """Mean and dispersion per variant/policy, plus paired tests vs baseline."""
    out = []
    by_variant: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        by_variant.setdefault(r['variant'], {}).setdefault(r['policy'], []).append(r)
    for vname, per_policy in sorted(by_variant.items()):
        base = {r['seed']: r for r in per_policy.get(BASELINE, [])}
        for pname, group in sorted(per_policy.items()):
            group = sorted(group, key=lambda r: r['seed'])
            entry = dict(variant=vname, policy=pname, n_seeds=len(group))
            for key in KEY_METRICS:
                vals = [r[key] for r in group if r.get(key) is not None]
                entry[f'{key}_mean'] = float(np.mean(vals)) if vals else None
                entry[f'{key}_sd'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            paired = {key: ([r[key] for r in group],
                            [base[r['seed']][key] for r in group if r['seed'] in base])
                      for key in ('total_cost', 'dispatches', 'below_ref_days')}
            for key, (a, b) in paired.items():
                if len(a) == len(b) and len(a) > 1:
                    st = wilcoxon_paired(a, b)
                    entry[f'{key}_wilcoxon_p'] = st['pvalue']
                    entry[f'{key}_share_worse_vs_base'] = st['share_worse']
                    entry[f'{key}_delta_vs_base'] = st['mean_diff']
            oracle = {r['seed']: r for r in per_policy.get('O1_oracle', [])}
            reg = [r['total_cost'] - oracle[r['seed']]['total_cost']
                   for r in group if r['seed'] in oracle]
            entry['regret_vs_oracle_mean'] = float(np.mean(reg)) if reg else None
            out.append(entry)
    return out


def write_outputs(out_dir: Path, rows: list[dict], summary: list[dict],
                  cfg_sample: Config) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv

    def dump(path: Path, records: list[dict]) -> None:
        if not records:
            return
        keys: list[str] = []
        for r in records:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with path.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in records:
                w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                            for k, v in r.items()})

    dump(out_dir / 'rows.csv', rows)
    dump(out_dir / 'summary.csv', summary)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    (out_dir / 'config.json').write_text(json.dumps(
        dict(variants={k: asdict(v) for k, v in VARIANTS.items()},
             scenario=asdict(cfg_sample.scenario),
             policy=asdict(cfg_sample.policy)), indent=1))
    print(f'wrote {out_dir}/rows.csv ({len(rows)} runs) and summary.csv '
          f'({len(summary)} cells)', flush=True)


def dump_trace(variant: str, seed: int, policy: str, out: Path,
               n_pmec_trace: int = 4) -> None:
    """Write a per-day trace of a few compartments for the timeline figure."""
    sc = VARIANTS[variant]
    cfg = Config(scenario=sc, seed=seed)
    world = World(cfg)
    strategy = next(s for s in strategies() if s.name == policy)
    # trace compartments with a long service history rather than a short one
    order = sorted(world.ids,
                   key=lambda m: -(world.initial_pressure(m)
                                   - cfg.scenario.p_ref) / world.alpha(m, 0))
    trace = tuple(order[:n_pmec_trace])
    run = OnlineRunner(cfg, world, strategy, trace=trace).run()
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    rows = sorted(run.trace, key=lambda r: (r['pmec'], r['day']))
    with out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out} ({len(rows)} rows, {len(trace)} compartments, '
          f'{variant} seed {seed} policy {policy})', flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variants', nargs='*', default=sorted(VARIANTS))
    ap.add_argument('--policies', nargs='*', default=[])
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--seed0', type=int, default=1)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--out', default='results/main')
    ap.add_argument('--set', dest='overrides', nargs='*', default=[],
                    help='scenario.FIELD=value / policy.FIELD=value, e.g. '
                         'policy.shock_rate=0.002 scenario.delta_delay=10')
    ap.add_argument('--label', default=None, help='suffix added to the variant names')
    ap.add_argument('--dump-trace', action='store_true',
                    help='write a single-run trace for the timeline figure and exit')
    ap.add_argument('--variant', default='D_high_cluster')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--policy', default='S5_proposed')
    args = ap.parse_args(argv)

    if args.dump_trace:
        dump_trace(args.variant, args.seed, args.policy,
                   Path(args.out) / 'trace.csv' if Path(args.out).is_file()
                   else Path(args.out) if str(args.out).endswith('.csv')
                   else Path(args.out) / 'trace.csv')
        return

    variants = {k: VARIANTS[k] for k in args.variants}
    sc_over: dict[str, object] = {}
    po_over: dict[str, object] = {}
    for item in args.overrides:
        section, _, rest = item.partition('.')
        field, _, raw = rest.partition('=')
        if not field or not raw:
            raise SystemExit(f'cannot parse override {item!r}')
        value: object = raw
        if raw.lower() in ('true', 'false'):
            value = raw.lower() == 'true'
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    pass
        (sc_over if section == 'scenario' else po_over)[field] = value
    if sc_over:
        variants = {k: dc_replace(sc, **sc_over) for k, sc in variants.items()}
    policy_cfg = dc_replace(PolicyConfig(), **po_over) if po_over else None
    if args.label:
        variants = {f'{k}_{args.label}': v for k, v in variants.items()}
    seeds = range(args.seed0, args.seed0 + args.seeds)
    rows = run_matrix(variants, seeds, args.policies, args.jobs, policy_cfg)
    summary = summarise(rows)
    write_outputs(Path(args.out), rows, summary,
                  Config(scenario=variants[sorted(variants)[0]]))


if __name__ == '__main__':
    main()
