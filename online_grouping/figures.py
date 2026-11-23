"""Vector figures for Chapter 6, generated only from the committed CSV files.

Every figure is produced from ``results/*/rows.csv`` written by
``online_grouping.experiment``; no value is entered by hand.  Run

    python -m online_grouping.figures --results results --out results/figures

after ``./run_experiments.sh``.  Output is PDF (vector) for direct inclusion in
the thesis.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

POLICY_ORDER = ('S1_reactive', 'S2_periodic', 'S3_event_ch5win',
                'S4_event_riskwin', 'S5_proposed', 'S6_daily',
                'S7_static', 'O2_offline_clair', 'O1_oracle')
POLICY_LABEL = {
    'S1_reactive': 'S1 reactive',
    'S2_periodic': 'S2 periodic',
    'S3_event_ch5win': 'S3 event, nominal',
    'S4_event_riskwin': 'S4 event, risk',
    'S5_proposed': 'S5 proposed',
    'S6_daily': 'S6 daily',
    'S7_static': 'S7 offline',
    'O2_offline_clair': 'O2 offline clairvoyant',
    'O1_oracle': 'O1 clairvoyant',
}
VARIANT_LABEL = {
    'A_low_uniform': 'A low density, uniform',
    'B_low_cluster': 'B low density, clustered',
    'C_high_uniform': 'C high density, uniform',
    'D_high_cluster': 'D high density, clustered',
}


def load(results: Path) -> dict[str, list[dict]]:
    """dataset name -> rows, for every results subdirectory with rows.csv."""
    out: dict[str, list[dict]] = {}
    for sub in sorted(p for p in results.iterdir() if p.is_dir()):
        path = sub / 'rows.csv'
        if not path.exists():
            continue
        # sweeps are addressed by their axis name as well
        with path.open() as fh:
            rows = []
            for r in csv.DictReader(fh):
                rows.append({k: _num(v) for k, v in r.items()})
        out[sub.name] = rows
        if sub.name.startswith('sweep_'):
            out[sub.name[6:]] = rows
    return out


def _num(value):
    if value in (None, '', 'None', 'nan'):
        return None
    try:
        return float(value)
    except ValueError:
        return value


def cell(rows, metric, variant=None, policy=None):
    vals = [r[metric] for r in rows
            if (variant is None or r['variant'] == variant)
            and (policy is None or r['policy'] == policy)
            and isinstance(r.get(metric), float)]
    return vals


def fig_policies(rows, out: Path) -> None:
    """Cost, consolidation and safety per policy and scenario variant."""
    variants = [v for v in VARIANT_LABEL if any(r['variant'] == v for r in rows)]
    panels = [('total_cost', 'total cost $C^{tot}$'),
              ('dispatches', 'interventions in the field $n_b$'),
              ('below_ref_days', 'asset-days below the alarm')]
    fig, axes = plt.subplots(len(variants), len(panels),
                             figsize=(11, 2.5 * len(variants)), sharex=True)
    if len(variants) == 1:
        axes = axes[np.newaxis, :]
    for i, v in enumerate(variants):
        for j, (metric, label) in enumerate(panels):
            ax = axes[i, j]
            data = [[r[metric] for r in rows if r['variant'] == v
                     and r['policy'] == p and isinstance(r[metric], float)]
                    for p in POLICY_ORDER]
            data = [d if d else [np.nan] for d in data]
            ax.boxplot(data, patch_artist=True, medianprops=dict(color='black'),
                       flierprops=dict(markersize=3))
            ax.set_xticks(range(1, len(POLICY_ORDER) + 1),
                          [POLICY_LABEL[p].split()[0] for p in POLICY_ORDER])
            if i == 0:
                ax.set_title(label)
            if j == 0:
                ax.set_ylabel(VARIANT_LABEL[v], fontsize=8)
            ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    fig.text(0.5, 0.005, 'policy (S1 reactive ... O1 clairvoyant)',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_calibration(rows, out: Path, policy: str = 'S5_proposed') -> None:
    """Realised coverage of the predictive band against the nominal tolerance."""
    epsilons = ['picp_0.01', 'picp_0.05', 'picp_0.1', 'picp_0.25']
    nominal = [0.01, 0.05, 0.10, 0.25]
    leads = [5, 10, 20, 40]
    variants = [v for v in VARIANT_LABEL if any(r['variant'] == v for r in rows)]
    fig, axes = plt.subplots(1, len(variants), figsize=(4.6 * len(variants), 3.2),
                             sharey=True)
    if len(variants) == 1:
        axes = [axes]
    for ax, v in zip(axes, variants):
        width = 0.18
        x = np.arange(len(nominal))
        for k, lead in enumerate(leads):
            vals = [np.mean(cell(rows, f'{e}_lead{lead}', variant=v, policy=policy)
                             or [np.nan]) for e in epsilons]
            ax.bar(x + (k - 1.5) * width, vals, width, label=f'lead {lead} d')
        ax.plot(x, nominal, 'k--', linewidth=1, label='nominal')
        ax.set_xticks(x, [f'{n:g}' for n in nominal])
        ax.set_xlabel(r'chance tolerance $\varepsilon$')
        ax.set_title(VARIANT_LABEL[v], fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel('realised coverage (PICP)')
            ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_sensitivity(results: Path, out: Path) -> None:
    """Knob value against total cost, one panel per sensitivity axis."""
    sweeps = load(results)

    def rows_for(name: str):
        return sweeps.get(name) or sweeps.get('sweep_' + name)

    spec = [
        (r'delay bound $\Delta^{delay}$ (d)', [3, 5, 10, 20],
         [('delay3_D', 3), ('main', 5), ('delay10_D', 10), ('delay20_D', 20)],
         'D_high_cluster'),
        (r'advance bound $\Delta^{adv}$ (d)', [3, 6],
         [('main', 3), ('adv6_D', 6)], 'D_high_cluster'),
        (r'chance tolerance $\varepsilon$', [0.01, 0.05, 0.10, 0.25],
         [('eps01_A', 0.01), ('main', 0.05), ('eps10_A', 0.10), ('eps25_A', 0.25)],
         'A_low_uniform'),
        (r'measurement noise $\sigma_y$ (bar)', [0.001, 0.008, 0.020],
         [('clean_A', 0.001), ('main', 0.008), ('noise_A', 0.020)],
         'A_low_uniform'),
        (r'capacity $K^{clus}$ (tasks per visit)', [3, 5, 10],
         [('cap3_D', 3), ('cap5_D', 5), ('main', 10)], 'D_high_cluster'),
        (r'cluster scatter (days, sd)', [3, 8, 15],
         [('main', 3), ('jitter8_D', 8), ('jitter15_D', 15)], 'D_high_cluster'),
        (r'shock rate (per comp.-day)', [0.0, 0.005],
         [('main', 0.0), ('shock_D', 0.005)], 'D_high_cluster'),
        (r'delay penalty $\eta^{+}$', [10, 20],
         [('main', 10), ('eta20_D', 20)], 'D_high_cluster'),
    ]
    policies = ('S3_event_ch5win', 'S4_event_riskwin', 'S5_proposed')
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.2))
    for ax, (xlab, xs, sets, variant) in zip(axes.ravel(), spec):
        for pol in policies:
            ys = []
            for ds, _x in sets:
                rows = rows_for(ds)
                if rows is None:
                    ys.append(np.nan)
                    continue
                vals = [r['total_cost'] for r in rows
                        if r['variant'] == variant and r['policy'] == pol
                        and isinstance(r.get('total_cost'), float)]
                ys.append(np.mean(vals) if vals else np.nan)
            ax.plot(xs, ys, marker='o', linewidth=1.0, markersize=4,
                    label=POLICY_LABEL[pol])
        ax.set_xlabel(xlab, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, linewidth=0.5)
    axes.ravel()[0].legend(fontsize=7, loc='upper left')
    axes.ravel()[0].set_ylabel('total cost $C^{tot}$', fontsize=8)
    axes.ravel()[4].set_ylabel('total cost $C^{tot}$', fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_timeline(trace_csv: Path, out: Path) -> None:
    """Realised pressure paths, plan dates and actual interventions for one run."""
    with trace_csv.open() as fh:
        rows = [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    pmecs = sorted({r['pmec'] for r in rows if r['pmec']})
    fig, axes = plt.subplots(len(pmecs), 1, figsize=(9, 1.9 * len(pmecs)),
                             sharex=True)
    if len(pmecs) == 1:
        axes = [axes]
    for ax, pmec in zip(axes, pmecs):
        sub = [r for r in rows if r['pmec'] == pmec]
        days = [r['day'] for r in sub]
        ax.plot(days, [r['p_true'] for r in sub], color='0.2', label='true pressure')
        hat = [r['p_hat'] for r in sub]
        ax.plot(days, hat, color='tab:blue', linewidth=0.9, label='filtered')
        lo = [None if h is None or s is None else h - 1.645 * s
              for h, s in zip(hat, [r['sd'] for r in sub])]
        ax.plot(days, lo, color='tab:blue', linestyle=':', linewidth=0.8,
                label=r'$5\%$ bound')
        ax.axhline(3.2, color='tab:orange', linewidth=0.8)
        ax.axhline(3.0, color='tab:red', linewidth=0.8)
        execs = [r['day'] for r in sub if r.get('executed')]
        plans = [r['assigned'] for r in sub if r.get('assigned') is not None]
        ax.plot(execs, [3.2] * len(execs), 'v', color='tab:green', ms=6,
                label='intervention')
        for d in sorted(set(int(p) for p in plans)):
            ax.axvline(d, color='tab:green', alpha=0.15, linewidth=0.7)
        ax.set_ylabel(f'{pmec}\npressure (bar)', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].legend(fontsize=7, ncol=5, loc='upper right')
    axes[-1].set_xlabel('day')
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default='results/figures')
    args = ap.parse_args(argv)
    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    main_rows = load(results).get('main', [])
    if not main_rows:
        raise SystemExit('no results/main/rows.csv found; run run_experiments.sh')
    fig_policies(main_rows, out / 'fig_ch6_policies.pdf')
    fig_calibration(main_rows, out / 'fig_ch6_calibration.pdf')
    fig_sensitivity(results, out / 'fig_ch6_sensitivity.pdf')
    trace = results / 'trace.csv'
    if trace.exists():
        fig_timeline(trace, out / 'fig_ch6_timeline.pdf')
    for p in sorted(out.glob('*.pdf')):
        print(f'wrote {p}')


if __name__ == '__main__':
    main()
