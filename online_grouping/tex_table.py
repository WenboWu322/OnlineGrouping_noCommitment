"""LaTeX tables for Chapter 6, generated from the committed result CSVs.

Every number that appears in the chapter's tables is produced here from
``results/*/summary.csv`` or ``results/*/rows.csv``.  Run

    python -m online_grouping.tex_table --results results/main \\
        --out results/tables

and paste (or ``\\input``) the generated files into the chapter.  The generator
also prints the values referenced in the running text, so the prose cannot
drift from the data.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

POLICY_ORDER = ('S1_reactive', 'S2_periodic', 'S3_event_ch5win',
                'S4_event_riskwin', 'S5_proposed', 'S6_daily',
                'S7_static', 'O2_offline_clair', 'O1_oracle')
POLICY_ROW = {
    'S1_reactive': 'S1 & alarm-triggered, no grouping',
    'S2_periodic': 'S2 & periodic re-solve, nominal window',
    'S3_event_ch5win': 'S3 & event-triggered, Chapter~5 window',
    'S4_event_riskwin': 'S4 & event-triggered, risk window',
    'S5_proposed': 'S5 & proposed (risk window + commitments)',
    'S6_daily': 'S6 & daily re-solve',
    'S7_static': 'S7 & Chapter~5 plan applied offline',
    'O2_offline_clair': 'O2 & offline clairvoyant reference',
    'O1_oracle': 'O1 & online clairvoyant reference',
}
VARIANT_LABEL = {
    'A_low_uniform': 'A: low density, uniform',
    'B_low_cluster': 'B: low density, clustered',
    'C_high_uniform': 'C: high density, uniform',
    'D_high_cluster': 'D: high density, clustered',
}


def read_summary(path: Path) -> list[dict]:
    with path.open() as fh:
        return [{k: _num(v) for k, v in r.items()}
                for r in csv.DictReader(fh)]


def _num(v):
    if v in (None, '', 'None', 'nan'):
        return None
    try:
        return float(v)
    except ValueError:
        return v


def get(rows, variant, policy, key):
    for r in rows:
        if r['variant'] == variant and r['policy'] == policy:
            return r.get(key)
    return None


def fmt(value, digits=1, plus=False):
    if value is None:
        return '--'
    spec = f'+.{digits}f' if plus else f'.{digits}f'
    return f'{value:{spec}}'.replace('nan', '--')


def main_table(rows, variant: str) -> str:
    """One scenario variant: cost, consolidation, safety, information use."""
    head = (
        r"""\begin{table}[htbp]
\centering
\caption{Online grouping under data uncertainty -- variant %(label)s
(median over 30 seeds; $C^{\mathrm{tot}}$ total cost, $n_b$ interventions in the field,
$n$ refilling tasks, $\Phi^{\mathrm{pen}}$ shift penalty, $A^{\mathrm{alarm}}$ asset-days below
the refilling alarm, $A^{\mathrm{floor}}$ asset-days below the pressure floor,
$N^{\mathrm{chg}}$ plan changes, $n/n_b$ tasks per intervention,
$\Delta C$ cost gap to the online clairvoyant reference O1).}
\label{tab:ch6-results-%(tag)s}
\footnotesize
\begin{tabular}{@{}l rrrrrrrrr@{}}
\toprule
Policy & $C^{\mathrm{tot}}$ & $n_b$ & $n$ & $n/n_b$ & $\Phi^{\mathrm{pen}}$
       & $A^{\mathrm{alarm}}$ & $A^{\mathrm{floor}}$ & $N^{\mathrm{chg}}$ & $\Delta C$ \\
\midrule
""") % dict(label=VARIANT_LABEL[variant], tag=variant.split('_')[0])
    lines = [head]
    for p in POLICY_ORDER:
        name, desc = POLICY_ROW[p].split(' & ', 1)
        lines.append('%(name)s & %(cost)s & %(nb)s & %(n)s & %(tpd)s & %(pen)s'
                     ' & %(alarm)s & %(floor)s & %(chg)s & %(reg)s \\' % dict(
            name=name, desc=desc,
            cost=fmt(get(rows, variant, p, 'total_cost_mean'), 0),
            nb=fmt(get(rows, variant, p, 'dispatches_mean'), 1),
            n=fmt(get(rows, variant, p, 'interventions_mean'), 1),
            tpd=fmt(get(rows, variant, p, 'tasks_per_dispatch_mean'), 2),
            pen=fmt(get(rows, variant, p, 'shift_penalty_mean'), 0),
            alarm=fmt(get(rows, variant, p, 'below_ref_days_mean'), 1),
            floor=fmt(get(rows, variant, p, 'below_min_days_mean'), 1),
            chg=fmt(get(rows, variant, p, 'plan_changes_mean'), 0),
            reg=fmt(get(rows, variant, p, 'regret_vs_oracle_mean'), 0)))
    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return '\n'.join(lines)


def calibration_table(rows, variant: str, policy: str = 'S5_proposed') -> str:
    head = (r"""\begin{table}[htbp]
\centering
\caption{Calibration of the predictive pressure distribution, %(label)s
(coverage of the $\varepsilon$-quantile at several lead times; the nominal
coverage equals $\varepsilon$).}
\label{tab:ch6-calibration-%(tag)s}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
& \multicolumn{5}{c}{Lead time (days)} \\
\cmidrule(r){2-6}
Tolerance & 1 & 5 & 10 & 20 & 40 \\
\midrule
""") % dict(label=VARIANT_LABEL[variant], tag=variant.split('_')[0])
    lines = [head]
    for eps, key in ((r'$\varepsilon=0.01$', 'picp_0.01'),
                     (r'$\varepsilon=0.05$', 'picp_0.05'),
                     (r'$\varepsilon=0.10$', 'picp_0.1'),
                     (r'$\varepsilon=0.25$', 'picp_0.25')):
        cells = [fmt(get(rows, variant, policy, f'{key}_lead{lead}' if lead != 1
                         else key), 3) for lead in (1, 5, 10, 20, 40)]
        lines.append(eps + ' & ' + ' & '.join(cells) + r' \\')
    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return '\n'.join(lines)


def paired_table(rows, variant: str, baseline: str = 'S3_event_ch5win') -> str:
    """Paired per-seed differences against the Chapter 5 online baseline."""
    keys = (('total_cost', r'$C^{\mathrm{tot}}$'), ('dispatches', r'$n_b$'),
            ('below_ref_days', r'$A^{\mathrm{alarm}}$'))
    head = (r"""\begin{table}[htbp]
\centering
\caption{Paired comparison against %(base)s (mean per-seed difference and
Wilcoxon signed-rank $p$-value over 30 seeds, variant %(label)s).}
\label{tab:ch6-paired-%(tag)s}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
& \multicolumn{2}{c}{%(c1)s} & \multicolumn{2}{c}{%(c2)s}
& \multicolumn{2}{c}{%(c3)s} \\
\cmidrule(r){2-3}\cmidrule(lr){4-5}\cmidrule(r){6-7}
Policy & $\Delta$ & $p$ & $\Delta$ & $p$ & $\Delta$ & $p$ \\
\midrule
""") % dict(base=baseline, label=VARIANT_LABEL[variant], tag=variant.split('_')[0],
            c1=r'$C^{\mathrm{tot}}$', c2=r'$n_b$', c3=r'$A^{\mathrm{alarm}}$')
    lines = [head]
    for p in POLICY_ORDER:
        if p == baseline:
            continue
        cells = []
        for key, _ in keys:
            d = get(rows, variant, p, f'{key}_delta_vs_base')
            pv = get(rows, variant, p, f'{key}_wilcoxon_p')
            cells += [fmt(d, 1, plus=True),
                      '--' if pv is None else (f'{pv:.3f}' if pv >= 1e-3
                                              else r'$<0.001$')]
        lines.append(POLICY_ROW[p].split(' & ')[0] + ' & ' + ' & '.join(cells)
                     + r' \\')
    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return '\n'.join(lines)


def sensitivity_table(main_rows, sweep_rows, sweep: str, variant: str) -> str:
    """One sensitivity axis: what happens to the four event-based policies."""
    head = (r"""\begin{table}[htbp]
\centering
\caption{Sensitivity of the online policies to %(sweep)s (variant %(label)s,
median over 20 seeds; ``base'' is the corresponding entry of
Table~\ref{tab:ch6-results-%(tag)s}).}
\label{tab:ch6-sens-%(tag)s-%(short)s}
\footnotesize
\begin{tabular}{@{}l rr rr rr rr@{}}
\toprule
& \multicolumn{2}{c}{$C^{\mathrm{tot}}$} & \multicolumn{2}{c}{$n_b$}
  & \multicolumn{2}{c}{$A^{\mathrm{alarm}}$} & \multicolumn{2}{c}{$N^{\mathrm{chg}}$} \\
\cmidrule(r){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(r){8-9}
Policy & base & pert. & base & pert. & base & pert. & base & pert. \\
\midrule
""") % dict(sweep=sweep, label=VARIANT_LABEL[variant], tag=variant.split('_')[0],
            short=sweep.split('_')[0])
    lines = [head]
    for p in POLICY_ORDER:
        if p not in ('S3_event_ch5win', 'S4_event_riskwin', 'S5_proposed',
                     'S6_daily'):
            continue
        cells = []
        for key, digits in (('total_cost_mean', 0), ('dispatches_mean', 1),
                            ('below_ref_days_mean', 1), ('plan_changes_mean', 0)):
            cells += [fmt(get(main_rows, variant, p, key), digits),
                      fmt(get(sweep_rows, variant, p, key), digits)]
        lines.append(POLICY_ROW[p].split(' & ')[0] + ' & ' + ' & '.join(cells)
                     + r' \\')
    lines.append(r"""ottomrule
\end{tabular}
\end{table}""")
    return '\n'.join(lines)


def prose_values(rows) -> str:
    """The numbers quoted in the running text, printed for copy-paste."""
    out = []
    for v in VARIANT_LABEL:
        for p in POLICY_ORDER:
            out.append(f"{v} {p}: cost={fmt(get(rows,v,p,'total_cost_mean'),0)}"
                       f" nb={fmt(get(rows,v,p,'dispatches_mean'),1)}"
                       f" n={fmt(get(rows,v,p,'interventions_mean'),1)}"
                       f" alarm={fmt(get(rows,v,p,'below_ref_days_mean'),1)}"
                       f" regret={fmt(get(rows,v,p,'regret_vs_oracle_mean'),0)}"
                       f" t/d={fmt(get(rows,v,p,'tasks_per_dispatch_mean'),2)}"
                       f" churn={fmt(get(rows,v,p,'plan_changes_mean'),0)}"
                       f" shift={fmt(get(rows,v,p,'shift_penalty_mean'),0)}"
                       f" emerg={fmt(get(rows,v,p,'emergency_mean'),1)}"
                       f" unserv={fmt(get(rows,v,p,'unserviced_alarms_mean'),1)}"
                       f" minp={fmt(get(rows,v,p,'min_pressure_mean'),3)}"
                       f" picp={fmt(get(rows,v,p,'picp_0.05_lead10_mean'),3)}"
                       f" p_cost={fmt(get(rows,v,p,'total_cost_wilcoxon_p'),3)}"
                       f" p_alarm={fmt(get(rows,v,p,'below_ref_days_wilcoxon_p'),3)}")
    return '\n'.join(out)


# ----------------------------------------------------------------------
# combined tables: one float for all four variants, so that the chapter
# body stays readable
# ----------------------------------------------------------------------

MAIN_COLS = (('$C^{\\mathrm{tot}}$', 'total_cost_mean', 0),
             ('$n_b$', 'dispatches_mean', 1),
             ('$n$', 'interventions_mean', 1),
             ('$n/n_b$', 'tasks_per_dispatch_mean', 2),
             ('$\\Phi^{\\mathrm{pen}}$', 'shift_penalty_mean', 0),
             ('$A^{\\mathrm{alarm}}$', 'below_ref_days_mean', 1),
             ('$A^{\\mathrm{floor}}$', 'below_min_days_mean', 1),
             ('$N^{\\mathrm{chg}}$', 'plan_changes_mean', 0),
             ('$\\Delta C$', 'regret_vs_oracle_mean', 0))


def main_combined(rows, seeds: int) -> str:
    head = (r"""\begin{table}[htbp]
\centering
\caption{Online grouping under data uncertainty: median over %(seeds)d seeds
per policy and variant. $C^{\mathrm{tot}}$ is the total cost, $n_b$ the
interventions in the field, $n$ the refilling tasks served, $n/n_b$ the
consolidation ratio, $\Phi^{\mathrm{pen}}$ the shift penalty evaluated
against the realised alarm dates, $A^{\mathrm{alarm}}$ and
$A^{\mathrm{floor}}$ the compartment-days below the refilling alarm and
below the pressure floor, $N^{\mathrm{chg}}$ the number of changed plan
dates, and $\Delta C$ the cost gap to the online clairvoyant reference O1.}
\label{tab:ch6-results}
\footnotesize
\begin{tabular}{@{}l %(cols)s@{}}
\toprule
Policy & %(head)s \\
\midrule
""") % dict(seeds=seeds, cols='r' * len(MAIN_COLS),
            head=' & '.join(c[0] for c in MAIN_COLS))
    lines = [head]
    for j, v in enumerate(VARIANT_LABEL):
        if not any(r['variant'] == v for r in rows):
            continue
        if j:
            lines.append(r'\midrule')
        lines.append(r'\multicolumn{%(n)d}{@{}l@{}}{\textbf{%(label)s}} \\'
                     % dict(n=len(MAIN_COLS) + 1, label=VARIANT_LABEL[v]))
        for p in POLICY_ORDER:
            if get(rows, v, p, 'total_cost_mean') is None:
                continue
            cells = [fmt(get(rows, v, p, key), digits) for _, key, digits in MAIN_COLS]
            lines.append(POLICY_ROW[p].split(' & ')[0] + ' & ' + ' & '.join(cells)
                         + r' \\')
    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return '\n'.join(lines)


PAIRED_KEYS = (('total_cost', 0, r'$C^{\mathrm{tot}}$'),
               ('below_ref_days', 1, r'$A^{\mathrm{alarm}}$'),
               ('plan_changes', 0, r'$N^{\mathrm{chg}}$'))


def paired_stats(main_rows, sweep_rows, variant, baseline, policy, key):
    """(median delta, Wilcoxon p) over the seeds shared by two row sets."""
    base = {(r['policy'], int(r['seed'])): r.get(key) for r in main_rows
            if r['variant'] == variant}
    d = []
    for r in sweep_rows:
        if r['variant'] != variant or r['policy'] != policy:
            continue
        a = base.get((baseline, int(r['seed'])))
        b = r.get(key)
        if a is not None and b is not None:
            d.append(b - a)
    if not d:
        return 0.0, 1.0
    try:
        from scipy.stats import wilcoxon
        pv = float(wilcoxon(d).pvalue)
    except Exception:
        pv = float('nan')
    return float(np.median(d)), pv


def paired_combined(rows, seeds: int, baseline: str = 'S3_event_ch5win') -> str:
    """`rows` are per-seed rows (rows.csv), not the summary."""
    head = (r"""\begin{table}[htbp]
\centering
\caption{Paired per-seed differences against the baseline %(base)s and Wilcoxon
signed-rank $p$-value over %(seeds)d seeds. A positive $\Delta$ is worse
for costs and for plan changes, and better for $A^{\mathrm{alarm}}$,
which counts exposure below the alarm.}
\label{tab:ch6-paired}
\footnotesize
\begin{tabular}{@{}l %(cols)s@{}}
\toprule
& \multicolumn{2}{c}{$C^{\mathrm{tot}}$} & \multicolumn{2}{c}{$A^{\mathrm{alarm}}$}
  & \multicolumn{2}{c}{$N^{\mathrm{chg}}$} \\
\cmidrule(r){2-3}\cmidrule(lr){4-5}\cmidrule(r){6-7}
Policy & $\Delta$ & $p$ & $\Delta$ & $p$ & $\Delta$ & $p$ \\
\midrule
""") % dict(base=baseline.split('_')[0], seeds=seeds, cols='rr' * len(PAIRED_KEYS))
    lines = [head]
    for j, v in enumerate(VARIANT_LABEL):
        if not any(r['variant'] == v for r in rows):
            continue
        if j:
            lines.append(r'\midrule')
        lines.append(r'\multicolumn{7}{@{}l@{}}{\textbf{%(label)s}} \\'
                     % dict(label=VARIANT_LABEL[v]))
        for p in POLICY_ORDER:
            if p == baseline or get(rows, v, p, 'total_cost') is None:
                continue
            cells = []
            for key, digits, _ in PAIRED_KEYS:
                d, pv = paired_stats(rows, rows, v, baseline, p, key)
                cells += [fmt(d, digits, plus=True),
                          r'$<0.001$' if pv < 1e-3 else f'{pv:.3f}']
            lines.append(POLICY_ROW[p].split(' & ')[0] + ' & '
                         + ' & '.join(cells) + r' \\')
    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return '\n'.join(lines)


def calibration_leads(rows, policy: str = 'S5_proposed') -> str:
    """Coverage of the lower predictive bound by lead time at the operating
    tolerance, with the sharpness and reference-date accuracy of the model."""
    head = (r"""\begin{table}[htbp]
\centering
\caption{Coverage of the lower predictive pressure bound under S5 at the
operating tolerance $\varepsilon=0.05$, by lead time, together with the mean
predictive standard deviation (sharpness) and the mean absolute error of the
predicted reference date. Horizons containing a planned intervention are
excluded: a refill is a structural break of the degradation model, not a
prediction failure. The nominal coverage is $0.05$.}
\label{tab:ch6-calibration}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
& \multicolumn{5}{c}{lead time (days)} & mean $\sigma$ & MAE $\widehat{\tau}^0$ \\
\cmidrule(r){2-6}
Variant & 1 & 5 & 10 & 20 & 40 & (bar) & (days) \\
\midrule
""")
    lines = [head]
    for v in VARIANT_LABEL:
        if not any(r['variant'] == v and r['policy'] == policy for r in rows):
            continue
        cells = [fmt(get(rows, v, policy, f'picp_0.05_lead{lead}_mean'
                         if lead != 1 else 'picp_0.05_mean'), 3)
                 for lead in (1, 5, 10, 20, 40)]
        cells += [fmt(get(rows, v, policy, 'mean_sd_mean'), 3),
                  fmt(get(rows, v, policy, 'mae_reference_mean'), 1)]
        lines.append(f'{v.split("_")[0]} & ' + ' & '.join(cells) + r' \\')
    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return '\n'.join(lines)


def calibration_eps(rows, policy: str = 'S5_proposed') -> str:
    """Coverage against the nominal tolerance, at a one-day lead."""
    head = (r"""\begin{table}[htbp]
\centering
\caption{Realised coverage of the one-day predictive bound under S5 against
the nominal tolerance, for the four chance-constraint levels. An entry below
the nominal value is conservative: the predicted pressure falls below its own
bound less often than the tolerance allows, so the chance constraint does not
bind.}
\label{tab:ch6-calibration-eps}
\begin{tabular}{@{}lrrrr@{}}
\toprule
& \multicolumn{4}{c}{nominal $\varepsilon$} \\
\cmidrule(r){2-5}
Variant & 0.01 & 0.05 & 0.10 & 0.25 \\
\midrule
""")
    lines = [head]
    for v in VARIANT_LABEL:
        if not any(r['variant'] == v and r['policy'] == policy for r in rows):
            continue
        cells = [fmt(get(rows, v, policy, f'{key}_mean'), 3)
                 for key in ('picp_0.01', 'picp_0.05', 'picp_0.1', 'picp_0.25')]
        lines.append(f'{v.split("_")[0]} & ' + ' & '.join(cells) + r' \\')
    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return '\n'.join(lines)


def _calibration_legacy(rows, policy: str = 'S5_proposed') -> str:
    head = (r"""\begin{table}[htbp]
\centering
\caption{Coverage of the lower predictive pressure bound under %(pol)s,
by lead time; the nominal coverage equals $\varepsilon$. Horizons
containing a planned intervention are excluded, since a refill is a
structural break of the model and not a prediction failure.}
\label{tab:ch6-calibration}
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Variant & $\varepsilon$ & 1 & 5 & 10 & 20 & 40 \\
\midrule
""") % dict(pol=policy)
    lines = [head]
    for j, v in enumerate(VARIANT_LABEL):
        if not any(r['variant'] == v and r['policy'] == policy for r in rows):
            continue
        if j:
            lines.append(r'\addlinespace')
        for eps, key in ((r'$0.01$', 'picp_0.01'), (r'$0.05$', 'picp_0.05'),
                         (r'$0.10$', 'picp_0.1'), (r'$0.25$', 'picp_0.25')):
            cells = [fmt(get(rows, v, policy,
                             f'{key}_lead{lead}_mean' if lead != 1
                             else f'{key}_mean'), 3)
                     for lead in (1, 5, 10, 20, 40)]
            lines.append(f'{v.split("_")[0]} & {eps} & ' + ' & '.join(cells)
                         + r' \\')
    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return '\n'.join(lines)


AXIS_LABEL = {
    'noise_A': (r'$\sigma_y=0.020$ ($3\times$ nominal)', 'A_low_uniform'),
    'noise_C': (r'$\sigma_y=0.020$ ($3\times$ nominal)', 'C_high_uniform'),
    'clean_A': (r'clean records', 'A_low_uniform'),
    'missing_A': (r'$20\%$ records missing', 'A_low_uniform'),
    'eps01_A': (r'$\varepsilon=0.01$', 'A_low_uniform'),
    'eps10_A': (r'$\varepsilon=0.10$', 'A_low_uniform'),
    'eps25_A': (r'$\varepsilon=0.25$', 'A_low_uniform'),
    'shock_C': (r'sudden leaks ($5\times10^{-3}$ per day)',
                'C_high_uniform'),
    'shock_D': (r'sudden leaks ($5\times10^{-3}$ per day)',
                'D_high_cluster'),
    'delay3_D': (r'$\Delta^{\mathrm{delay}}=3$ days', 'D_high_cluster'),
    'delay10_D': (r'$\Delta^{\mathrm{delay}}=10$ days', 'D_high_cluster'),
    'delay20_D': (r'$\Delta^{\mathrm{delay}}=20$ days', 'D_high_cluster'),
    'adv6_D': (r'$\Delta^{\mathrm{adv}}=6$ days', 'D_high_cluster'),
    'win20_D': (r'$L^{\mathrm{win}}=20$ days', 'D_high_cluster'),
    'cap3_D': (r'$K^{\mathrm{clus}}=3$ tasks per visit', 'D_high_cluster'),
    'cap5_D': (r'$K^{\mathrm{clus}}=5$ tasks per visit', 'D_high_cluster'),
    'jitter8_D': (r'cluster scatter, sd 8 d', 'D_high_cluster'),
    'jitter15_D': (r'cluster scatter, sd 15 d', 'D_high_cluster'),
    'eta20_D': (r'$\eta^{+}=20$ per day of delay', 'D_high_cluster'),
    'nolock_S5': (r'commitments removed ($L^{\mathrm{lock}}=0$,'
                  r' $\eta^{\mathrm{stab}}=0$)', 'D_high_cluster'),
}


def read_rows(path: Path) -> list[dict]:
    with open(path) as fh:
        rows = []
        for r in csv.DictReader(fh):
            for k, v in list(r.items()):
                if k in ('variant', 'policy'):
                    continue
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r.pop(k, None)
            rows.append(r)
        return rows


def paired_delta(main_rows, sweep_rows, variant, policy, key):
    """Median per-seed difference between a sweep and the main comparison.

    Sweeps share the seed sequence with the main comparison, so for axes that
    do not alter the environment itself the paired difference isolates the
    axis; axes that resample the instance are flagged in the caption.
    """
    base = {(r['policy'], int(r['seed'])): r.get(key)
            for r in main_rows if r['variant'] == variant}
    out = []
    for r in sweep_rows:
        if r['variant'] != variant or r['policy'] != policy:
            continue
        a = base.get((r['policy'], int(r['seed'])))
        b = r.get(key)
        if a is not None and b is not None:
            out.append(b - a)
    return out


def sensitivity_combined(main_csv, sweeps: dict[str, list[dict]]) -> str:
    """One row per sensitivity axis: change against the main comparison."""
    head = r"""\begin{table}[htbp]
\centering
\caption{Effect of the sensitivity axes on total cost and on alarm
exposure, as the median difference to the corresponding entry of
Table~\ref{tab:ch6-results} (20 seeds). S3 is the deterministic
Chapter~\ref{chap:grouping} method operated online, S5 the proposed
scheme. A positive entry means higher cost or higher exposure.}
\label{tab:ch6-sensitivity}
%(note)s
\footnotesize
\begin{tabular}{@{}l l rr rr@{}}
\toprule
Axis & Variant & \multicolumn{2}{c}{$\Delta C^{\mathrm{tot}}$}
     & \multicolumn{2}{c}{$\Delta A^{\mathrm{alarm}}$} \\
\cmidrule(r){3-4}\cmidrule(r){5-6}
& & S3 & S5 & S3 & S5 \\
\midrule
"""
    head = head % dict(note='\\setlength{\\tabcolsep}{4pt}')
    lines = [head]
    for name, (label, variant) in AXIS_LABEL.items():
        srows = sweeps.get(name)
        if not srows:
            continue
        cells = []
        for key, digits in (('total_cost', 0), ('below_ref_days', 1)):
            for pol in ('S3_event_ch5win', 'S5_proposed'):
                d = paired_delta(main_csv, srows, variant, pol, key)
                cells.append('--' if not d
                             else f'{float(np.median(d)):+.{digits}f}')
        lines.append(f'{label} & {variant.split("_")[0]} & '
                     + ' & '.join(cells) + r' \\')
    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return '\n'.join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', default='results/main')
    ap.add_argument('--out', default='results/tables')
    args = ap.parse_args(argv)
    src = Path(args.results)
    summary = src / 'summary.csv' if src.is_dir() else src
    rows = read_summary(summary)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in VARIANT_LABEL:
        if not any(r['variant'] == v for r in rows):
            continue
        (out / f'tab_ch6_main_{v}.tex').write_text(main_table(rows, v) + '\n')
        (out / f'tab_ch6_paired_{v}.tex').write_text(paired_table(rows, v) + '\n')
        (out / f'tab_ch6_calibration_{v}.tex').write_text(
            calibration_table(rows, v) + '\n')
    (out / 'tab_ch6_main_combined.tex').write_text(
        main_combined(rows, int(rows[0]['n_seeds']) if rows else 0) + '\n')
    per_seed = read_rows(src / 'rows.csv')
    (out / 'tab_ch6_paired_combined.tex').write_text(
        paired_combined(per_seed, int(rows[0]['n_seeds']) if rows else 0) + '\n')
    (out / 'tab_ch6_calibration_leads.tex').write_text(
        calibration_leads(rows) + '\n')
    (out / 'tab_ch6_calibration_eps.tex').write_text(
        calibration_eps(rows) + '\n')
    (out / 'prose_values.txt').write_text(prose_values(rows) + '\n')
    root = src.parent if src.is_dir() else src.parent
    sweeps = {}
    for sub in sorted(root.glob('sweep_*')):
        srows = sub / 'rows.csv'
        if srows.exists():
            sweeps[sub.name[6:]] = read_rows(srows)
    if sweeps:
        (out / 'tab_ch6_sensitivity.tex').write_text(
            sensitivity_combined(read_rows(src / 'rows.csv'), sweeps) + '\n')
    for sub in sorted(root.glob('sweep_*')):
        ssum = sub / 'summary.csv'
        if not ssum.exists():
            continue
        srows = read_summary(ssum)
        variants = sorted({r['variant'] for r in srows} & set(VARIANT_LABEL))
        for v in variants:
            (out / f'tab_ch6_sens_{v.split("_")[0]}_{sub.name[6:]}.tex').write_text(
                sensitivity_table(rows, srows, sub.name[6:], v) + '\n')
    for p in sorted(out.glob('*.tex')):
        print(f'wrote {p}')
    print(prose_values(rows))


if __name__ == '__main__':
    main()
