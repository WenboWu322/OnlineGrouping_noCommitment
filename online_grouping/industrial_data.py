"""Statistics and figures for the real RTE data described in Chapter 2.

The thesis never uses these records as model inputs: the scheduling and
grouping instances are synthetic.  What Chapter 2 reports is the shape of
the real data (fields, coverage, ranges) and the evidence behind the model
parameters.  Every number quoted in the chapter is recomputed here and
written to ``results/industrial/industrial_data_summary.json``.

Run::

    python -m online_grouping.industrial_data \
        --data /path/to/data \
        --fits /path/to/Uncertainty_20261012/cycle_fit_results \
        --out results/industrial --figures results/figures

Inputs:

* ``Données patrimoniales.xlsm``  -- equipment register, one row per CSEM,
* ``operations_CR_2005_2024.xlsx`` -- refilling (CR) history per equipment,
* ``reduced length monitoring data.xlsx`` -- continuous pressure monitoring,
* ``cycle_fit_summary.csv`` and ``pressure_data_with_cycle_fits.csv`` --
  cycle-level slope fits over a monitoring extract.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def q(v: pd.Series) -> dict:
    v = pd.to_numeric(v, errors='coerce')
    return {
        'n': int(len(v)),
        'non_null': int(v.notna().sum()),
        'missing': int(v.isna().sum()),
        'min': round(float(v.min()), 4),
        'p05': round(float(v.quantile(.05)), 4),
        'median': round(float(v.median()), 4),
        'p90': round(float(v.quantile(.90)), 4),
        'p99': round(float(v.quantile(.99)), 4),
        'max': round(float(v.max()), 4),
    }


# --------------------------------------------------------------------------
# equipment register
# --------------------------------------------------------------------------

def register(path: Path) -> dict:
    xl = pd.ExcelFile(path)
    out = {}
    for sheet in xl.sheet_names:
        d = xl.parse(sheet)
        d.columns = [str(c).strip() for c in d.columns]
        if 'Pression Nominale Relative' not in d.columns:
            continue
        for c in ('Pression Nominale Relative', 'Masse de Gaz',
                  'Rapport Masse/Pression', 'Volume'):
            d[c] = pd.to_numeric(d[c], errors='coerce')
        s = {
            'rows': int(len(d)),
            'distinct_equipment': int(d['N° eqt'].nunique()),
            'nominal_pressure_counts': {str(k): int(v) for k, v in
                                        d['Pression Nominale Relative'].value_counts().head(8).items()},
            'nominal_pressure_missing': int(d['Pression Nominale Relative'].isna().sum()),
            'gas_mass': q(d['Masse de Gaz']),
            'gas_mass_zero': int((d['Masse de Gaz'] == 0).sum()),
            'volume': q(d['Volume']),
            'volume_zero': int((d['Volume'] == 0).sum()),
        }
        both = d.dropna(subset=['Rapport Masse/Pression', 'Volume'])
        both = both[(both['Volume'] > 0) & (both['Rapport Masse/Pression'] > 0)]
        k = both['Rapport Masse/Pression'] / both['Volume']
        s['kappa_ratio_over_volume'] = {
            'rows': int(len(both)),
            'min': round(float(k.min()), 3),
            'median': round(float(k.median()), 3),
            'max': round(float(k.max()), 3),
        }
        sub = d[d['Pression Nominale Relative'] == 3.5]
        s['nominal_3p5_bar'] = {
            'rows': int(len(sub)),
            'volume': q(sub['Volume']),
            'volume_zero': int((sub['Volume'] == 0).sum()),
            'gas_mass': q(sub['Masse de Gaz']),
        }
        out[sheet] = s
    return out


# --------------------------------------------------------------------------
# continuous monitoring
# --------------------------------------------------------------------------

LONG_DIGITS = re.compile(r'^\d{8,}$')
SHORT_INT = re.compile(r'^-?\d{1,3}$')


def parse_monitoring_row(cells: list) -> tuple[list[str], int]:
    """One record reached us as a ';'-separated string that the spreadsheet
    broke across four cells, and the decimal separator of every float
    survived as another ';' delimiter.  Re-join the cells and repair the
    numbers: a long digit run belongs to the short integer before it."""
    s = ';'.join('' if c is None else str(c) for c in cells)
    out: list[str] = []
    repaired = 0
    for tok in s.split(';'):
        tok = tok.strip()
        if LONG_DIGITS.match(tok) and out and SHORT_INT.match(out[-1]):
            out[-1] = out[-1] + '.' + tok
            repaired += 1
        else:
            out.append(tok)
    return out, repaired


def load_monitoring(path: Path) -> tuple[pd.DataFrame, int]:
    import openpyxl
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)['Sheet1']
    rows = list(ws.iter_rows(values_only=True))
    header = str(rows[0][0]).split(';')
    parsed = [parse_monitoring_row(r) for r in rows[1:]]
    d = pd.DataFrame([p[0] for p in parsed], columns=header)
    for c in ('temperature', 'compensatedRelativePressure', 'pressure'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d['ts'] = pd.to_datetime(d['dateMeasure'], errors='coerce', format='ISO8601')
    d = d.sort_values(['csemCur', 'ts']).reset_index(drop=True)
    return d, int(sum(p[1] for p in parsed))


def monitoring(path: Path) -> dict:
    d, repaired = load_monitoring(path)
    d = d.copy()
    d['dp'] = d.groupby('csemCur').compensatedRelativePressure.diff()
    d['dt_h'] = d.groupby('csemCur').ts.diff().dt.total_seconds() / 3600.0
    span = d.groupby('csemCur').ts.agg(['min', 'max', 'size'])
    span_h = (span['max'] - span['min']).dt.total_seconds() / 3600.0
    per_std = d.groupby('csemCur').compensatedRelativePressure.std()
    per_min = d.groupby('csemCur').compensatedRelativePressure.min()
    per_rng = d.groupby('csemCur').compensatedRelativePressure.agg(lambda s: s.max() - s.min())
    sites = sorted({str(g).split('-')[0] for g in d['csemGroupName'].dropna().unique()})
    grp = d.groupby('csemGroupName')['csemCur'].nunique()
    ok = d.dropna(subset=['pressure', 'temperature'])
    return {
        'records': int(len(d)),
        'fields': d.columns.tolist()[:10],
        'rows_repaired_for_decimal_separator': repaired,
        'compartments': int(d['csemCur'].nunique()),
        'monitoring_groups': int(d['csemGroupName'].nunique()),
        'compartments_per_group': {'min': int(grp.min()), 'median': float(grp.median()),
                                   'max': int(grp.max())},
        'sites': len(sites),
        'first_record': str(d['ts'].min()),
        'last_record': str(d['ts'].max()),
        'group_types': {str(k): int(v) for k, v in d['csemGroupType'].value_counts().head(5).items()},
        'functionalities_top': {str(k): int(v) for k, v in
                                d['csemFunctionality'].value_counts().head(8).items()},
        'sensor_category': {str(k): int(v) for k, v in d['sensorCategory'].value_counts().head(3).items()},
        'compensated_pressure': q(d['compensatedRelativePressure']),
        'raw_pressure': q(d['pressure']),
        'temperature': q(d['temperature']),
        'corr_raw_pressure_temperature': round(float(np.corrcoef(ok['pressure'], ok['temperature'])[0, 1]), 3),
        'corr_compensated_temperature': round(float(np.corrcoef(
            d.loc[d.compensatedRelativePressure.notna(), 'compensatedRelativePressure'],
            d.loc[d.compensatedRelativePressure.notna(), 'temperature'])[0, 1]), 3),
        'records_per_compartment': {'min': int(span['size'].min()),
                                    'median': float(span['size'].median()),
                                    'max': int(span['size'].max())},
        'span_days_per_compartment': {'median': round(float(span_h.median() / 24), 1),
                                      'max': round(float(span_h.max() / 24), 1)},
        'coverage_of_hourly_sampling_pct': round(float(100 * span['size'].median() / span_h.median()), 2),
        'gap_hours': {'median': float(d['dt_h'].median()),
                      'p95': round(float(d['dt_h'].quantile(.95)), 0),
                      'max': round(float(d['dt_h'].max()), 0),
                      'n_over_24h': int((d['dt_h'] > 24).sum()),
                      'n_over_7d': int((d['dt_h'] > 168).sum())},
        'consecutive_pairs': int(d['dp'].notna().sum()),
        'abs_dp_bar': {'median': round(float(d['dp'].abs().median()), 4),
                       'p90': round(float(d['dp'].abs().quantile(.90)), 4),
                       'p99': round(float(d['dp'].abs().quantile(.99)), 4),
                       'max': round(float(d['dp'].abs().max()), 4),
                       'n_over_0p05': int((d['dp'].abs() > 0.05).sum()),
                       'n_over_0p20': int((d['dp'].abs() > 0.20).sum())},
        'per_compartment_std_bar': {'median': round(float(per_std.median()), 4),
                                    'p90': round(float(per_std.quantile(.90)), 4),
                                    'max': round(float(per_std.max()), 4)},
        'within_compartment_range_bar': {'median': round(float(per_rng.median()), 3),
                                         'p90': round(float(per_rng.quantile(.90)), 3),
                                         'max': round(float(per_rng.max()), 3)},
        'compartments_ever_below_3p2': int((per_min < 3.2).sum()),
        'compartments_ever_below_3p0': int((per_min < 3.0).sum()),
        'duplicate_compartment_timestamp_rows': int(d.duplicated(['csemCur', 'ts']).sum()),
    }


# --------------------------------------------------------------------------
# refilling history
# --------------------------------------------------------------------------

HIST_SHEET = 'Patrimoine vs CR_2010-2024'


def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=HIST_SHEET)
    df.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in df.columns]
    return df


def history(path: Path) -> dict:
    df = load_history(path)
    mass = 'Masse de Gaz CSEM (kg)'
    pct = '% de la masse de gaz rejetée par CR'
    mrej = 'Masse rejetée à chaque CR (kg)'
    crit = "Nb atteinte critère d'exploit 12 mois"
    m12 = 'Masse rejetée 1 an glissant (kg)'
    pn = 'Pression Nominale (bar)'
    years = [c for c in df.columns if c.startswith('Nb CR')]
    for c in [mass, pct, mrej, crit, m12, pn] + years:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    tot = df[years].sum(axis=1)
    dmes = pd.to_datetime(df['Date MES CSEM'], errors='coerce')
    age = (pd.Timestamp('2024-12-31') - dmes).dt.days / 365.25
    return {
        'sheet': HIST_SHEET,
        'rows': int(len(df)),
        'distinct_equipment': int(df['N Equipement'].nunique()),
        'duplicate_equipment_rows': int(len(df) - df['N Equipement'].nunique()),
        'nominal_pressure_counts': {str(k): int(v) for k, v in df[pn].value_counts().head(8).items()},
        'nominal_pressure_missing': int(df[pn].isna().sum()),
        'nominal_pressure_outside_2p9_3p6': int(((df[pn] < 2.9) | (df[pn] > 3.6)).sum()),
        'gas_mass': q(df[mass]),
        'gas_mass_zero': int((df[mass] == 0).sum()),
        'pct_mass_per_refill': q(df[pct]),
        'mass_per_refill_kg': q(df[mrej]),
        'refills_by_year': {c.replace('Nb CR ', ''): int(df[c].sum()) for c in years},
        'refills_total': int(df[years].sum().sum()),
        'refills_per_equipment': {'median': float(tot.median()),
                                  'p75': float(tot.quantile(.75)),
                                  'p90': float(tot.quantile(.90)),
                                  'p99': float(tot.quantile(.99)),
                                  'max': float(tot.max()),
                                  'share_ge_3': round(float((tot >= 3).mean()), 3),
                                  'share_ge_5': round(float((tot >= 5).mean()), 3)},
        'equipment_with_at_least_one_refill': int((tot > 0).sum()),
        'operational_criterion_hits': {'flagged_equipment': int((df[crit] >= 1).sum()),
                                       'flagged_twice_or_more': int((df[crit] >= 2).sum()),
                                       'max': float(df[crit].max()),
                                       'non_null': int(df[crit].notna().sum())},
        'rolling_year_rejected_mass_kg': q(df[m12]),
        'last_work_order_nature': {str(k): int(v) for k, v in
                                   df['Nature du dernier OT EVA realized'].value_counts(dropna=False).head(5).items()},
        'last_leak_treated': {str(k): int(v) for k, v in
                              df['La dernière Fuite est-elle traitée?'].value_counts(dropna=False).head(5).items()},
        'commissioning_dates': {'min': str(dmes.min().date()),
                                'max': str(dmes.max().date()),
                                'unparsed': int(dmes.isna().sum())},
        'age_years_2024': {'median': round(float(age.median()), 1),
                           'p90': round(float(age.quantile(.90)), 1),
                           'max': round(float(age.max()), 1)},
    }


# --------------------------------------------------------------------------
# cycle-level slope fits
# --------------------------------------------------------------------------

def cycles(root: Path) -> dict:
    fits = pd.read_csv(root / 'cycle_fit_summary.csv')
    raw = pd.read_csv(root / 'pressure_data_with_cycle_fits.csv')
    fits['alpha'] = pd.to_numeric(fits['alpha_hat'], errors='coerce')
    f = fits[fits['alpha'].notna()].copy()
    lo = pd.to_numeric(f['alpha_ci_low'], errors='coerce')
    hi = pd.to_numeric(f['alpha_ci_high'], errors='coerce')
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    per = raw.groupby('pmec_id')
    missing_days = [len(pd.date_range(g['timestamp'].min(), g['timestamp'].max())) - len(g)
                    for _, g in per]
    rmse = pd.to_numeric(f['residual_rmse'], errors='coerce')
    r2 = pd.to_numeric(f['r_squared'], errors='coerce')
    out = {
        'cycles': int(len(fits)),
        'fits_with_supported_trend': int(len(f)),
        'compartments': int(fits['pmec_id'].nunique()),
        'alpha_bar_per_day': {'min': round(float(f['alpha'].min()), 5),
                              'p25': round(float(f['alpha'].quantile(.25)), 5),
                              'median': round(float(f['alpha'].median()), 5),
                              'p75': round(float(f['alpha'].quantile(.75)), 5),
                              'max': round(float(f['alpha'].max()), 5)},
        'alpha_ci_width_bar_per_day': {'median': round(float((hi - lo).median()), 5),
                                       'max': round(float((hi - lo).max()), 5)},
        'r_squared': {'min': round(float(r2.min()), 4), 'median': round(float(r2.median()), 4)},
        'residual_rmse_bar': {'min': round(float(rmse.min()), 5),
                              'median': round(float(rmse.median()), 5),
                              'max': round(float(rmse.max()), 5)},
        'fit_span_days': {'median': float(pd.to_numeric(fits['fit_span_days']).median()),
                          'max': float(pd.to_numeric(fits['fit_span_days']).max())},
        'fit_points': {'median': float(pd.to_numeric(fits['n_fit_points']).median()),
                       'max': float(pd.to_numeric(fits['n_fit_points']).max()),
                       'without_points': int((pd.to_numeric(fits['n_fit_points']) == 0).sum())},
        'extract_rows': int(len(raw)),
        'extract_compartments': int(raw['pmec_id'].nunique()),
        'extract_first': str(raw['timestamp'].min().date()),
        'extract_last': str(raw['timestamp'].max().date()),
        'extract_calendar_days': int((raw['timestamp'].max() - raw['timestamp'].min()).days + 1),
        'extract_days_per_compartment': {'min': int(per.size().min()),
                                         'median': float(per.size().median()),
                                         'max': int(per.size().max())},
        'extract_missing_days_per_compartment': {'median': float(np.median(missing_days)),
                                                 'max': int(max(missing_days))},
        'raw_measurements_per_day': {'min': int(raw['n_measurements'].min()),
                                     'median': float(raw['n_measurements'].median()),
                                     'max': int(raw['n_measurements'].max()),
                                     'single_measurement_days': int((raw['n_measurements'] == 1).sum())},
        'refill_days_in_extract': int(raw['is_gr_day'].sum()),
        'pressure_obs_range_bar': [round(float(pd.to_numeric(raw['pressure_numeric']).min()), 3),
                                   round(float(pd.to_numeric(raw['pressure_numeric']).max()), 3)],
        'unsupported_fits': int((r2 < 0.05).sum()),
        'compartments_with_several_fits': int((f.groupby('pmec_id').size() >= 3).sum()),
        'within_compartment_rate_ratio_max_over_min': {},
        'within_compartment_factor_two': 0,
    }
    per_fit = f.groupby('pmec_id')['alpha']
    stats = {}
    for label, (pid, g) in enumerate(per_fit, start=1):
        if len(g) >= 3:
            ratio = float(g.max() / g.min())
            # register ids are not reproduced in artifacts; use ordered letters
            stats[chr(ord('A') + label - 1)] = {'n': int(len(g)),
                                               'min': round(float(g.min()), 5),
                                               'max': round(float(g.max()), 5),
                                               'ratio': round(ratio, 2)}
    out['within_compartment_rate_ratio_max_over_min'] = stats
    out['within_compartment_factor_two'] = int(sum(1 for v in stats.values() if v['ratio'] <= 2.0))
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_monitoring(fits_dir: Path, out: Path) -> None:
    raw = pd.read_csv(fits_dir / 'pressure_data_with_cycle_fits.csv')
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    ids = sorted(raw['pmec_id'].unique())[:2]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4), sharey=True)
    for k, (ax, pid) in enumerate(zip(axes, ids)):
        g = raw[raw['pmec_id'] == pid].sort_values('timestamp')
        p = pd.to_numeric(g['pressure_numeric'], errors='coerce')
        ax.plot(g['timestamp'], p, lw=0.9, color='#1f4e79')
        gr = g[pd.to_numeric(g['is_gr_day'], errors='coerce') == 1]
        ax.vlines(pd.to_datetime(gr['timestamp']), 2.95, 3.70, color='#b4451f', lw=0.8, alpha=.8)
        used = g[g['used_in_cycle_fit'] == True]  # noqa: E712
        ax.plot(pd.to_datetime(used['timestamp']),
                pd.to_numeric(used['fitted_pressure'], errors='coerce'), '.', ms=3, color='#2e7d32')
        ax.axhline(3.2, color='grey', ls='--', lw=.7)
        ax.axhline(3.0, color='k', ls=':', lw=.7)
        ax.set_title('instrumented compartment %d' % (k + 1), fontsize=9)
        ax.tick_params(labelsize=7)
        for lab in ax.get_xticklabels():
            lab.set_rotation(20)
    axes[0].set_ylim(2.95, 3.70)
    axes[0].set_ylabel('compensated pressure (bar)', fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_quality(mon_path: Path, out: Path) -> None:
    d, _ = load_monitoring(mon_path)
    d = d.copy()
    d['dp'] = d.groupby('csemCur').compensatedRelativePressure.diff()
    d['dt_h'] = d.groupby('csemCur').ts.diff().dt.total_seconds() / 3600.0
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))
    axes[0].hist(np.log10(d['dt_h'].dropna()), bins=40, color='#4a6f8a')
    for h, lab in ((1, '1 h'), (24, '1 d'), (168, '1 w')):
        axes[0].axvline(np.log10(h), color='#b4451f', lw=.8, ls='--')
        axes[0].text(np.log10(h) + .05, axes[0].get_ylim()[1] * .88, lab, fontsize=7)
    axes[0].set_xlabel(r'log$_{10}$ gap between consecutive records (h)', fontsize=8)
    axes[0].set_ylabel('pairs of records', fontsize=8)
    axes[0].set_title('transmission gaps', fontsize=9)
    axes[1].hist(d['dp'].abs().dropna() * 1000, bins=np.arange(0, 101, 2), color='#4a6f8a')
    axes[1].axvline(50, color='#b4451f', lw=.8, ls='--')
    axes[1].axvline(200, color='#b4451f', lw=.8, ls=':')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('|change in compensated pressure| (mbar)', fontsize=8)
    axes[1].set_ylabel('pairs of records (log)', fontsize=8)
    axes[1].set_title('apparent pressure jumps', fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_fleet(data_dir: Path, out: Path) -> None:
    reg = pd.ExcelFile(data_dir / 'Données patrimoniales.xlsm').parse('Patrimoine CSEM')
    reg.columns = [str(c).strip() for c in reg.columns]
    reg['Pression Nominale Relative'] = pd.to_numeric(reg['Pression Nominale Relative'], errors='coerce')
    hist = load_history(data_dir / 'operations_CR_2005_2024.xlsx')
    years = [c for c in hist.columns if c.startswith('Nb CR')]
    for c in years:
        hist[c] = pd.to_numeric(hist[c], errors='coerce')
    tot = hist[years].sum(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.1))
    vc = reg['Pression Nominale Relative'].value_counts().head(9)
    axes[0].bar([str(k) for k in vc.index], vc.values, color='#4a6f8a')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('nominal relative pressure (bar)', fontsize=8)
    axes[0].set_ylabel('compartments (log)', fontsize=8)
    axes[0].set_title('equipment register', fontsize=9)
    axes[1].hist(tot[tot > 0], bins=np.arange(0, 131, 5), color='#4a6f8a')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('refillings per compartment, 2010-2024', fontsize=8)
    axes[1].set_ylabel('compartments (log)', fontsize=8)
    axes[1].set_title('refilling history', fontsize=9)
    yr = {c.replace('Nb CR ', ''): int(hist[c].sum()) for c in years}
    axes[2].plot(list(yr), yr.values(), marker='o', ms=4, color='#1f4e79')
    axes[2].set_xlabel('year', fontsize=8)
    axes[2].set_ylabel('refillings', fontsize=8)
    axes[2].set_title('refillings per year', fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def fig_slopes(fits_dir: Path, out: Path) -> None:
    f = pd.read_csv(fits_dir / 'cycle_fit_summary.csv')
    f['alpha'] = pd.to_numeric(f['alpha_hat'], errors='coerce')
    f = f[f['alpha'].notna()].copy()
    f['lo'] = pd.to_numeric(f['alpha_ci_low'], errors='coerce')
    f['hi'] = pd.to_numeric(f['alpha_ci_high'], errors='coerce')
    f = f.sort_values('alpha')
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    x = np.arange(len(f))
    ax.errorbar(x, f['alpha'], yerr=[f['alpha'] - f['lo'], f['hi'] - f['alpha']],
                fmt='o', ms=4, lw=.9, ecolor='#8a8a8a', color='#1f4e79')
    for (a, b, lab) in ((0.001, 0.003, 'Chapter 6, variants A and B'),
                        (0.004, 0.008, 'Chapter 6, variants C and D')):
        ax.axhspan(a, b, color='#f2e6c8', alpha=.9, zorder=0)
        ax.text(len(f) - .5, (a + b) / 2, lab, fontsize=7, va='center', ha='right')
    ax.axhspan(0.012, 0.030, color='#dbe7f3', alpha=.9, zorder=0)
    ax.text(len(f) - .5, 0.019, 'Chapter 5 instance', fontsize=7, va='center', ha='right')
    ax.axhline(0.028, color='#b4451f', lw=1.0, ls='--')
    ax.text(0.2, 0.0295, 'Chapter 4 rate at 3.5 bar', fontsize=7, color='#b4451f')
    ax.set_yscale('log')
    ax.set_ylim(8e-5, 4e-2)
    ax.set_xlabel('refilling cycle, slowest to fastest', fontsize=8)
    ax.set_ylabel(r'fitted leak rate $\hat\alpha$ (bar/day)', fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description='Chapter 2 data statistics and figures')
    ap.add_argument('--data', required=True, help='directory with the RTE extracts')
    ap.add_argument('--fits', required=True, help='directory with cycle_fit_*.csv')
    ap.add_argument('--out', default='results/industrial')
    ap.add_argument('--figures', default='results/figures')
    a = ap.parse_args()
    data, fits = Path(a.data), Path(a.fits)
    out, figs = Path(a.out), Path(a.figures)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    summary = {
        'register': register(data / 'Données patrimoniales.xlsm'),
        'monitoring': monitoring(data / 'reduced length monitoring data.xlsx'),
        'refill_history': history(data / 'operations_CR_2005_2024.xlsx'),
        'cycle_fits': cycles(fits),
    }
    (out / 'industrial_data_summary.json').write_text(json.dumps(summary, indent=2))
    print('wrote', out / 'industrial_data_summary.json')

    fig_monitoring(fits, figs / 'fig_data_monitoring.pdf')
    fig_quality(data / 'reduced length monitoring data.xlsx', figs / 'fig_data_quality.pdf')
    fig_fleet(data, figs / 'fig_data_fleet.pdf')
    fig_slopes(fits, figs / 'fig_data_slopes.pdf')
    print('wrote four figures to', figs)


if __name__ == '__main__':
    main()
