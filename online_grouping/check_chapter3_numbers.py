"""Assert that Chapter 3 only claims what the data artifacts support.

The statistics of the three RTE extracts live in
``results/industrial/industrial_data_summary.json`` (written by
``industrial_data.py``).  The chapter quotes only a few of them, so the check has
two halves.  First, every statistic the pipeline produces is pinned against a
known value, so a change in the extracts or in the code is caught.  Second, the
qualitative claims the chapter makes ("about half the register", "two orders of
magnitude", "gaps of days to months") are recomputed from the artifacts, and the
few numbers the chapter does print are matched against them.  Run this after
re-running the statistics, before trusting the chapter.

Run::

    python -m online_grouping.check_chapter3_numbers \
        --json results/industrial/industrial_data_summary.json \
        --tex ../thesis-overleaf/Thesis/chapters/"INDUSTRIAL AND TECHNICAL BACKGROUND.tex"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json', default='results/industrial/industrial_data_summary.json')
    ap.add_argument('--tex', default='../thesis-overleaf/Thesis/chapters/'
                                     'INDUSTRIAL AND TECHNICAL BACKGROUND.tex')
    a = ap.parse_args()
    j = json.loads(Path(a.json).read_text())
    tex = Path(a.tex).read_text()

    reg = j['register']['Patrimoine CSEM']
    sub = reg['nominal_3p5_bar']
    mon = j['monitoring']
    hist = j['refill_history']
    cyc = j['cycle_fits']

    # (label, value derived from the artifact, tolerance) -> compared with the text
    numeric = [
        ('register entries', reg['rows'], 24978),
        ('register 3.5 bar class', reg['nominal_pressure_counts']['3.5'], 13827),
        ('register 6.3 bar class', reg['nominal_pressure_counts']['6.3'], 5055),
        ('register 6.5 bar class', reg['nominal_pressure_counts']['6.5'], 1813),
        ('register 5.5 bar class', reg['nominal_pressure_counts']['5.5'], 1685),
        ('register 7.0 bar class', reg['nominal_pressure_counts']['7.0'], 927),
        ('register 3.3 bar class', reg['nominal_pressure_counts']['3.3'], 736),
        ('register missing nominal pressure', reg['nominal_pressure_missing'], 59),
        ('rows with charge and volume', reg['kappa_ratio_over_volume']['rows'], 24495),
        ('register mass/volume ratio', reg['kappa_ratio_over_volume']['median'], 5.99),
        ('3.5 bar volume median', sub['volume']['median'], 0.6439),
        ('3.5 bar volume p90', sub['volume']['p90'], 4.0732),
        ('3.5 bar volume max', sub['volume']['max'], 17.9574),
        ('3.5 bar charge median', sub['gas_mass']['median'], 13.5),
        ('3.5 bar charge max', sub['gas_mass']['max'], 376.5),
        ('3.5 bar rows without usable volume', sub['volume']['missing'] + sub['volume_zero'], 242),
        ('register missing charge or volume', reg['gas_mass']['missing'], 285),
        ('register zero volume', reg['volume_zero'], 198),
        ('monitoring records', mon['records'], 9999),
        ('monitoring compartments', mon['compartments'], 299),
        ('monitoring groups', mon['monitoring_groups'], 17),
        ('monitoring sites', mon['sites'], 2),
        ('feeder-bay records', mon['group_types']['D\u00e9part'], 7256),
        ('line-section records', mon['group_types']['tron\u00e7on'], 1509),
        ('busbar records', mon['group_types']['Barre'], 1234),
        ('compartments per group min', mon['compartments_per_group']['min'], 5),
        ('compartments per group median', mon['compartments_per_group']['median'], 21),
        ('compartments per group max', mon['compartments_per_group']['max'], 42),
        ('compensated pressure min', mon['compensated_pressure']['min'], 2.71),
        ('compensated pressure median', mon['compensated_pressure']['median'], 3.44),
        ('compensated pressure max', mon['compensated_pressure']['max'], 3.73),
        ('raw pressure min', mon['raw_pressure']['min'], 3.51),
        ('raw pressure max', mon['raw_pressure']['max'], 4.84),
        ('temperature min', mon['temperature']['min'], 2.89),
        ('temperature max', mon['temperature']['max'], 32.97),
        ('corr raw pressure temperature', mon['corr_raw_pressure_temperature'], 0.595),
        ('corr compensated temperature', mon['corr_compensated_temperature'], 0.007),
        ('compartments below 3.2', mon['compartments_ever_below_3p2'], 33),
        ('compartments below 3.0', mon['compartments_ever_below_3p0'], 14),
        ('records per compartment median', mon['records_per_compartment']['median'], 38),
        ('coverage of hourly sampling pct', mon['coverage_of_hourly_sampling_pct'], 1.31),
        ('gap p95 hours', mon['gap_hours']['p95'], 6),
        ('gap max hours', mon['gap_hours']['max'], 2857),
        ('gaps over 24 h', mon['gap_hours']['n_over_24h'], 464),
        ('gaps over 7 d', mon['gap_hours']['n_over_7d'], 464),
        ('consecutive pairs', mon['consecutive_pairs'], 9700),
        ('abs dp median', mon['abs_dp_bar']['median'], 0.01),
        ('abs dp max', mon['abs_dp_bar']['max'], 0.71),
        ('jumps over 0.05', mon['abs_dp_bar']['n_over_0p05'], 204),
        ('jumps over 0.20', mon['abs_dp_bar']['n_over_0p20'], 22),
        ('per-compartment std median', mon['per_compartment_std_bar']['median'], 0.0246),
        ('per-compartment std p90', mon['per_compartment_std_bar']['p90'], 0.0614),
        ('per-compartment std max', mon['per_compartment_std_bar']['max'], 0.3476),
        ('within-compartment range max', mon['within_compartment_range_bar']['max'], 0.72),
        ('history rows', hist['rows'], 2903),
        ('history distinct', hist['distinct_equipment'], 2842),
        ('history duplicate rows', hist['duplicate_equipment_rows'], 61),
        ('history 3.5 bar', hist['nominal_pressure_counts']['3.5'], 2416),
        ('history missing charge', hist['gas_mass']['missing'], 125),
        ('history zero charge', hist['gas_mass_zero'], 2),
        ('history missing nominal', hist['nominal_pressure_missing'], 130),
        ('history nominal outside band', hist['nominal_pressure_outside_2p9_3p6'], 110),
        ('charge median', hist['gas_mass']['median'], 12.0),
        ('charge max', hist['gas_mass']['max'], 325.0),
        ('share per refilling p05', hist['pct_mass_per_refill']['p05'], 5.8),
        ('share per refilling median', hist['pct_mass_per_refill']['median'], 7.1),
        ('share per refilling max', hist['pct_mass_per_refill']['max'], 9.0),
        ('mass per refilling median', hist['mass_per_refill_kg']['median'], 0.75),
        ('mass per refilling p90', hist['mass_per_refill_kg']['p90'], 5.5025),
        ('mass per refilling max', hist['mass_per_refill_kg']['max'], 23.075),
        ('refillings total', hist['refills_total'], 21515),
        ('refillings 2016', hist['refills_by_year']['2016'], 1805),
        ('refillings 2024', hist['refills_by_year']['2024'], 818),
        ('refillings per compartment median', hist['refills_per_equipment']['median'], 3),
        ('refillings per compartment p90', hist['refills_per_equipment']['p90'], 19),
        ('refillings per compartment max', hist['refills_per_equipment']['max'], 127),
        ('share at least 3 refillings', hist['refills_per_equipment']['share_ge_3'], 0.534),
        ('share at least 5 refillings', hist['refills_per_equipment']['share_ge_5'], 0.395),
        ('criterion flagged', hist['operational_criterion_hits']['flagged_equipment'], 51),
        ('criterion flagged twice', hist['operational_criterion_hits']['flagged_twice_or_more'], 24),
        ('criterion max', hist['operational_criterion_hits']['max'], 13),
        ('criterion column populated', hist['operational_criterion_hits']['non_null'], 51),
        ('rolling-year mass p90', hist['rolling_year_rejected_mass_kg']['p90'], 0.9902),
        ('rolling-year mass max', hist['rolling_year_rejected_mass_kg']['max'], 47.2221),
        ('work orders recorded', hist['rows'] - hist['last_work_order_nature'].get('nan', 0), 1220),
        ('accessory replacements', hist['last_work_order_nature']['SF6ACCES'], 393),
        ('manual repairs', hist['last_work_order_nature']['SF6MAN'], 349),
        ('temporary seals', hist['last_work_order_nature']['SF6COTMP'], 222),
        ('upgrades', hist['last_work_order_nature']['SF6MANT'], 81),
        ('untreated last leak', hist['last_leak_treated']['NON'], 1723),
        ('commissioning median age', hist['age_years_2024']['median'], 39.0),
        ('commissioning unparsed dates', hist['commissioning_dates']['unparsed'], 5),
        ('cycle extract rows', cyc['extract_rows'], 1685),
        ('cycle extract compartments', cyc['extract_compartments'], 8),
        ('cycle extract calendar days', cyc['extract_calendar_days'], 690),
        ('refill days in extract', cyc['refill_days_in_extract'], 28),
        ('measurements per day max', cyc['raw_measurements_per_day']['max'], 25),
        ('measurements per day median', cyc['raw_measurements_per_day']['median'], 24),
        ('cycles', cyc['cycles'], 30),
        ('cycles without usable interval', cyc['fit_points']['without_points'], 6),
        ('fits with an estimated rate', cyc['fits_with_supported_trend'], 22),
        ('fits without a trend', cyc['unsupported_fits'], 1),
        ('alpha min', cyc['alpha_bar_per_day']['min'], 0.00003),
        ('alpha p25', cyc['alpha_bar_per_day']['p25'], 0.00222),
        ('alpha median', cyc['alpha_bar_per_day']['median'], 0.00425),
        ('alpha p75', cyc['alpha_bar_per_day']['p75'], 0.01021),
        ('alpha max', cyc['alpha_bar_per_day']['max'], 0.02038),
        ('r2 median', cyc['r_squared']['median'], 0.9928),
        ('r2 min', cyc['r_squared']['min'], 0.0005),
        ('ci width median', cyc['alpha_ci_width_bar_per_day']['median'], 0.00049),
        ('ci width max', cyc['alpha_ci_width_bar_per_day']['max'], 0.00758),
        ('residual rmse min', cyc['residual_rmse_bar']['min'], 0.00479),
        ('residual rmse median', cyc['residual_rmse_bar']['median'], 0.00714),
        ('residual rmse max', cyc['residual_rmse_bar']['max'], 0.03796),
        ('fit span median', cyc['fit_span_days']['median'], 31.5),
        ('fit span max', cyc['fit_span_days']['max'], 371),
        ('observed days min', cyc['extract_days_per_compartment']['min'], 37),
        ('observed days median', cyc['extract_days_per_compartment']['median'], 137),
        ('observed days max', cyc['extract_days_per_compartment']['max'], 536),
        ('missing days max', cyc['extract_missing_days_per_compartment']['max'], 46),
        ('compartments with several fits', cyc['compartments_with_several_fits'], 4),
        ('within-compartment factor two', cyc['within_compartment_factor_two'], 3),
    ]

    bad = 0
    for label, got, want in numeric:
        if abs(float(got) - float(want)) > 1e-9:
            print('ARTIFACT MISMATCH  %-45s artifact=%s  text=%s' % (label, got, want))
            bad += 1

    # the chapter now quotes only the magnitudes it argues with; each of those
    # strings has to be present, in the form used in the text
    strings = [
        # Table~\ref{tab:data-sources}: what the extracts cover
        '24{,}978 compartments', '2{,}903 compartments', '9{,}999 records, 299',
        '30 cycles, 8 compartments',
        # register
        '55\\%', '2{,}416 of the 2{,}903',
        '0.64~m\\textsuperscript{3} to 17.96~m\\textsuperscript{3}',
        '5.99~kg of SF$_6$', '$\\kappa=5.993$~kg/bar',
        # monitoring
        '299 instrumented \\ac{PMEC}s', '17 electrical groups',
        # refilling history
        'more than half of the file answers no',
        # leak rates
        '$10^{-3}$ to\n$10^{-2}$~bar/day', 'nearer $10^{-4}$~bar/day',
        'about 1\\% of the charge per year at 3.5~bar',
        # data quality
        'a fifth of a nominal charge', 'gaps of days to\nmonths',
        'scatters by a few hundredths of a bar',
        # summary
        '25{,}000 compartments', 'about 7\\% of the charge gone',
        # parameter tables: the values the modeling chapters use
        '$\\kappa = 5.993$~kg/bar per m\\textsuperscript{3}',
        '0.012--0.030~bar/day', '0.001--0.003 and 0.004--0.008~bar/day',
        '$\\sigma_y = 0.008$~bar', '2.5~kg per compartment per 30 days',
        '5~kg per rolling year',
    ]
    missing = [s for s in strings if s not in tex]
    for s in missing:
        print('NOT IN TEXT  %r' % s)
    if missing:
        bad += len(missing)

    # qualitative claims the slimmed chapter makes have to follow from the data
    alpha = cyc['alpha_bar_per_day']
    derived = [
        ('about half the register is 3.5 bar class', 0.50 <= 13827 / 24978 <= 0.60),
        ('3.5 bar class dominates the refilling history',
         hist['nominal_pressure_counts']['3.5'] / hist['rows'] > 0.8),
        ('the refilling history is mostly old equipment',
         30 <= hist['age_years_2024']['median'] <= 45),
        ('refilling comes back often on part of the population',
         hist['refills_per_equipment']['p90'] >= 10),
        ('each refilling puts back a few percent of the charge',
         0.05 <= hist['pct_mass_per_refill']['median'] / 100 <= 0.10),
        ('most rows have no repair behind them',
         (hist['rows'] - hist['last_work_order_nature'].get('nan', 0)) < hist['rows'] / 2
         and hist['last_leak_treated']['NON'] > hist['rows'] / 2),
        ('leak rates span about two orders of magnitude',
         100 <= alpha['max'] / alpha['min'] <= 1000),
        ('fitted rates are of the order 1e-3 to 1e-2 bar/day',
         1e-3 <= alpha['median'] <= 1e-2 and 1e-2 <= alpha['max'] <= 3e-2),
        ('the fleet average of 1%/yr is nearer 1e-4 bar/day at 3.5 bar',
         8e-5 <= 0.01 * 3.5 / 365 <= 1.1e-4),
        ('the instrumented population leaks far faster than that average',
         alpha['median'] / (0.01 * 3.5 / 365) > 20),
        ('the alarm drop at 3.2 bar is about 7-9% of the charge',
         0.90 <= 3.2 / 3.5 <= 0.93),
        ('sensor gaps run to months',
         30 <= mon['gap_hours']['max'] / 24 <= 180),
        ('per-compartment scatter is a few hundredths of a bar',
         0.01 <= mon['per_compartment_std_bar']['median'] <= 0.05),
        ('consecutive readings can disagree by several tenths of a bar',
         0.2 <= mon['abs_dp_bar']['max'] <= 1.0),
        ('a few cycles give no usable trend',
         cyc['fit_points']['without_points'] + cyc['unsupported_fits'] >= 5),
        ('a single rate interval is wide against the slow/fast split',
         cyc['alpha_ci_width_bar_per_day']['max'] > 0.004 - 0.001),
        ('a few percent of register rows cannot be turned into a mass',
         0.005 <= (sub['volume']['missing'] + sub['volume_zero']) / reg['nominal_pressure_counts']['3.5'] <= 0.05),
    ]
    for label, ok in derived:
        if not ok:
            print('DERIVED CHECK FAILED  %s' % label)
            bad += 1

    # no site or group names, no raw identifiers
    for leak in re.findall(r'PENL5[\w-]*|COULA[\w-]*|\b50008\d{5}\b|\b50013\d{5}\b', tex):
        print('ANONYMISATION LEAK  %s' % leak)
        bad += 1

    print('%d checks failed' % bad)
    raise SystemExit(1 if bad else 0)


if __name__ == '__main__':
    main()
