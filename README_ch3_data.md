# Chapter 3 (Industrial Background): fleet-record statistics

Thesis Chapter 3 describes three RTE extracts and the derived refilling-cycle
fits. The chapter quotes only the handful of magnitudes it argues with; the full
statistics stay in this repository, produced by
`online_grouping/industrial_data.py`, so that every quoted value and every
qualitative claim traces back to a computed artifact. The chapter does not use
these records as model data — Chapters 4, 5 and 6 run on synthetic instances — it
uses them to justify the parameter ranges, and to argue that leak-rate estimation
under sparse, noisy records is where the uncertainty chapter comes from.

## Running it

```bash
python -m online_grouping.industrial_data \
    --data ../data \
    --fits ../RTE_SchedulingGrouping/Uncertainty_20261012/cycle_fit_results \
    --out results/industrial \
    --figures results/figures
```

Inputs (not committed here, they stay on the local machine):

| File | Content |
|---|---|
| `data/Données patrimoniales.xlsm`, sheet `Patrimoine CSEM` | equipment register |
| `data/operations_CR_2005_2024.xlsx` | compartments with refilling history |
| `data/reduced length monitoring data.xlsx` | monitoring transmissions |
| `.../cycle_fit_results/cycle_fit_summary.csv` | refilling-cycle trend fits |

## Artifacts

| Path | Content |
|---|---|
| `results/industrial/industrial_data_summary.json` | every statistic of the three extracts |
| `results/figures/fig_data_monitoring.pdf` | measured pressure of two instrumented PMECs, titled PMEC 1 / PMEC 2 (thesis Figure 3.2) |
| `results/figures/fig_data_slopes.pdf` | fitted leak rates per refilling cycle against the modelled bands (Figure 3.3) |
| `results/figures/fig_data_quality.pdf` | distribution of transmission gaps (Figure 3.4) |
| `results/figures/fig_data_fleet.pdf` | register and refilling distributions: diagnostic only, no longer included in the chapter |

The series in `fig_data_monitoring.pdf` are measurements, not simulations: they are
`pressure_obs` (daily means of the transmissions, `n_measurements` transmissions per
day) of two real instrumented PMECs in `pressure_data_with_cycle_fits.csv`. Only the
green dots are model output — the value of the fitted cycle trend at the points used
in the fit.

`sync_thesis_assets.sh` copies the three included PDFs to `Thesis/figures/` and the
JSON to `Thesis/tables/industrial_data_summary.json`.

## What the artifact contains

The chapter prints a few of these; the rest are kept so that its
qualitative wording can be re-checked.

| Quantity | Value |
|---|---|
| Register entries (compartments) | 24,978; 3.5 bar class 13,827 (55%) |
| Register mass-per-bar over volume | 5.99 in all 24,495 rows carrying both fields |
| Monitored compartments / records | 299 / 9,999 over 2022-02-10 → 2022-06-11 |
| Compartments per monitoring group | 5 to 42, median 21 |
| Corr(raw pressure, temperature) | 0.60; after compensation 0.01 |
| Compartments ever below 3.2 / 3.0 bar | 33 / 14 |
| Compartments with history / refillings | 2,903 / 21,515 (2010–2024) |
| Median release per refilling | 7.1% of charge, 0.75 kg |
| Operational criterion reached | 51 compartments |
| Last leak untreated | 1,723 (59%) |
| Fitted leak rates (22 fits) | 0.00003–0.0204 bar/day, median 0.0043 |
| Fitted-cycle RMSE | median 0.0071 bar (0.0048–0.038) |
| Reporting gaps | median 1 h, 464 above 24 h, longest 2,857 h (119 days) |
| Within-hour compensated steps | 204 of 9,700 above 0.05 bar, 22 above 0.2 bar, max 0.71 bar |

## Checking the chapter against the artifact

```bash
python -m online_grouping.check_chapter3_numbers
```

Three things are checked: every statistic of the pipeline is pinned against a known
value (so a change in the extracts or the code is caught); the strings the chapter
does print are matched against the artifact; and the qualitative claims it makes
("about half the register", "two orders of magnitude", "gaps of days to months") are
recomputed. It also fails if any raw site, group, or register identifier appears in
the chapter. Exit code 0 means the chapter says no more than the data support.

## Anonymity

The extracts carry operator identifiers (group names such as the two sites and the
17 electrical groups, 10-digit register ids). None of them appear in the thesis,
in the figures, or in the committed JSON: aggregates only, four example
compartments labelled A–H or "PMEC 1 / PMEC 2", and no raw record dumps.
