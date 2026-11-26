# Chapter 3 (Industrial Background): fleet-record statistics

Thesis Chapter 3 quotes statistics of three RTE extracts. Every quoted number is
produced by `online_grouping/industrial_data.py`; nothing in the chapter was typed
by hand. The chapter does not use these records as model data — Chapters 4, 5 and
6 run on synthetic instances — it uses them to justify the parameter ranges and to
argue that uncertainty is a property of the records, not a modelling choice.

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
| `results/industrial/industrial_data_summary.json` | every statistic quoted in the chapter |
| `results/figures/fig_data_monitoring.pdf` | two example pressure series (chapter Figure 3.2) |
| `results/figures/fig_data_fleet.pdf` | nominal-pressure classes and volume classes (Figure 3.3) |
| `results/figures/fig_data_slopes.pdf` | 22 fitted leak rates against the modelled bands (Figure 3.4) |
| `results/figures/fig_data_quality.pdf` | reporting gaps and within-hour scatter (Figure 3.5) |

`sync_thesis_assets.sh` copies the four PDFs to `Thesis/figures/` and the JSON to
`Thesis/tables/industrial_data_summary.json`.

## Numbers the chapter rests on

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

It compares the JSON against the strings in the chapter, recomputes the derived
claims (0.0043 bar/day × 30 days × κ = 0.76 kg, etc.), and fails if any raw site,
group, or register identifier appears in the chapter. Exit code 0 means the
chapter is consistent with the data as processed.

## Anonymity

The extracts carry operator identifiers (group names such as the two sites and the
17 electrical groups, 10-digit register ids). None of them appear in the thesis,
in the figures, or in the committed JSON: aggregates only, four example
compartments labelled A–H or "first/second", and no raw record dumps.
