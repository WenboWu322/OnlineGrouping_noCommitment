# Chapter 6 — online predictive grouping under data uncertainty

This directory contains the simulation package behind Chapter 6
(`STOCHASTIC PROBLEM EXPLORATION`) of the thesis. Every number, table and
figure in that chapter is produced here; nothing is entered by hand in the
LaTeX source.

## Layout

| path | role |
| --- | --- |
| `online_grouping/config.py` | scenario, policy and configuration dataclasses; the single place where parameters live |
| `online_grouping/process.py` | local-linear-trend propagation, closed-form $h$-step mean/variance, risk-adjusted crossing times |
| `online_grouping/filter.py` | recursive filter, robust event classification (reset / shock / outlier), covariance inflation |
| `online_grouping/scenarios.py` | world generator, seeded shocks, missing-data and outlier channels, realised (truth) plan |
| `online_grouping/grouping.py` | window grouping MILP of Chapter 5 (Models 1 and 2), commitments and locks, greedy fallback |
| `online_grouping/controller.py` | rolling loop: observe, filter, create tasks, trigger, solve, commit, execute |
| `online_grouping/policies.py` | the compared strategies S1–S7, O1, O2 |
| `online_grouping/metrics.py` | cost and safety metrics, calibration (PICP), paired Wilcoxon statistics, regret |
| `online_grouping/experiment.py` | experiment driver, writes `rows.csv`, `summary.csv`, `summary.json` |
| `online_grouping/figures.py` | vector PDF figures used by the chapter |
| `online_grouping/tex_table.py` | LaTeX tables and the list of quoted prose values |
| `run_experiments.sh` | the whole suite: main comparison plus 21 sensitivity sweeps |
| `sync_thesis_assets.sh` | copies generated figures and tables into the thesis repository |

## Regenerating everything

```bash
./run_experiments.sh > results/suite.log 2>&1      # ~45 min, 8 jobs
python3 -m online_grouping.figures --results results --out results/figures
python3 -m online_grouping.tex_table --results results/main --out results/tables
./sync_thesis_assets.sh                            # THESIS=../thesis-overleaf
```

Individual commands:

```bash
# main comparison, 30 seeds
python3 -m online_grouping.experiment --seeds 30 --jobs 8 --out results/main

# one sensitivity axis
python3 -m online_grouping.experiment --variants D_high_cluster \
        --set scenario.delta_delay=10 --seeds 20 --out results/sweep_delay10_D

# a single run, with the trace used by the timeline figure
python3 -m online_grouping.experiment --dump-trace --variant D_high_cluster \
        --seed 3 --policy S5_proposed --out results/trace_run
```

Any dataclass field can be overridden with `--set section.name=value`, e.g.
`--set policy.sigma_meas=0.020`, `--set scenario.epsilon=0.10`,
`--set scenario.K_clus=3`, `--set scenario.cluster_jitter=8.0`.

## Solver

Window sub-problems are the Chapter 5 MILPs. Gurobi is used while the model
fits the licence size limit; larger windows fall back to the branch-and-bound
of `scipy.optimize.milp` under the same gap tolerance and time limit, and a
greedy assignment is used if neither finds a feasible plan. The solver used
per window is recorded in the run's solver log.

## What the chapter quotes

* `results/main/summary.csv` — medians per variant and policy, and the paired
  Wilcoxon statistics against S3.
* `results/tables/tab_ch6_main_combined.tex` — Table `tab:ch6-results`.
* `results/tables/tab_ch6_paired_combined.tex` — Table `tab:ch6-paired`.
* `results/tables/tab_ch6_calibration_combined.tex` — Table `tab:ch6-calibration`.
* `results/tables/tab_ch6_sensitivity.tex` — Table `tab:ch6-sensitivity`.
* `results/tables/prose_values.txt` — every number quoted in running text.
* `results/figures/fig_ch6_policies.pdf`, `fig_ch6_calibration.pdf`,
  `fig_ch6_sensitivity.pdf`, `fig_ch6_timeline.pdf`.

## Notes on the experimental design

* The shift penalty reported in the results is evaluated against the
  *realised* alarm date of the cycle (`metrics.realized_references`), so a
  prediction error shows up in the cost instead of cancelling against the
  estimate that produced it. The objective optimised inside a window still
  uses the predicted reference date.
* Forecast horizons that contain a planned intervention are excluded from
  the calibration metrics: a refill is a structural break of the degradation
  model, not a prediction failure.
* Own refills are fed back to the filter as known process inputs; otherwise
  the filter keeps degrading a compartment that has just been refilled and
  invents duplicate tasks.
* The offline clairvoyant reference O2 (`O2_offline_clair`) separates the
  price of imperfect data from the price of rolling visibility.
