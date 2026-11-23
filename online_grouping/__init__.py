"""Online predictive grouping of SF6 gas-refilling tasks under data uncertainty.

Thesis Chapter 6 (``STOCHASTIC PROBLEM EXPLORATION'').  The grouping model, the
task-window structure and the evaluation indicators are those of Chapter 5
(``TASK GROUPING PROBLEM''); this package adds the measurement, prediction and
rolling-replanning layers that turn that deterministic grouping problem into an
online one.

Modules
-------
config      frozen parameter set (single source of truth for every run)
process     predictive pressure moments, threshold crossing day, risk-adjusted
            deadline and admissible day set (Chapter 5 limit preserved)
filter      local-linear-trend Kalman filter with robust update and event flags
scenarios   synthetic truth process, observation channel and shock injection
grouping    window-level grouping MILP (M1 / M2 + commitment/stability)
controller  daily information loop, re-solve triggers, execution bookkeeping
policies    strategy definitions compared in the chapter
metrics     Chapter 5 indicators plus safety, stability, calibration and regret
experiment  scenario x seed x policy matrix with paired statistics
figures     vector figures for the thesis
"""

from .config import Config, ScenarioConfig, PolicyConfig
from .process import (Prediction, propagate, noise_covariance,
                      predictive_arrays, crossing_lead, risk_deadline_lead,
                      deterministic_deadline_lead, admissible_days)

__all__ = [
    'Config', 'ScenarioConfig', 'PolicyConfig',
    'predict', 'propagate', 'crossing_lead', 'risk_deadline_lead',
    'deterministic_deadline_lead', 'admissible_days',
]
