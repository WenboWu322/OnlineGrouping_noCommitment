"""Frozen parameter set for the online predictive-grouping study.

Every number used by a run lives here, so a saved ``config.json`` fully
reproduces an experiment.  Symbols follow the thesis notation of
Chapter 5 (``tab:notation``): ``p_nom`` = p_bar, ``p_min`` = p_under,
``p_ref`` = p^ref, ``delta_adv`` = Delta^adv, ``delta_safe`` = Delta^safe,
``L_win``, ``K_clus``, ``c_set``, ``eta_minus``, ``eta_plus``.

Chapter 5 quantities are kept unchanged so that the Chapter 6 model reduces
to the Chapter 5 model when the measurement uncertainty is removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import math

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class ScenarioConfig:
    """Physical / operational parameters shared with Chapters 4 and 5."""

    # --- time and fleet -------------------------------------------------
    T: int = 180                       # planning horizon, days (Ch.5 sensitivity setting)
    n_pmec: int = 30                   # number of PMEC compartments |I|

    # --- pressure levels (bar), identical to Chapter 4 tab:parameters ---
    p_nom: float = 3.5                 # p_bar   restored by a GR intervention
    p_ref: float = 3.2                 # p^ref   refilling alarm threshold
    p_min: float = 3.0                 # p_under minimum admissible pressure

    # --- task-window and grouping parameters (Ch.5 tab:grouping-experiment) --
    L_win: int = 10                    # window length, days
    delta_adv: int = 3                 # maximum permitted advancement, days
    delta_safe: int = 2                # safety margin before p_min, days
    delta_delay: int = 5               # maximum lag of a routine GR behind the alarm
    K_clus: int = 10                   # cluster capacity, tasks per day
    c_set: float = 100.0               # setup cost per activated cluster
    eta_minus: float = 5.0             # penalty per day of advancement
    eta_plus: float = 10.0             # penalty per day of delay
    lead_days: int = 1                 # minimum notice between decision and execution

    # --- degradation (bar/day) ------------------------------------------
    alpha_low: float = 0.001           # low-density scenario range
    alpha_high: float = 0.003
    p0_low: float = 3.21               # initial pressure range, bar
    p0_high: float = 3.50
    aperiodic_pct: float = 0.05        # Ch.5 aperiodic perturbation of the rate

    # --- risk level of the predictive window ---------------------------
    epsilon: float = 0.05              # tolerance of the chance constraint
    horizon_forecast: int = 60         # days ahead searched for crossing / deadline

    # --- periodicity / temporal concentration (Ch.5 scenario axes) -----
    periodic: bool = True
    clustered: bool = False
    n_clusters: int = 3                # temporal clusters in a clustered instance
    cluster_jitter: float = 3.0        # sd of the date scatter inside a cluster, days

    def kappa(self) -> float:
        """Standard-normal quantile z_{1-epsilon} used by the window bound."""
        return float(norm.ppf(1.0 - self.epsilon))

    def alpha_range(self) -> tuple[float, float]:
        return (self.alpha_low, self.alpha_high)


@dataclass(frozen=True)
class PolicyConfig:
    """Measurement, filtering and replanning parameters (Chapter 6 only)."""

    # --- observation channel -------------------------------------------
    # Values marked (RTE) are measured on the cleaned daily monitoring extract
    # (Uncertainty_20261012): sensor resolution 0.01 bar, per-cycle linear-fit
    # residual sigma 0.007-0.011 bar, alpha_hat 0.0024-0.0031 bar/day.
    sigma_meas: float = 0.008          # (RTE) daily measurement noise, bar
    quantisation: float = 0.01         # (RTE) reported density resolution, bar
    p_missing: float = 0.05            # probability a daily record is absent
    missing_run_mean: float = 3.0      # mean length of an outage spell, days
    shock_rate: float = 0.0            # expected alpha jumps per PMEC-day
    shock_alpha_mult: float = 3.0      # leak-rate multiplier applied by a shock
    outlier_rate: float = 0.0          # spurious reading probability
    outlier_sigma: float = 0.10        # spurious-reading scale, bar

    # --- Kalman filter (local linear trend, state = [pressure, slope]) --
    q_p: float = 1e-8                  # process noise on pressure
    q_alpha: float = 1e-9              # process noise on the leak rate: the
                                       # truth keeps a constant rate per cycle
    alpha_prior_mean: float = 0.003    # (RTE) prior mean of the leak rate
    alpha_prior_sd: float = 0.003      # prior sd, broad: one PMEC is unknown
    alpha_min: float = 0.0             # the filter never assumes re-pressurisation
    alpha_max: float = 0.05            # numerical guard on the leak rate
    gate_z: float = 3.0                # innovation gate; 0 disables the robust step
    outlier_streak: int = 2            # consecutive out-of-band readings that
                                       # make the model adapt instead of reject
    jump_z: float = 6.0                # innovation above which a cycle reset is tried
    jump_bar: float = 0.10             # (RTE) GR refills raise pressure by ~0.3 bar
    reset_sd: float = 0.01             # uncertainty of the restored level, bar
    cooldown_days: int = 3             # no new task this soon after our own refill

    # --- rolling replanning --------------------------------------------
    replan_max_interval: int = 7       # force a re-solve at most every N days
    replan_drift_days: int = 3         # re-solve when a reference date drifts by N days
    lock_days: int = 2                 # L^lock: announced dates frozen within N days
    eta_stab: float = 2.0              # anchoring penalty per announced day moved
    emergency_extra: float = 200.0     # extra cost of an intervention with < lead notice
    risk_margin_days: float = 0.0      # re-solve when deadline margin falls below this

    # --- solver ---------------------------------------------------------
    solver: str = 'auto'               # auto | gurobi | scipy
    time_limit: float = 60.0
    mip_gap: float = 1e-4


@dataclass(frozen=True)
class Config:
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    seed: int = 1
    days: int | None = None            # simulation length; defaults to scenario.T

    def __post_init__(self):
        self.validate()

    # ------------------------------------------------------------------
    def validate(self) -> None:
        s, p = self.scenario, self.policy
        for name in ('T', 'n_pmec', 'L_win', 'delta_adv', 'delta_safe', 'K_clus',
                     'lead_days', 'horizon_forecast'):
            v = getattr(s, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f'{name} must be an integer.')
        if s.T < 2 or s.n_pmec < 1 or s.L_win < 1 or s.K_clus < 1:
            raise ValueError('T, n_pmec, L_win and K_clus must be positive.')
        if not 0 < s.epsilon < 0.5:
            raise ValueError('epsilon must lie in (0, 0.5).')
        if not s.p_min < s.p_ref < s.p_nom:
            raise ValueError('Require p_min < p_ref < p_nom.')
        if s.delta_delay < 0:
            raise ValueError('delta_delay must be non-negative.')
        if s.lead_days < 0 or s.delta_adv < 0 or s.delta_safe < 0:
            raise ValueError('lead_days, delta_adv and delta_safe must be non-negative.')
        if s.L_win + s.lead_days > s.T:
            raise ValueError('Window longer than the planning horizon.')
        if min(s.c_set, s.eta_minus, s.eta_plus) < 0:
            raise ValueError('Cost coefficients must be non-negative.')
        if not s.alpha_low > 0 or s.alpha_low > s.alpha_high:
            raise ValueError('Invalid degradation-rate range.')
        if not s.p0_low >= s.p_min or s.p0_high > s.p_nom:
            raise ValueError('Initial pressure range must lie within (p_min, p_nom].')
        if p.sigma_meas <= 0 or p.quantisation < 0:
            raise ValueError('Measurement noise must be positive.')
        if not 0 <= p.p_missing < 1 or p.missing_run_mean < 1:
            raise ValueError('Invalid missing-data parameters.')
        if p.shock_rate < 0 or p.shock_alpha_mult < 1:
            raise ValueError('Invalid shock parameters.')
        if 0 < p.outlier_rate < 1 and p.outlier_sigma > 0:
            pass
        elif p.outlier_rate and p.outlier_sigma <= 0:
            raise ValueError('Invalid outlier parameters.')
        if min(p.q_p, p.q_alpha, p.alpha_prior_sd) <= 0:
            raise ValueError('Filter variances must be positive.')
        if p.alpha_min < 0 or p.alpha_max <= p.alpha_min:
            raise ValueError('Invalid leak-rate guard.')
        if p.replan_max_interval < 1 or p.lock_days < 0:
            raise ValueError('Invalid replanning parameters.')
        if p.lock_days > s.L_win:
            raise ValueError('lock_days must not exceed L_win.')
        if p.solver not in {'auto', 'gurobi', 'scipy'}:
            raise ValueError("solver must be 'auto', 'gurobi' or 'scipy'.")
        if p.time_limit <= 0 or p.mip_gap < 0:
            raise ValueError('Invalid solver settings.')
        if self.days is not None and self.days < 2:
            raise ValueError('days must be at least 2.')

    # ------------------------------------------------------------------
    @property
    def n_days(self) -> int:
        return self.days if self.days is not None else self.scenario.T

    def to_dict(self) -> dict:
        return {'scenario': asdict(self.scenario), 'policy': asdict(self.policy),
                'seed': self.seed, 'days': self.days}

    def write(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'config.json').write_text(json.dumps(self.to_dict(), indent=2),
                                               encoding='utf-8')

    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        return cls(scenario=ScenarioConfig(**data.get('scenario', {})),
                   policy=PolicyConfig(**data.get('policy', {})),
                   seed=int(data.get('seed', 1)),
                   days=data.get('days'))

    @classmethod
    def from_json(cls, path: Path) -> 'Config':
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    # ------------------------------------------------------------------
    def scenario_variants(self) -> dict[str, 'Config']:
        """Named ablations used by the experiment matrix (see README)."""
        s, p = self.scenario, self.policy
        variants: dict[str, Config] = {}

        # Exact Chapter 5 setting: no measurement noise, no missing data, no
        # shocks, deterministic window, daily re-solve without commitments.
        exact5 = Config(
            scenario=replace(s, epsilon=0.5),
            policy=replace(p, sigma_meas=1e-9, quantisation=0.0, p_missing=0.0,
                           shock_rate=0.0, outlier_rate=0.0, q_p=1e-12,
                           q_alpha=1e-12, alpha_prior_sd=1e-9, gate_z=0.0,
                           jump_z=0.0, lock_days=0, eta_stab=0.0,
                           replan_max_interval=1, replan_drift_days=10**6),
            seed=self.seed, days=self.days)
        variants['ch5_exact'] = exact5

        for eps in (0.01, 0.05, 0.10, 0.25):
            variants[f'eps{eps:g}'] = Config(replace(s, epsilon=eps), p,
                                             self.seed, self.days)
        for miss in (0.0, 0.05, 0.15, 0.30):
            variants[f'miss{miss:g}'] = Config(s, replace(p, p_missing=miss),
                                               self.seed, self.days)
        for sd in (0.002, 0.008, 0.020):
            variants[f'noise{sd:g}'] = Config(s, replace(p, sigma_meas=sd),
                                              self.seed, self.days)
        for shock in (0.0, 0.002, 0.010, 0.030):
            variants[f'shock{shock:g}'] = Config(s, replace(p, shock_rate=shock),
                                                 self.seed, self.days)
        for n in (10, 30, 100, 150):
            variants[f'size{n}'] = Config(replace(s, n_pmec=n), p, self.seed, self.days)
        return variants


def replace(obj, **changes):
    """dataclasses.replace without importing the helper at module level."""
    from dataclasses import replace as _replace
    return _replace(obj, **changes)


def gaussian_quantile(epsilon: float) -> float:
    return float(norm.ppf(1.0 - epsilon))


def is_finite(value: float) -> bool:
    return bool(math.isfinite(value)) and bool(np.isfinite(value))
