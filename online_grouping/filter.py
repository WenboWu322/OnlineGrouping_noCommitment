"""Local-linear-trend Kalman filter with robust update and event detection.

The filter is the estimator side of the state-space model documented in
``process.py``.  Only information available up to the current day is used:
the observation of day ``d`` is incorporated before the day-``d`` decision is
taken, and the execution outcome is applied at the start of the next day.

Missing records are handled by simply omitting the update, so a measurement
outage degrades the estimate through the accumulated process noise instead of
requiring imputation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

F = np.array([[1.0, -1.0], [0.0, 1.0]])
H = np.array([1.0, 0.0])


@dataclass
class FilterOutput:
    day: int
    observation: float | None
    innovation: float
    innovation_sd: float
    z_score: float
    updated: bool
    event: str | None      # 'reset' | 'shock' | 'outlier' | None


class TrendFilter:
    """Estimates the pressure and the leak rate of one PMEC."""

    def __init__(self, cfg):
        self.cfg = cfg
        p = cfg.policy
        self.q = np.diag([p.q_p, p.q_alpha])
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.last_day = -1
        self.cycle = 0
        self.observed_days = 0
        self.suspect_streak = 0
        self.events: list[tuple[int, str]] = []

    # ------------------------------------------------------------------
    def initialise(self, y0: float) -> None:
        p = self.cfg.policy
        self.mean = np.array([float(y0), p.alpha_prior_mean])
        self.cov = np.diag([max(p.sigma_meas, p.q_p) ** 2, p.alpha_prior_sd ** 2])
        self.observed_days = 1

    def advance_to(self, day: int) -> None:
        """Propagate the estimate to ``day`` without observations."""
        if self.mean is None:
            return
        steps = day - self.last_day
        for _ in range(steps):
            self.mean = F @ self.mean
            self.cov = F @ self.cov @ F.T + self.q
        self.last_day = day

    def reset_cycle(self, day: int, pressure: float | None = None) -> None:
        """Re-initialise after a gas-refilling intervention or a level jump.

        A refilling restores the compartment to the nominal pressure, so the
        post-intervention level is treated as known up to the reporting
        resolution, while the degradation rate of the new cycle is unknown
        again and reverts to the prior.
        """
        p = self.cfg.policy
        p0 = self.cfg.scenario.p_nom if pressure is None else float(pressure)
        self.mean = np.array([p0, p.alpha_prior_mean])
        self.cov = np.diag([p.reset_sd ** 2, p.alpha_prior_sd ** 2])
        self.last_day = day
        self.cycle += 1
        self.observed_days += 1

    # ------------------------------------------------------------------
    def update(self, day: int, y: float | None) -> FilterOutput:
        """Incorporate the observation of ``day`` (``None`` = missing record)."""
        if day <= self.last_day:
            raise ValueError(f'day {day} is not after {self.last_day}')
        if self.mean is None:
            if y is None:
                self.last_day = day
                return FilterOutput(day, None, 0.0, 0.0, 0.0, False, None)
            self.initialise(y)
            self.last_day = day
            return FilterOutput(day, y, 0.0, 0.0, 0.0, True, None)

        self.advance_to(day)
        assert self.mean is not None and self.cov is not None
        if y is None:
            return FilterOutput(day, None, 0.0, 0.0, 0.0, False, None)

        pred = float(H @ self.mean)
        S = float(H @ self.cov @ H) + self.cfg.policy.sigma_meas ** 2
        sd = float(np.sqrt(max(S, 1e-18)))
        innov = float(y) - pred
        z = innov / sd
        event = self._classify(z, innov, sd)

        if event == 'reset':
            # An unexplained step upward is treated as an intervention of the
            # distribution system: the degradation cycle restarts here.
            self.reset_cycle(day, pressure=float(y))
            return FilterOutput(day, y, innov, sd, z, True, event)

        if event in ('shock', 'bias'):
            # Sustained or large out-of-band evidence: the model, not the
            # reading, is at fault.  The state covariance is inflated in
            # proportion to the unexplained innovation so the update can move
            # the level and the slope; this is what keeps the filter from going
            # blind once a rate estimate has drifted.
            scale = min(abs(innov), self.cfg.scenario.p_nom -
                        self.cfg.scenario.p_min)
            self.cov = self.cov + np.diag([
                (0.5 * scale) ** 2, (0.5 * scale / 10.0) ** 2])
            S = float(H @ self.cov @ H) + self.cfg.policy.sigma_meas ** 2

        elif event == 'outlier':
            # An isolated suspect reading: Huber-style down-weighting instead
            # of rejection, so a single faulty densistat record cannot break
            # the estimate but also cannot dominate it.
            r_eff = self.cfg.policy.sigma_meas ** 2 * max(
                1.0, (abs(z) / self.cfg.policy.gate_z) ** 2)
            S = float(H @ self.cov @ H) + r_eff

        K = (self.cov @ H) / S
        self.mean = self.mean + K * innov
        A = np.eye(2) - np.outer(K, H)
        self.cov = A @ self.cov @ A.T + S * np.outer(K, K)
        self.cov = (self.cov + self.cov.T) / 2.0
        self._constrain()
        self.observed_days += 1
        if event:
            self.events.append((day, event))
        return FilterOutput(day, y, innov, sd, z, True, event)

    # ------------------------------------------------------------------
    def _classify(self, z: float, innov: float, sd: float) -> str | None:
        """Three-channel event classification, plus the streak safeguard.

        ``reset``    unexplained step up (re-pressurisation of the compartment)
        ``shock``    one large step down (accelerated degradation)
        ``bias``     several consecutive out-of-band readings (drifting rate:
                     the model must adapt, rejection is not an option)
        ``outlier``  a single suspect reading (down-weighted, not discarded)
        """
        p = self.cfg.policy
        if self.mean is None:
            return None
        if not p.gate_z:
            return None
        if innov > 0 and p.jump_z and innov > p.jump_z * sd and innov > p.jump_bar:
            self.suspect_streak = 0
            return 'reset'
        if abs(z) <= p.gate_z:
            self.suspect_streak = 0
            return None
        self.suspect_streak += 1
        if self.suspect_streak >= p.outlier_streak:
            return 'bias'
        return 'shock' if innov < 0 else 'outlier'

    def suspect(self) -> bool:
        """True when the latest reading was out of band (sensor suspicion)."""
        return self.suspect_streak > 0

    def _constrain(self) -> None:
        """Physical guards: a PMEC does not re-pressurise on its own."""
        p = self.cfg.policy
        self.mean[1] = float(np.clip(self.mean[1], p.alpha_min, p.alpha_max))
        self.mean[0] = float(min(self.mean[0], self.cfg.scenario.p_nom))

    # ------------------------------------------------------------------
    def state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.cov is None:
            raise RuntimeError('filter not initialised')
        return self.mean.copy(), self.cov.copy()

    def pressure_hat(self) -> float | None:
        return None if self.mean is None else float(self.mean[0])

    def rate_hat(self) -> float | None:
        return None if self.mean is None else float(self.mean[1])
