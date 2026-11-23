"""Prediction and admissible-day machinery (Chapter 6, "Model").

Physics is the Chapter 4 / Chapter 5 model: between interventions the pressure
of a PMEC falls at a constant rate, and a gas-refilling intervention restores
``p_nom``.  Chapter 6 replaces the *known* rate by a filtered estimate, so the
predicted pressure at lead time ``h`` is a random variable and the admissible
day set becomes a chance-constrained set.

State-space model (local linear trend, one day per step)::

    p_{k+1} = p_k - a_k + w_p ,      a_{k+1} = a_k + w_a
    y_k     = p_k + v_k

with ``w_p ~ N(0, q_p)``, ``w_a ~ N(0, q_alpha)`` and ``a`` the leak rate in
bar/day.  With ``F = [[1, -1], [0, 1]]`` the h-step propagation is closed form,
``F^h = [[1, -h], [0, 1]]``, and the accumulated process-noise covariance has
the entries used below (the leak-rate term grows cubically in ``h``).

All queries are vectorised over ``h = 0 .. H`` so that the full planning
horizon can be scanned at every decision day at negligible cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prediction:
    """Distribution of the pressure ``h`` days after the information time."""
    lead: int
    mean: float
    sd: float

    def quantile(self, prob: float) -> float:
        from scipy.stats import norm
        return self.mean + float(norm.ppf(prob)) * self.sd


def propagate(state_mean: np.ndarray, state_cov: np.ndarray, h: int,
              q_p: float, q_alpha: float) -> tuple[float, float, np.ndarray]:
    """h-step mean pressure, variance of the pressure and full state covariance."""
    if h < 0:
        raise ValueError('lead time must be non-negative')
    Fh = np.array([[1.0, -float(h)], [0.0, 1.0]])
    mean = Fh @ np.asarray(state_mean, dtype=float)
    cov = noise_covariance(h, q_p, q_alpha) + Fh @ np.asarray(state_cov, float) @ Fh.T
    return float(mean[0]), float(cov[0, 0]), cov


def noise_covariance(h: int, q_p: float, q_alpha: float) -> np.ndarray:
    """Closed form of ``sum_{k=0}^{h-1} F^k Q F^kT``."""
    if h <= 0:
        return np.zeros((2, 2))
    return np.array([
        [h * q_p + q_alpha * (h - 1) * h * (2 * h - 1) / 6.0,
         -q_alpha * h * (h - 1) / 2.0],
        [-q_alpha * h * (h - 1) / 2.0,
         h * q_alpha],
    ])


def predictive_arrays(state_mean: np.ndarray, state_cov: np.ndarray, horizon: int,
                      q_p: float, q_alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the pressure for leads ``0..horizon``."""
    h = np.arange(horizon + 1, dtype=float)
    m = np.asarray(state_mean, dtype=float)
    P = np.asarray(state_cov, dtype=float)
    mean = m[0] - m[1] * h
    var = (P[0, 0] - 2.0 * h * P[0, 1] + h * h * P[1, 1]
           + h * q_p + q_alpha * h * (h - 1.0) * (2.0 * h - 1.0) / 6.0)
    return mean, np.sqrt(np.maximum(var, 0.0))


def crossing_lead(state_mean, state_cov, p_ref: float, q_p: float, q_alpha: float,
                  max_lead: int) -> int | None:
    """First lead time at which the *mean* predicted pressure reaches ``p_ref``.

    ``None`` when the filtered state does not reach the threshold within
    ``max_lead`` days (very small or non-positive leak rate).
    """
    mean, _ = predictive_arrays(state_mean, state_cov, max_lead, q_p, q_alpha)
    hit = np.flatnonzero(mean <= p_ref)
    return int(hit[0]) if hit.size else None


def risk_deadline_lead(state_mean, state_cov, p_min: float, kappa: float,
                       q_p: float, q_alpha: float,
                       max_lead: int) -> tuple[int | None, int | None]:
    """Leads at which the chance constraint ``P(p >= p_min) >= 1 - eps`` holds.

    Returns ``(first, last)`` for the contiguous run of admissible leads
    starting at the first admissible one; ``(None, None)`` when the bound is
    violated at the information time itself, and ``(0, None)`` when the
    constraint never fails inside the scan horizon.
    """
    mean, sd = predictive_arrays(state_mean, state_cov, max_lead, q_p, q_alpha)
    ok = mean - kappa * sd >= p_min
    if not ok[0]:
        return None, None
    bad = np.flatnonzero(~ok)
    if bad.size == 0:
        return 0, None
    return 0, int(bad[0] - 1)


def deterministic_deadline_lead(pressure: float, rate: float, p_min: float,
                                max_lead: int) -> int | None:
    """Chapter 5 deadline (without the ``Delta^safe`` subtraction).

    ``floor((p - p_min) / delta)`` is the integer day at which the linear
    trajectory reaches ``p_min``; Chapter 5 then subtracts the safety margin.
    """
    if rate <= 0:
        return None
    return int(min(np.floor((pressure - p_min) / rate), max_lead))


def admissible_days(reference: int | None, deadline: int | None, horizon: set[int],
                    delta_adv: int) -> list[int]:
    """Chapter 5 eq. (5.9): ``t >= tau_n^0 - delta_adv`` and ``t <= d_n^safe``."""
    if reference is None or deadline is None:
        return []
    lo = int(np.ceil(reference - delta_adv))
    hi = int(np.floor(deadline))
    return sorted(t for t in horizon if lo <= t <= hi)
