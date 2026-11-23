"""Synthetic environment: truth process and observation channel.

Everything the environment draws is keyed by ``(seed, pmec, cycle)`` or
``(seed, day, pmec)`` rather than by a sequentially advanced generator.  The
realised world is therefore *independent of the policy under test*: the same
seed gives the same leak rates, the same shock instants, the same measurement
noise and the same missing-data pattern for every strategy, which is what makes
the paired per-seed comparison valid.

The truth follows the Chapter 4 / Chapter 5 physics: constant leak rate inside
a cycle, pressure restored to ``p_nom`` by a gas-refilling intervention.
Uncertainty enters exclusively through the observation channel and the shock
process, which is the Chapter 6 object of study.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np


_MASK = (1 << 64) - 1


def _part(value) -> int:
    if isinstance(value, str):
        return int(hashlib.sha256(value.encode('utf-8')).hexdigest()[:16], 16)
    return int(value) & _MASK


def _key(*parts) -> int:
    """Deterministic 64-bit FNV-1a hash of integer / string coordinates."""
    h = 1469598103934665603
    for value in parts:
        h = ((h ^ _part(value)) * 1099511628211) & _MASK
    return h >> 16


def draw(*parts) -> np.random.Generator:
    """Reproducible generator attached to a coordinate key."""
    return np.random.default_rng(_key(*parts))


class World:
    """Policy-independent realisation of the environment for one seed."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.seed = cfg.seed
        self.ids = pmec_ids(cfg.scenario.n_pmec)
        self._cache: dict = {}
        self.shock_log: list[dict] = []
        self.shock_seen: set[tuple[int, str]] = set()

    # ------------------------------------------------------------------
    def alpha(self, pmec: str, cycle: int) -> float:
        """True leak rate (bar/day) of a cycle.

        Periodic configurations keep one rate per PMEC across cycles; aperiodic
        configurations re-draw and perturb it by the Chapter 5 +-5 % rule, which
        is a property of the *test instance*, not of the optimisation model.
        """
        s = self.cfg.scenario
        idx = self._idx(pmec)
        if s.periodic and cycle > 0:
            cycle = 0
        g = draw(self.seed, 'alpha', idx, cycle)
        base = g.uniform(*s.alpha_range())
        if not s.periodic:
            base *= 1.0 + g.uniform(-s.aperiodic_pct, s.aperiodic_pct)
        return float(max(base, 1e-5))

    def n_clusters(self) -> int:
        return int(getattr(self.cfg.scenario, 'n_clusters', 3))

    def cluster_centre(self, pmec: str) -> float:
        """Day around which this compartment's alarms fall in a clustered instance.

        The Chapter 5 clustered configuration concentrates the reference dates
        of a group of PMECs, which is what makes consolidation valuable; the
        centres are spread over the horizon and each compartment is scattered
        around its centre.
        """
        s = self.cfg.scenario
        k = self.n_clusters()
        g = draw(self.seed, 'cluster', self._idx(pmec))
        centre = s.T * (2 * int(g.integers(0, k)) + 1) / (2.0 * k)
        # the Chapter 5 clustered configuration concentrates reference dates
        # of a group of compartments on (nearly) the same day, which is what
        # makes consolidation valuable; the scatter is therefore measured in
        # days rather than as a fraction of the horizon
        return float(g.normal(centre, max(s.cluster_jitter, 0.5)))

    def initial_pressure(self, pmec: str) -> float:
        """Initial pressure; in a clustered instance it is set so that the first
        alarm lands on the compartment's cluster centre."""
        s = self.cfg.scenario
        g = draw(self.seed, 'init-p', self._idx(pmec))
        if not s.clustered:
            return float(g.uniform(s.p0_low, s.p0_high))
        alpha = self.alpha(pmec, 0)
        centre = self.cluster_centre(pmec)
        p = s.p_ref + alpha * max(centre, 1.0)
        return float(min(max(p, s.p0_low), s.p0_high))

    # --- shocks -----------------------------------------------------------
    def shocks_on(self, day: int) -> list[tuple[str, float]]:
        """Leak-rate shocks that take effect at ``day`` (policy independent)."""
        p = self.cfg.policy
        if p.shock_rate <= 0:
            return []
        out = []
        for pmec in self.ids:
            g = draw(self.seed, 'shock', day, self._idx(pmec))
            if g.random() < p.shock_rate:
                mult = 1.0 + p.shock_alpha_mult * g.random()
                out.append((pmec, mult))
                if (day, pmec) not in self.shock_seen:
                    self.shock_seen.add((day, pmec))
                    self.shock_log.append(dict(day=day, pmec=pmec, multiplier=mult))
        return out

    # --- observation channel ---------------------------------------------
    def measurement_noise(self, day: int, pmec: str) -> float:
        return float(draw(self.seed, 'noise', day, self._idx(pmec)).normal(
            0.0, self.cfg.policy.sigma_meas))

    def outlier(self, day: int, pmec: str) -> float | None:
        p = self.cfg.policy
        if p.outlier_rate <= 0:
            return None
        g = draw(self.seed, 'outlier', day, self._idx(pmec))
        if g.random() < p.outlier_rate:
            return float(g.normal(0.0, p.outlier_sigma))
        return None

    def missing(self, day: int, pmec: str) -> bool:
        """Markov missingness: no record, or a spell of consecutive absences."""
        p = self.cfg.policy
        if p.p_missing <= 0:
            return False
        key = ('miss', self._idx(pmec))
        chain = self._cache.get(key)
        if chain is None:
            g = draw(self.seed, 'miss', self._idx(pmec))
            stay = 1.0 - 1.0 / p.missing_run_mean      # P(absent | absent)
            enter = p.p_missing / (1.0 - p.p_missing) * (1.0 - stay)
            chain, absent = [], False
            for _ in range(self.cfg.scenario.T + 1):
                u = g.random()
                absent = absent if u < stay else (u < stay + enter)
                chain.append(absent)
            self._cache[key] = chain
        return chain[day]

    # ------------------------------------------------------------------
    def _idx(self, pmec: str) -> int:
        return self.ids.index(pmec)


def pmec_ids(n: int) -> list[str]:
    """PMEC identifiers, PMEC_001 ... PMEC_{n:03d}."""
    return [f'PMEC_{j:03d}' for j in range(1, n + 1)]


@dataclass
class TruthState:
    """True compartment state maintained by the simulator."""
    pressure: float
    alpha: float
    cycle: int
    shocks: int = 0


class Simulator:
    """Steps the true pressure path and emits the daily observation."""

    def __init__(self, cfg, world: World):
        self.cfg = cfg
        self.world = world
        s = cfg.scenario
        self.state = {pmec: TruthState(world.initial_pressure(pmec),
                                       world.alpha(pmec, 0), 0)
                      for pmec in world.ids}
        self.leaked_mass = 0.0                # cumulative bar-days, proxy only
        self.observed: dict[str, float | None] = {}
        self.truth_log: list[dict] = []
        self.day = 0

    # ------------------------------------------------------------------
    def observe(self, day: int) -> dict[str, float | None]:
        p = self.cfg.policy
        out: dict[str, float | None] = {}
        for pmec, st in self.state.items():
            if self.world.missing(day, pmec):
                out[pmec] = None
                continue
            y = st.pressure + self.world.measurement_noise(day, pmec)
            bad = self.world.outlier(day, pmec)
            if bad is not None:
                y += bad
            if p.quantisation > 0:
                y = round(y / p.quantisation) * p.quantisation
            out[pmec] = float(y)
        self.observed = out
        return out

    def step(self, day: int, refilled: set[str]) -> None:
        """Advance one day: degradation, shocks, then the interventions of ``day``."""
        for pmec, st in self.state.items():
            st.pressure -= st.alpha
            self.leaked_mass += max(self.cfg.scenario.p_ref - st.pressure, 0.0) \
                if st.pressure < self.cfg.scenario.p_ref else 0.0
        for pmec, mult in self.world.shocks_on(day):
            st = self.state[pmec]
            st.alpha *= mult
            st.shocks += 1
        for pmec in sorted(refilled):
            st = self.state[pmec]
            st.pressure = self.cfg.scenario.p_nom
            st.cycle += 1
            st.alpha = self.world.alpha(pmec, st.cycle)
        self.day = day

    # ------------------------------------------------------------------
    def log(self, day: int) -> None:
        for pmec, st in self.state.items():
            self.truth_log.append(dict(
                day=day, pmec=pmec, cycle=st.cycle, alpha=st.alpha,
                pressure=st.pressure, shocks=st.shocks,
                observation=self.observed.get(pmec),
                below_ref=bool(st.pressure < self.cfg.scenario.p_ref),
                below_min=bool(st.pressure < self.cfg.scenario.p_min)))

    def violations(self) -> tuple[int, int]:
        """(asset-days below p_ref, asset-days below p_min) over the run."""
        return (sum(r['below_ref'] for r in self.truth_log),
                sum(r['below_min'] for r in self.truth_log))

    def path_no_action(self, pmec: str, to_day: int) -> list[float]:
        """True pressure path from the current state assuming no further work.

        This is the quantity a forecast at the current decision day is about:
        the compartment is left to degrade, with the shocks the environment
        happens to apply.  Used only for calibration bookkeeping.
        """
        st = self.state[pmec]
        p, alpha = st.pressure, st.alpha
        out = [p]
        for d in range(self.day + 1, to_day + 1):
            p -= alpha
            for sm, mult in self.world.shocks_on(d):
                if sm == pmec:
                    alpha *= mult
            out.append(p)
        return out

    # --- oracle helpers: perfect knowledge of the physical process -------
    def true_pressure_path(self, pmec: str, from_day: int, to_day: int,
                           planned: dict[int, set[str]]) -> list[float]:
        """True pressure of ``pmec`` given its current state and ``planned`` work.

        Used only by the clairvoyant strategies; ``planned`` maps a day to the
        set of PMEC refilled at that day, including interventions decided later.
        """
        st = self.state[pmec]
        p = st.pressure
        alpha = st.alpha
        path = []
        for d in range(from_day, to_day + 1):
            if d > from_day:
                p -= alpha
                for shmec, mult in self.world.shocks_on(d):
                    if shmec == pmec:
                        alpha *= mult
                if pmec in planned.get(d, ()):
                    p = self.cfg.scenario.p_nom
                    alpha = self.world.alpha(pmec, st.cycle + 1)
            path.append(p)
        return path


# ----------------------------------------------------------------------
def nominal_plan(cfg, world: World) -> list[dict]:
    """Chapter 5 style input plan built from a deterministic prediction.

    Each PMEC contributes a chain of GR tasks whose reference dates follow from
    the constant-rate pressure model; nothing about shocks, noise or missing
    data is known here, which is exactly the information state of
    Chapter 5's rolling-window procedure.
    """
    s = cfg.scenario
    tasks = []
    for pmec in world.ids:
        pressure = world.initial_pressure(pmec)
        alpha = world.alpha(pmec, 0)
        cycle, t, k = 0, 0, 0
        while True:
            if alpha <= 0:
                break
            wait = int(np.ceil((pressure - s.p_ref) / alpha))
            t_next = t + max(wait, 1) + (world.offset(pmec) if k == 0 else 0)
            if t_next > s.T:
                break
            tasks.append(dict(task_id=f'{pmec}#{cycle}', pmec=pmec, cycle=cycle,
                              created_day=t, nominal_day=t_next))
            t, cycle, k = t_next, cycle + 1, k + 1
            pressure, alpha = s.p_nom, world.alpha(pmec, cycle)
    return tasks


def realized_plan(cfg, world: World, executions: list[dict]) -> list[dict]:
    """Reference dates taken from the realised pressure paths, not from a forecast.

    For every compartment the true trajectory is walked day by day; the
    reference date of a cycle is the first day on which the true pressure
    reaches the refilling alarm.  Pairing those with the interventions actually
    performed gives the shift penalties an operator would have been charged
    after the fact, which is the Chapter 6 evaluation of a plan made under
    imperfect information.  A cycle whose alarm is never answered is reported
    with ``serviced = False``.
    """
    s = cfg.scenario
    by_day: dict[int, set[str]] = {}
    for e in executions:
        day = int(e['executed_day'] if 'executed_day' in e else e['day'])
        by_day.setdefault(day, set()).add(e['pmec'])
    out = []
    for pmec in world.ids:
        pressure = world.initial_pressure(pmec)
        alpha = world.alpha(pmec, 0)
        cycle, alarm, serviced = 0, None, False
        for t in range(0, s.T + 1):
            if t and pressure > s.p_nom:
                pressure = s.p_nom
            if t:
                pressure -= alpha
                for shmec, mult in world.shocks_on(t):
                    if shmec == pmec:
                        alpha *= mult
            if alarm is None and pressure <= s.p_ref:
                alarm = t
            if pmec in by_day.get(t, ()):
                if alarm is None and alpha > 0:
                    # the intervention anticipated the alarm: the reference date
                    # of the cycle is the day the alarm *would* have occurred
                    alarm = int(round(t + (pressure - s.p_ref) / alpha))
                out.append(dict(task_id=f'{pmec}#{cycle}', pmec=pmec, cycle=cycle,
                                reference_real=alarm, executed_day=t,
                                serviced=True))
                cycle, alarm, serviced = cycle + 1, None, False
                pressure = s.p_nom
                alpha = world.alpha(pmec, cycle)
        if alarm is not None and not serviced:
            out.append(dict(task_id=f'{pmec}#{cycle}', pmec=pmec, cycle=cycle,
                            reference_real=alarm, executed_day=None,
                            serviced=False))
    return out
