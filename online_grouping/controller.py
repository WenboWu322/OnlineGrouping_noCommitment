"""Rolling online controller: forecast -> window -> trigger -> group -> execute.

One loop implements every strategy of the computational study; the strategies
differ only by the switches collected in :class:`Strategy`.  The loop is
information-causal: at day ``d`` the controller knows the observation of day
``d`` and nothing later, and it may only assign work from day ``d + lead_days``
onwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from . import metrics, process
from .filter import TrendFilter
from .grouping import WindowSpec, greedy_assignment, solve_window
from .metrics import cost_metrics
from .scenarios import Simulator, World, pmec_ids


# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Strategy:
    name: str
    grouping: str = 'M2'          # 'M1' | 'M2' | 'none' (dispatch per task)
    replan: str = 'event'         # 'none' | 'periodic' | 'event' | 'daily'
    window: str = 'risk'          # 'risk' (chance constraint) | 'nominal' (Ch.5)
    stability: bool = True        # anchoring + date lock
    detect: bool = True           # robust update and event handling
    clairvoyant: bool = False     # perfect physical knowledge (oracle)
    creation: str = 'proactive'   # 'proactive' (forecast horizon) | 'reactive' (alarm)
    periodic_days: int = 7
    lead: int | None = None       # override of scenario.lead_days

    def key(self) -> tuple:
        return (self.grouping, self.replan, self.window, self.stability,
                self.detect, self.clairvoyant, self.periodic_days, self.lead)


@dataclass
class PendingTask:
    task_id: str
    pmec: str
    cycle: int
    created_day: int
    tau0: int | None = None
    deadline: int | None = None
    assigned: int | None = None
    announced: int | None = None
    anchor: int | None = None
    tau0_at_solve: int | None = None
    status: str = 'pending'       # pending | executed | dropped


class Forecaster:
    """Interface returning lead times for crossing and deadline."""

    def crossing_lead(self, pmec: str, day: int) -> int | None:
        raise NotImplementedError

    def deadline_lead(self, pmec: str, day: int, kappa: float | None) -> int | None:
        raise NotImplementedError

    def pressure_hat(self, pmec: str) -> float | None:
        raise NotImplementedError


class FilterForecaster(Forecaster):
    """Predictive window bounds from the Kalman estimate (imperfect data)."""

    def __init__(self, cfg, filters: dict[str, TrendFilter]):
        self.cfg = cfg
        self.filters = filters

    def _state(self, pmec):
        f = self.filters[pmec]
        return None if f.mean is None else f.state()

    def _max_lead(self, day: int) -> int:
        return max(self.cfg.scenario.T - day, 0)

    def crossing_reference(self, pmec, day):
        st = self._state(pmec)
        if st is None:
            return None
        s = self.cfg.scenario
        lead = process.crossing_lead(st[0], st[1], s.p_ref, self.cfg.policy.q_p,
                                     self.cfg.policy.q_alpha, self._max_lead(day))
        return None if lead is None else day + lead

    def deadline_lead(self, pmec, day, kappa):
        st = self._state(pmec)
        if st is None:
            return None
        s = self.cfg.scenario
        if kappa is None:
            return process.deterministic_deadline_lead(
                float(st[0][0]), float(st[0][1]), s.p_min, self._max_lead(day))
        first, last = process.risk_deadline_lead(
            st[0], st[1], s.p_min, kappa, self.cfg.policy.q_p,
            self.cfg.policy.q_alpha, self._max_lead(day))
        return None if first is None else (last if last is not None
                                           else self._max_lead(day))

    def horizon_entry(self, pmec, day, lead_end, kappa):
        st = self._state(pmec)
        if st is None:
            return False
        s = self.cfg.scenario
        mean, sd = process.predictive_arrays(st[0], st[1], lead_end,
                                             self.cfg.policy.q_p,
                                             self.cfg.policy.q_alpha)
        bound = mean[-1] if kappa is None else mean[-1] - kappa * sd[-1]
        return bool(bound <= s.p_ref)

    def rate_estimate(self, pmec):
        return self.filters[pmec].rate_hat()

    def pressure_hat(self, pmec):
        return self.filters[pmec].pressure_hat()


class PerfectForecaster(Forecaster):
    """Clairvoyant reference: exact physical state and known shock schedule."""

    def __init__(self, cfg, world: World, sim: Simulator):
        self.cfg = cfg
        self.world = world
        self.sim = sim

    def _path(self, pmec, day, max_lead):
        st = self.sim.state[pmec]
        p, alpha = st.pressure, st.alpha
        out = []
        for h in range(0, max_lead + 1):
            if h:
                d = day + h
                p -= alpha
                for sm, mult in self.world.shocks_on(d):
                    if sm == pmec:
                        alpha *= mult
            out.append(p)
        return out

    def crossing_reference(self, pmec, day):
        s = self.cfg.scenario
        path = self._path(pmec, day, max(s.T - day, 0))
        for h, p in enumerate(path):
            if p <= s.p_ref:
                return day + h
        return None

    def deadline_lead(self, pmec, day, kappa):
        s = self.cfg.scenario
        st = self.sim.state[pmec]
        if st.alpha <= 0:
            return None
        return int(np.floor((st.pressure - s.p_min) / st.alpha))

    def horizon_entry(self, pmec, day, lead_end, kappa):
        s = self.cfg.scenario
        path = self._path(pmec, day, lead_end)
        return bool(path[-1] <= s.p_ref)

    def rate_estimate(self, pmec):
        return float(self.sim.state[pmec].alpha)

    def pressure_hat(self, pmec):
        return float(self.sim.state[pmec].pressure)


# ----------------------------------------------------------------------
@dataclass
class RunResult:
    strategy: str
    seed: int
    cost: dict = field(default_factory=dict)
    execution_log: list[dict] = field(default_factory=list)
    decision_log: list[dict] = field(default_factory=list)
    solver_log: list[dict] = field(default_factory=list)
    change_log: list[dict] = field(default_factory=list)
    truth_log: list[dict] = field(default_factory=list)
    unserviced_alarms: int = 0
    record_forecasts: bool = True
    task_log: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    forecast_log: list[dict] = field(default_factory=list)
    fallback_log: list[dict] = field(default_factory=list)
    event_log: list[dict] = field(default_factory=list)


class OnlineRunner:
    """One simulated maintenance campaign."""

    def __init__(self, cfg, world: World, strategy: Strategy, trace=()):
        self.cfg = cfg
        self.world = world
        self.strategy = strategy
        self.sim = Simulator(cfg, world)
        self.lead = cfg.scenario.lead_days if strategy.lead is None else strategy.lead
        if strategy.clairvoyant:
            self.filters = {}
            self.forecaster: Forecaster = PerfectForecaster(cfg, world, self.sim)
        else:
            self.filters = {m: TrendFilter(cfg) for m in world.ids}
            self.forecaster = FilterForecaster(cfg, self.filters)
        self.pending: dict[str, PendingTask] = {}
        # number of refilling interventions already carried out per compartment:
        # an operational record, identical for every strategy on a given seed
        self._cycles = {pid: 0 for pid in self.world.ids}
        self._self_refill: dict[str, int] = {}
        self.res: RunResult = RunResult(strategy.name, cfg.seed)
        self.FORECAST_LEADS = (1, 5, 10, 20, 40, 60)
        self.last_solve_day: int | None = None
        self.trace = set(trace or ())
        self._plan_day_of: dict[str, int] = {}
        self.first_execution_of_task: dict[str, tuple[int, int]] = {}

    # ------------------------------------------------------------------
    def run(self) -> RunResult:
        s = self.cfg.scenario
        for day in range(s.T + 1):
            if day > 0:
                # interventions decided earlier are carried out at the start of the day
                due = {t.pmec for t in self.pending.values()
                       if t.status == 'pending' and t.assigned == day}
                self._execute(day, due)
                self.sim.step(day, due)
                self._known_interventions(day, due)
            obs = self.sim.observe(day)
            self._update_filters(day, obs)
            self._log_forecasts(day)
            self.sim.log(day)
            self._refresh_tasks(day)
            if day < s.T:
                self._plan(day)
        self._finalise()
        self.res.truth_log = self.sim.truth_log
        self.res.event_log = [dict(pmec=m, day=d, event=e)
                              for m, f in self.filters.items() for d, e in f.events]
        return self.res

    # ------------------------------------------------------------------
    def _execute(self, day: int, due: set[str]) -> None:
        for task in list(self.pending.values()):
            if task.status == 'pending' and task.assigned == day:
                task.status = 'executed'
                self._cycles[task.pmec] += 1
                self._self_refill[task.pmec] = day
                self.res.execution_log.append(dict(
                    task_id=task.task_id, pmec=task.pmec, cycle=task.cycle,
                    executed_day=day, created_day=task.created_day,
                    reference_hat=(task.tau0_at_solve
                                   if task.tau0_at_solve is not None else task.tau0),
                    reference_latest=task.tau0, anchor=task.anchor,
                    announced_day=task.announced,
                    notice=day - (self.last_planned_day(day, task)),
                    unscheduled=bool(task.announced is None)))
                self.first_execution_of_task[task.task_id] = (day, task.tau0)
                del self.pending[task.task_id]

    def last_planned_day(self, day: int, task: PendingTask) -> int:
        """Decision day that produced the executed assignment (for notice time)."""
        return self._plan_day_of.get(task.task_id, day)

    # ------------------------------------------------------------------
    def _update_filters(self, day: int, obs: dict[str, float | None]) -> None:
        if self.strategy.clairvoyant:
            return
        for pmec, f in self.filters.items():
            out = f.update(day, obs.get(pmec))
            if out.event == 'reset':
                # A step up that is not our own intervention means the
                # compartment was re-pressurised from outside: its open task
                # and its cycle are no longer valid.
                if self._self_refill.get(pmec) == day:
                    continue
                for key in [k for k, t in self.pending.items()
                            if t.pmec == pmec and t.status == 'pending']:
                    self.res.fallback_log.append(dict(
                        day=day, pmec=pmec, task_id=key,
                        reason='observed_repressurisation'))
                    del self.pending[key]

    @staticmethod
    def _task_key(pmec: str, cycle: int) -> str:
        """A task is identified by its compartment and its intervention cycle,
        so a withdrawn task and its later replacement are the same object."""
        return f'{pmec}#{cycle}'

    def _known_interventions(self, day: int, due: set[str]) -> None:
        """Feed our own interventions back into the estimator.

        A refilling carried out by the operator is a *known* input to the
        process, not an anomaly to be detected: afterwards the compartment is
        at the nominal pressure.  Leaving it out lets the estimate keep
        degrading whenever the daily record happens to be missing, which in
        turn invites a second intervention on the same compartment.
        """
        if self.strategy.clairvoyant or not due:
            return
        for pmec in due:
            # the state is anchored at the start of the day, so the day's own
            # record can still be used as a check on the restored level
            self.filters[pmec].reset_cycle(max(day - 1, 0))

    def _log_forecasts(self, day: int) -> None:
        """Record the predictive distribution at the study lead times (E1)."""
        if self.strategy.clairvoyant or not self.res.record_forecasts:
            return
        s = self.cfg.scenario
        pol = self.cfg.policy
        for pmec, f in self.filters.items():
            if f.mean is None:
                continue
            m, sd = process.predictive_arrays(f.mean, f.cov,
                                              max(self.FORECAST_LEADS),
                                              pol.q_p, pol.q_alpha)
            for lead in self.FORECAST_LEADS:
                if day + lead > s.T:
                    continue
                self.res.forecast_log.append(dict(day=day, pmec=pmec, lead=lead,
                                                  mean=float(m[lead]),
                                                  sd=float(sd[lead])))

    # ------------------------------------------------------------------
    # task lifecycle
    # ------------------------------------------------------------------
    def _window(self, day: int) -> list[int]:
        s = self.cfg.scenario
        if self.strategy.replan == 'none':
            return list(range(day + self.lead, s.T + 1))
        return list(range(day + self.lead,
                          min(day + self.lead + s.L_win - 1, s.T) + 1))

    def _in_horizon(self, pmec: str, day: int) -> bool:
        """Risk-adjusted horizon entry (Chapter 6 replacement of the Chapter 5
        reference-date test): a task enters the planning window when the lower
        confidence bound of the predicted pressure at the window end has
        reached the refilling alarm.  Using the mean crossing instead would
        submit tasks months before they are actionable, because the leak rate
        is only weakly identified early in a cycle.
        """
        s = self.cfg.scenario
        window = self._window(day)
        if not window:
            return False
        kappa = (None if self.strategy.window == 'nominal'
                 else norm.ppf(1.0 - s.epsilon))
        return self.forecaster.horizon_entry(pmec, day, window[-1] - day, kappa)

    def _offline_chain(self) -> None:
        """Reference date of every future cycle, forecast at the information time.

        The offline benchmark (Chapter 5 applied once) must know which tasks
        exist before it plans them, otherwise it would only ever see the first
        cycle of each compartment.  The chain is built from the day-0 filtered
        state: each cycle contributes one task, its reference date following
        from the restored pressure and the estimated leak rate.
        """
        s = self.cfg.scenario
        for pmec in pmec_ids(s.n_pmec):
            if not self.strategy.clairvoyant:
                f = self.filters.get(pmec)
                if f is None or f.mean is None:
                    continue
            rate = self.forecaster.rate_estimate(pmec)
            if rate is None or rate <= 0:
                continue
            tau = self.forecaster.crossing_reference(pmec, 0)
            if tau is None:
                continue
            period = (s.p_nom - s.p_ref) / rate
            cycle, k = 0, 0
            while tau <= s.T:
                key = self._task_key(pmec, cycle)
                task = PendingTask(task_id=key, pmec=pmec, cycle=cycle,
                                   created_day=0, tau0=int(round(tau)))
                task.deadline = min(int(round(tau)) + s.delta_delay,
                                    int(round(tau + (s.p_ref - s.p_min) / rate)))
                self.pending[key] = task
                self.res.task_log.append(dict(day=0, action='task_added',
                                              task_id=key, pmec=pmec))
                k += 1
                tau = tau + period
                cycle += 1
                if k > 1 + s.T // max(int(period), 1):
                    break

    def _refresh_tasks(self, day: int) -> None:
        """Update reference dates; enter and leave the horizon explicitly."""
        s = self.cfg.scenario
        if self.strategy.replan == 'none':
            if day == 0:
                self._offline_chain()
            return
        window = self._window(day)
        if not window:
            return          # horizon boundary: nothing can be scheduled any more
        for pmec in pmec_ids(s.n_pmec):
            cycle = self._cycles[pmec]
            key = self._task_key(pmec, cycle)
            task = self.pending.get(key)
            if task is None:
                if self._should_create(pmec, day):
                    task = PendingTask(task_id=key, pmec=pmec, cycle=cycle,
                                       created_day=day)
                    self.pending[key] = task
                    self.res.task_log.append(dict(day=day, action='task_added',
                                                   task_id=key, pmec=pmec))
            elif task.status != 'pending':
                task = None
            if task is not None:
                task.tau0 = self.forecaster.crossing_reference(pmec, day)
                task.deadline = self._deadline(pmec, day, task.tau0)
                hi = window[-1]
                if task.tau0 is None or task.tau0 > hi:
                    # The task left the planning horizon: withdraw it (Chapter 5
                    # task_removed) rather than carry a plan date it cannot use.
                    # It re-enters through the horizon test when it is actionable.
                    task.status = 'cancelled'
                    del self.pending[key]
                    self.res.task_log.append(dict(day=day, action='task_removed',
                                                   task_id=key, pmec=pmec,
                                                   reason='outside_horizon'))
            if self.trace and pmec in self.trace and pmec in self.filters:
                f = self.filters[pmec]
                hi = window[-1]
                self.res.trace.append(dict(
                    day=day, pmec=pmec,
                    p_true=self.sim.state[pmec].pressure,
                    alpha_true=self.sim.state[pmec].alpha,
                    p_hat=None if f.mean is None else float(f.mean[0]),
                    alpha_hat=None if f.mean is None else float(f.mean[1]),
                    sd=None if f.mean is None else float(np.sqrt(f.cov[0, 0])),
                    tau0=None if task is None else task.tau0,
                    deadline=None if task is None else task.deadline,
                    assigned=None if task is None else task.assigned,
                    in_window=bool(task is not None and task.tau0 is not None
                                   and window[0] <= task.tau0 <= hi)))

    def _deadline(self, pmec: str, day: int, tau0: int | None) -> int | None:
        """Upper edge of the admissible set.

        Two limits act on a refilling: the chance-constrained physical deadline
        (Chapter 6 eq. for the risk-adjusted deadline, or the Chapter 5
        deterministic one under the nominal window, less the safety margin) and
        the operational limit on how long routine work may lag its own alarm.
        The safety margin is subtracted from the physical lead before the
        horizon cap, never after it.
        """
        s = self.cfg.scenario
        kappa = (None if self.strategy.window == 'nominal'
                 else norm.ppf(1.0 - s.epsilon))
        lead = self.forecaster.deadline_lead(pmec, day, kappa)
        if lead is None:
            physical = None
        else:
            if self.strategy.window == 'nominal':
                lead -= s.delta_safe
            physical = min(s.T, day + max(lead, 0))
        alarm_bound = None if tau0 is None else tau0 + s.delta_delay
        if physical is None:
            return alarm_bound
        if alarm_bound is None:
            return physical
        return min(physical, alarm_bound)

    def _should_create(self, pmec: str, day: int) -> bool:
        if self.strategy.replan == 'none' and day > 0:
            return False                      # offline plan: no new tasks
        # A compartment that has just been refilled sits at the nominal
        # pressure; opening a new task for it this soon would duplicate an
        # intervention that has no effect.
        last = self._self_refill.get(pmec)
        if last is not None and day - last < self.cfg.policy.cooldown_days:
            return False
        if self.strategy.creation == 'reactive':
            p_hat = self.forecaster.pressure_hat(pmec)
            return p_hat is not None and p_hat <= self.cfg.scenario.p_ref
        return self._in_horizon(pmec, day)

    # ------------------------------------------------------------------
    def _plan(self, day: int) -> None:
        s = self.cfg.scenario
        pending = [t for t in self.pending.values() if t.status == 'pending']
        if self.strategy.replan == 'none':
            window = list(range(day + self.lead, s.T + 1))
        else:
            window = list(range(day + self.lead,
                                min(day + self.lead + s.L_win - 1, s.T) + 1))
        active, deferred, overdue, notes = self._partition(day, pending, window)
        for note in notes:
            self.res.fallback_log.append(dict(day=day, **note))
            self._emergency(day, self.pending[note['task_id']], window)

        reasons = self._trigger(day, active + overdue)
        if not reasons:
            self.res.decision_log.append(dict(day=day, solved=False, reasons='',
                                              n_pending=len(pending),
                                              n_active=len(active),
                                              n_deferred=len(deferred)))
            return

        spec = self._build(day, active, window)
        status, assign = 'optimal', {}
        obj = gap = None
        secs, model = 0.0, ''
        if active:
            if self.strategy.grouping == 'none':
                assign, clusters, status = self._assign_independent(day, active, window)
            elif self.strategy.grouping == 'greedy':
                r = greedy_assignment(spec, self.cfg)
                assign, clusters, status = r.assignment, r.clusters, r.status
                obj, gap, secs, model = r.objective, r.gap, r.seconds, r.model
            else:
                r = solve_window(spec, self.cfg)
                assign, clusters, status = r.assignment, r.clusters, r.status
                obj, gap, secs, model = r.objective, r.gap, r.seconds, r.model
                if status == 'infeasible':
                    # Daily capacity over the window: keep the tasks whose
                    # deadline forces action now and defer the rest.  A
                    # documented relaxation, never silent.
                    urgent = [t for t in active
                              if t.deadline is not None and t.deadline <= window[-1]]
                    if urgent and len(urgent) < len(active):
                        for t in active:
                            if t not in urgent:
                                self.res.fallback_log.append(
                                    dict(day=day, pmec=t.pmec, task_id=t.task_id,
                                         reason='deferred_capacity'))
                        spec = self._build(day, urgent, window)
                        r = solve_window(spec, self.cfg)
                        assign, clusters, status = r.assignment, r.clusters, r.status
                        obj, gap, secs, model = (r.objective, r.gap, r.seconds,
                                                 r.model)
                        active = [t for t in active if t.task_id in assign] \
                            if assign else urgent
        self.res.solver_log.append(dict(day=day, status=status, objective=obj,
                                        gap=gap, seconds=secs, model=model,
                                        n_tasks=len(active), n_days=len(window),
                                        n_deferred=len(deferred),
                                        reasons=';'.join(reasons)))
        if status == 'infeasible':
            for t in active:
                self.res.fallback_log.append(dict(day=day, pmec=t.pmec,
                                                  task_id=t.task_id,
                                                  reason='infeasible_window'))
                self._emergency(day, t, window)
        else:
            self._apply(day, [t for t in active if t.task_id in assign], assign)
            if status == 'time_limit':
                for t in active:
                    if t.task_id not in assign and t.assigned is None:
                        self.res.fallback_log.append(dict(day=day, pmec=t.pmec,
                                                          task_id=t.task_id,
                                                          reason='time_limit'))
                        self._emergency(day, t, window)
        self.last_solve_day = day

    # ------------------------------------------------------------------
    def _emergency(self, day: int, task: PendingTask, window: list[int]) -> None:
        """Place a task as early as physically possible, outside the window.

        The window restrictions (advance limit, risk deadline, capacity) may
        leave no admissible day; the intervention is then placed at the earliest
        day the pressure physics still allows and recorded as a fallback, so
        that no emergency work is hidden from the accounting.
        """
        s = self.cfg.scenario
        horizon = list(range(day + 1, s.T + 1))
        days = self._feasible(task, day, horizon)
        if not days:
            days = [d for d in horizon if task.tau0 is None
                    or d >= task.tau0 - s.delta_adv]
        if not days:
            self.res.fallback_log.append(dict(day=day, pmec=task.pmec,
                                              task_id=task.task_id,
                                              reason='abandoned'))
            return
        task.assigned = min(days)
        if task.anchor is None:
            task.anchor = task.tau0
        self._plan_day_of[task.task_id] = day
        self.res.fallback_log.append(dict(
            day=day, pmec=task.pmec, task_id=task.task_id,
            reason='urgent' if task.assigned <= day + self.lead else 'relaxed',
            assigned=task.assigned))

    # ------------------------------------------------------------------
    def _trigger(self, day: int, pending: list[PendingTask]) -> list[str]:
        s, pol = self.cfg.scenario, self.cfg.policy
        if not pending and self.strategy.replan != 'none':
            return []
        if self.strategy.replan == 'none':
            return ['offline'] if day == 0 else []
        reasons: list[str] = []
        if self.last_solve_day is None:
            reasons.append('initial')
        if self.strategy.replan == 'daily':
            reasons.append('daily')
        elif self.strategy.replan == 'periodic':
            if (self.last_solve_day is None
                    or day - self.last_solve_day >= self.strategy.periodic_days):
                reasons.append('periodic')
        if any(t.assigned is None for t in pending):
            reasons.append('unassigned')
        if any(t.tau0 is not None and t.tau0 <= day for t in pending):
            reasons.append('overdue')
        if self.strategy.replan == 'event':
            for t in pending:
                if t.deadline is not None and t.assigned is not None \
                        and t.deadline - t.assigned <= pol.risk_margin_days:
                    reasons.append('margin')
                # new information since the previous solve, not since the
                # task was first created
                if t.tau0 is not None and t.tau0_at_solve is not None \
                        and abs(t.tau0 - t.tau0_at_solve) >= pol.replan_drift_days:
                    reasons.append('drift')
        return list(dict.fromkeys(reasons))

    def _window(self, day: int) -> list[int]:
        s = self.cfg.scenario
        return list(range(day + self.lead,
                          min(day + self.lead + s.L_win - 1, s.T) + 1))

    # ------------------------------------------------------------------
    def _partition(self, day: int, pending: list[PendingTask],
                   window: list[int]) -> tuple[list, list, list, list[dict]]:
        """Split pending tasks into schedulable, deferred and overdue groups.

        A task whose admissible days lie entirely *after* the window is not
        infeasible: it simply does not need a date yet, and forcing one would
        create a premature intervention.  A task whose admissible days lie
        entirely *before* the window is overdue and needs the emergency path.
        """
        s = self.cfg.scenario
        active, deferred, overdue, notes = [], [], [], []
        for t in pending:
            if t.tau0 is None:
                deferred.append(t)
                continue
            hi = t.deadline
            if hi is not None and hi < window[0]:
                overdue.append(t)
                notes.append(dict(pmec=t.pmec, task_id=t.task_id,
                                  reason='overdue', tau0=t.tau0, deadline=hi,
                                  window_lo=window[0], window_hi=window[-1]))
            elif t.tau0 > window[-1]:
                # Chapter 5 selection rule: only tasks whose current reference
                # date falls in the window are submitted to the sub-problem.
                deferred.append(t)
            else:
                active.append(t)
        return active, deferred, overdue, notes

    def _feasible(self, task: PendingTask, day: int, window: list[int]) -> list[int]:
        s = self.cfg.scenario
        lo = None if task.tau0 is None else task.tau0 - s.delta_adv
        hi = task.deadline
        return [d for d in window if (lo is None or d >= lo) and (hi is None or d <= hi)]

    def _build(self, day: int, active: list[PendingTask],
               window: list[int]) -> WindowSpec:
        """Window sub-problem in Chapter 5 notation.

        Tasks already announced for the near term are date-locked (soft
        stability in the objective plus hard locks inside the lock horizon).  A
        commitment whose announced day has ceased to be admissible is released
        rather than made infeasible: the release is recorded, charged through
        the shift and stability terms, and appears in the plan-change metric.
        """
        s = self.cfg.scenario
        domains = {t.task_id: self._feasible(t, day, window) for t in active}
        announced: dict[str, int] = {}
        # the anchoring term and the date locks belong to the same commitment
        # mechanism: a baseline without commitments must not carry either, or
        # the ablation S3 -> S4 -> S5 no longer isolates them
        for t in (active if self.strategy.stability else []):
            if t.announced is None:
                continue
            if t.announced in domains[t.task_id]:
                announced[t.task_id] = t.announced
            else:
                self.res.fallback_log.append(dict(day=day, pmec=t.pmec,
                                                  task_id=t.task_id,
                                                  reason='commitment_released',
                                                  announced=t.announced))
        return WindowSpec(tasks=[t.task_id for t in active], days=window,
                          domains=domains,
                          tau0={t.task_id: (t.tau0 if t.tau0 is not None else window[0])
                                for t in active},
                          announced=announced,
                          lock_until=(day + self.cfg.policy.lock_days
                                      if self.strategy.stability else None),
                          objective=(self.strategy.grouping
                                     if self.strategy.grouping in ('M1', 'M2')
                                     else 'M2'))

    # ------------------------------------------------------------------
    def _assign_independent(self, day: int, pending, window):
        """Grouping switched off: each task gets its own nearest-date dispatch."""
        assign, clusters, status = {}, [], 'optimal'
        for t in sorted(pending, key=lambda x: (x.tau0 or 10**6, x.task_id)):
            days = self._feasible(t, day, window)
            if not days:
                status = 'infeasible'
                continue
            d = min(days, key=lambda d: abs(d - (t.tau0 or d)))
            assign[t.task_id] = d
            if d not in clusters:
                clusters.append(d)
        return assign, clusters, status

    def _feasible(self, t: PendingTask, day: int, window: list[int]) -> list[int]:
        s = self.cfg.scenario
        lo = None if t.tau0 is None else t.tau0 - s.delta_adv
        return [d for d in window if (lo is None or d >= lo)
                and (t.deadline is None or d <= t.deadline)]

    # ------------------------------------------------------------------
    def _apply(self, day: int, pending: list[PendingTask],
               assign: dict[str, int]) -> None:
        for t in pending:
            new = assign[t.task_id]
            if t.assigned is not None and t.assigned != new:
                self.res.change_log.append(dict(day=day, task_id=t.task_id,
                                                pmec=t.pmec, old=t.assigned,
                                                new=new, reference=t.tau0))
            t.assigned = new
            t.announced = new
            t.tau0_at_solve = t.tau0
            if t.anchor is None:
                t.anchor = t.tau0
            self._plan_day_of[t.task_id] = day

    # ------------------------------------------------------------------
    def _finalise(self) -> None:
        open_tasks = sum(1 for t in self.pending.values() if t.status == 'pending')
        # evaluation against the realised alarm dates rather than the estimate
        # the decision was taken on: prediction error then shows up in the
        # shift penalties instead of cancelling itself out
        if self.res.forecast_log:
            truth = {(r['day'], r['pmec']): r['pressure']
                     for r in self.sim.truth_log}
            executed: dict[str, list[int]] = {}
            for e in self.res.execution_log:
                executed.setdefault(e['pmec'], []).append(e['executed_day'])
            clean = []
            for f in self.res.forecast_log:
                key = (f['day'] + f['lead'], f['pmec'])
                if key not in truth:
                    continue
                # a planned refilling inside the horizon is a structural break
                f['clean'] = not any(f['day'] < d <= f['day'] + f['lead']
                                     for d in executed.get(f['pmec'], ()))
                clean.append(dict(f, true_pressure=truth[key]))
            self.res.forecast_log = clean
        realised = metrics.realized_references(self.cfg, self.world,
                                              self.res.execution_log)
        if self.res.trace:
            done = {(e['pmec'], e['executed_day']) for e in self.res.execution_log}
            shocks = {(s['pmec'], s['day']) for s in self.world.shock_log}
            for t in self.res.trace:
                t['executed'] = (t['pmec'], t['day']) in done
                t['shock'] = (t['pmec'], t['day']) in shocks
        by_key = {(r['pmec'], r['cycle']): r for r in realised}
        for e in self.res.execution_log:
            r = by_key.get((e['pmec'], e['cycle']))
            if r is not None:
                e['reference_real'] = r['reference_real']
        self.res.unserviced_alarms = sum(1 for r in realised if not r['serviced'])
        self.res.cost['unserviced_alarms'] = self.res.unserviced_alarms
        self.res.cost = cost_metrics(
            self.cfg, self.strategy.name, self.res.execution_log,
            self.sim.truth_log, self.res.solver_log, self.res.change_log,
            self.res.fallback_log, self.sim.leaked_mass,
            grouping_off=(self.strategy.grouping == 'none'),
            pending_open=open_tasks)
