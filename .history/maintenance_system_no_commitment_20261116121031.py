"""Event-triggered Kalman/B0/MILP maintenance demonstrator WITHOUT commitments.
Python >= 3.10. Dependencies: numpy, pandas, scipy.
All unexecuted assignments may move at a subsequent successful solve.
"""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from scipy.stats import norm


@dataclass
class Config:
    assets: int = 12
    days: int = 120
    seed: int = 42
    horizon: int = 30
    preparation_days: int = 2
    max_advance_days: int = 5
    capacity: int = 4
    p_trigger: float = 3.2
    p_safe: float = 3.0
    p_plan: float = 3.05
    epsilon: float = 0.05
    replanning_policy: str = 'event'
    max_replan_interval: int = 7
    risk_trigger_margin: float = 0.0
    nominal_shift_trigger_days: int | None = None
    reset_pressure: float = 3.5
    reset_pressure_var: float = 0.0
    R: float = 0.0001
    q_p: float = 0.000001
    q_alpha: float = 0.000000001
    alpha_prior_mean: float = 0.006
    alpha_prior_var: float = 0.000004
    sim_R: float = 0.0001
    sim_q_p: float = 0.000001
    sim_alpha_min: float = 0.004
    sim_alpha_max: float = 0.009
    missing_probability: float = 0.02
    setup_cost: float = 100.0
    early_penalty: float = 5.0
    late_penalty: float = 10.0
    time_limit: float = 15.0
    mip_gap: float = 0.001

    def validate(self):
        for name in ['assets', 'days', 'seed', 'horizon', 'preparation_days',
                     'max_advance_days', 'capacity', 'max_replan_interval']:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'{name} must be an integer.')
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise ValueError(f'{name} must be finite.')
        if self.assets < 1 or self.capacity < 1 or self.days < 2 or self.seed < 0:
            raise ValueError('Invalid assets, capacity, duration or seed.')
        if not 1 <= self.preparation_days <= self.horizon:
            raise ValueError('Require 1 <= preparation_days <= horizon.')
        if not 0 < self.epsilon < 0.5:
            raise ValueError('Require 0 < epsilon < 0.5.')
        if not self.p_safe <= self.p_plan < self.p_trigger < self.reset_pressure:
            raise ValueError('Require p_safe <= p_plan < p_trigger < reset_pressure.')
        if min(self.R, self.alpha_prior_var) <= 0 or min(
                self.q_p, self.q_alpha, self.sim_q_p, self.sim_R,
                self.reset_pressure_var) < 0:
            raise ValueError('Invalid noise/prior variances.')
        if not 0 <= self.missing_probability < 1:
            raise ValueError('Invalid missing probability.')
        if not 0 < self.sim_alpha_min <= self.sim_alpha_max:
            raise ValueError('Invalid simulator leakage-rate range.')
        if self.alpha_prior_mean <= 0:
            raise ValueError('alpha_prior_mean must be positive.')
        if self.max_advance_days < 0 or min(
                self.setup_cost, self.early_penalty, self.late_penalty,
                self.risk_trigger_margin) < 0:
            raise ValueError('Invalid costs, advancement or risk margin.')
        if self.replanning_policy not in {'event', 'daily', 'periodic'}:
            raise ValueError('Policy must be event, daily or periodic.')
        if self.max_replan_interval < 1:
            raise ValueError('max_replan_interval must be positive.')
        if self.nominal_shift_trigger_days is not None:
            if (isinstance(self.nominal_shift_trigger_days, bool)
                    or not isinstance(self.nominal_shift_trigger_days, int)
                    or self.nominal_shift_trigger_days < 1):
                raise ValueError('nominal_shift_trigger_days must be a positive integer or null.')
        if self.time_limit <= 0 or self.mip_gap < 0:
            raise ValueError('Invalid solver settings.')


class PlanningError(RuntimeError):
    """Stop automated operation; an external intervention procedure is required."""


@dataclass
class FilterState:
    mean: np.ndarray | None = None
    covariance: np.ndarray | None = None
    last_day: int = -1
    cycle: int = 0

    def process(self, day, observation, executed_gr, cfg):
        if day <= self.last_day:
            raise ValueError('Duplicate or out-of-order daily update.')
        F = np.array([[1., -1.], [0., 1.]])
        H = np.array([1., 0.])
        Q = np.diag([cfg.q_p, cfg.q_alpha])
        for d in range(self.last_day + 1, day + 1):
            if d == day and executed_gr:
                self.cycle += 1
                self.mean = np.array([cfg.reset_pressure, cfg.alpha_prior_mean])
                self.covariance = np.diag([cfg.reset_pressure_var, cfg.alpha_prior_var])
                continue
            y = observation if d == day else None
            if self.mean is None:
                if y is not None:
                    self.mean = np.array([float(y), cfg.alpha_prior_mean])
                    self.covariance = np.diag([cfg.R, cfg.alpha_prior_var])
                continue
            m = F @ self.mean
            P = F @ self.covariance @ F.T + Q
            if y is not None:
                residual = float(y - H @ m)
                S = float(H @ P @ H + cfg.R)
                K = P @ H / S
                m = m + K * residual
                A = np.eye(2) - np.outer(K, H)
                P = A @ P @ A.T + cfg.R * np.outer(K, K)
            self.mean = m
            self.covariance = (P + P.T) / 2
        self.last_day = day


def forecast(state, h, cfg):
    if state.mean is None or h < 0 or int(h) != h:
        raise ValueError('Forecast requires initialized state and integer h >= 0.')
    h = int(h)
    v = np.array([1., -float(h)])
    mu = float(v @ state.mean)
    variance = float(v @ state.covariance @ v + h * cfg.q_p
                     + h * (h - 1) * (2 * h - 1) / 6 * cfg.q_alpha)
    if not np.isfinite(mu) or not np.isfinite(variance):
        raise PlanningError('Non-finite forecast.')
    if variance < -1e-10:
        raise PlanningError('Materially negative forecast variance.')
    return mu, max(0., variance)


def reference_date(state, day, cfg):
    if state.mean is None:
        return None
    p, alpha = state.mean
    if p <= cfg.p_trigger:
        return day
    if alpha <= 0:
        return None
    return day + int(np.ceil((p - cfg.p_trigger) / alpha))


@dataclass
class Task:
    task_id: str
    asset: str
    cycle: int
    created_day: int
    ready_day: int
    nominal_day: int | None
    status: str = 'PROPOSED'
    proposed_day: int | None = None
    nominal_at_last_solve: int | None = None
    executed_day: int | None = None
    nominal_at_execution: int | None = None


class Controller:
    def __init__(self, ids, cfg):
        self.cfg = cfg
        self.states = {asset: FilterState() for asset in ids}
        self.tasks = {}
        self.forecasts, self.states_log, self.plans = [], [], []
        self.solver_log, self.changes, self.issues = [], [], []
        self.monitoring, self.decisions = [], []
        self.last_solve_day = None
        self.k = float(norm.ppf(1 - cfg.epsilon))

    def issue(self, day, asset, reason):
        self.issues.append(dict(day=day, asset=asset, reason=reason))
        raise PlanningError(f'Day {day}, {asset}: {reason}')

    def ingest(self, day, observations, executed):
        if set(executed) - set(self.states):
            self.issue(day, 'execution', 'Unknown asset in execution confirmation')
        for asset, state in self.states.items():
            if asset in executed:
                matches = [task for task in self.tasks.values()
                           if task.asset == asset and task.status == 'PROPOSED'
                           and task.proposed_day == day]
                if len(matches) != 1:
                    self.issue(day, asset, 'Unexpected GR or missing scheduled execution mapping')
                task = matches[0]
                task.status = 'EXECUTED'
                task.executed_day = day
                # Use the latest reference available BEFORE today's execution.
                task.nominal_at_execution = task.nominal_day
            y = observations.get(asset)
            if y is not None and not np.isfinite(y):
                self.issue(day, asset, 'Non-finite pressure observation')
            state.process(day, y, asset in executed, self.cfg)
            if state.mean is not None:
                self.states_log.append(dict(
                    day=day, asset=asset, cycle=state.cycle,
                    p_hat=state.mean[0], alpha_hat=state.mean[1],
                    P_pp=state.covariance[0, 0], P_aa=state.covariance[1, 1],
                    P_pa=state.covariance[0, 1]))
        for task in self.tasks.values():
            if (task.status == 'PROPOSED' and task.proposed_day is not None
                    and task.proposed_day <= day):
                self.issue(day, task.asset, 'Scheduled action overdue; execution confirmation missing')

    def check_current_safety(self, day):
        for asset, state in self.states.items():
            if state.mean is None:
                self.issue(day, asset, 'Current-cycle filter uninitialized; no risk certificate')
            mu, var = forecast(state, 0, self.cfg)
            if mu - self.k * np.sqrt(var) < self.cfg.p_safe:
                self.issue(day, asset, 'Current modeled physical safety bound violated')

    def prepare(self, day):
        c = self.cfg
        self.check_current_safety(day)
        selected, domains = [], {}
        for asset, state in self.states.items():
            nominal = reference_date(state, day, c)
            key = f'{asset}:cycle{state.cycle}'
            task = self.tasks.get(key)
            if task is not None:
                if task.status == 'EXECUTED':
                    self.issue(day, asset, 'Executed task still attached to current cycle')
                task.nominal_day = nominal
            grid = {}
            for d in range(day + 1, day + c.horizon + 1):
                mu, var = forecast(state, d - day, c)
                lb = mu - self.k * np.sqrt(var)
                grid[d] = (mu, var, lb)
                self.forecasts.append(dict(
                    decision_day=day, asset=asset, target_day=d,
                    mean=mu, variance=var, lower_bound=lb,
                    safe=bool(lb >= c.p_plan), physical_safe=bool(lb >= c.p_safe)))
            nominal_near = (nominal is not None
                            and nominal - c.max_advance_days <= day + c.horizon)
            risk_near = any(lb < c.p_plan for _, _, lb in grid.values())
            if task is None and (nominal_near or risk_near):
                task = Task(key, asset, state.cycle, day,
                            day + c.preparation_days, nominal)
                self.tasks[key] = task
            if task is None:
                continue
            if not (nominal_near or risk_near):
                if task.proposed_day is not None:
                    self.changes.append(dict(
                        day=day, task_id=key, old_day=task.proposed_day,
                        new_day=None, reason='deferred_outside_horizon'))
                task.proposed_day = None
                task.status = 'DEFERRED'
                continue
            if task.status == 'DEFERRED':
                task.status = 'PROPOSED'
            if nominal is None:
                self.issue(day, asset, 'Pending task has no mean crossing; fallback policy required')
            earliest = max(day + 1, task.ready_day,
                           nominal - c.max_advance_days)
            feasible = [d for d in grid if d >= earliest and grid[d][2] >= c.p_plan]
            if not feasible:
                self.issue(day, asset, 'No B0-safe date satisfying readiness/date restrictions')
            selected.append(task)
            domains[key] = feasible
        return selected, domains

    def inspect_plan(self, day, tasks, domains):
        c = self.cfg
        reasons = []
        if self.last_solve_day is None:
            reasons.append('initial_plan')
        elif day - self.last_solve_day >= c.max_replan_interval:
            reasons.append('maximum_interval')
        if c.replanning_policy == 'daily':
            reasons.append('daily_policy')
        counts = {}
        for task in tasks:
            target = task.proposed_day
            if target is None:
                reasons.append('unassigned_task')
                self.monitoring.append(dict(
                    day=day, task_id=task.task_id, asset=task.asset,
                    status=task.status, target_day=None, lower_bound=None,
                    margin_plan=None, margin_physical=None))
                continue
            if target <= day:
                self.issue(day, task.asset, 'Unexecuted retained assignment is overdue')
            mu, var = forecast(self.states[task.asset], target - day, c)
            lb = mu - self.k * np.sqrt(var)
            margin = float(lb - c.p_plan)
            self.monitoring.append(dict(
                day=day, task_id=task.task_id, asset=task.asset,
                status=task.status, target_day=target, lower_bound=lb,
                margin_plan=margin, margin_physical=lb - c.p_safe))
            if target not in domains[task.task_id]:
                reasons.append('retained_assignment_infeasible')
            if c.replanning_policy in {'event', 'daily'}:
                if margin <= c.risk_trigger_margin:
                    reasons.append('risk_margin')
                limit = c.nominal_shift_trigger_days
                if (limit is not None and task.nominal_day is not None
                        and task.nominal_at_last_solve is not None
                        and abs(task.nominal_day - task.nominal_at_last_solve) >= limit):
                    reasons.append('nominal_shift')
            counts[target] = counts.get(target, 0) + 1
        if any(n > c.capacity for n in counts.values()):
            reasons.append('retained_capacity_violation')
        return list(dict.fromkeys(reasons))

    def optimize(self, day, tasks, domains):
        c = self.cfg
        pairs = [(task.task_id, d) for task in tasks for d in domains[task.task_id]]
        dates = sorted({d for _, d in pairs})
        xi = {pair: j for j, pair in enumerate(pairs)}
        zi = {d: len(pairs) + j for j, d in enumerate(dates)}
        nvars = len(pairs) + len(dates)
        objective = np.zeros(nvars)
        for task in tasks:
            ref = task.nominal_day
            for d in domains[task.task_id]:
                objective[xi[task.task_id, d]] = (
                    c.early_penalty * max(ref - d, 0)
                    + c.late_penalty * max(d - ref, 0))
        for d in dates:
            objective[zi[d]] = c.setup_cost
        rows, cols, values, lows, highs = [], [], [], [], []
        def add(coefficients, low, high):
            row = len(lows)
            for j, value in coefficients.items():
                rows.append(row); cols.append(j); values.append(value)
            lows.append(low); highs.append(high)
        for task in tasks:
            add({xi[task.task_id, d]: 1. for d in domains[task.task_id]}, 1., 1.)
        for d in dates:
            assignment = {xi[key, dt]: 1. for key, dt in pairs if dt == d}
            add({**assignment, zi[d]: -float(c.capacity)}, -np.inf, 0.)
            add({**{j: -v for j, v in assignment.items()}, zi[d]: 1.}, -np.inf, 0.)
        lo, hi = np.zeros(nvars), np.ones(nvars)
        A = coo_matrix((values, (rows, cols)), shape=(len(lows), nvars)).tocsc()
        start = time.perf_counter()
        result = milp(
            objective, integrality=np.ones(nvars), bounds=Bounds(lo, hi),
            constraints=LinearConstraint(A, lows, highs),
            options={'time_limit': c.time_limit, 'mip_rel_gap': c.mip_gap})
        elapsed = time.perf_counter() - start
        self.solver_log.append(dict(
            day=day, status=int(result.status), message=str(result.message),
            objective=None if result.fun is None else float(result.fun),
            gap=None if getattr(result, 'mip_gap', None) is None else float(result.mip_gap),
            solve_seconds=elapsed))
        if result.x is None or not np.all(np.isfinite(result.x)):
            self.issue(day, 'ILP', 'No valid feasible solver incumbent')
        rounded = np.rint(result.x)
        if (np.max(np.abs(result.x - rounded)) > 1e-5
                or np.any(A @ rounded < np.asarray(lows) - 1e-6)
                or np.any(A @ rounded > np.asarray(highs) + 1e-6)
                or np.any(rounded < lo - 1e-6)
                or np.any(rounded > hi + 1e-6)):
            self.issue(day, 'ILP', 'Returned incumbent failed feasibility validation')
        assignment = {}
        for task in tasks:
            chosen = [d for d in domains[task.task_id]
                      if rounded[xi[task.task_id, d]] == 1]
            if len(chosen) != 1:
                self.issue(day, 'ILP', 'Invalid assignment count')
            assignment[task.task_id] = chosen[0]
        return assignment

    def apply_plan(self, day, tasks, assignment):
        for task in tasks:
            date = assignment[task.task_id]
            previous = task.proposed_day
            if previous is not None and previous != date:
                self.changes.append(dict(
                    day=day, task_id=task.task_id, old_day=previous,
                    new_day=date, reason='reoptimized'))
            task.proposed_day = date
            task.nominal_at_last_solve = task.nominal_day

    def finalize_daily_plan(self, day, tasks, domains, replanned):
        counts = {}
        for task in tasks:
            date = task.proposed_day
            if (date is None or date not in domains[task.task_id]
                    or date <= day or date < task.ready_day):
                self.issue(day, task.asset, 'Invalid retained assignment before execution')
            counts[date] = counts.get(date, 0) + 1
        if any(n > self.cfg.capacity for n in counts.values()):
            self.issue(day, 'capacity', 'Daily capacity exceeded')
        for task in tasks:
            self.plans.append(dict(
                decision_day=day, task_id=task.task_id, asset=task.asset,
                cycle=task.cycle, execution_day=task.proposed_day,
                nominal_day=task.nominal_day, status=task.status,
                ready_day=task.ready_day, replanned_today=replanned))

    def decide(self, day):
        tasks, domains = self.prepare(day)
        reasons = self.inspect_plan(day, tasks, domains)
        replanned = False
        if tasks and reasons:
            assignment = self.optimize(day, tasks, domains)
            self.apply_plan(day, tasks, assignment)
            self.last_solve_day = day
            replanned = True
        self.finalize_daily_plan(day, tasks, domains, replanned)
        self.decisions.append(dict(
            day=day, replanned=replanned,
            reasons=';'.join(reasons) if tasks else 'no_active_tasks',
            active_tasks=len(tasks), last_solve_day=self.last_solve_day))
        return {task.asset for task in tasks
                if task.status == 'PROPOSED' and task.proposed_day == day + 1}

    def terminal_monitor(self, day):
        # Do not create tasks, defer plans, solve, or issue new actions at termination.
        self.check_current_safety(day)
        for task in self.tasks.values():
            if task.status != 'PROPOSED' or task.proposed_day is None:
                continue
            if task.proposed_day <= day:
                self.issue(day, task.asset, 'Unexecuted assignment overdue at termination')
            mu, var = forecast(self.states[task.asset], task.proposed_day - day, self.cfg)
            lb = mu - self.k * np.sqrt(var)
            self.monitoring.append(dict(
                day=day, task_id=task.task_id, asset=task.asset,
                status=task.status, target_day=task.proposed_day, lower_bound=lb,
                margin_plan=lb - self.cfg.p_plan,
                margin_physical=lb - self.cfg.p_safe))


class Simulator:
    """Private truth is used only by the environment and evaluation exporter."""
    def __init__(self, ids, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.p = {asset: self.rng.uniform(3.32, 3.5) for asset in ids}
        self.alpha = {asset: self.draw_alpha() for asset in ids}
        self.cycle = {asset: 0 for asset in ids}
        self.observation_log, self.truth_log, self.executions = [], [], []

    def draw_alpha(self):
        return self.rng.uniform(self.cfg.sim_alpha_min, self.cfg.sim_alpha_max)

    def initial_observations(self):
        return self.observe(0, set(), self.p.copy(), force_observed=True)

    def observe(self, day, executed, pre_gr, force_observed=False):
        observations = {}
        for asset in self.p:
            y = self.p[asset] + self.rng.normal(0, np.sqrt(self.cfg.sim_R))
            if not force_observed and self.rng.random() < self.cfg.missing_probability:
                y = None
            if asset in executed:
                y = None
            observations[asset] = y
            self.observation_log.append(dict(
                day=day, asset=asset, pressure_obs=y, gr_day=asset in executed))
            self.truth_log.append(dict(
                day=day, asset=asset, cycle=self.cycle[asset],
                pre_gr_pressure=pre_gr[asset], true_pressure=self.p[asset],
                true_alpha=self.alpha[asset],
                below_safe=bool(pre_gr[asset] < self.cfg.p_safe),
                below_plan=bool(pre_gr[asset] < self.cfg.p_plan)))
        return observations

    def advance(self, day, actions):
        pre_gr = {}
        for asset in self.p:
            self.p[asset] += (-self.alpha[asset]
                             + self.rng.normal(0, np.sqrt(self.cfg.sim_q_p)))
            pre_gr[asset] = self.p[asset]
        for asset in sorted(actions):
            self.p[asset] = self.cfg.reset_pressure
            self.cycle[asset] += 1
            self.alpha[asset] = self.draw_alpha()
            self.executions.append(dict(
                day=day, asset=asset, reset_pressure=self.p[asset]))
        return self.observe(day, actions, pre_gr), set(actions)


SCHEMAS = {
    'observations': ['day', 'asset', 'pressure_obs', 'gr_day'],
    'simulation_truth': ['day', 'asset', 'cycle', 'pre_gr_pressure', 'true_pressure', 'true_alpha', 'below_safe', 'below_plan'],
    'executions': ['day', 'asset', 'reset_pressure'],
    'state_estimates': ['day', 'asset', 'cycle', 'p_hat', 'alpha_hat', 'P_pp', 'P_aa', 'P_pa'],
    'forecasts': ['decision_day', 'asset', 'target_day', 'mean', 'variance', 'lower_bound', 'safe', 'physical_safe'],
    'plans': ['decision_day', 'task_id', 'asset', 'cycle', 'execution_day', 'nominal_day', 'status', 'ready_day', 'replanned_today'],
    'tasks': list(Task.__dataclass_fields__),
    'solver_log': ['day', 'status', 'message', 'objective', 'gap', 'solve_seconds'],
    'schedule_changes': ['day', 'task_id', 'old_day', 'new_day', 'reason'],
    'exceptions': ['day', 'asset', 'reason'],
    'plan_monitoring': ['day', 'task_id', 'asset', 'status', 'target_day', 'lower_bound', 'margin_plan', 'margin_physical'],
    'replanning_log': ['day', 'replanned', 'reasons', 'active_tasks', 'last_solve_day'],
}


def run(cfg, directory):
    cfg.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'commitment_log.csv').exists():
        raise ValueError('Output directory contains old commitment results; use a new directory.')
    ids = [f'PMEC_{j:03d}' for j in range(1, cfg.assets + 1)]
    env, ctl = Simulator(ids, cfg), Controller(ids, cfg)
    observations, executed = env.initial_observations(), set()
    status, message, last_day = 'completed', '', 0
    try:
        for day in range(cfg.days):
            last_day = day
            ctl.ingest(day, observations, executed)
            if day == cfg.days - 1:
                ctl.terminal_monitor(day)
                break
            actions = ctl.decide(day)
            observations, executed = env.advance(day + 1, actions)
    except PlanningError as exc:
        status, message = 'stopped_for_intervention', str(exc)
    tables = {
        'observations': env.observation_log,
        'simulation_truth': env.truth_log,
        'executions': env.executions,
        'state_estimates': ctl.states_log,
        'forecasts': ctl.forecasts,
        'plans': ctl.plans,
        'tasks': [asdict(task) for task in ctl.tasks.values()],
        'solver_log': ctl.solver_log,
        'schedule_changes': ctl.changes,
        'exceptions': ctl.issues,
        'plan_monitoring': ctl.monitoring,
        'replanning_log': ctl.decisions,
    }
    for name, records in tables.items():
        pd.DataFrame(records, columns=SCHEMAS[name]).to_csv(directory / f'{name}.csv', index=False)
    est = pd.DataFrame(ctl.states_log, columns=SCHEMAS['state_estimates'])
    truth = pd.DataFrame(env.truth_log, columns=SCHEMAS['simulation_truth'])
    joined = est.merge(truth, on=['day', 'asset', 'cycle']) if not est.empty else pd.DataFrame()
    executed_tasks = [task for task in ctl.tasks.values() if task.status == 'EXECUTED']
    penalty = sum(
        cfg.early_penalty * max(task.nominal_at_execution - task.executed_day, 0)
        + cfg.late_penalty * max(task.executed_day - task.nominal_at_execution, 0)
        for task in executed_tasks)
    dispatches = len({row['day'] for row in env.executions})
    future_margins = [row['margin_plan'] for row in ctl.monitoring
                      if row['day'] == last_day and row['margin_plan'] is not None]
    summary = dict(
        status=status, stop_reason=message, last_processed_day=last_day,
        commitment_enabled=False, replanning_policy=cfg.replanning_policy,
        executed_tasks=len(executed_tasks), actual_dispatches=dispatches,
        setup_cost=dispatches * cfg.setup_cost,
        executed_shift_penalty_at_execution_reference=penalty,
        total_recorded_cost=dispatches * cfg.setup_cost + penalty,
        pending_scheduled_tasks=sum(task.status == 'PROPOSED' and task.proposed_day is not None
                                    for task in ctl.tasks.values()),
        deferred_tasks=sum(task.status == 'DEFERRED' for task in ctl.tasks.values()),
        plan_date_changes=sum(row['reason'] == 'reoptimized' for row in ctl.changes),
        provisional_assignment_withdrawals=sum(row['reason'] == 'deferred_outside_horizon' for row in ctl.changes),
        actual_low_pressure_asset_days=int(truth['below_safe'].sum()),
        actual_below_plan_asset_days=int(truth['below_plan'].sum()),
        solver_calls=len(ctl.solver_log),
        total_solver_seconds=sum(row['solve_seconds'] for row in ctl.solver_log),
        days_without_solve=sum(not row['replanned'] for row in ctl.decisions),
        terminal_min_retained_plan_margin=min(future_margins) if future_margins else None,
        pressure_state_rmse=None if joined.empty else float(np.sqrt(np.mean((joined.p_hat - joined.true_pressure) ** 2))),
        alpha_state_rmse=None if joined.empty else float(np.sqrt(np.mean((joined.alpha_hat - joined.true_alpha) ** 2))),
    )
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (directory / 'config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--out', type=Path, default=Path('results_no_commitment'))
    for flag in ['days', 'assets', 'seed']:
        parser.add_argument('--' + flag, type=int)
    parser.add_argument('--policy', choices=['event', 'daily', 'periodic'])
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    # Accept the previous configuration without silently using its old window.
    if 'commitment_days' in values:
        values.pop('commitment_days')
        print('Note: commitment_days is ignored; commitments are disabled.')
    for key in ['days', 'assets', 'seed']:
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    if args.policy is not None:
        values['replanning_policy'] = args.policy
    summary = run(Config(**values), args.out)
    print(json.dumps(summary, indent=2))
    if summary['status'] != 'completed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
