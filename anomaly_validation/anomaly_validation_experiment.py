#!/usr/bin/env python3
"""Standalone anomaly-validation experiment for online GR grouping.

Compares:
  M0: reactive 3.2-bar threshold-triggered GR policy.
  M1: online Kalman + B0 chance-constraint + MILP grouping policy.

A PMEC experiences an abrupt (or ramped) increase in leakage rate.
Both policies run in the same closed-loop simulator with the same seeds.

Outputs:
  - config.json
  - runs/<scenario>_<seed>_<strategy>/... (detailed logs for main scenario)
  - aggregate_results.csv
  - paired_comparison.csv
  - figures/*.png
  - report.md
"""
from __future__ import annotations
import argparse, json, math, time
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from scipy.stats import norm, ttest_rel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ExpConfig:
    assets: int = 12
    days: int = 180
    base_seed: int = 42
    n_seeds: int = 10
    horizon: int = 30
    preparation_days: int = 1
    max_advance_days: int = 5
    capacity: int = 4
    p_trigger: float = 3.2
    p_safe: float = 3.0
    p_plan: float = 3.05
    epsilon: float = 0.05
    R: float = 0.0001
    q_p: float = 0.000001
    q_alpha: float = 0.000001
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
    max_replan_interval: int = 7
    risk_trigger_margin: float = 0.0
    nominal_shift_trigger_days: int = 2
    reset_pressure: float = 3.5
    reset_pressure_var: float = 0.0
    time_limit: float = 10.0
    mip_gap: float = 0.001
    # anomaly scenario
    anomaly_asset_index: int = 6
    anomaly_start_day: int = 50
    anomaly_alpha: float = 0.022
    anomaly_ramp_days: int = 0
    # experiment
    anomaly_magnitudes: list = field(default_factory=lambda: [0.012, 0.018, 0.022, 0.026, 0.030])
    anomaly_onset_day: int = 50

    def validate(self):
        assert self.assets >= 2 and self.days >= 30 and self.capacity >= 1
        assert self.p_safe <= self.p_plan < self.p_trigger < self.reset_pressure
        assert 0 < self.epsilon < 0.5
        assert self.sim_alpha_min <= self.sim_alpha_max
        assert self.anomaly_alpha > self.sim_alpha_max, "anomaly alpha should exceed normal range"


@dataclass
class FilterState:
    mean: np.ndarray | None = None
    cov: np.ndarray | None = None
    last_day: int = -1
    cycle: int = 0

    def process(self, day, obs, executed, cfg):
        F = np.array([[1., -1.], [0., 1.]])
        H = np.array([1., 0.])
        Q = np.diag([cfg.q_p, cfg.q_alpha])
        for d in range(self.last_day + 1, day + 1):
            if d == day and executed:
                self.cycle += 1
                self.mean = np.array([cfg.reset_pressure, cfg.alpha_prior_mean])
                self.cov = np.diag([cfg.reset_pressure_var, cfg.alpha_prior_var])
                continue
            y = obs if d == day else None
            if self.mean is None:
                if y is not None:
                    self.mean = np.array([float(y), cfg.alpha_prior_mean])
                    self.cov = np.diag([cfg.R, cfg.alpha_prior_var])
                continue
            m = F @ self.mean
            P = F @ self.cov @ F.T + Q
            if y is not None:
                resid = float(y - H @ m)
                S = float(H @ P @ H + cfg.R)
                K = P @ H / S
                m = m + K * resid
                A = np.eye(2) - np.outer(K, H)
                P = A @ P @ A.T + cfg.R * np.outer(K, K)
            self.mean = m
            self.cov = (P + P.T) / 2
        self.last_day = day


def forecast(state, h, cfg):
    h = int(h)
    v = np.array([1., -float(h)])
    mu = float(v @ state.mean)
    var = float(v @ state.cov @ v + h * cfg.q_p
                + h * (h - 1) * (2 * h - 1) / 6 * cfg.q_alpha)
    return mu, max(0., var)


def reference_date(state, day, cfg):
    p, alpha = state.mean
    if p <= cfg.p_trigger:
        return day
    return day + int(np.ceil((p - cfg.p_trigger) / alpha))


@dataclass
class Task:
    task_id: str
    asset: str
    cycle: int
    created_day: int
    ready_day: int
    nominal_day: int
    status: str = "PROPOSED"
    proposed_day: int | None = None
    nominal_at_last_solve: int | None = None
    executed_day: int | None = None
    nominal_at_execution: int | None = None
    emergency: bool = False


class Simulator:
    def __init__(self, ids, cfg, seed):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.ids = ids
        self.p = {a: self.rng.uniform(3.32, 3.5) for a in ids}
        self.alpha = {a: self.rng.uniform(cfg.sim_alpha_min, cfg.sim_alpha_max) for a in ids}
        self.cycle = {a: 0 for a in ids}
        self.anomaly_active = {a: False for a in ids}
        self.anomaly_gr_done = False
        self.anomaly_asset = ids[min(cfg.anomaly_asset_index, len(ids) - 1)]
        # ensure the anomalous asset survives until the anomaly onset
        self.p[self.anomaly_asset] = cfg.reset_pressure
        self.alpha[self.anomaly_asset] = cfg.sim_alpha_min
        self.obs_log, self.truth_log, self.exec_log = [], [], []

    def _eff_alpha(self, asset, day):
        a = self.alpha[asset]
        if (asset == self.anomaly_asset and not self.anomaly_gr_done
                and day >= self.cfg.anomaly_start_day):
            if self.cfg.anomaly_ramp_days > 0:
                frac = min(1.0, (day - self.cfg.anomaly_start_day) / self.cfg.anomaly_ramp_days)
                a = self.alpha[asset] * (1 - frac) + self.cfg.anomaly_alpha * frac
            else:
                a = self.cfg.anomaly_alpha
        return a

    def initial_observations(self):
        return self.observe(0, set(), self.p.copy(), force=True)

    def observe(self, day, executed, pre_gr, force=False):
        obs = {}
        for a in self.ids:
            y = self.p[a] + self.rng.normal(0, np.sqrt(self.cfg.sim_R))
            if not force and self.rng.random() < self.cfg.missing_probability:
                y = None
            if a in executed:
                y = None
            obs[a] = y
            self.obs_log.append(dict(day=day, asset=a, pressure_obs=y, gr_day=a in executed))
            self.truth_log.append(dict(
                day=day, asset=a, cycle=self.cycle[a],
                pre_gr_pressure=pre_gr[a], true_pressure=self.p[a],
                true_alpha=self.alpha[a],
                below_safe=bool(pre_gr[a] < self.cfg.p_safe),
                below_plan=bool(pre_gr[a] < self.cfg.p_plan)))
        return obs

    def advance(self, day, actions):
        pre_gr = {}
        for a in self.ids:
            if a in actions:
                self.p[a] = self.cfg.reset_pressure
                self.cycle[a] += 1
                self.alpha[a] = self.rng.uniform(self.cfg.sim_alpha_min, self.cfg.sim_alpha_max)
                self.anomaly_active[a] = False
                if a == self.anomaly_asset:
                    self.anomaly_gr_done = True
                self.exec_log.append(dict(day=day, asset=a, reset_pressure=self.p[a]))
            else:
                self.p[a] += (-self._eff_alpha(a, day)
                              + self.rng.normal(0, np.sqrt(self.cfg.sim_q_p)))
            pre_gr[a] = self.p[a]
        return self.observe(day, actions, pre_gr), set(actions)


class ThresholdController:
    """M0: reactive 3.2-bar trigger, greedy earliest-safe-day execution."""
    def __init__(self, ids, cfg):
        self.cfg = cfg
        self.states = {a: FilterState() for a in ids}
        self.scheduled = {}
        self.cap_used = {}
        self.est_log, self.plan_log, self.dec_log = [], [], []

    def ingest(self, day, obs, executed):
        for a, st in self.states.items():
            st.process(day, obs.get(a), a in executed, self.cfg)
            if st.mean is not None:
                self.est_log.append(dict(day=day, asset=a, cycle=st.cycle,
                                         p_hat=st.mean[0], alpha_hat=st.mean[1]))
        for a, d in list(self.scheduled.items()):
            if d == day and a not in executed:
                raise RuntimeError(f"baseline scheduled GR for {a} on {day} not executed")
        for a in executed:
            self.scheduled.pop(a, None)

    def decide(self, day):
        cfg = self.cfg
        candidates = []
        for a, st in self.states.items():
            if a in self.scheduled:
                continue
            if st.mean is None:
                continue
            mu, var = forecast(st, 0, cfg)
            if mu <= cfg.p_trigger:
                candidates.append((a, mu, var))
        candidates.sort(key=lambda x: x[1])
        actions = set()
        for a, mu, var in candidates:
            # find earliest day with capacity and B0-safe
            assigned = None
            emergency = False
            for d in range(day + 1, day + cfg.horizon + 1):
                if self.cap_used.get(d, 0) >= cfg.capacity:
                    continue
                mu_d, var_d = forecast(self.states[a], d - day, cfg)
                lb = mu_d - norm.ppf(1 - cfg.epsilon) * np.sqrt(var_d)
                if lb >= cfg.p_safe:
                    assigned = d
                    break
            if assigned is None:
                assigned = day + 1
                emergency = True
            self.scheduled[a] = assigned
            self.cap_used[assigned] = self.cap_used.get(assigned, 0) + 1
            self.plan_log.append(dict(day=day, asset=a, trigger_day=day,
                                      execution_day=assigned, emergency=emergency,
                                      nominal_day=day))
            if assigned == day + 1:
                actions.add(a)
        self.dec_log.append(dict(day=day, new_triggers=len(candidates),
                                 scheduled=len(self.scheduled)))
        return actions


class OnlineGroupingController:
    """M1: online Kalman + B0 + MILP grouping."""
    def __init__(self, ids, cfg):
        self.cfg = cfg
        self.states = {a: FilterState() for a in ids}
        self.tasks = {}
        self.last_solve_day = None
        self.kappa = float(norm.ppf(1 - cfg.epsilon))
        self.est_log, self.forecast_log, self.plan_log = [], [], []
        self.solver_log, self.change_log, self.dec_log = [], [], []

    def ingest(self, day, obs, executed):
        for a, st in self.states.items():
            st.process(day, obs.get(a), a in executed, self.cfg)
            if st.mean is not None:
                self.est_log.append(dict(day=day, asset=a, cycle=st.cycle,
                                         p_hat=st.mean[0], alpha_hat=st.mean[1]))
        for t in self.tasks.values():
            if t.status == "PROPOSED" and t.proposed_day == day and t.asset not in executed:
                raise RuntimeError(f"online scheduled GR for {t.asset} on {day} not executed")
            if t.status == "PROPOSED" and t.proposed_day == day and t.asset in executed:
                t.status = "EXECUTED"
                t.executed_day = day
                t.nominal_at_execution = t.nominal_day

    def _grid(self, day, st):
        cfg = self.cfg
        grid = {}
        for d in range(day + 1, day + cfg.horizon + 1):
            mu, var = forecast(st, d - day, cfg)
            lb = mu - self.kappa * np.sqrt(var)
            grid[d] = (mu, var, lb)
            self.forecast_log.append(dict(decision_day=day, asset=st.cycle, target_day=d,
                                          mean=mu, variance=var, lower_bound=lb))
        return grid

    def decide(self, day):
        cfg = self.cfg
        tasks, domains = [], {}
        for a, st in self.states.items():
            if st.mean is None:
                continue
            nominal = reference_date(st, day, cfg)
            key = f"{a}:cycle{st.cycle}"
            task = self.tasks.get(key)
            if task and task.status == "EXECUTED":
                continue
            grid = self._grid(day, st)
            nominal_near = nominal - cfg.max_advance_days <= day + cfg.horizon
            risk_near = any(lb < cfg.p_plan for _, _, lb in grid.values())
            if task is None and (nominal_near or risk_near):
                task = Task(key, a, st.cycle, day, day + cfg.preparation_days, nominal)
                self.tasks[key] = task
            if task is None:
                continue
            if nominal <= day + cfg.horizon:
                earliest = max(day + 1, task.ready_day, nominal - cfg.max_advance_days)
            else:
                earliest = max(day + 1, task.ready_day)
            feas = [d for d in grid if d >= earliest and grid[d][2] >= cfg.p_plan]
            if not feas:
                feas_safe = [d for d in grid
                             if d >= max(day + 1, task.ready_day)
                             and grid[d][2] >= cfg.p_safe]
                if feas_safe:
                    feas = feas_safe
                else:
                    task.emergency = True
                    feas = [day + 1]
            tasks.append(task)
            domains[key] = feas
        # replan check
        reasons = []
        if self.last_solve_day is None:
            reasons.append("initial")
        elif day - self.last_solve_day >= cfg.max_replan_interval:
            reasons.append("max_interval")
        counts = {}
        for t in tasks:
            if t.proposed_day is None:
                reasons.append("unassigned")
                continue
            mu, var = forecast(self.states[t.asset], t.proposed_day - day, cfg)
            lb = mu - self.kappa * np.sqrt(var)
            margin = lb - cfg.p_plan
            self.plan_log.append(dict(day=day, task_id=t.task_id, asset=t.asset,
                                      status=t.status, target_day=t.proposed_day,
                                      lower_bound=lb, margin_plan=margin))
            if t.proposed_day not in domains[t.task_id]:
                reasons.append("infeasible_retained")
            if margin <= cfg.risk_trigger_margin:
                reasons.append("risk_margin")
            if (t.nominal_at_last_solve is not None and
                    abs(t.nominal_day - t.nominal_at_last_solve) >= cfg.nominal_shift_trigger_days):
                reasons.append("nominal_shift")
            counts[t.proposed_day] = counts.get(t.proposed_day, 0) + 1
        if any(n > cfg.capacity for n in counts.values()):
            reasons.append("capacity_violation")
        replanned = False
        if tasks and reasons:
            assign = self._solve(day, tasks, domains)
            for t in tasks:
                old = t.proposed_day
                new = assign[t.task_id]
                if old is not None and old != new:
                    self.change_log.append(dict(day=day, task_id=t.task_id,
                                                old_day=old, new_day=new, reason="reoptimized"))
                t.proposed_day = new
                t.nominal_at_last_solve = t.nominal_day
            self.last_solve_day = day
            replanned = True
        # finalize: validate capacity
        counts = {}
        for t in tasks:
            if t.proposed_day is None:
                raise RuntimeError("unassigned task at finalize")
            counts[t.proposed_day] = counts.get(t.proposed_day, 0) + 1
        if any(n > cfg.capacity for n in counts.values()):
            raise RuntimeError("capacity exceeded after finalize")
        actions = {t.asset for t in tasks
                   if t.status == "PROPOSED" and t.proposed_day == day + 1}
        self.dec_log.append(dict(day=day, replanned=replanned,
                                 reasons=";".join(reasons) if tasks else "no_tasks",
                                 active=len(tasks)))
        return actions

    def _solve(self, day, tasks, domains):
        cfg = self.cfg
        pairs = [(t.task_id, d) for t in tasks for d in domains[t.task_id]]
        dates = sorted({d for _, d in pairs})
        xi = {p: j for j, p in enumerate(pairs)}
        zi = {d: len(pairs) + j for j, d in enumerate(dates)}
        n = len(pairs) + len(dates)
        obj = np.zeros(n)
        for t in tasks:
            for d in domains[t.task_id]:
                obj[xi[t.task_id, d]] = (cfg.early_penalty * max(t.nominal_day - d, 0)
                                         + cfg.late_penalty * max(d - t.nominal_day, 0))
        for d in dates:
            obj[zi[d]] = cfg.setup_cost
        rows, cols, vals, lo, hi = [], [], [], [], []
        def add(coefs, low, high):
            r = len(lo)
            for j, v in coefs.items():
                rows.append(r); cols.append(j); vals.append(v)
            lo.append(low); hi.append(high)
        for t in tasks:
            add({xi[t.task_id, d]: 1. for d in domains[t.task_id]}, 1., 1.)
        for d in dates:
            assign = {xi[k, dd]: 1. for k, dd in pairs if dd == d}
            add(assign, -np.inf, float(cfg.capacity))
            add(assign, -np.inf, np.inf)  # placeholder, replaced below
        # rebuild properly: need z_d linking
        # redo constraints from scratch
        rows, cols, vals, lo, hi = [], [], [], [], []
        def add2(coefs, low, high):
            r = len(lo)
            for j, v in coefs.items():
                rows.append(r); cols.append(j); vals.append(v)
            lo.append(low); hi.append(high)
        for t in tasks:
            add2({xi[t.task_id, d]: 1. for d in domains[t.task_id]}, 1., 1.)
        for d in dates:
            assign = {xi[k, dd]: 1. for k, dd in pairs if dd == d}
            # sum_assign <= capacity * z_d
            coefs = dict(assign)
            coefs[zi[d]] = -float(cfg.capacity)
            add2(coefs, -np.inf, 0.)
            # sum_assign >= z_d
            coefs = {j: -v for j, v in assign.items()}
            coefs[zi[d]] = 1.
            add2(coefs, -np.inf, 0.)
        A = coo_matrix((vals, (rows, cols)), shape=(len(lo), n)).tocsc()
        start = time.perf_counter()
        res = milp(obj, integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)),
                   constraints=LinearConstraint(A, lo, hi),
                   options={"time_limit": cfg.time_limit, "mip_rel_gap": cfg.mip_gap})
        elapsed = time.perf_counter() - start
        self.solver_log.append(dict(day=day, status=int(res.status),
                                    message=str(res.message),
                                    obj=None if res.fun is None else float(res.fun),
                                    seconds=elapsed))
        if res.x is None:
            raise RuntimeError(f"MILP failed on day {day}: {res.message}")
        rounded = np.rint(res.x)
        assign = {}
        for t in tasks:
            chosen = [d for d in domains[t.task_id] if rounded[xi[t.task_id, d]] == 1]
            if len(chosen) != 1:
                raise RuntimeError("invalid assignment count")
            assign[t.task_id] = chosen[0]
        return assign


def run_single(cfg, seed, strategy, outdir=None, detailed=False):
    ids = [f"PMEC_{j:03d}" for j in range(1, cfg.assets + 1)]
    env = Simulator(ids, cfg, seed)
    ctl = ThresholdController(ids, cfg) if strategy == "baseline" else OnlineGroupingController(ids, cfg)
    obs, executed = env.initial_observations(), set()
    status, msg = "completed", ""
    try:
        for day in range(cfg.days):
            ctl.ingest(day, obs, executed)
            if day == cfg.days - 1:
                break
            actions = ctl.decide(day)
            obs, executed = env.advance(day + 1, actions)
    except Exception as exc:
        status, msg = "stopped", str(exc)
    truth = pd.DataFrame(env.truth_log)
    execs = pd.DataFrame(env.exec_log)
    summary = dict(
        strategy=strategy, seed=seed, status=status, message=msg,
        anomaly_alpha=cfg.anomaly_alpha, anomaly_start_day=cfg.anomaly_start_day,
        anomaly_asset=env.anomaly_asset,
        executed_tasks=len(execs),
        dispatch_days=execs["day"].nunique() if not execs.empty else 0,
        setup_cost=(execs["day"].nunique() if not execs.empty else 0) * cfg.setup_cost,
        below_safe_asset_days=int(truth["below_safe"].sum()),
        below_plan_asset_days=int(truth["below_plan"].sum()),
        min_true_pressure=float(truth["pre_gr_pressure"].min()),
        total_recorded_cost=0.0,  # filled below
    )
    # shift penalty
    if strategy == "baseline":
        plan = pd.DataFrame(ctl.plan_log) if ctl.plan_log else pd.DataFrame()
        if not plan.empty:
            exec_map = dict(zip(execs["asset"], execs["day"])) if not execs.empty else {}
            pen = sum(cfg.late_penalty * max(row["execution_day"] - row["trigger_day"], 0)
                      for _, row in plan.iterrows())
            summary["total_recorded_cost"] = summary["setup_cost"] + pen
            summary["executed_shift_penalty"] = pen
        else:
            summary["executed_shift_penalty"] = 0
            summary["total_recorded_cost"] = summary["setup_cost"]
    else:
        tasks = [t for t in ctl.tasks.values() if t.status == "EXECUTED"]
        pen = sum(cfg.early_penalty * max(t.nominal_at_execution - t.executed_day, 0)
                  + cfg.late_penalty * max(t.executed_day - t.nominal_at_execution, 0)
                  for t in tasks)
        summary["executed_shift_penalty"] = pen
        summary["total_recorded_cost"] = summary["setup_cost"] + pen
    # anomaly-specific metrics
    anom = env.anomaly_asset
    truth_anom = truth[truth["asset"] == anom]
    exec_anom = execs[execs["asset"] == anom] if not execs.empty else pd.DataFrame()
    summary["anomaly_gr_day"] = int(exec_anom["day"].min()) if not exec_anom.empty else None
    summary["anomaly_gr_count"] = len(exec_anom)
    if strategy == "baseline":
        plan = pd.DataFrame(ctl.plan_log) if ctl.plan_log else pd.DataFrame()
        if not plan.empty:
            ap = plan[plan["asset"] == anom]
            summary["anomaly_detection_day"] = int(ap["trigger_day"].min()) if not ap.empty else None
        else:
            summary["anomaly_detection_day"] = None
    else:
        created = [t for t in ctl.tasks.values()
                   if t.asset == anom and t.created_day >= cfg.anomaly_start_day]
        summary["anomaly_detection_day"] = int(min(t.created_day for t in created)) if created else None
    if summary["anomaly_detection_day"] is not None:
        summary["anomaly_detection_lead"] = summary["anomaly_detection_day"] - cfg.anomaly_start_day
    else:
        summary["anomaly_detection_lead"] = None
    if summary["anomaly_gr_day"] is not None:
        summary["anomaly_gr_lead"] = summary["anomaly_gr_day"] - cfg.anomaly_start_day
    else:
        summary["anomaly_gr_lead"] = None
    # emergency count
    if strategy == "baseline":
        plan = pd.DataFrame(ctl.plan_log) if ctl.plan_log else pd.DataFrame()
        summary["emergency_gr_count"] = int(plan["emergency"].sum()) if not plan.empty else 0
    else:
        summary["emergency_gr_count"] = sum(1 for t in ctl.tasks.values() if t.emergency)
    # grouping rate
    if summary["executed_tasks"] > 0:
        summary["tasks_per_dispatch"] = summary["executed_tasks"] / summary["dispatch_days"]
    else:
        summary["tasks_per_dispatch"] = 0.0
    if detailed and outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        truth.to_csv(outdir / "simulation_truth.csv", index=False)
        pd.DataFrame(env.obs_log).to_csv(outdir / "observations.csv", index=False)
        execs.to_csv(outdir / "executions.csv", index=False)
        pd.DataFrame(ctl.est_log).to_csv(outdir / "state_estimates.csv", index=False)
        if hasattr(ctl, "plan_log"):
            pd.DataFrame(ctl.plan_log).to_csv(outdir / "plans.csv", index=False)
        if hasattr(ctl, "dec_log"):
            pd.DataFrame(ctl.dec_log).to_csv(outdir / "decisions.csv", index=False)
        if hasattr(ctl, "solver_log"):
            pd.DataFrame(ctl.solver_log).to_csv(outdir / "solver_log.csv", index=False)
        if hasattr(ctl, "change_log"):
            pd.DataFrame(ctl.change_log).to_csv(outdir / "changes.csv", index=False)
        if hasattr(ctl, "forecast_log"):
            pd.DataFrame(ctl.forecast_log).to_csv(outdir / "forecasts.csv", index=False)
    return summary, env, ctl


def make_plots(results_dir, detailed_run_baseline, detailed_run_online, cfg):
    fig_dir = Path(results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    anom = detailed_run_baseline[1].anomaly_asset
    # pressure trajectory plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for env, ctl, label, color in [
        (detailed_run_baseline[1], detailed_run_baseline[2], "M0 baseline", "tab:red"),
        (detailed_run_online[1], detailed_run_online[2], "M1 online grouping", "tab:blue")]:
        truth = pd.DataFrame(env.truth_log)
        t = truth[truth["asset"] == anom]
        ax.plot(t["day"], t["true_pressure"], label=f"{label} true pressure", color=color)
        execs = pd.DataFrame(env.exec_log)
        e = execs[execs["asset"] == anom]
        if not e.empty:
            ax.scatter(e["day"], [3.5] * len(e), marker="v", color=color, s=80,
                       label=f"{label} GR", zorder=5)
    ax.axhline(cfg.p_trigger, color="orange", ls="--", label="3.2 bar trigger")
    ax.axhline(cfg.p_plan, color="green", ls="--", label="3.05 bar plan")
    ax.axhline(cfg.p_safe, color="red", ls="--", label="3.0 bar safe")
    ax.axvline(cfg.anomaly_start_day, color="purple", ls=":", label="anomaly onset")
    ax.set_xlabel("Day"); ax.set_ylabel("Pressure (bar)")
    ax.set_title(f"Anomalous PMEC ({anom}) pressure trajectory")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "pressure_trajectory.png", dpi=150)
    plt.close(fig)
    # bar chart across magnitudes
    agg = pd.read_csv(Path(results_dir) / "aggregate_results.csv")
    mags = sorted(agg["anomaly_alpha"].unique())
    metrics = ["below_safe_asset_days", "dispatch_days", "total_recorded_cost"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, metrics):
        for strat, color in [("baseline", "tab:red"), ("online", "tab:blue")]:
            sub = agg[agg["strategy"] == strat].groupby("anomaly_alpha")[m].mean()
            ax.plot(mags, [sub.get(x, np.nan) for x in mags], marker="o", color=color, label=strat)
        ax.set_xlabel("Anomaly alpha (bar/day)")
        ax.set_title(m)
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "metric_sweep.png", dpi=150)
    plt.close(fig)


def generate_report(results_dir, cfg, agg, paired):
    results_dir = Path(results_dir)
    lines = []
    A = lines.append
    A("# 异常 PMEC 情景下在线 GR 分组方法验证报告\n")
    A("## 1. 实验目的\n")
    A("本实验旨在验证：当某一台 PMEC 出现突发泄漏加速（异常）时，基于在线 Kalman 状态估计、"
      "B0 概率安全约束与 MILP 任务组合的维护策略（M1），相较于传统的“压力达到 3.2 bar 即触发 GR”"
      "响应式策略（M0），能否在降低低压风险的同时减少紧急出勤并保持任务组合效率。\n")
    A("## 2. 系统与异常模型\n")
    A(f"- PMEC 数量：{cfg.assets} 台，仿真天数：{cfg.days} 天。")
    A(f"- 正常泄漏率范围：{cfg.sim_alpha_min}–{cfg.sim_alpha_max} bar/day。")
    A(f"- 异常 PMEC：{results_dir / 'config.json'} 中 `anomaly_asset_index` 对应的设备，"
      f"异常起始日：第 {cfg.anomaly_start_day} 天，异常泄漏率：{cfg.anomaly_alpha} bar/day，"
      f"渐变天数：{cfg.anomaly_ramp_days} 天。")
    A("- 异常持续到该 PMEC 的下一次 GR；GR 后泄漏率恢复为正常抽样。")
    A(f"- 压力阈值：触发 3.2 bar，计划 3.05 bar，安全 3.0 bar；GR 后压力重置为 {cfg.reset_pressure} bar。\n")
    A("## 3. 被比较策略\n")
    A("| 策略 | 决策规则 | 说明 |")
    A("|---|---|---|")
    A("| M0 基线 | 滤波均值 ≤ 3.2 bar 即触发，安排最早安全可行日执行，容量不足则排队 | 响应式，无前瞻 |")
    A("| M1 在线分组 | 每日更新 Kalman，预测均值接近 3.2 bar 或 B0 下界接近 3.05 bar 时创建任务，MILP 组合排程 | 前瞻、风险感知、可组合 |")
    A("\n## 4. 实验设计\n")
    A(f"- 异常幅度扫描：{cfg.anomaly_magnitudes} bar/day。")
    A(f"- 异常起始日固定为第 {cfg.anomaly_onset_day} 天。")
    A(f"- 每个情景 × 每个策略运行 {cfg.n_seeds} 个随机种子（配对比较）。")
    A("- 所有策略共享相同的初始条件、观测噪声、缺测概率、GR 重置规则与容量约束。")
    A("- 容量约束：每日最多 " + str(cfg.capacity) + " 次 GR。\n")
    A("## 5. 评价指标\n")
    A("| 类别 | 指标 | 定义 |")
    A("|---|---|---|")
    A("| 安全 | below_safe_asset_days | 真实压力低于 3.0 bar 的设备—日数 |")
    A("| 安全 | below_plan_asset_days | 真实压力低于 3.05 bar 的设备—日数 |")
    A("| 安全 | min_true_pressure | 全期真实压力最低值 |")
    A("| 及时性 | anomaly_detection_lead | 方法首次创建/触发异常任务日 − 异常开始日 |")
    A("| 及时性 | anomaly_gr_lead | 异常 PMEC 的 GR 执行日 − 异常开始日 |")
    A("| 经济 | dispatch_days | 实际出勤日期数 |")
    A("| 经济 | setup_cost | 出勤次数 × 单次出勤成本 |")
    A("| 经济 | executed_shift_penalty | 实际执行日相对名义参考日的早晚惩罚 |")
    A("| 经济 | total_recorded_cost | setup 成本 + 移动惩罚 |")
    A("| 组合 | tasks_per_dispatch | 每次出勤平均完成的 GR 数 |")
    A("| 稳健性 | emergency_gr_count | 因无安全可行日而被迫紧急执行的 GR 数 |")
    A("| 算法 | solver 次数/耗时 | MILP 调用次数与总耗时（仅 M1） |")
    A("\n## 6. 实验结果\n")
    A("### 6.1 聚合结果（各情景 × 策略，跨种子均值）\n")
    A(agg.to_markdown(index=False))
    A("\n### 6.2 配对比较（M1 − M0，同一情景同一种子）\n")
    A(paired.to_markdown(index=False))
    A("\n### 6.3 图表\n")
    A("![压力轨迹](figures/pressure_trajectory.png)")
    A("![指标扫描](figures/metric_sweep.png)")
    A("\n## 7. 结果分析\n")
    A("- **安全性**：若 M1 的 `below_safe_asset_days` 显著低于 M0，说明在线方法通过提前识别异常趋势并预留安全执行窗口，有效避免了压力跌破安全下限。")
    A("- **及时性**：M1 的 `anomaly_detection_lead` 应为正且大于 M0 的对应值，说明方法在异常发生后更早进入计划阶段。")
    A("- **经济性**：若 M1 的 `dispatch_days` 与 `total_recorded_cost` 不高于 M0，说明提前规划并未以更多出勤为代价；在多任务场景下，组合效应应进一步降低出勤。")
    A("- **组合效应**：`tasks_per_dispatch` 越高，说明单次出勤完成的 GR 越多，组合收益越明显。")
    A("- **稳健性**：随着异常幅度增大，若 M0 的安全指标恶化而 M1 保持稳定，说明方法对异常强度具有鲁棒性。")
    A("- **代价**：M1 可能产生更多计划重排（`changes.csv`），这是动态响应的固有代价，应在报告中透明呈现。\n")
    A("## 8. 局限性与后续工作\n")
    A("- 当前异常模型为单设备、单次突变/渐变；后续可扩展为多设备同时异常、周期性异常或随机异常概率。")
    A("- Kalman 假设高斯噪声与线性退化；真实系统可能存在厚尾噪声或非线性退化，需做模型失配敏感性分析。")
    A("- 容量约束为每日硬上限；实际中可能存在多班组、多站点等更复杂的资源约束。")
    A("- 建议后续在真实历史数据上做 out-of-sample 回测，验证预测区间校准与 B0 风险解释。\n")
    (results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("anomaly_validation_results"))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--assets", type=int)
    p.add_argument("--days", type=int)
    p.add_argument("--seeds", type=int)
    args = p.parse_args()
    cfg = ExpConfig()
    if args.assets: cfg.assets = args.assets
    if args.days: cfg.days = args.days
    if args.seeds: cfg.n_seeds = args.seeds
    if args.quick:
        cfg.n_seeds = 2
        cfg.anomaly_magnitudes = [0.022]
        cfg.days = min(cfg.days, 120)
    cfg.validate()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    runs_dir = out / "runs"
    runs_dir.mkdir(exist_ok=True)
    all_summaries = []
    detailed_baseline = detailed_online = None
    for mag in cfg.anomaly_magnitudes:
        for seed in range(cfg.n_seeds):
            for strategy in ["baseline", "online"]:
                c = cfg
                c.anomaly_alpha = mag
                detailed = (mag == cfg.anomaly_magnitudes[0] and seed == 0)
                run_dir = runs_dir / f"alpha{mag}_seed{seed}_{strategy}" if detailed else None
                summary, env, ctl = run_single(c, cfg.base_seed + seed, strategy,
                                               outdir=run_dir, detailed=detailed)
                summary["anomaly_alpha"] = mag
                summary["seed"] = seed
                all_summaries.append(summary)
                if detailed and strategy == "baseline":
                    detailed_baseline = (summary, env, ctl)
                if detailed and strategy == "online":
                    detailed_online = (summary, env, ctl)
    agg = pd.DataFrame(all_summaries)
    agg.to_csv(out / "aggregate_results.csv", index=False)
    # paired comparison
    rows = []
    metrics = ["below_safe_asset_days", "below_plan_asset_days", "dispatch_days",
               "total_recorded_cost", "anomaly_detection_lead", "anomaly_gr_lead",
               "tasks_per_dispatch", "emergency_gr_count"]
    for mag in cfg.anomaly_magnitudes:
        for seed in range(cfg.n_seeds):
            b = agg[(agg["anomaly_alpha"] == mag) & (agg["seed"] == seed) & (agg["strategy"] == "baseline")]
            o = agg[(agg["anomaly_alpha"] == mag) & (agg["seed"] == seed) & (agg["strategy"] == "online")]
            if b.empty or o.empty:
                continue
            row = {"anomaly_alpha": mag, "seed": seed}
            for m in metrics:
                row[f"{m}_baseline"] = b.iloc[0][m]
                row[f"{m}_online"] = o.iloc[0][m]
                row[f"{m}_diff"] = o.iloc[0][m] - b.iloc[0][m]
            rows.append(row)
    paired = pd.DataFrame(rows)
    paired.to_csv(out / "paired_comparison.csv", index=False)
    # statistical test for key metrics
    stat_rows = []
    for mag in cfg.anomaly_magnitudes:
        for m in metrics:
            b = paired[(paired["anomaly_alpha"] == mag)][f"{m}_baseline"].dropna()
            o = paired[(paired["anomaly_alpha"] == mag)][f"{m}_online"].dropna()
            if len(b) > 1 and len(o) > 1:
                t, pv = ttest_rel(o, b)
                stat_rows.append(dict(anomaly_alpha=mag, metric=m,
                                      mean_baseline=b.mean(), mean_online=o.mean(),
                                      mean_diff=(o - b).mean(), t_stat=t, p_value=pv))
    pd.DataFrame(stat_rows).to_csv(out / "statistical_tests.csv", index=False)
    make_plots(out, detailed_baseline, detailed_online, cfg)
    generate_report(out, cfg, agg, paired)
    print(json.dumps({"status": "done", "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
