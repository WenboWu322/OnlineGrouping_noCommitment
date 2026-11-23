"""Window-level grouping MILP: the Chapter 5 model plus an anchoring term.

Constraint block and objective are those of Chapter 5
(``eq:grouping-assignment`` ... ``eq:grouping-penalty-coefficient``).  The only
addition is the anchoring term of eq. (6.x) of the thesis, which prices changes
to intervention dates that have already been announced::

    min  c_set * (|N| - sum_t y_t)                      (setup expenditure)
         + sum_{n,t} pi_nt x_nt                          (time-shift penalty)
         + eta_stab * sum_{n,t} sigma_nt x_nt            (plan stability)
    with  pi_nt = eta_minus (tau_n^0 - t)  if t <= tau_n^0
                  eta_plus  (t - tau_n^0)  otherwise
          sigma_nt = |t - a_n| for announced tasks (a_n <= d + L^lock), else 0

Tasks whose announced date lies within the lock horizon are fixed instead of
penalised, which is what makes "not re-solving every day" operationally
meaningful.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class WindowSpec:
    """One grouping sub-problem, all day indices absolute."""
    tasks: list[str]
    days: list[int]
    domains: dict[str, list[int]]
    tau0: dict[str, int]
    announced: dict[str, int] = field(default_factory=dict)
    lock_until: int | None = None       # announced dates <= this are frozen
    objective: str = 'M2'               # 'M2' economic, 'M1' cluster count

    def locked(self) -> dict[str, int]:
        if self.lock_until is None:
            return {}
        return {n: t for n, t in self.announced.items() if t <= self.lock_until}

    def penalty(self, n: str, t: int, cfg) -> float:
        ref = self.tau0[n]
        return (cfg.scenario.eta_minus * (ref - t) if t <= ref
                else cfg.scenario.eta_plus * (t - ref))

    def stability(self, n: str, t: int, cfg) -> float:
        a = self.announced.get(n)
        return 0.0 if a is None else cfg.policy.eta_stab * abs(t - a)


@dataclass
class SolveResult:
    status: str                       # 'optimal' | 'time_limit' | 'infeasible' | 'error'
    assignment: dict[str, int] = field(default_factory=dict)
    clusters: list[int] = field(default_factory=list)
    objective: float | None = None
    gap: float | None = None
    seconds: float = 0.0
    model: str = ''


def solve_window(spec: WindowSpec, cfg, solver: str | None = None) -> SolveResult:
    """Solve one window with Gurobi (preferred) or SciPy's MILP branch-and-bound."""
    for n in spec.tasks:
        if not spec.domains[n]:
            return SolveResult('infeasible', model='pre-check')
    fixed = spec.locked()
    for n, t in fixed.items():
        if t not in spec.domains[n]:
            return SolveResult('infeasible', model='lock-check')

    which = solver or cfg.policy.solver
    if which in ('auto', 'gurobi'):
        try:
            res = _solve_gurobi(spec, cfg)
            if which == 'gurobi' or res.status in ('optimal', 'infeasible'):
                return res
        except ImportError:
            if which == 'gurobi':
                raise
        except Exception:
            # a restricted Gurobi licence refuses models beyond a size limit;
            # fall back to the open-source branch-and-bound instead of failing
            if which == 'gurobi':
                raise
    return _solve_scipy(spec, cfg)


# ----------------------------------------------------------------------
def _coefficients(spec: WindowSpec, cfg) -> dict[tuple[str, int], float]:
    cost = {}
    for n in spec.tasks:
        for t in spec.domains[n]:
            c = spec.penalty(n, t, cfg)
            if spec.objective == 'M2':
                c += spec.stability(n, t, cfg)
                c += cfg.scenario.c_set          # constant part handled below
            cost[(n, t)] = c
    return cost


def _solve_gurobi(spec: WindowSpec, cfg) -> SolveResult:
    import gurobipy as gp
    from gurobipy import GRB

    start = time.perf_counter()
    m = gp.Model('grouping')
    m.params.OutputFlag = 0
    m.params.TimeLimit = cfg.policy.time_limit
    m.params.MIPGap = cfg.policy.mip_gap
    x = {(n, t): m.addVar(vtype=GRB.BINARY, name=f'x_{n}_{t}')
         for n in spec.tasks for t in spec.domains[n]}
    y = {t: m.addVar(vtype=GRB.BINARY, name=f'y_{t}') for t in spec.days}

    for n in spec.tasks:
        if n in spec.locked():
            m.addConstr(x[(n, spec.locked()[n])] == 1, name=f'lock_{n}')
        else:
            m.addConstr(gp.quicksum(x[(n, t)] for t in spec.domains[n]) == 1,
                        name=f'assign_{n}')
    for t in spec.days:
        col = [x[(n, t)] for n in spec.tasks if t in spec.domains[n]]
        if col:
            m.addConstr(gp.quicksum(col) <= cfg.scenario.K_clus * y[t],
                        name=f'cap_{t}')
            m.addConstr(gp.quicksum(col) >= y[t], name=f'act_{t}')
        else:
            m.addConstr(y[t] == 0, name=f'act_{t}')

    if spec.objective == 'M1':
        m.setObjective(gp.quicksum(y.values()), GRB.MINIMIZE)
    else:
        pen = gp.quicksum(spec.penalty(n, t, cfg) * x[(n, t)]
                         for n in spec.tasks for t in spec.domains[n])
        stab = gp.quicksum(spec.stability(n, t, cfg) * x[(n, t)]
                           for n in spec.tasks for t in spec.domains[n]
                           if n in spec.announced)
        # Chapter 5 Model 2 maximises the setup-cost saving minus the shift
        # penalty, eq. (eq:m2-obj); the equivalent minimisation form charges a
        # setup cost for every activated cluster.
        m.setObjective(c_set_of(y, gp, cfg.scenario.c_set) + pen + stab,
                       GRB.MINIMIZE)

    m.optimize()
    secs = time.perf_counter() - start
    if m.Status == GRB.INFEASIBLE:
        return SolveResult('infeasible', seconds=secs, model='gurobi')
    if m.SolCount == 0:
        return SolveResult('time_limit', seconds=secs, model='gurobi')
    assign = {n: t for n in spec.tasks for t in spec.domains[n]
              if x[(n, t)].X > 0.5}
    if len(assign) != len(spec.tasks):
        return SolveResult('time_limit', seconds=secs, model='gurobi')
    clusters = [t for t in spec.days if y[t].X > 0.5]
    gap = float(m.MIPGap) if m.SolCount else None
    return SolveResult('optimal' if m.Status == GRB.OPTIMAL else 'time_limit',
                       assign, clusters, float(m.ObjVal), gap, secs, 'gurobi')


def c_set_of(y, gp, c_set: float | None = None):
    """Setup-cost term: one c_set per activated cluster."""
    total = gp.quicksum(y.values()) if isinstance(y, dict) else y.sum()
    return total if c_set in (None, 1.0) else c_set * total


def _solve_scipy(spec: WindowSpec, cfg) -> SolveResult:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix

    pairs = [(n, t) for n in spec.tasks for t in spec.domains[n]]
    pidx = {p: k for k, p in enumerate(pairs)}
    yidx = {t: len(pairs) + k for k, t in enumerate(spec.days)}
    nvars = len(pairs) + len(spec.days)

    obj = np.zeros(nvars)
    for (n, t), k in pidx.items():
        c = spec.penalty(n, t, cfg)
        if spec.objective == 'M2':
            c += spec.stability(n, t, cfg)
        obj[k] = c
    if spec.objective == 'M2':
        # one setup cost per activated cluster (minimisation form of
        # Chapter 5 eq. (eq:m2-obj))
        obj[len(pairs):] = cfg.scenario.c_set
    fixed = spec.locked()
    big = 1e6
    for n, t in fixed.items():
        obj[pidx[(n, t)]] -= big

    rows, lo, hi = [], [], []
    for n in spec.tasks:
        row = np.zeros(nvars)
        for t in spec.domains[n]:
            row[pidx[(n, t)]] = 1.0
        rows.append(row); lo.append(1.0); hi.append(1.0)
    for t in spec.days:
        row = np.zeros(nvars)
        for n in spec.tasks:
            if t in spec.domains[n]:
                row[pidx[(n, t)]] = 1.0
        row[yidx[t]] = -cfg.scenario.K_clus
        rows.append(row); lo.append(-np.inf); hi.append(0.0)
        row2 = np.zeros(nvars)
        for n in spec.tasks:
            if t in spec.domains[n]:
                row2[pidx[(n, t)]] = -1.0
        row2[yidx[t]] = 1.0
        rows.append(row2); lo.append(-np.inf); hi.append(0.0)

    A = lil_matrix(np.vstack(rows)) if rows else lil_matrix((0, nvars))
    start = time.perf_counter()
    res = milp(c=obj, integrality=np.ones(nvars),
               bounds=Bounds(np.zeros(nvars), np.ones(nvars)),
               constraints=LinearConstraint(A, np.array(lo), np.array(hi)),
               options={'time_limit': cfg.policy.time_limit,
                        'mip_rel_gap': cfg.policy.mip_gap})
    secs = time.perf_counter() - start
    if res.x is None:
        return SolveResult('infeasible' if res.status == 2 else 'time_limit',
                           seconds=secs, model='scipy')
    xv = np.rint(res.x)
    assign = {}
    for (n, t), k in pidx.items():
        if xv[k] > 0.5:
            assign[n] = t
    clusters = [t for t in spec.days if xv[yidx[t]] > 0.5]
    value = _true_objective(spec, cfg, assign, clusters)
    return SolveResult('optimal', assign, clusters, value, None, secs, 'scipy')


def _true_objective(spec: WindowSpec, cfg, assign: dict[str, int],
                    clusters: list[int]) -> float:
    if spec.objective == 'M1':
        return float(len(clusters))
    value = cfg.scenario.c_set * len(clusters)
    for n, t in assign.items():
        value += spec.penalty(n, t, cfg)
        value += spec.stability(n, t, cfg)
    return float(value)


# ----------------------------------------------------------------------
def greedy_assignment(spec: WindowSpec, cfg) -> SolveResult:
    """Chapter 5 strategy M3: earliest feasible day, capacity respecting."""
    start = time.perf_counter()
    used: dict[int, int] = {t: 0 for t in spec.days}
    assign: dict[str, int] = {}
    for n in sorted(spec.tasks, key=lambda m: (spec.tau0[m], m)):
        cands = [t for t in spec.domains[n] if used[t] < cfg.scenario.K_clus]
        if not cands:
            return SolveResult('infeasible', seconds=time.perf_counter() - start,
                               model='greedy')
        t = min(cands)
        assign[n] = t
        used[t] += 1
    clusters = [t for t in spec.days if used[t] > 0]
    return SolveResult('optimal', assign, clusters,
                       _true_objective(spec, cfg, assign, clusters), None,
                       time.perf_counter() - start, 'greedy')
