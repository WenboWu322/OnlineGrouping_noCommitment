"""Regression tests for the no-commitment maintenance controller."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from maintenance_system_no_commitment import (
    Config, FilterState, Task, Controller, PlanningError, forecast, run)


def main():
    checks=[]
    cfg=Config()
    cfg.validate()
    assert not hasattr(cfg,'commitment_days')
    assert 'committed_day' not in Task.__dataclass_fields__
    checks.append('No commitment config/state')

    s=FilterState(np.array([3.4,.006]),np.array([[1e-4,-2e-6],[-2e-6,4e-6]]))
    F=np.array([[1.,-1.],[0.,1.]])
    mean,P=s.mean.copy(),s.covariance.copy()
    for h in range(31):
        mu,var=forecast(s,h,cfg)
        assert np.isclose(mu,mean[0],atol=1e-12)
        assert np.isclose(var,P[0,0],rtol=1e-12,atol=1e-15)
        mean=F@mean
        P=F@P@F.T+np.diag([cfg.q_p,cfg.q_alpha])
    checks.append('Forecast variance matches recursive covariance for h=0..30')

    s=FilterState()
    s.process(0,3.4,False,cfg)
    assert np.isclose(s.covariance[0,0],cfg.R)
    s.process(1,None,False,cfg)
    s.process(2,2.5,True,cfg)
    assert s.cycle==1 and s.mean[0]==cfg.reset_pressure
    checks.append('Single-use initialization, missing observation and GR reset')

    ctl=Controller(['A'],cfg)
    t=Task('A:cycle0','A',0,0,1,5,proposed_day=1)
    assignment=ctl.optimize(0,[t],{t.task_id:[2]})
    ctl.apply_plan(0,[t],assignment)
    assert t.proposed_day==2 and t.status=='PROPOSED'
    checks.append('Imminent assignment can move without a fixed-date constraint')

    ctl=Controller(['A'],cfg)
    ctl.states['A']=FilterState(np.array([3.2,.01]),np.diag([1e-8,1e-10]),last_day=1)
    t=Task('A:cycle0','A',0,0,2,1,proposed_day=20)
    ctl.tasks[t.task_id]=t;ctl.last_solve_day=0
    actions=ctl.decide(1)
    assert t.proposed_day==2 and actions=={'A'}
    assert 'retained_assignment_infeasible' in ctl.decisions[-1]['reasons']
    checks.append('Unsafe retained date is replanned when alternate dates exist')

    old_reference=t.nominal_day
    ctl.ingest(2,{'A':None},{'A'})
    assert t.status=='EXECUTED' and t.nominal_at_execution==old_reference
    assert ctl.states['A'].cycle==1
    checks.append('Execution matched to proposed_day and pre-execution reference frozen')

    ctl=Controller(['A'],cfg)
    ctl.states['A']=FilterState(np.array([3.2,.01]),np.diag([1e-8,1e-10]),last_day=0)
    t=Task('A:cycle0','A',0,0,1,0,proposed_day=1)
    ctl.tasks[t.task_id]=t; ctl.last_solve_day=0
    actions=ctl.decide(0)
    assert actions=={'A'} and not ctl.solver_log
    checks.append('Due action dispatched on a day without an ILP solve')
    try:
        ctl.ingest(1,{'A':3.19},set())
    except PlanningError as exc:
        assert 'execution confirmation missing' in str(exc)
    else:
        raise AssertionError('Missing execution was not detected')
    checks.append('Missing execution confirmation stops operation')

    ctl=Controller(['A'],cfg)
    ctl.states['A']=FilterState(np.array([3.055,.01]),np.diag([1e-8,1e-10]),last_day=0)
    try:
        ctl.decide(0)
    except PlanningError as exc:
        assert 'No B0-safe date' in str(exc)
    else:
        raise AssertionError('Empty feasible-date set did not stop operation')
    checks.append('No feasible date stops operation rather than violating B0')

    summaries=[]
    with tempfile.TemporaryDirectory() as tmp:
        for policy in ['event','daily','periodic']:
            path=Path(tmp)/policy
            summary=run(Config(replanning_policy=policy),path)
            assert summary['status']=='completed'
            tasks=pd.read_csv(path/'tasks.csv')
            assert set(tasks.status)<= {'PROPOSED','DEFERRED','EXECUTED'}
            assert not (path/'commitment_log.csv').exists()
            done=tasks.loc[tasks.status=='EXECUTED']
            assert done.nominal_at_execution.notna().all()
            executions=pd.read_csv(path/'executions.csv')
            assert executions.groupby('day').size().max()<=cfg.capacity
            assert len(done)==len(executions)
            assert summary['actual_low_pressure_asset_days']==0
            summaries.append(summary)
        assert summaries[0]['solver_calls']<summaries[1]['solver_calls']
    checks.append('Default event/daily/periodic full runs; capacity, execution and schema checks')
    return {'status':'passed','checks':checks,'default_policy_runs':summaries}


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
