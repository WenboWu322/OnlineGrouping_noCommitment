"""Plot observed/estimated pressure and optional event-trigger diagnostics.
Truth is hidden by default, including GR-day true-pressure markers and axis scaling.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_utils_event import read_table, bool_values, numeric, config

def load_sources(directory,show_observations,show_estimate,show_truth):
    obs=read_table(directory/'observations.csv',['day','asset','pressure_obs'],optional=True,unique=['day','asset']) if show_observations else pd.DataFrame()
    est=read_table(directory/'state_estimates.csv',['day','asset','p_hat'],optional=True,unique=['day','asset']) if show_estimate else pd.DataFrame()
    truth=read_table(directory/'simulation_truth.csv',['day','asset','pre_gr_pressure','true_pressure'],optional=False,unique=['day','asset']) if show_truth else pd.DataFrame()
    executions=read_table(directory/'executions.csv',['day','asset'],optional=True,unique=['day','asset'])
    return obs,est,truth,executions

def plot_trajectories(results_dir,assets=None,output=None,show_observations=True,show_estimate=True,show_truth=False,show_replanning=False):
    directory=Path(results_dir); cfg=config(directory)
    trigger,safe=cfg.get('p_trigger',3.2),cfg.get('p_safe',3.0)
    plan=cfg.get('p_plan',safe)  # Legacy outputs used p_safe for B0.
    obs,est,truth,executions=load_sources(directory,show_observations,show_estimate,show_truth)
    sources=[df for df in [obs,est,truth] if not df.empty]
    if not sources: raise ValueError('No visible pressure data. Enable observations, estimates or truth.')
    available=sorted(set().union(*(set(df.asset.astype(str)) for df in sources)))
    selected=available if assets is None else list(dict.fromkeys(assets))
    if not selected or set(selected)-set(available): raise ValueError(f'Unknown/empty asset selection. Available: {available}')
    replan=read_table(directory/'replanning_log.csv',['day','replanned'],optional=True,unique=['day']) if show_replanning else pd.DataFrame()
    solve_days=replan.loc[bool_values(replan.replanned),'day'].tolist() if not replan.empty else []
    ncols=1 if len(selected)==1 else 2; nrows=math.ceil(len(selected)/ncols)
    fig,axes=plt.subplots(nrows,ncols,figsize=(14,4.2*nrows),squeeze=False,constrained_layout=True)
    for ax,asset in zip(axes.flat,selected):
        segments=[]; visible=[]; days=[]
        gr=executions.loc[executions.asset==asset].sort_values('day')
        gr_days=set(gr.day.tolist())
        if show_truth:
            data=truth.loc[truth.asset==asset].sort_values('day')
            numeric(data,['day','pre_gr_pressure','true_pressure'])
            x,y=[],[]
            for r in data.itertuples(index=False):
                if r.day in gr_days: x.extend([r.day,r.day]); y.extend([r.pre_gr_pressure,r.true_pressure])
                else: x.append(r.day); y.append(r.true_pressure)
            ax.plot(x,y,color='#1764ab',lw=1.6,zorder=3)
            visible.extend(y); days.extend(data.day)
            for d in gr_days:
                match=data.loc[data.day==d]
                if match.empty: raise ValueError(f'GR {asset} day {d} has no truth row')
                r=match.iloc[0]
                ax.scatter([d],[r.true_pressure],marker='v',s=40,color='#218c55',zorder=5)
                ax.scatter([d],[r.pre_gr_pressure],s=22,facecolors='white',edgecolors='#218c55',zorder=5)
        if show_observations and not obs.empty:
            data=obs.loc[obs.asset==asset].sort_values('day')
            days.extend(data.day)
            valid=data.dropna(subset=['pressure_obs']).copy(); numeric(valid,['day','pressure_obs'])
            ax.scatter(valid.day,valid.pressure_obs,s=13,color='#777777',alpha=.55,zorder=2)
            visible.extend(valid.pressure_obs)
        if show_estimate and not est.empty:
            data=est.loc[est.asset==asset].sort_values('day').copy(); numeric(data,['day','p_hat'])
            days.extend(data.day); visible.extend(data.p_hat)
            cycle=data.cycle if 'cycle' in data else data.day.map(lambda d:sum(g<=d for g in gr_days))
            # Also break missing-day gaps rather than invent a continuous trajectory.
            cuts=((cycle.ne(cycle.shift())) | (data.day.diff().ne(1))).cumsum()
            for _,part in data.groupby(cuts,sort=False):
                ax.plot(part.day,part.p_hat,'--',color='#e48b21',lw=1.2,zorder=4)
        if not visible:
            raise ValueError(f'No finite visible pressure data for {asset}')
        ax.axhline(trigger,color='#b47e00',ls='--',lw=1)
        ax.axhline(safe,color='#c43232',ls=':',lw=1.4)
        if not np.isclose(plan,safe): ax.axhline(plan,color='#9a4fa3',ls='-.',lw=1.2)
        for d in solve_days:
            if min(days)<=d<=max(days): ax.axvline(d,color='#8f96a3',ls=':',lw=.65,alpha=.35,zorder=0)
        for d in sorted(gr_days):
            days.append(d)
            ax.axvline(d,color='#218c55',ls='--',lw=1,alpha=.6)
            # Marker in axis coordinates: no hidden true-pressure values are used.
            ax.plot([d],[.96],marker='v',color='#218c55',transform=ax.get_xaxis_transform(),ls='None')
            ax.annotate(f'GR d{d:g}',xy=(d,.96),xycoords=('data','axes fraction'),xytext=(3,-16),textcoords='offset points',fontsize=7,color='#17643c',rotation=20)
        ax.set_ylim(min(safe-.04,min(visible)-.04),max(3.60,max(visible)+.07))
        ax.set_xlim(min(days)-.5,max(days)+.5)
        ax.set_title(f'{asset} | Executed GRs: {len(gr)}'); ax.set_xlabel('Simulation day'); ax.set_ylabel('Pressure (bar)'); ax.grid(alpha=.16)
    for ax in list(axes.flat)[len(selected):]: ax.set_visible(False)
    handles=[Line2D([],[],color='#b47e00',ls='--',label=f'Task reference ({trigger:g} bar)'),Line2D([],[],color='#c43232',ls=':',label=f'Physical threshold ({safe:g} bar)'),Line2D([],[],color='#218c55',ls='--',marker='v',label='Actual GR date')]
    if not np.isclose(plan,safe): handles.append(Line2D([],[],color='#9a4fa3',ls='-.',label=f'Planning threshold ({plan:g} bar)'))
    if show_observations and not obs.empty: handles.append(Line2D([],[],color='#777',marker='.',ls='None',label='Sensor observation'))
    if show_estimate and not est.empty: handles.append(Line2D([],[],color='#e48b21',ls='--',label='Kalman posterior mean'))
    if show_truth: handles.append(Line2D([],[],color='#1764ab',label='Simulated true pressure'))
    if solve_days: handles.append(Line2D([],[],color='#8f96a3',ls=':',label='System-wide ILP solve day'))
    axes.flat[0].legend(handles=handles,fontsize=8,loc='lower left',ncol=2)
    fig.suptitle('Daily pressure monitoring and actual GR execution',fontsize=14)
    if output is None:
        label=selected[0] if len(selected)==1 else 'all_assets'
        output=directory/'figures'/f'pressure_{label}.png'
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=200,bbox_inches='tight'); plt.close(fig)
    return output

def plot_plan_margins(results_dir,assets=None,output=None):
    directory=Path(results_dir); cfg=config(directory)
    monitor=read_table(directory/'plan_monitoring.csv',['day','task_id','status','target_day','margin_plan'])
    # The new monitoring log has task IDs but no asset column.
    tasks=read_table(directory/'tasks.csv',['task_id','asset'],unique=['task_id'])
    if 'asset' not in monitor:
        monitor=monitor.merge(tasks[['task_id','asset']],on='task_id',how='left',validate='many_to_one')
    if monitor.asset.isna().any(): raise ValueError('Monitoring task missing from tasks.csv')
    if monitor.duplicated(['day','task_id']).any(): raise ValueError('Duplicate monitoring record')
    available=sorted(monitor.asset.unique())
    selected=available if assets is None else list(dict.fromkeys(assets))
    if not selected or set(selected)-set(available): raise ValueError(f'No monitoring data for selection. Available: {available}')
    replan=read_table(directory/'replanning_log.csv',['day','replanned'],optional=True,unique=['day'])
    solve_days=replan.loc[bool_values(replan.replanned),'day'].tolist() if not replan.empty else []
    threshold=cfg.get('risk_trigger_margin',0.)
    safe=cfg.get('p_safe',3.0); plan=cfg.get('p_plan',safe)
    fig,axes=plt.subplots(len(selected),1,figsize=(13,max(3.5,2.8*len(selected))),squeeze=False,sharex=True,constrained_layout=True)
    exported=[]
    for ax,asset in zip(axes.flat,selected):
        data=monitor.loc[monitor.asset==asset].sort_values(['task_id','day']).copy()
        numeric(data,['day','target_day','margin_plan'],allow_missing=True)
        exported.append(data)
        plotted=False
        for tid,part in data.groupby('task_id',sort=False):
            # Do not join across date changes, missing days, missing margins or status changes.
            cut=(part.day.diff().ne(1)|part.target_day.ne(part.target_day.shift())|part.status.ne(part.status.shift())|part.margin_plan.isna()|part.margin_plan.shift().isna()).cumsum()
            for _,seg in part.groupby(cut,sort=False):
                seg=seg.dropna(subset=['margin_plan','target_day'])
                if seg.empty: continue
                committed=seg.status.iloc[0]=='COMMITTED'
                ax.plot(seg.day,seg.margin_plan,marker='s' if committed else 'o',ms=3,lw=1,color='#9467bd' if committed else '#1764ab')
                plotted=True
        if not plotted: ax.text(.5,.6,'No retained assignment yet',transform=ax.transAxes,ha='center')
        ax.axhline(0,color='#c43232',ls='--',lw=1)
        if not np.isclose(threshold,0): ax.axhline(threshold,color='#b47e00',ls=':',lw=1)
        if not np.isclose(plan,safe): ax.axhline(safe-plan,color='#555',ls='-.',lw=1)
        ax.plot(solve_days,[.97]*len(solve_days),'|',color='#218c55',ms=10,transform=ax.get_xaxis_transform())
        ax.set_title(asset,fontsize=10); ax.set_ylabel('M_plan (bar)'); ax.grid(alpha=.16)
    axes.flat[-1].set_xlabel('Monitoring day')
    handles=[Line2D([],[],color='#1764ab',marker='o',label='Uncommitted retained assignment'),Line2D([],[],color='#9467bd',marker='s',label='Committed assignment'),Line2D([],[],color='#c43232',ls='--',label='Planning B0 boundary (M = 0)'),Line2D([],[],color='#218c55',marker='|',ls='None',label='System-wide ILP solve day')]
    if not np.isclose(plan,safe): handles.append(Line2D([],[],color='#555',ls='-.',label=f'Physical B0 boundary (M = {safe-plan:g})'))
    if not np.isclose(threshold,0): handles.append(Line2D([],[],color='#b47e00',ls=':',label=f'Risk trigger (M = {threshold:g})'))
    axes.flat[0].legend(handles=handles,fontsize=7,ncol=2,loc='best')
    fig.suptitle('Retained-plan safety margins BEFORE replanning\nEach point concerns its retained future GR date, not today\'s pressure',fontsize=12)
    if output is None: output=directory/'figures'/('plan_margins_'+(selected[0] if len(selected)==1 else 'all_assets')+'.png')
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=200,bbox_inches='tight'); plt.close(fig)
    pd.concat(exported,ignore_index=True).to_csv(output.with_name(output.stem+'_data.csv'),index=False)
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results_event'))
    selection=p.add_mutually_exclusive_group(); selection.add_argument('--assets',nargs='+'); selection.add_argument('--all',action='store_true')
    p.add_argument('--out',type=Path)
    p.add_argument('--no-observations',action='store_true'); p.add_argument('--no-estimate',action='store_true')
    p.add_argument('--show-truth',action='store_true'); p.add_argument('--show-replanning',action='store_true')
    p.add_argument('--diagnostics',action='store_true',help='Additionally plot pre-replanning retained-plan margins')
    p.add_argument('--diagnostics-out',type=Path)
    args=p.parse_args(); assets=None if args.all else (args.assets or ['PMEC_001'])
    print(plot_trajectories(args.results,assets,args.out,not args.no_observations,not args.no_estimate,args.show_truth,args.show_replanning))
    if args.diagnostics: print(plot_plan_margins(args.results,assets,args.diagnostics_out))
if __name__=='__main__': main()
