"""Plot actual, committed or SINGLE-day planned maintenance grouping."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_utils_event import read_table, bool_values, numeric, config

def snapshot_kind(directory, data, day):
    flags=[]
    if 'replanned_today' in data and not data.empty:
        flags=bool_values(data.replanned_today).unique().tolist()
    log=read_table(directory/'replanning_log.csv',['day','replanned'],optional=True,unique=['day'])
    match=log.loc[log.day==day]
    if not match.empty: flags+=bool_values(match.replanned).unique().tolist()
    flags=set(flags)
    if len(flags)>1: raise ValueError('Inconsistent replanned_today flags in selected snapshot')
    return 'ILP solved today' if flags=={True} else 'retained plan; no ILP solve today' if flags=={False} else 'solve status unavailable (legacy output)'

def plot_grouping(results_dir,mode='executed',decision_day=None,output=None):
    directory=Path(results_dir)
    if mode=='planned':
        if decision_day is None: raise ValueError('--decision-day is required for planned mode')
        tasks=read_table(directory/'plans.csv',['decision_day','task_id','asset','execution_day','nominal_day','status'],optional=True)
        data=tasks.loc[tasks.decision_day==decision_day].copy()
        kind=snapshot_kind(directory,data,decision_day)
        data['reference_day']=data.nominal_day
        data['target_day']=data.execution_day
        subtitle=f'Daily snapshot at day {decision_day}: {kind}\nReference = nominal date updated today; not actual execution'
        target_label='Planned GR date'; stem=f'grouping_plan_day_{decision_day}'
    elif mode in {'executed','committed'}:
        tasks=read_table(directory/'tasks.csv',['task_id','asset','status','nominal_at_commitment','executed_day','committed_day'],optional=True)
        status='EXECUTED' if mode=='executed' else 'COMMITTED'
        data=tasks.loc[tasks.status==status].copy()
        data['reference_day']=data.nominal_at_commitment
        data['target_day']=data.executed_day if mode=='executed' else data.committed_day
        subtitle=('Actual executions' if mode=='executed' else 'Pending commitments in final exported task table')+'\nReference = nominal date frozen at commitment'
        target_label='Actual GR date' if mode=='executed' else 'Committed GR date (pending)'
        stem=f'grouping_{mode}'
    else: raise ValueError('Invalid mode')
    if data.task_id.duplicated().any(): raise ValueError('Duplicate task IDs: select exactly one snapshot')
    # DEFERRED tasks are not scheduled tasks. Do not silently drop invalid active rows.
    if data.status.isin(['DEFERRED']).any():
        raise ValueError('DEFERRED tasks must not appear as scheduled entries')
    numeric(data,['reference_day','target_day'])
    assets=set(tasks.asset.dropna().astype(str))
    for name in ['state_estimates','observations','tasks']:
        extra=read_table(directory/f'{name}.csv',['asset'],optional=True)
        assets.update(extra.asset.dropna().astype(str))
    assets=sorted(assets)
    ypos={a:j for j,a in enumerate(assets)}
    data['shift_days']=data.target_day-data.reference_day
    data['movement']=np.where(data.shift_days<0,'advanced',np.where(data.shift_days>0,'delayed','unchanged'))
    data['group_size']=data.groupby('target_day').task_id.transform('size')
    groups=(data.groupby('target_day').agg(task_count=('task_id','size'),pmec_count=('asset','nunique'),assets=('asset',lambda x:';'.join(sorted(set(x.astype(str))))),task_ids=('task_id',lambda x:';'.join(x.astype(str)))).reset_index().sort_values('target_day'))
    if output is None: output=directory/'figures'/f'{stem}.png'
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    if data.empty:
        fig,ax=plt.subplots(figsize=(12,4))
        ax.axis('off'); ax.set_title('Maintenance grouping\n'+subtitle)
        ax.text(.5,.45,'No scheduled tasks in this selection',ha='center',transform=ax.transAxes)
    else:
        xmin=min(data.reference_day.min(),data.target_day.min()); xmax=max(data.reference_day.max(),data.target_day.max())
        fig,(ax,bars)=plt.subplots(2,1,sharex=True,figsize=(max(13,min(24,(xmax-xmin)/8)),max(7,.53*len(assets)+3.6)),gridspec_kw={'height_ratios':[max(3,.53*len(assets)),1.65]},constrained_layout=True)
        for j in range(len(assets)):
            if j%2==0: ax.axhspan(j-.5,j+.5,color='#f3f5f7',zorder=0)
        for row in groups.itertuples(index=False):
            if row.task_count>=2: ax.axvline(row.target_day,color='#4477aa',alpha=.3,ls=':',lw=1.3)
        for row in data.sort_values(['asset','target_day']).itertuples(index=False):
            y=ypos[str(row.asset)]; color='#198754' if row.shift_days<0 else '#c4473d' if row.shift_days>0 else '#666666'
            ax.scatter(row.reference_day,y,s=42,facecolors='white',edgecolors=color,zorder=4)
            if row.shift_days!=0:
                ax.annotate('',xy=(row.target_day,y),xytext=(row.reference_day,y),arrowprops=dict(arrowstyle='-|>',color=color,lw=1.4,shrinkA=5,shrinkB=6),zorder=3)
            ax.scatter(row.target_day,y,marker='D',color=color,s=43,zorder=5)
            cycle=getattr(row,'cycle',None)
            label=(f'C{int(cycle)}: ' if cycle is not None and pd.notna(cycle) else '')+f'd{row.target_day:g}'
            ax.annotate(label,(row.target_day,y),xytext=(0,-14),textcoords='offset points',ha='center',fontsize=7,color=color)
        ax.set_yticks(range(len(assets)),assets); ax.set_ylim(len(assets)-.4,-.7)
        ax.set_xlim(xmin-3,xmax+3); ax.set_ylabel('PMEC'); ax.grid(axis='x',alpha=.12)
        ax.set_title('Task grouping and date shifts across all PMECs\n'+subtitle,fontsize=12)
        ax.legend(handles=[Line2D([],[],marker='o',mfc='white',color='#333',ls='None',label='Nominal reference'),Line2D([],[],marker='D',color='#333',ls='None',label=target_label),Line2D([],[],color='#198754',label='Advanced'),Line2D([],[],color='#c4473d',label='Delayed'),Line2D([],[],color='#666',label='Unchanged'),Line2D([],[],color='#4477aa',ls=':',label='Shared execution day (2+ tasks)')],loc='lower left',bbox_to_anchor=(0,1.14),ncol=3,fontsize=8)
        bars.bar(groups.target_day,groups.task_count,width=.7,color=['#4477aa' if n>=2 else '#aab3bc' for n in groups.task_count])
        for r in groups.itertuples(index=False): bars.text(r.target_day,r.task_count+.05,str(r.task_count),ha='center',fontsize=8)
        cap=config(directory).get('capacity')
        if cap is not None:
            bars.axhline(cap,color='#9b3333',ls='--',lw=.9,label=f'Capacity = {cap}')
            bars.legend(fontsize=8,loc='upper right')
        bars.set_ylim(0,max(float(groups.task_count.max()),float(cap or 0))+1)
        bars.yaxis.set_major_locator(plt.MaxNLocator(integer=True)); bars.set_ylabel('Tasks / day'); bars.set_xlabel('Simulation day'); bars.grid(axis='y',alpha=.15)
    fig.savefig(output,dpi=200,bbox_inches='tight'); plt.close(fig)
    cols=['task_id','asset','reference_day','target_day','shift_days','movement','group_size','status']
    cols += [c for c in ['cycle','replanned_today'] if c in data]
    data[cols].to_csv(output.with_name(output.stem+'_tasks.csv'),index=False)
    groups.to_csv(output.with_name(output.stem+'_groups.csv'),index=False)
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results_event'))
    p.add_argument('--mode',choices=['executed','committed','planned'],default='executed')
    p.add_argument('--decision-day',type=int); p.add_argument('--out',type=Path)
    args=p.parse_args()
    print(plot_grouping(args.results,args.mode,args.decision_day,args.out))
if __name__=='__main__': main()
