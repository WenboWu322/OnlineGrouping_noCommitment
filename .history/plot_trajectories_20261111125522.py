"""Pressure and retained-plan-margin plots for the no-commitment controller."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read_csv(path, required, optional=False):
    if not path.exists():
        if optional:
            return pd.DataFrame(columns=required)
        raise FileNotFoundError(path)

    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        if optional:
            return pd.DataFrame(columns=required)
        raise ValueError(f'Empty file: {path.name}')

    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(
            f'{path.name}: missing {sorted(missing)}'
        )

    return data


def settings(directory):
    path = directory / 'config.json'

    if path.exists():
        return json.loads(
            path.read_text(encoding='utf-8')
        )

    return {}


def solve_days(directory):
    data = read_csv(
        directory / 'replanning_log.csv',
        ['day', 'replanned'],
        optional=True,
    )

    if data.day.duplicated().any():
        raise ValueError('Duplicate replanning days')

    flags = (
        data.replanned.astype(str)
        .str.lower()
        .str.strip()
    )

    if not flags.isin(
        ['true', 'false', '1', '0']
    ).all():
        raise ValueError('Invalid replanning flag')

    return data.loc[
        flags.isin(['true', '1']),
        'day',
    ].tolist()


def plot_trajectories(
    results_dir,
    assets=None,
    output=None,
    show_observations=True,
    show_estimate=True,
    show_truth=False,
    show_replanning=False,
):
    directory = Path(results_dir)
    cfg = settings(directory)

    safe = cfg.get('p_safe', 3.0)
    trigger = cfg.get('p_trigger', 3.2)
    plan = cfg.get('p_plan', safe)

    obs = (
        read_csv(
            directory / 'observations.csv',
            ['day', 'asset', 'pressure_obs'],
            optional=True,
        )
        if show_observations
        else pd.DataFrame()
    )

    est = (
        read_csv(
            directory / 'state_estimates.csv',
            ['day', 'asset', 'p_hat'],
            optional=True,
        )
        if show_estimate
        else pd.DataFrame()
    )

    truth = (
        read_csv(
            directory / 'simulation_truth.csv',
            [
                'day',
                'asset',
                'pre_gr_pressure',
                'true_pressure',
            ],
        )
        if show_truth
        else pd.DataFrame()
    )

    gr = read_csv(
        directory / 'executions.csv',
        ['day', 'asset'],
        optional=True,
    )

    for frame in [obs, est, truth, gr]:
        if (
            not frame.empty
            and frame.duplicated(
                ['day', 'asset']
            ).any()
        ):
            raise ValueError(
                'Duplicate day/asset rows'
            )

    sources = [
        frame
        for frame in [obs, est, truth]
        if not frame.empty
    ]

    available = sorted(
        set().union(
            *(set(frame.asset) for frame in sources)
        )
    )

    selected = (
        available
        if assets is None
        else list(dict.fromkeys(assets))
    )

    if (
        not selected
        or set(selected) - set(available)
    ):
        raise ValueError(
            'Invalid asset selection. '
            f'Available: {available}'
        )

    solves = (
        solve_days(directory)
        if show_replanning
        else []
    )

    ncols = 1 if len(selected) == 1 else 2
    nrows = math.ceil(len(selected) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        squeeze=False,
        figsize=(14, 4.2 * nrows),
        constrained_layout=True,
    )

    for ax, asset in zip(axes.flat, selected):
        events = sorted(
            gr.loc[
                gr.asset == asset,
                'day',
            ].tolist()
        )

        values = []
        days = []

        if show_truth:
            data = truth.loc[
                truth.asset == asset
            ].sort_values('day')

            x, y = [], []

            for row in data.itertuples():
                if row.day in events:
                    x.extend([row.day, row.day])
                    y.extend([
                        row.pre_gr_pressure,
                        row.true_pressure,
                    ])
                else:
                    x.append(row.day)
                    y.append(row.true_pressure)

            ax.plot(
                x,
                y,
                color='#1764ab',
                lw=1.6,
            )

            values.extend(y)
            days.extend(data.day)

        if show_observations and not obs.empty:
            data = obs.loc[
                obs.asset == asset
            ].sort_values('day')

            days.extend(data.day)

            valid = data.dropna(
                subset=['pressure_obs']
            )

            ax.scatter(
                valid.day,
                valid.pressure_obs,
                s=13,
                color='#777',
                alpha=.55,
            )

            values.extend(valid.pressure_obs)

        if show_estimate and not est.empty:
            data = est.loc[
                est.asset == asset
            ].sort_values('day').copy()

            if 'cycle' in data:
                cycle = data.cycle
            else:
                cycle = data.day.map(
                    lambda d: sum(
                        g <= d for g in events
                    )
                )

            cuts = (
                cycle.ne(cycle.shift())
                | data.day.diff().ne(1)
            ).cumsum()

            for _, part in data.groupby(cuts):
                ax.plot(
                    part.day,
                    part.p_hat,
                    '--',
                    color='#e48b21',
                    lw=1.2,
                )

            days.extend(data.day)
            values.extend(data.p_hat)

        if (
            not values
            or not np.isfinite(
                np.asarray(values, dtype=float)
            ).all()
        ):
            raise ValueError(
                f'No finite visible pressure data for {asset}'
            )

        ax.axhline(
            trigger,
            color='#b47e00',
            ls='--',
            lw=1,
        )
        ax.axhline(
            safe,
            color='#c43232',
            ls=':',
            lw=1.4,
        )

        if not np.isclose(plan, safe):
            ax.axhline(
                plan,
                color='#9a4fa3',
                ls='-.',
                lw=1.2,
            )

        for day in solves:
            if min(days) <= day <= max(days):
                ax.axvline(
                    day,
                    color='#8f96a3',
                    ls=':',
                    alpha=.35,
                    lw=.7,
                )

        for day in events:
            days.append(day)

            ax.axvline(
                day,
                color='#218c55',
                ls='--',
                alpha=.6,
            )

            # Axis-coordinate marker: no true pressure is used.
            ax.plot(
                [day],
                [.96],
                marker='v',
                color='#218c55',
                transform=ax.get_xaxis_transform(),
            )

            ax.annotate(
                f'GR d{day:g}',
                (day, .96),
                xycoords=(
                    'data',
                    'axes fraction',
                ),
                xytext=(3, -16),
                textcoords='offset points',
                fontsize=7,
                color='#17643c',
            )

        ax.set_ylim(
            min(safe - .04, min(values) - .04),
            max(3.6, max(values) + .07),
        )
        ax.set_xlim(
            min(days) - .5,
            max(days) + .5,
        )

        ax.set_title(
            f'{asset} | Executed GRs: {len(events)}'
        )
        ax.set_xlabel('Simulation day')
        ax.set_ylabel('Pressure (bar)')
        ax.grid(alpha=.16)

    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)

    handles = [
        Line2D(
            [], [],
            color='#b47e00',
            ls='--',
            label=f'Task reference ({trigger:g} bar)',
        ),
        Line2D(
            [], [],
            color='#c43232',
            ls=':',
            label=f'Physical threshold ({safe:g} bar)',
        ),
        Line2D(
            [], [],
            color='#218c55',
            ls='--',
            marker='v',
            label='Actual GR date',
        ),
    ]

    if not np.isclose(plan, safe):
        handles.append(
            Line2D(
                [], [],
                color='#9a4fa3',
                ls='-.',
                label=f'Planning threshold ({plan:g} bar)',
            )
        )

    if show_observations and not obs.empty:
        handles.append(
            Line2D(
                [], [],
                color='#777',
                marker='.',
                ls='None',
                label='Sensor observation',
            )
        )

    if show_estimate and not est.empty:
        handles.append(
            Line2D(
                [], [],
                color='#e48b21',
                ls='--',
                label='Kalman posterior mean',
            )
        )

    if show_truth:
        handles.append(
            Line2D(
                [], [],
                color='#1764ab',
                label='Simulated true pressure',
            )
        )

    if solves:
        handles.append(
            Line2D(
                [], [],
                color='#8f96a3',
                ls=':',
                label='System-wide ILP solve day',
            )
        )

    axes.flat[0].legend(
        handles=handles,
        fontsize=8,
        ncol=2,
        loc='lower left',
    )

    fig.suptitle(
        'Pressure monitoring and actual GR execution'
        ' — no commitment locking'
    )

    label = (
        selected[0]
        if len(selected) == 1
        else 'all_assets'
    )

    output = (
        Path(output)
        if output
        else directory / 'figures' / f'pressure_{label}.png'
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output,
        dpi=200,
        bbox_inches='tight',
    )
    plt.close(fig)

    return output


def plot_plan_margins(
    results_dir,
    assets=None,
    output=None,
):
    directory = Path(results_dir)
    cfg = settings(directory)

    data = read_csv(
        directory / 'plan_monitoring.csv',
        [
            'day',
            'task_id',
            'target_day',
            'margin_plan',
        ],
    )

    # New logs already contain asset.
    # Legacy logs can obtain it from tasks.csv.
    if 'asset' not in data:
        tasks = read_csv(
            directory / 'tasks.csv',
            ['task_id', 'asset'],
        )

        data = data.merge(
            tasks[['task_id', 'asset']],
            on='task_id',
            how='left',
            validate='many_to_one',
        )

    if data.asset.isna().any():
        raise ValueError(
            'Unknown task-to-asset mapping'
        )

    if data.duplicated(
        ['day', 'task_id']
    ).any():
        raise ValueError(
            'Duplicate monitoring records'
        )

    available = sorted(data.asset.unique())

    selected = (
        available
        if assets is None
        else list(dict.fromkeys(assets))
    )

    if (
        not selected
        or set(selected) - set(available)
    ):
        raise ValueError(
            'No monitoring data for selection. '
            f'Available: {available}'
        )

    for col in [
        'day',
        'target_day',
        'margin_plan',
    ]:
        data[col] = pd.to_numeric(
            data[col],
            errors='raise',
        )

        if not np.isfinite(
            data[col].dropna().to_numpy(dtype=float)
        ).all():
            raise ValueError(f'Invalid {col}')

    safe = cfg.get('p_safe', 3.0)
    plan = cfg.get('p_plan', safe)
    threshold = cfg.get('risk_trigger_margin', 0.)
    solves = solve_days(directory)

    fig, axes = plt.subplots(
        len(selected),
        1,
        squeeze=False,
        sharex=True,
        figsize=(
            13,
            max(3.5, 2.8 * len(selected)),
        ),
        constrained_layout=True,
    )

    exported = []

    for ax, asset in zip(axes.flat, selected):
        part = data.loc[
            data.asset == asset
        ].sort_values(['task_id', 'day'])

        exported.append(part)

        for _, task in part.groupby('task_id'):
            # Break at target-date changes and missing records.
            cuts = (
                task.day.diff().ne(1)
                | task.target_day.ne(
                    task.target_day.shift()
                )
                | task.margin_plan.isna()
                | task.margin_plan.shift().isna()
            ).cumsum()

            for _, seg in task.groupby(cuts):
                seg = seg.dropna(
                    subset=[
                        'target_day',
                        'margin_plan',
                    ]
                )

                if not seg.empty:
                    ax.plot(
                        seg.day,
                        seg.margin_plan,
                        '-o',
                        color='#1764ab',
                        ms=3,
                        lw=1,
                    )

        ax.axhline(
            0,
            color='#c43232',
            ls='--',
        )

        if not np.isclose(plan, safe):
            ax.axhline(
                safe - plan,
                color='#555',
                ls='-.',
            )

        if not np.isclose(threshold, 0):
            ax.axhline(
                threshold,
                color='#b47e00',
                ls=':',
            )

        ax.plot(
            solves,
            [.97] * len(solves),
            '|',
            color='#218c55',
            ms=10,
            transform=ax.get_xaxis_transform(),
        )

        if part.margin_plan.notna().sum() == 0:
            ax.text(
                .5,
                .6,
                'No retained date yet',
                transform=ax.transAxes,
                ha='center',
            )

        ax.set_title(asset)
        ax.set_ylabel('M_plan (bar)')
        ax.grid(alpha=.16)

    handles = [
        Line2D(
            [], [],
            color='#1764ab',
            marker='o',
            label='Retained assignment (modifiable)',
        ),
        Line2D(
            [], [],
            color='#c43232',
            ls='--',
            label='Planning B0 boundary (M = 0)',
        ),
        Line2D(
            [], [],
            color='#218c55',
            marker='|',
            ls='None',
            label='System-wide ILP solve day',
        ),
    ]

    if not np.isclose(plan, safe):
        handles.append(
            Line2D(
                [], [],
                color='#555',
                ls='-.',
                label=(
                    'Physical B0 boundary '
                    f'(M = {safe - plan:g})'
                ),
            )
        )

    if not np.isclose(threshold, 0):
        handles.append(
            Line2D(
                [], [],
                color='#b47e00',
                ls=':',
                label=f'Risk trigger (M = {threshold:g})',
            )
        )

    axes.flat[0].legend(
        handles=handles,
        fontsize=7,
        ncol=2,
    )
    axes.flat[-1].set_xlabel('Monitoring day')

    fig.suptitle(
        'Retained-plan margins BEFORE replanning\n'
        'Each point concerns a retained future GR date, '
        "not today's pressure"
    )

    label = (
        selected[0]
        if len(selected) == 1
        else 'all_assets'
    )

    output = (
        Path(output)
        if output
        else directory / 'figures'
        / f'plan_margins_{label}.png'
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output,
        dpi=200,
        bbox_inches='tight',
    )
    plt.close(fig)

    pd.concat(
        exported,
        ignore_index=True,
    ).to_csv(
        output.with_name(
            output.stem + '_data.csv'
        ),
        index=False,
    )

    return output


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        '--results',
        type=Path,
        default=Path('results_no_commitment'),
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        '--assets',
        nargs='+',
    )
    selection.add_argument(
        '--all',
        action='store_true',
    )

    parser.add_argument(
        '--out',
        type=Path,
    )
    parser.add_argument(
        '--diagnostics-out',
        type=Path,
    )
    parser.add_argument(
        '--no-observations',
        action='store_true',
    )
    parser.add_argument(
        '--no-estimate',
        action='store_true',
    )
    parser.add_argument(
        '--show-truth',
        action='store_true',
    )
    parser.add_argument(
        '--show-replanning',
        action='store_true',
    )
    parser.add_argument(
        '--diagnostics',
        action='store_true',
    )

    args = parser.parse_args()

    assets = (
        None
        if args.all
        else (args.assets or ['PMEC_001'])
    )

    print(
        plot_trajectories(
            args.results,
            assets,
            args.out,
            not args.no_observations,
            not args.no_estimate,
            args.show_truth,
            args.show_replanning,
        )
    )

    if args.diagnostics:
        print(
            plot_plan_margins(
                args.results,
                assets,
                args.diagnostics_out,
            )
        )


if __name__ == '__main__':
    main()