"""Grouping plots for the no-commitment maintenance controller."""
import argparse
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
        raise ValueError(f'{path.name}: missing {sorted(missing)}')

    return data


def as_bool(values):
    values = values.astype(str).str.lower().str.strip()

    if not values.isin(['true', 'false', '1', '0']).all():
        raise ValueError('Invalid replanning flags')

    return values.isin(['true', '1'])


def plot_grouping(
    results_dir,
    mode='executed',
    decision_day=None,
    output=None,
):
    directory = Path(results_dir)

    if mode == 'planned':
        if decision_day is None:
            raise ValueError('--decision-day is required')

        tasks = read_csv(
            directory / 'plans.csv',
            [
                'decision_day',
                'task_id',
                'asset',
                'execution_day',
                'nominal_day',
                'status',
            ],
            optional=True,
        )

        data = tasks.loc[
            tasks.decision_day == decision_day
        ].copy()

        data['reference_day'] = data.nominal_day
        data['target_day'] = data.execution_day

        flags = []

        if 'replanned_today' in data and not data.empty:
            flags.extend(
                as_bool(data.replanned_today).tolist()
            )

        log = read_csv(
            directory / 'replanning_log.csv',
            ['day', 'replanned'],
            optional=True,
        )

        if log.day.duplicated().any():
            raise ValueError('Duplicate replanning days')

        match = log.loc[log.day == decision_day]
        flags.extend(as_bool(match.replanned).tolist())

        if len(set(flags)) > 1:
            raise ValueError('Inconsistent solve flags')

        if flags:
            kind = (
                'ILP solved today'
                if flags[0]
                else 'retained plan; no solve today'
            )
        else:
            kind = 'solve status unavailable'

        subtitle = (
            f'Day {decision_day}: {kind}\n'
            'Reference = current nominal date; '
            'not actual execution'
        )

        target_label = 'Planned GR date'
        stem = f'grouping_plan_day_{decision_day}'

    elif mode == 'executed':
        tasks = read_csv(
            directory / 'tasks.csv',
            [
                'task_id',
                'asset',
                'status',
                'nominal_at_execution',
                'executed_day',
            ],
            optional=True,
        )

        data = tasks.loc[
            tasks.status == 'EXECUTED'
        ].copy()

        data['reference_day'] = data.nominal_at_execution
        data['target_day'] = data.executed_day

        subtitle = (
            'Actual execution; reference = latest nominal '
            'date saved before execution'
        )

        target_label = 'Actual GR date'
        stem = 'grouping_executed'

    else:
        raise ValueError('mode must be executed or planned')

    if data.task_id.duplicated().any():
        raise ValueError('Duplicate selected task IDs')

    if data.status.eq('DEFERRED').any():
        raise ValueError(
            'DEFERRED task has no valid scheduled date'
        )

    for col in ['reference_day', 'target_day']:
        data[col] = pd.to_numeric(
            data[col],
            errors='raise',
        )

        if not np.isfinite(
            data[col].to_numpy(dtype=float)
        ).all():
            raise ValueError(f'Invalid {col}')

    assets = set(tasks.asset.dropna().astype(str))

    for name in ['state_estimates', 'observations']:
        extra = read_csv(
            directory / f'{name}.csv',
            ['asset'],
            optional=True,
        )
        assets.update(extra.asset.dropna().astype(str))

    assets = sorted(assets)

    data['shift_days'] = (
        data.target_day - data.reference_day
    )

    data['movement'] = np.where(
        data.shift_days < 0,
        'advanced',
        np.where(
            data.shift_days > 0,
            'delayed',
            'unchanged',
        ),
    )

    data['group_size'] = (
        data.groupby('target_day')
        .task_id.transform('size')
    )

    groups = (
        data.groupby('target_day')
        .agg(
            task_count=('task_id', 'size'),
            pmec_count=('asset', 'nunique'),
            assets=(
                'asset',
                lambda x: ';'.join(sorted(set(x))),
            ),
            task_ids=(
                'task_id',
                lambda x: ';'.join(x),
            ),
        )
        .reset_index()
    )

    output = (
        Path(output)
        if output
        else directory / 'figures' / f'{stem}.png'
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    if data.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        ax.set_title('Maintenance grouping\n' + subtitle)

        ax.text(
            .5,
            .45,
            'No tasks in this selection',
            ha='center',
            transform=ax.transAxes,
        )

    else:
        fig, (ax, bars) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(14, max(7, .55 * len(assets) + 3)),
            gridspec_kw={
                'height_ratios': [
                    max(3, .55 * len(assets)),
                    1.5,
                ]
            },
            constrained_layout=True,
        )

        ypos = {
            asset: j
            for j, asset in enumerate(assets)
        }

        for j in range(len(assets)):
            if j % 2 == 0:
                ax.axhspan(
                    j - .5,
                    j + .5,
                    color='#f3f5f7',
                )

        for row in groups.itertuples():
            if row.task_count >= 2:
                ax.axvline(
                    row.target_day,
                    color='#4477aa',
                    ls=':',
                    alpha=.35,
                )

        for row in data.sort_values(
            ['asset', 'target_day']
        ).itertuples():
            y = ypos[str(row.asset)]

            color = (
                '#198754'
                if row.shift_days < 0
                else '#c4473d'
                if row.shift_days > 0
                else '#666666'
            )

            ax.scatter(
                row.reference_day,
                y,
                s=42,
                facecolors='white',
                edgecolors=color,
                zorder=4,
            )

            if row.shift_days != 0:
                ax.annotate(
                    '',
                    (row.target_day, y),
                    (row.reference_day, y),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color=color,
                        shrinkA=5,
                        shrinkB=6,
                    ),
                    zorder=3,
                )

            ax.scatter(
                row.target_day,
                y,
                marker='D',
                color=color,
                s=42,
                zorder=5,
            )

            cycle = getattr(row, 'cycle', None)

            label = (
                f'C{int(cycle)}: '
                if pd.notna(cycle)
                else ''
            ) + f'd{row.target_day:g}'

            ax.annotate(
                label,
                (row.target_day, y),
                xytext=(0, -13),
                textcoords='offset points',
                ha='center',
                fontsize=7,
            )

        ax.set_yticks(range(len(assets)), assets)
        ax.set_ylim(len(assets) - .4, -.7)
        ax.set_ylabel('PMEC')

        ax.set_title(
            'Task grouping and date shifts\n' + subtitle
        )
        ax.grid(axis='x', alpha=.15)

        ax.legend(
            handles=[
                Line2D(
                    [], [],
                    marker='o',
                    mfc='white',
                    color='#333',
                    ls='None',
                    label='Nominal reference',
                ),
                Line2D(
                    [], [],
                    marker='D',
                    color='#333',
                    ls='None',
                    label=target_label,
                ),
                Line2D(
                    [], [],
                    color='#198754',
                    label='Advanced',
                ),
                Line2D(
                    [], [],
                    color='#c4473d',
                    label='Delayed',
                ),
                Line2D(
                    [], [],
                    color='#666',
                    label='Unchanged',
                ),
                Line2D(
                    [], [],
                    color='#4477aa',
                    ls=':',
                    label='Shared date (2+ tasks)',
                ),
            ],
            fontsize=8,
            ncol=3,
            loc='lower left',
            bbox_to_anchor=(0, 1.14),
        )

        bars.bar(
            groups.target_day,
            groups.task_count,
            color=[
                '#4477aa' if n >= 2 else '#aab3bc'
                for n in groups.task_count
            ],
        )

        for row in groups.itertuples():
            bars.text(
                row.target_day,
                row.task_count + .05,
                str(row.task_count),
                ha='center',
                fontsize=8,
            )

        bars.set_ylim(
            0,
            groups.task_count.max() + 1,
        )
        bars.yaxis.set_major_locator(
            plt.MaxNLocator(integer=True)
        )

        bars.set_ylabel('Tasks / day')
        bars.set_xlabel('Simulation day')
        bars.grid(axis='y', alpha=.15)

    fig.savefig(
        output,
        dpi=200,
        bbox_inches='tight',
    )
    plt.close(fig)

    data.to_csv(
        output.with_name(
            output.stem + '_tasks.csv'
        ),
        index=False,
    )

    groups.to_csv(
        output.with_name(
            output.stem + '_groups.csv'
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
    parser.add_argument(
        '--mode',
        choices=['executed', 'planned'],
        default='executed',
    )
    parser.add_argument(
        '--decision-day',
        type=int,
    )
    parser.add_argument(
        '--out',
        type=Path,
    )

    args = parser.parse_args()

    print(
        plot_grouping(
            args.results,
            args.mode,
            args.decision_day,
            args.out,
        )
    )


if __name__ == '__main__':
    main()