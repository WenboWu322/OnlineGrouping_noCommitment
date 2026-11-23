"""Strategy families compared in Chapter 6.

Naming follows the thesis tables:

===========  ============================================================
S1           reactive, no grouping (alarm-triggered, one trip per task)
S2           grouping, periodic re-solve, Chapter 5 window
S3           grouping, event-triggered, Chapter 5 window
S4           grouping, event-triggered, chance-constrained window
S5           S4 + plan-stability anchoring  (the proposed method)
S6           S5 with daily re-solve (upper bound on adaptivity)
S7           static plan, never re-optimised (open-loop Chapter 5 usage)
O1           clairvoyant online re-planning (regret reference)
===========  ============================================================
"""
from __future__ import annotations

from .controller import Strategy

# Chapter 5 window parameters used by the "nominal" window family.
_NOMINAL = dict(window='nominal', stability=False, detect=True)
_RISK = dict(window='risk', stability=False, detect=True)


def strategies() -> list[Strategy]:
    return [
        Strategy('S1_reactive', grouping='none', replan='daily',
                 window='risk', stability=False, creation='reactive'),
        Strategy('S2_periodic', grouping='M2', replan='periodic', **_NOMINAL),
        Strategy('S3_event_ch5win', grouping='M2', replan='event', **_NOMINAL),
        Strategy('S4_event_riskwin', grouping='M2', replan='event', **_RISK),
        Strategy('S5_proposed', grouping='M2', replan='event',
                 window='risk', stability=True),
        Strategy('S6_daily', grouping='M2', replan='daily',
                 window='risk', stability=True),
        Strategy('S7_static', grouping='M2', replan='none', **_NOMINAL),
        Strategy('O2_offline_clair', grouping='M2', replan='none',
                 window='risk', clairvoyant=True, stability=False,
                 creation='proactive', detect=False),
        Strategy('O1_oracle', grouping='M2', replan='event', window='risk',
                 stability=True, clairvoyant=True),
    ]


def by_name(names: list[str] | None) -> list[Strategy]:
    table = {s.name: s for s in strategies()}
    if not names:
        return strategies()
    missing = [n for n in names if n not in table]
    if missing:
        raise KeyError(f'unknown strategies: {missing}')
    return [table[n] for n in names]


ORACLE = 'O1_oracle'
