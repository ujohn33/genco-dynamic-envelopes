"""case33bw populated with real Fluvius 2024 households and simulated EV charging sessions.

Each load bus of case33bw stands for a distribution transformer serving K_b homes, K_b ~ the bus's
nominal load in kW / HOME_KW. Homes are real Fluvius meters from the groups WITHOUT an EV (their
EV charging would otherwise be counted twice): 30% heat pump, 40% PV (independent draws, so some
have both), the rest plain. 30% of homes own an EV whose charging comes from gencodoe.ev_sessions
with the season-specific parameters calibrated to Fluvius (configs/ev_sessions_fluvius2024.json).

Rows are the Fluvius 15-min steps (366 days x 96, row hours = UTC). Days are split by index:
day % 5 == 0 test, == 1 validation, otherwise training. The population (which meters sit where, who
owns an EV) is fixed: the operator's own network, observed over a year.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ev_sessions import STEPS, EVParams, simulate

CACHE = Path.home() / "gencodoe_work" / "fluvius_cache"
EV_PARAMS = Path(__file__).resolve().parents[2] / "configs" / "ev_sessions_fluvius2024.json"
GROUP = {(0, 0): "geen_ZP", (0, 1): "enkel_ZP", (1, 0): "WP_geen_ZP", (1, 1): "WP_met_ZP"}   # (heat pump, PV)
N_DAYS = 366
HOME_KW = 1.0
MIX = {"hp": 0.3, "pv": 0.4, "ev": 0.3}


def season_of(day: int) -> str:
    month = (day * 12) // N_DAYS
    return "winter" if month in (0, 1, 10, 11) else "summer" if month in (5, 6, 7) else "shoulder"


def split_of(day: int) -> str:
    return "test" if day % 5 == 0 else "val" if day % 5 == 1 else "train"


@dataclass
class Population:
    bus: np.ndarray        # (n_homes,) load bus of each home
    hp: np.ndarray         # (n_homes,) bool
    pv: np.ndarray
    ev: np.ndarray
    meter: np.ndarray      # (n_homes,) meter row within its Fluvius group


def build_population(pd_kw: np.ndarray, seed: int = 0, home_kw: float = HOME_KW, mix: dict = MIX) -> Population:
    rng = np.random.default_rng(seed)
    k = np.where(pd_kw > 0, np.maximum(1, np.round(pd_kw / home_kw)), 0).astype(int)
    bus = np.repeat(np.arange(len(pd_kw)), k)
    n = len(bus)
    hp, pv, ev = (rng.uniform(size=n) < mix["hp"]), (rng.uniform(size=n) < mix["pv"]), (rng.uniform(size=n) < mix["ev"])
    meter = rng.integers(0, 300, n)
    return Population(bus, hp, pv, ev, meter)


def bus_load_kw(pop: Population, n_bus: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Net consumption per bus and step (kW, + = import): homes' metered net + simulated EV charging.
    Returns (net_kw, ev_kw), each (n_bus, 366 * 96)."""
    net = np.zeros((n_bus, N_DAYS * STEPS))
    for (h, p), name in GROUP.items():
        sel = (pop.hp == bool(h)) & (pop.pv == bool(p))
        if not sel.any():
            continue
        d = np.load(CACHE / f"{name}.npz")
        x = d["offtake_kw"].astype(np.float64) - d["injection_kw"].astype(np.float64)
        np.add.at(net, pop.bus[sel], x[pop.meter[sel]])
    params = {k: EVParams(**v) for k, v in json.loads(EV_PARAMS.read_text()).items()}
    rng = np.random.default_rng(seed)
    ev = np.zeros_like(net)
    owners = np.flatnonzero(pop.ev)
    days = np.arange(N_DAYS)
    for season in ("winter", "summer", "shoulder"):
        ds = days[[season_of(d) == season for d in days]]
        # simulate contiguous runs of days of this season (sessions may spill into the next day)
        runs = np.split(ds, np.flatnonzero(np.diff(ds) > 1) + 1)
        for run in runs:
            x = simulate(params[season], len(owners), len(run), rng)
            np.add.at(ev[:, run[0] * STEPS:(run[-1] + 1) * STEPS], pop.bus[owners], x)
    return net + ev, ev


def day_ahead_forecast(net_kw: np.ndarray, window: int = 7) -> np.ndarray:
    """Per bus and step: mean of the same step over the previous `window` days (day 0: itself)."""
    x = net_kw.reshape(net_kw.shape[0], N_DAYS, STEPS)
    out = np.empty_like(x)
    for d in range(N_DAYS):
        lo = max(0, d - window)
        out[:, d] = x[:, lo:d].mean(axis=1) if d > 0 else x[:, 0]
    return out.reshape(net_kw.shape)
