"""EV charging sessions: time-varying Poisson arrivals and survival-function plug-in durations.

Per EV and day:
  arrivals   non-homogeneous Poisson process with rate
               lambda(t) = n_per_day * [ base / 24 + (1 - base) * g(t; peak, width) ]
             (g a wrapped Gaussian density over the 24-h day), sampled by thinning; an arrival
             while the EV is still plugged in is dropped
  duration   plug-in time D from a Weibull survival function S(d | h) = exp(-(d / scale_h)^shape),
             scale_h long for evening/night arrivals and short for daytime arrivals; drawn by
             inverse transform d = scale_h * (-ln U)^(1/shape)
  energy     log-normal kWh per session (mean, cv), capped at the battery size
  power      charger mix 3.7 / 7.4 / 11 kW (single- and three-phase home chargers)
  charging   uncontrolled from plug-in, or for a share of evening sessions delayed to the off-peak
             start (timer charging, as Belgian night tariffs encourage), until the energy is delivered
             or the EV leaves

fit_to_profile() calibrates the season-specific parameters so that the mean simulated charging
profile matches a target profile, here the Fluvius 2024 EV households minus non-EV households.
Time is in row hours (the Fluvius rows, UTC), 96 steps of 15 minutes per day.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

STEPS = 96
DT_H = 0.25
POWER_MIX = ((3.7, 0.3), (7.4, 0.5), (11.0, 0.2))
BATTERY_KWH = 60.0


@dataclass
class EVParams:
    sessions_per_day: float = 0.7
    arrival_peak_h: float = 18.0
    arrival_width_h: float = 2.5
    base_share: float = 0.3          # share of arrivals spread uniformly over the day
    timer_share: float = 0.4         # share of evening (15-22 h) arrivals that wait for the off-peak start
    timer_start_h: float = 22.0
    timer_spread_h: float = 2.0      # timer starts uniformly in [timer_start, timer_start + spread]
    energy_mean_kwh: float = 12.0
    energy_cv: float = 0.6
    dur_shape: float = 1.8
    dur_scale_night_h: float = 12.0  # arrivals 15:00-05:00
    dur_scale_day_h: float = 4.0     # arrivals 05:00-15:00


def arrival_rate(p: EVParams, t_h: np.ndarray) -> np.ndarray:
    d = (t_h - p.arrival_peak_h + 12.0) % 24.0 - 12.0
    g = np.exp(-0.5 * (d / p.arrival_width_h) ** 2) / (p.arrival_width_h * np.sqrt(2 * np.pi))
    return p.sessions_per_day * (p.base_share / 24.0 + (1 - p.base_share) * g)


def simulate(p: EVParams, n_ev: int, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Charging power (kW) per EV and 15-min step: (n_ev, n_days * 96). Sessions may run into the next day."""
    total = n_days * STEPS
    out = np.zeros((n_ev, total + 2 * STEPS))                # spill-over buffer past the last day
    grid = np.linspace(0, 24, 241)
    lam_max = arrival_rate(p, grid).max() * 1.05
    powers = np.array([kw for kw, _ in POWER_MIX])
    probs = np.array([w for _, w in POWER_MIX])
    mu = np.log(p.energy_mean_kwh / np.sqrt(1 + p.energy_cv ** 2))
    sig = np.sqrt(np.log(1 + p.energy_cv ** 2))
    for e in range(n_ev):
        kw = rng.choice(powers, p=probs)
        busy_until = -1.0                                    # hours since the start of day 0
        for day in range(n_days):
            n_cand = rng.poisson(lam_max * 24.0)
            t = np.sort(rng.uniform(0, 24, n_cand))
            t = t[rng.uniform(size=n_cand) < arrival_rate(p, t) / lam_max]   # thinning
            for h in t:
                start = day * 24.0 + h
                if start < busy_until:
                    continue
                scale = p.dur_scale_day_h if 5.0 <= h < 15.0 else p.dur_scale_night_h
                dur = scale * (-np.log(rng.uniform())) ** (1.0 / p.dur_shape)
                leave = start + dur
                busy_until = leave
                energy = min(rng.lognormal(mu, sig), BATTERY_KWH)
                c0 = start
                if 15.0 <= h < p.timer_start_h and rng.uniform() < p.timer_share:
                    c0 = day * 24.0 + p.timer_start_h + p.timer_spread_h * rng.uniform()
                c1 = min(c0 + energy / kw, leave)
                if c1 <= c0:
                    continue
                a, b = c0 / DT_H, c1 / DT_H                   # fractional step indices
                i0, i1 = int(np.floor(a)), int(np.ceil(b))
                for i in range(i0, min(i1, out.shape[1])):
                    frac = min(b, i + 1) - max(a, i)
                    out[e, i] += kw * max(frac, 0.0)
    return out[:, :total]


def mean_profile(p: EVParams, n_ev: int, n_days: int, seed: int = 0) -> np.ndarray:
    """Mean kW per EV by hour of day (24,), averaged over EVs and days (first day dropped as warm-up)."""
    x = simulate(p, n_ev, n_days + 1, np.random.default_rng(seed))[:, STEPS:]
    return x.reshape(n_ev, n_days, 24, 4).mean(axis=(0, 1, 3))


FIT_KEYS = ("sessions_per_day", "arrival_peak_h", "arrival_width_h", "base_share", "timer_share",
            "timer_spread_h", "energy_mean_kwh")
BOUNDS = {"sessions_per_day": (0.2, 1.5), "arrival_peak_h": (14.0, 21.0), "arrival_width_h": (1.0, 5.0),
          "base_share": (0.05, 0.9), "timer_share": (0.0, 0.95), "timer_spread_h": (0.0, 5.0),
          "energy_mean_kwh": (4.0, 30.0)}


def target_from_groups(ev_net: np.ndarray, base_net: np.ndarray, floor_quantile: float = 0.0) -> np.ndarray:
    """Hourly EV charging target (24,) from Fluvius EV and non-EV households (kW, (meters, days, 96)):
    mean difference, minus its daytime floor (attributed to EV households being larger homes, not to
    charging) when floor_quantile is given; the floor is the minimum of the hourly difference."""
    diff = ev_net.reshape(ev_net.shape[0], -1, 24, 4).mean(axis=(0, 1, 3)) - \
        base_net.reshape(base_net.shape[0], -1, 24, 4).mean(axis=(0, 1, 3))
    return diff - diff.min() if floor_quantile is not None else diff


def fit_to_profile(target: np.ndarray, init: EVParams | None = None, n_ev: int = 120, n_days: int = 5,
                   maxiter: int = 15, popsize: int = 6, seed: int = 0) -> tuple[EVParams, float]:
    """Differential evolution within BOUNDS on the hourly profile error (common random numbers)."""
    from scipy.optimize import differential_evolution
    p0 = init or EVParams()

    def unpack(z):
        return replace(p0, **{k: float(v) for k, v in zip(FIT_KEYS, z)})

    def loss(z):
        return float(np.mean((mean_profile(unpack(z), n_ev, n_days) - target) ** 2))

    res = differential_evolution(loss, [BOUNDS[k] for k in FIT_KEYS], maxiter=maxiter, popsize=popsize,
                                 seed=seed, polish=False, tol=1e-6)
    return unpack(res.x), float(np.sqrt(res.fun))
