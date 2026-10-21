"""Real-time phase switching study on ENWL1 (exploratory; see gencodoe.phase_switching).

Per interval (one 15-min step per chosen hour) and direction, uniform per-household envelope
under the doe_day design rule (S design scenarios, eps = beta = 0.05 -> acceptance threshold),
for three ways of setting the phases of the customers that have a phase-switching device (PSD):

  fixed      the feeder's phases (no switching)
  balance    greedy per-feeder phase balancing of the expected net load at that step (no power flow)
  search     sweep-guided greedy local search from the better-scoring of `fixed` and `balance`: each round moves the one PSD customer
             whose move most improves the binding-constraint margin at a test level just above the
             current envelope; all candidate moves are scored in one batched four-wire sweep

Every method's envelope is then checked exactly with OpenDSS on the design scenarios (scaled down
by bisection if it fails the rule) and on fresh held-out scenarios. Switches are counted against
the previous interval's assignment.

    python -m gencodoe.phase_switching_run --day 2024-12-15 --hours 7 18 19 20 --out results/...json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import envelopes
from .datakit_fluvius import FLEX_EXPORT_KW, FLEX_IMPORT_KW
from .doe_day import DayScenarios
from .enwl import load_enwl1
from .fourwire_net import load_fourwire
from .four_wire import four_wire_ok
from .limits import VMAX_PU, VMIN_PU, record as limits_record
from .phase_switching import PHASES, PhaseSweep, feasible, fit_thevenin_opendss, opendss_with_phases, uniform_level
from .fluvius import CACHE_DIR, GROUPS


def feeder_of(net) -> np.ndarray:
    """Main feeder (child of the LV root) of every customer."""
    top = np.empty(net.n_bus, int)
    for i in net.order:
        top[i] = i if net.parent[i] in (0, -1) else top[net.parent[i]]
    return top[net.customer_bus]


def margins(res: dict, direction: str) -> np.ndarray:
    """Binding-constraint margin per row (pu of voltage; 5% cable/transformer headroom ~ 0.05)."""
    v = res["v_pn_min"] - VMIN_PU if direction == "import" else VMAX_PU - res["v_pn_max"]
    m = np.minimum(v, 0.05 * (1 - res["line_loading_max"]))
    m = np.minimum(m, 0.05 * (1 - res["transformer_loading"]))
    return np.where(res["converged"], m, -1.0)


def balance(base_phase, switchable, feeder, value, direction):
    """Greedy phase balancing per feeder: switchable homes, largest |expected net load| first, each go to
    the phase that minimises the largest absolute phase total after adding them. value: expected net
    consumption per home (+ = import, - = export) including the flexible level. Signed totals let
    importing and exporting homes cancel on a phase (what reduces phase current); a plain min-max on
    signed values breaks once totals turn negative (PV at noon, export envelopes at night)."""
    phase = base_phase.copy()
    for f in np.unique(feeder):
        here = feeder == f
        tot = np.zeros(PHASES)
        fixed = here & ~switchable
        np.add.at(tot, phase[fixed], value[fixed])
        homes = np.flatnonzero(here & switchable)
        for c in homes[np.argsort(-np.abs(value[homes]))]:
            cost = [np.abs(tot + value[c] * (np.arange(PHASES) == p)).max() for p in range(PHASES)]
            p = int(np.argmin(cost))
            phase[c] = p
            tot[p] += value[c]
    return phase


def greedy(sw, s_rank, phase0, switchable, direction, level, threshold, rounds, max_cands, rng):
    sign = 1.0 if direction == "import" else -1.0
    rows = s_rank + sign * level
    q = 1.0 - threshold                                   # tail share allowed to fail

    def score(phases):                                    # phases (K, n_c) -> (K,)
        k = len(phases)
        res = sw.solve(np.tile(rows, (k, 1)), np.repeat(phases, len(rows), axis=0))
        m = margins(res, direction).reshape(k, len(rows))
        return np.quantile(m, q, axis=1)

    starts = np.atleast_2d(phase0)                       # several starts: keep the best-scoring one
    sc0 = score(starts)
    cur = starts[int(np.argmax(sc0))].copy()
    cur_score = float(sc0.max())
    moves, evals = 0, len(rows) * len(starts)
    for _ in range(rounds):
        cands = [(c, p) for c in np.flatnonzero(switchable) for p in range(PHASES) if p != cur[c]]
        if len(cands) > max_cands:
            cands = [cands[i] for i in rng.choice(len(cands), max_cands, replace=False)]
        ph = np.repeat(cur[None], len(cands), axis=0)
        for j, (c, p) in enumerate(cands):
            ph[j, c] = p
        sc = score(ph)
        evals += len(cands) * len(rows)
        j = int(np.argmax(sc))
        if sc[j] <= cur_score + 1e-6:
            break
        cur, cur_score = ph[j], float(sc[j])
        moves += 1
    return cur, moves, evals


def exact_level(ub, s_design, direction, level, threshold, iters=6):
    """OpenDSS design check of a level; bisection downwards if it fails. Returns (level, share, solves)."""
    sign = 1.0 if direction == "import" else -1.0
    solves = 0

    def share(lv):
        nonlocal solves
        p = (s_design.real + sign * lv)
        q = s_design.imag
        solves += len(p)
        return float(np.mean([four_wire_ok(ub.solve(p[i], q[i]), direction) for i in range(len(p))]))

    sh = share(level)
    if sh >= threshold or level == 0.0:
        return level, sh, solves
    lo, hi = 0.0, level
    for _ in range(iters):
        m = (lo + hi) / 2
        (lo, hi) = (m, hi) if share(m) >= threshold else (lo, m)
    return lo, share(lo), solves


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--hours", type=int, nargs="+", required=True)
    ap.add_argument("--directions", nargs="+", default=["import", "export"])
    ap.add_argument("--scenarios", type=int, default=200)
    ap.add_argument("--rank-scenarios", type=int, default=40, help="design scenarios used to rank moves")
    ap.add_argument("--verify", type=int, default=200)
    ap.add_argument("--psd", default="hp_ev", help="hp_ev (every HP or EV home) or an integer share in %% of customers")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--max-cands", type=int, default=120)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--save-phases", action="store_true", help="store each method's phase assignment in the rows")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    synthetic = any("synthetic" in np.load(CACHE_DIR / f"{g}.npz") for g in GROUPS)

    bal = load_enwl1()
    fw = load_fourwire()
    scen = DayScenarios(bal.customer_bus, bal.n_bus, seed=a.seed)
    base_phase = fw.customer_phase.copy()
    if a.psd == "hp_ev":
        switchable = scen.flags[:, 0] | scen.flags[:, 1]
    else:
        switchable = np.random.default_rng(a.seed).uniform(size=len(base_phase)) < int(a.psd) / 100
    feeder = feeder_of(fw)
    th = fit_thevenin_opendss()
    sw = PhaseSweep(fw, th)
    threshold = envelopes.acceptance_threshold(a.scenarios, 0.05, 0.05)
    rng = np.random.default_rng(a.seed)
    def engine(phase):          # opendssdirect is one engine per process: a new model replaces the last
        return opendss_with_phases(phase)

    t0 = int((np.datetime64(a.day) - np.datetime64("2024-01-01")).astype(int)) * 96
    prev = {d: {"balance": base_phase, "search": base_phase} for d in a.directions}
    rows = []
    for h in a.hours:
        step = t0 + 4 * h
        p, q = scen.sample(step, a.scenarios, rng)
        pv, qv = scen.sample(step, a.verify, rng)
        cb = bal.customer_bus
        s_design = (-p[:, cb] - 1j * q[:, cb]) * 1e3          # customer consumption, kVA (+ = import)
        s_verify = (-pv[:, cb] - 1j * qv[:, cb]) * 1e3
        rank = s_design[: a.rank_scenarios]
        for d in a.directions:
            cap = FLEX_IMPORT_KW if d == "import" else FLEX_EXPORT_KW
            sign = 1.0 if d == "import" else -1.0

            def level_of(phase):
                return uniform_level(lambda s: feasible(sw.solve(s, phase), d), s_design, d, cap, threshold)

            t = time.perf_counter()
            l_fixed, _ = level_of(base_phase)
            t_fixed = time.perf_counter() - t
            guess = min(l_fixed + 0.5, cap)
            mean_c = s_design.real.mean(axis=0) + sign * guess
            t = time.perf_counter()
            ph_bal = balance(base_phase, switchable, feeder, mean_c, d)
            l_bal, _ = level_of(ph_bal)
            t_bal = time.perf_counter() - t
            t = time.perf_counter()
            ph_s, moves, evals = greedy(sw, rank, np.stack([base_phase, ph_bal]), switchable, d,
                                        min(max(l_bal, l_fixed) + 0.5, cap),
                                        threshold, a.rounds, a.max_cands, rng)
            l_s, _ = level_of(ph_s)
            t_search = time.perf_counter() - t
            for name, ph, lv, dt in (("fixed", base_phase, l_fixed, t_fixed), ("balance", ph_bal, l_bal, t_bal),
                                     ("search", ph_s, l_s, t_search)):
                ub = engine(ph)
                lx, share_design, solves = exact_level(ub, s_design, d, lv, threshold)
                pc, qc = s_verify.real + sign * lx, s_verify.imag
                held = float(np.mean([four_wire_ok(ub.solve(pc[i], qc[i]), d) for i in range(len(pc))]))
                switches = 0 if name == "fixed" else int((ph != prev[d][name]).sum())
                if name != "fixed":
                    prev[d][name] = ph
                rows.append({"hour": h, "step": step, "direction": d, "method": name, "level_sweep_kw": lv,
                             "level_kw": lx, "design_share_opendss": share_design, "held_out_ok": held,
                             "moved_from_feeder_phases": int((ph != base_phase).sum()), "switches": switches,
                             "search_moves": moves if name == "search" else None,
                             "sweep_rows": evals if name == "search" else None, "opendss_solves": solves,
                             "time_s": dt, **({"phases": ph.tolist()} if a.save_phases else {})})
                print(f"{a.day} {h:02d}h {d:6s} {name:7s} level {lx:5.2f} kW (sweep {lv:5.2f}) design {share_design:.3f} "
                      f"held-out {held:.3f} switches {switches:3d} moved {rows[-1]['moved_from_feeder_phases']:3d} "
                      f"{dt:6.1f}s", flush=True)
    meta = {"feeder_phases": base_phase.tolist(), "psd_mask": switchable.tolist(), "day": a.day, "hours": a.hours, "synthetic_profiles": synthetic, "psd": a.psd,
            "psd_customers": int(switchable.sum()), "scenarios": a.scenarios, "rank_scenarios": a.rank_scenarios,
            "verify": a.verify, "threshold": threshold, "rounds": a.rounds, "max_cands": a.max_cands, "seed": a.seed,
            "limits": limits_record(), "export_cap_kw": FLEX_EXPORT_KW, "import_cap_kw": FLEX_IMPORT_KW}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
