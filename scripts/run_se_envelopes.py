"""Real-time envelopes from sparse meters on case33bw: forecast vs classical WLS vs GENCO SE.

    python scripts/run_se_envelopes.py --states fluvius --placement telemetry [--run <GENCO SE run dir>]

States: 'fluvius' = case33bw with real Fluvius 2024 homes and simulated EV sessions
(scripts/make_se_fluvius_data.py; validation and test days only, the prior and GENCO see training
days only), or 'synthetic' = gencodoe.se_envelopes.make_states.
Estimators: forecast only; WLS with independent pseudo-measurements (std tuned on validation); WLS
with the full covariance of training-day forecast errors (Bayesian WLS); GENCO SE; GENCO + WLS
(GENCO's voltages as pseudo-measurements, std from validation errors); perfect information.
Every method's envelope is scaled by a factor kappa calibrated on validation states to <= 5% unsafe.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from gencodoe import demo, feeder, se_envelopes as se

WORK = Path(os.environ.get("GENCODOE_WORK", Path.home() / "gencodoe_work"))


def load_fluvius(path, split, n, rng, fd):
    d = np.load(path)
    idx = np.flatnonzero(d["split"] == split)
    if n and n < len(idx):
        idx = np.sort(rng.choice(idx, n, replace=False))
    p, q = d["p"][idx], d["q"][idx]
    return se.States(d["step"][idx] // 4, p, q, d["p_fc"][idx], d["q_fc"][idx], feeder.sweep_pf(fd, p, q))


def train_residual_states(path):
    d = np.load(path)
    m = d["split"] == "train"
    return se.States(d["step"][m] // 4, d["p"][m], d["q"][m], d["p_fc"][m], d["q_fc"][m], None)


def levels(fd, cust, p, q):
    return {d: se.envelope_level(fd, cust, p, q, d) for d in ("export", "import")}


def calibrate_kappa(fd, cust, st, lv, target=0.05):
    out = {}
    for d in ("export", "import"):
        out[d] = 0.0
        for k in np.linspace(1.0, 0.0, 41):
            safe = se.apply_level(fd, cust, st.p, st.q, k * lv[d], d) | (k * lv[d] == 0)
            if 1 - safe.mean() <= target:
                out[d] = float(k)
                break
    return out


def evaluate(fd, cust, st, p, q, lt, kappa):
    r = feeder.sweep_pf(fd, p, q)
    row = {"vm_mae": float(np.abs(r.vm - st.truth.vm)[:, 1:].mean()),
           "p_mae_kw": float(np.abs(p - st.p)[:, 1:].mean() * 1e3)}
    lv = levels(fd, cust, p, q)
    for d in ("export", "import"):
        safe = se.apply_level(fd, cust, st.p, st.q, lv[d], d) | (lv[d] == 0)
        lk = kappa[d] * lv[d]
        safe_k = se.apply_level(fd, cust, st.p, st.q, lk, d) | (lk == 0)
        row[d] = {"unsafe": float(1 - safe.mean()), "level": float(lv[d].mean()), "true": float(lt[d].mean()),
                  "kappa": kappa[d], "unsafe_k": float(1 - safe_k.mean()),
                  "kept_k": float(lk.sum() / max(lt[d].sum(), 1e-9))}
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", choices=["fluvius", "synthetic"], default="fluvius")
    ap.add_argument("--placement", choices=["telemetry", "feeder_ends"], default="telemetry")
    ap.add_argument("--share", type=float, default=0.3, help="share of load buses monitored (telemetry)")
    ap.add_argument("--placement-seed", type=int, default=0)
    ap.add_argument("--run", default=None, help="GENCO SE run dir")
    ap.add_argument("--config", default="configs/genco_se_case33bw.yaml")
    ap.add_argument("--raw", default=str(WORK / "data_se/case33bw/raw"))
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    fd, pd, qd, cust = demo.build(case33bw=True)
    se.set_placement(a.placement, fd.n_bus, a.share, a.placement_seed)
    path = WORK / "se_fluvius/states.npz"
    if a.states == "fluvius":
        val = load_fluvius(path, "val", a.n_val, np.random.default_rng(11), fd)
        st = load_fluvius(path, "test", a.n_test, np.random.default_rng(22), fd)
        hist = train_residual_states(path)
    else:
        val = se.make_states(fd, pd, qd, cust, a.n_val, np.random.default_rng(1001))
        st = se.make_states(fd, pd, qd, cust, a.n_test, np.random.default_rng(2002))
        hist = se.make_states(fd, pd, qd, cust, 4000, np.random.default_rng(3003))
    mval = se.measure(fd, val.truth, val.p, val.q, np.random.default_rng(33))
    meas = se.measure(fd, st.truth, st.p, st.q, np.random.default_rng(44))
    lt = levels(fd, cust, st.p, st.q)
    lt_val = levels(fd, cust, val.p, val.q)
    sigma = se.prior_covariance(hist)

    best = None                                                     # tune the independent-pseudo WLS on validation
    for rel in (0.25, 0.5, 1.0, 2.0):
        se.PSEUDO_REL = rel
        p, q = se.wls(fd, mval, val.p_fc, val.q_fc)
        lv = levels(fd, cust, p, q)
        score = sum(np.abs(lv[d] - lt_val[d]).mean() for d in lv)
        best = (score, rel) if best is None or score < best[0] else best
    se.PSEUDO_REL = best[1]

    methods = {"forecast only": lambda S, M: (S.p_fc, S.q_fc),
               f"WLS, independent prior (std {best[1]:g}x)": lambda S, M: se.wls(fd, M, S.p_fc, S.q_fc),
               "WLS, correlated prior": lambda S, M: se.wls(fd, M, S.p_fc, S.q_fc, prior_cov=sigma)}
    if a.run:
        g = se.GencoSE(a.config, a.run, a.raw, device=a.device)
        cache = {}

        def genco_out(S):
            key = id(S)
            if key not in cache:
                cache[key] = g(fd, S, np.random.default_rng(len(S.p)))
            return cache[key]

        def genco(S, M):
            vm, va, pg, qg = genco_out(S)
            pg, qg = pg.copy(), qg.copy()
            pg[:, 0] = qg[:, 0] = 0.0
            return pg, qg

        vm_val = genco_out(val)[0]
        vm_sd = float(np.std((vm_val - val.truth.vm)[:, 1:]))       # calibrated on validation only
        methods["GENCO SE"] = genco
        methods["GENCO + WLS (correlated prior)"] = lambda S, M: se.wls(
            fd, M, S.p_fc, S.q_fc, prior_cov=sigma, vm_prior=genco_out(S)[0], vm_prior_sd=vm_sd)
    methods["perfect information"] = lambda S, M: (S.p, S.q)

    rows, times = {}, {}
    for name, fn in methods.items():
        pv, qv = fn(val, mval)
        kappa = calibrate_kappa(fd, cust, val, levels(fd, cust, pv, qv))
        t = time.perf_counter()
        p, q = fn(st, meas)
        times[name] = (time.perf_counter() - t) / len(p)
        rows[name] = evaluate(fd, cust, st, p, q, lt, kappa)
        print(f"{name}: done", flush=True)

    head = (f"{a.states} states, placement {a.placement}"
            + (f" ({a.share:.0%} of load buses, seed {a.placement_seed})" if a.placement == "telemetry" else "")
            + f": {len(st.p)} test / {len(val.p)} validation states")
    lines = [f"# Real-time envelopes from sparse meters, case33bw ({head})", "",
             f"Meters (1% noise): voltage at buses {list(se.METER_VM_BUSES)}; P/Q at buses {list(se.METER_INJ_BUSES)}; "
             "feeder-head flow. Envelope: largest proportional export / import level keeping the estimated state "
             "within limits, applied to the true state. 'At <= 5% unsafe': scaled by kappa calibrated on validation; "
             "'kept' = issued envelope / perfect-information envelope, summed over test states.", "",
             "| Method | Vm MAE (pu) | Load P MAE (kW/bus) | Export raw unsafe | Import raw unsafe | Export kept at <= 5% unsafe | "
             "Import kept at <= 5% unsafe (unsafe) | ms/state |", "|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        e, i = r["export"], r["import"]
        lines.append(f"| {name} | {r['vm_mae']:.4f} | {r['p_mae_kw']:.1f} | {e['unsafe']:.1%} | {i['unsafe']:.1%} | "
                     f"{e['kept_k']:.1%} | {i['kept_k']:.1%} ({i['unsafe_k']:.1%}) | {1e3 * times[name]:.2f} |")
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).parent.mkdir(exist_ok=True)
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
