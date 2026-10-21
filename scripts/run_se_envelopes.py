"""Real-time envelopes from sparse meters on case33bw: forecast vs classical WLS vs GENCO SE.

    python scripts/run_se_envelopes.py [--run <mlflow SE run dir>] [--out results/se_envelopes_case33bw.md]

WLS's pseudo-measurement uncertainty is tuned on separate validation states (lowest envelope MAE
over both directions) before the test states are touched. See gencodoe.se_envelopes.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from gencodoe import demo, feeder, se_envelopes as se

WORK = Path(os.environ.get("GENCODOE_WORK", Path.home() / "gencodoe_work"))


def levels(fd, cust, p, q):
    return {d: se.envelope_level(fd, cust, p, q, d) for d in ("export", "import")}


def calibrate_kappa(fd, cust, st, lv, target=0.05):
    """Largest shrink factor per direction with at most `target` unsafe envelopes on these states."""
    out = {}
    for d in ("export", "import"):
        best = 0.0
        for k in np.linspace(1.0, 0.0, 41):
            safe = se.apply_level(fd, cust, st.p, st.q, k * lv[d], d) | (k * lv[d] == 0)
            if 1 - safe.mean() <= target:
                best = float(k)
                break
        out[d] = best
    return out


def evaluate(fd, cust, st, p, q, levels_true, kappa=None):
    r = feeder.sweep_pf(fd, p, q)
    row = {"vm_mae": float(np.abs(r.vm - st.truth.vm)[:, 1:].mean()),
           "p_mae_kw": float(np.abs(p - st.p)[:, 1:].mean() * 1e3 * fd.base_mva)}
    lv_all = levels(fd, cust, p, q)
    for d in ("export", "import"):
        lv = lv_all[d]
        safe = se.apply_level(fd, cust, st.p, st.q, lv, d) | (lv == 0)
        row[d] = {"unsafe": float(1 - safe.mean()), "level": float(lv.mean()), "true": float(levels_true[d].mean()),
                  "mae": float(np.abs(lv - levels_true[d]).mean())}
        if kappa is not None:
            lk = kappa[d] * lv
            safe_k = se.apply_level(fd, cust, st.p, st.q, lk, d) | (lk == 0)
            row[d].update({"kappa": kappa[d], "unsafe_k": float(1 - safe_k.mean()),
                           "kept_k": float(lk.sum() / max(levels_true[d].sum(), 1e-9))})
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="GENCO SE run dir (omit: classical methods only)")
    ap.add_argument("--config", default="configs/genco_se_case33bw.yaml")
    ap.add_argument("--raw", default=str(WORK / "data_doe/case33bw/raw"))
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/se_envelopes_case33bw.md")
    a = ap.parse_args(argv)
    fd, pd, qd, cust = demo.build(case33bw=True)

    # tune WLS pseudo-measurement uncertainty on validation states
    rng = np.random.default_rng(1001)
    val = se.make_states(fd, pd, qd, cust, a.n_val, rng)
    mval = se.measure(fd, val.truth, val.p, val.q, rng)
    lt_val = {d: se.envelope_level(fd, cust, val.p, val.q, d) for d in ("export", "import")}
    best = None
    for rel in (0.25, 0.5, 1.0, 2.0):
        se.PSEUDO_REL = rel
        p, q = se.wls(fd, mval, val.p_fc, val.q_fc)
        r = evaluate(fd, cust, val, p, q, lt_val)
        score = r["export"]["mae"] + r["import"]["mae"]
        print(f"WLS pseudo std {rel}: val envelope MAE {score:.3f}, unsafe {r['export']['unsafe']:.1%}/{r['import']['unsafe']:.1%}")
        if best is None or score < best[0]:
            best = (score, rel)
    se.PSEUDO_REL = best[1]

    # test states; every method's shrink factor kappa is calibrated on the validation states
    rng = np.random.default_rng(2002)
    st = se.make_states(fd, pd, qd, cust, a.n_test, rng)
    meas = se.measure(fd, st.truth, st.p, st.q, rng)
    lt = levels(fd, cust, st.p, st.q)
    methods = {"forecast only": lambda S, M: (S.p_fc, S.q_fc),
               f"WLS (pseudo std {best[1]:g})": lambda S, M: se.wls(fd, M, S.p_fc, S.q_fc)}
    if a.run:
        g = se.GencoSE(a.config, a.run, a.raw, device=a.device)

        def genco(S, M, _g=g):
            vm, va, pg, qg = _g(fd, S, np.random.default_rng(len(S.p)))
            pg[:, 0] = qg[:, 0] = 0.0
            genco.vm = vm
            return pg, qg
        methods["GENCO SE"] = genco
    methods["perfect information"] = lambda S, M: (S.p, S.q)
    rows, times = {}, {}
    for name, fn in methods.items():
        pv, qv = fn(val, mval)
        kappa = calibrate_kappa(fd, cust, val, levels(fd, cust, pv, qv))
        t = time.perf_counter()
        p, q = fn(st, meas)
        times[name] = (time.perf_counter() - t) / len(p)
        rows[name] = evaluate(fd, cust, st, p, q, lt, kappa)
        if name == "GENCO SE":
            rows[name]["vm_mae_direct"] = float(np.abs(methods[name].vm - st.truth.vm)[:, 1:].mean())

    lines = ["# Real-time envelopes from sparse meters, case33bw", "",
             f"{a.n_test} test states (envelope-like: passive load and PV at a random hour plus unobserved flexible "
             f"export/import). Meters, 1% relative noise: voltage at buses {list(se.METER_VM_BUSES)}, substation P/Q, "
             f"feeder-head flow. Envelope: largest proportional export / import level (share of each connection's cap) "
             f"keeping the estimated state within limits, then applied to the true state.", "",
             "## Raw envelopes", "",
             "| Method | Vm MAE (pu) | Load P MAE (kW/bus) | Export: unsafe | Export: mean level (true) | Export MAE | "
             "Import: unsafe | Import: mean level (true) | Import MAE | ms/state |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        e, i = r["export"], r["import"]
        lines.append(f"| {name} | {r['vm_mae']:.4f} | {r['p_mae_kw']:.1f} | {e['unsafe']:.1%} | {e['level']:.3f} ({e['true']:.3f}) | "
                     f"{e['mae']:.3f} | {i['unsafe']:.1%} | {i['level']:.3f} ({i['true']:.3f}) | {i['mae']:.3f} | "
                     f"{1e3 * times[name]:.2f} |")
    lines += ["", "## At matched safety", "",
              f"Each method's envelopes are scaled by a factor kappa chosen on {a.n_val} validation states so that at most "
              "5% are unsafe; test results below. 'Kept' = issued envelope summed over states / perfect-information envelope.", "",
              "| Method | Export: kappa | Export: unsafe | Export: kept | Import: kappa | Import: unsafe | Import: kept |",
              "|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        e, i = r["export"], r["import"]
        lines.append(f"| {name} | {e['kappa']:.3f} | {e['unsafe_k']:.1%} | {e['kept_k']:.1%} | {i['kappa']:.3f} | "
                     f"{i['unsafe_k']:.1%} | {i['kept_k']:.1%} |")
    if "GENCO SE" in rows:
        lines += ["", f"GENCO's own voltage estimate: Vm MAE {rows['GENCO SE']['vm_mae_direct']:.4f} pu "
                  "(the table's Vm MAE re-solves the estimated injections)."]
    lines += ["", f"WLS pseudo-measurement std tuned on {a.n_val} validation states: {best[1]:g} x forecast."]
    text = "\n".join(lines) + "\n"
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
