"""Figures explaining the real-time phase-switching setup on ENWL1 (figures/psd_*.png).

    GENCODOE_VOLTAGE_LIMITS=uk_esqcr python scripts/make_psd_figures.py \
        --example ~/gencodoe_work/psd/example_2024-12-15_18h.json

The example file is a phase_switching_run output with --save-phases (one interval).
Colours: reference palette of the dataviz method, validated all-pairs: phases a/b/c =
blue/orange/aqua; methods fixed/balance/search = violet/magenta/green; seasons Dec/Jun = blue/orange.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from gencodoe import fluvius  # noqa: E402
from gencodoe.doe_day import DayScenarios  # noqa: E402
from gencodoe.enwl import ENWL1_DIR, load_enwl1  # noqa: E402
from gencodoe.fourwire_net import load_fourwire  # noqa: E402
from gencodoe.limits import VMIN_PU, VMIN_V, V_BASE  # noqa: E402
from gencodoe.phase_switching import PhaseSweep, fit_thevenin_opendss  # noqa: E402
from gencodoe.phase_switching_run import feeder_of  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
PHASE = ["#2a78d6", "#eb6834", "#1baf7a"]
METHOD = {"fixed": "#6250d6", "balance": "#e87ba4", "search": "#008300"}
METHOD_LABEL = {"fixed": "Fixed phases (as built)", "balance": "Balance heuristic", "search": "Sweep-guided search"}
SEASON = {"December": "#2a78d6", "June": "#eb6834"}

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"], "font.size": 9.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2.0,
})


def headline(fig, title, subtitle):
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=12.5, weight="bold", color=INK)
    fig.text(0.012, 0.935, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)


def coords():
    xy = {}
    for line in (ENWL1_DIR / "Buscoords.dss").read_text().splitlines():
        p = [s.strip() for s in line.split(",")]
        if len(p) == 3:
            xy[p[0].lower()] = (float(p[1]), float(p[2]))
    segs = []
    for m in re.finditer(r"Bus1=(\w+)\.\S*\s+Bus2=(\w+)\.", (ENWL1_DIR / "Lines.dss").read_text(), re.I):
        a, b = m.group(1).lower(), m.group(2).lower()
        if a in xy and b in xy:
            segs.append((xy[a], xy[b]))
    load_bus = [m.group(1).lower() for m in re.finditer(r"New Load\.\w+.*?Bus1=(\w+)\.", (ENWL1_DIR / "Loads.dss").read_text(), re.I)]
    return xy, segs, load_bus


def fig_feeder(ex, out):
    from matplotlib.collections import LineCollection
    xy, segs, load_bus = coords()
    phase0 = np.array(ex["meta"]["feeder_phases"])
    psd = np.array(ex["meta"]["psd_mask"])
    fw = load_fourwire()
    feeder = feeder_of(fw)
    pts = np.array([xy[b] for b in load_bus])
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.84)
    ax.add_collection(LineCollection(segs, colors=AXIS, linewidths=0.9, zorder=1))
    for ph in range(3):
        for dev, marker, size in ((False, "o", 22), (True, "D", 34)):
            m = (phase0 == ph) & (psd == dev)
            ax.scatter(pts[m, 0], pts[m, 1], s=size, marker=marker, color=PHASE[ph], edgecolor=SURFACE,
                       linewidth=0.8, zorder=3)
    tr = xy["trbus"]
    ax.scatter(*tr, s=170, marker="s", color=INK, zorder=4)
    ax.annotate("800 kVA transformer\n11 / 0.416 kV", tr, xytext=(60, 70), textcoords="offset points",
                fontsize=9, color=INK, va="bottom", bbox=dict(fc=SURFACE, ec="none", pad=1.5),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    for k, f in enumerate(np.unique(feeder)):
        far = pts[feeder == f][np.argmax(np.hypot(*(pts[feeder == f] - np.array(tr)).T))]
        ax.annotate(f"feeder {k + 1}", far, xytext=(6, -14), textcoords="offset points", fontsize=8.5, color=INK2,
                    bbox=dict(fc=SURFACE, ec="none", pad=1.0), zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, color=PHASE[p], mec=SURFACE,
                          label=f"phase {'abc'[p]}: {(phase0 == p).sum()} homes") for p in range(3)]
    handles += [plt.Line2D([], [], ls="", marker="D", ms=7, color=MUTED, mec=SURFACE,
                           label=f"phase-switching device: {psd.sum()} heat-pump / EV homes"),
                plt.Line2D([], [], ls="", marker="o", ms=6, color=MUTED, mec=SURFACE, label=f"no device: {(~psd).sum()} homes")]
    ax.legend(handles=handles, loc="lower left", fontsize=9, labelcolor=INK2)
    headline(fig, "ENWL1 low-voltage feeder: 200 single-phase homes, 104 can switch phase",
             "Four-wire UK benchmark, populated with Fluvius 2024 smart-meter homes (30% heat pump, 30% EV, 40% PV).\n"
             "A device moves a home between phases each interval.")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_profiles(out):
    groups = [("geen_ZP", "No heat pump, EV or PV"), ("enkel_ZP", "PV only"), ("WP_geen_ZP", "Heat pump"),
              ("EV_geen_ZP", "Electric vehicle")]
    months = {"December": (334, 366), "June": (152, 182)}
    hours = np.arange(96) / 4
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.9), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.17, top=0.74, wspace=0.12)
    for ax, (g, title) in zip(axes, groups):
        d = fluvius.load_group(g)
        net = d.offtake_kw - d.injection_kw                                 # (meters, steps), + = consumption
        for month, (a, b) in months.items():
            prof = net[:, a * 96:b * 96].reshape(len(d.ean), b - a, 96).mean(axis=(0, 1))
            ax.plot(hours, prof, color=SEASON[month], label=month)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlim(0, 24)
        ax.set_xlabel("hour of day")
    axes[0].set_ylabel("mean net consumption per home (kW)\nnegative = export")
    axes[0].legend(loc="upper left", fontsize=9, labelcolor=INK2)
    headline(fig, "Why phases matter: heat pumps and EVs load winter evenings, PV exports at summer noon",
             "Mean daily profile per household type, Fluvius 2024 (300 meters per type). Each home is single-phase, "
             "so these loads pile up on whichever phase it is wired to.")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def example_states(ex):
    bal = load_enwl1()
    scen = DayScenarios(bal.customer_bus, bal.n_bus, seed=ex["meta"]["seed"])
    rng = np.random.default_rng(ex["meta"]["seed"])
    step = ex["rows"][0]["step"]
    p, q = scen.sample(step, ex["meta"]["scenarios"], rng)
    cb = bal.customer_bus
    return (-p[:, cb] - 1j * q[:, cb]) * 1e3


def fig_balance(ex, s_design, out):
    fw = load_fourwire()
    feeder = feeder_of(fw)
    fids = np.unique(feeder)
    mean_kw = s_design.real.mean(axis=0)
    rows = {r["method"]: r for r in ex["rows"]}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.15, top=0.74, wspace=0.08)
    w = 0.26
    ymax = 0
    for ax, m in zip(axes, ("fixed", "balance", "search")):
        ph = np.array(rows[m]["phases"])
        tot = np.zeros((len(fids), 3))
        for k, f in enumerate(fids):
            for p in range(3):
                tot[k, p] = mean_kw[(feeder == f) & (ph == p)].sum()
        ymax = max(ymax, tot.max())
        for p in range(3):
            ax.bar(np.arange(len(fids)) + (p - 1) * (w + 0.02), tot[:, p], width=w, color=PHASE[p],
                   label=f"phase {'abc'[p]}", zorder=2)
        spread = (tot.max(1) - tot.min(1)).sum()
        ax.set_title(f"{METHOD_LABEL[m]}", fontsize=10)
        ax.text(0.98, 0.97, f"phase spread {spread:.0f} kW\n{rows[m]['moved_from_feeder_phases']} homes moved",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
        ax.set_xticks(np.arange(len(fids)), [f"feeder {k + 1}" for k in range(len(fids))])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("expected load on the phase (kW)")
    axes[0].legend(loc="upper left", fontsize=9, labelcolor=INK2)
    for ax in axes:
        ax.set_ylim(0, ymax * 1.18)
    headline(fig, "Switching rebalances each feeder's phases (15 Dec 2024, 18:00)",
             "Expected load per phase = mean over 200 forecast scenarios of the homes connected to it. "
             "Phase spread = sum over feeders of (most loaded - least loaded phase).")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_voltage(ex, s_design, out):
    from gencodoe.phase_switching import feasible
    sw = PhaseSweep(load_fourwire(), fit_thevenin_opendss())
    rows = {r["method"]: r for r in ex["rows"]}
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.76)
    for i, m in enumerate(("fixed", "balance", "search")):
        res = sw.solve(s_design, np.array(rows[m]["phases"]))
        v = np.sort(res["v_pn_min"] * V_BASE)
        share = np.arange(1, len(v) + 1) / len(v)
        below = float((v < VMIN_V).mean())
        all_ok = float(feasible(res, "import").mean())
        ax.step(v, share, where="post", color=METHOD[m],
                label=f"{METHOD_LABEL[m]}: {1 - below:.0%} above {VMIN_V:.1f} V, {all_ok:.0%} within all limits")
        ax.plot([VMIN_V], [below], "o", ms=8, color=METHOD[m], mec=SURFACE, mew=2, zorder=4)
    ax.axvline(VMIN_V, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(VMIN_V, 1.02, f"statutory minimum {VMIN_V:.1f} V", color=INK, fontsize=9, va="bottom", ha="center",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("lowest phase-to-neutral voltage of any home in the scenario (V)")
    ax.set_ylabel("share of forecast scenarios at or below")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK2)
    headline(fig, "Same homes, same loads: switching lifts the worst voltage in more scenarios",
             "15 Dec 2024, 18:00, no flexible import added; 200 forecast scenarios. 'All limits' adds cable and transformer "
             "loading:\nthe remaining failures are cable overloads. An envelope needs >= 98% within all limits, so it stays "
             "0 kW here, but fewer scenarios violate.")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_loop(out):
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.84)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 26)
    steps = [("Forecast", "200 scenarios of every\nhome's load for the\nnext 15-min interval"),
             ("Candidates", "phase moves for the\n104 homes with a\nswitching device"),
             ("Fast scoring", "batched four-wire sweep\nscores all candidates\n(~1 ms per state)"),
             ("Envelope", "largest uniform kW limit\npassing >= 98% of\nscenarios (k-section)"),
             ("Exact check", "OpenDSS four-wire model;\nscale down if the rule\nfails"),
             ("Issue", "phases to the devices,\nimport / export limit\nto every home")]
    w, gap = 14.2, 2.4
    for i, (t, body) in enumerate(steps):
        x = 1 + i * (w + gap)
        exact = t == "Exact check"
        ax.add_patch(FancyBboxPatch((x, 6), w, 18, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="#f0efec" if not exact else "#e3eefb", ec=AXIS if not exact else PHASE[0], lw=1))
        ax.text(x + w / 2, 21.0, t, ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
        ax.text(x + w / 2, 13.0, body, ha="center", va="center", fontsize=8.6, color=INK2, linespacing=1.35)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.3, 15), (x + w + gap - 0.3, 15), arrowstyle="-|>",
                                         mutation_scale=11, color=MUTED, lw=1.2))
    ax.add_patch(FancyArrowPatch((1 + 5 * (w + gap) + w / 2, 5.6), (1 + w / 2, 5.6), connectionstyle="arc3,rad=-0.08",
                                 arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1, ls=(0, (3, 3))))
    ax.text(50, -3.6, "next interval: repeat, counting how many devices switched", ha="center", fontsize=8.6, color=MUTED)
    headline(fig, "The decision every interval: which phase each switchable home uses, and the envelope that follows",
             "Fixed phases skip steps 2-3; the balance heuristic replaces step 3 by per-feeder load balancing "
             "(no power flow); the search uses the sweep. Only the exact check decides what is issued.")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("figures"))
    a = ap.parse_args()
    ex = json.loads(a.example.read_text())
    if ex["meta"].get("synthetic_profiles"):
        raise SystemExit("example was run on synthetic profiles; figures must use Fluvius data")
    a.outdir.mkdir(exist_ok=True)
    fig_feeder(ex, a.outdir / "psd_1_feeder.png")
    fig_profiles(a.outdir / "psd_2_profiles.png")
    s_design = example_states(ex)
    fig_balance(ex, s_design, a.outdir / "psd_3_phase_balance.png")
    fig_voltage(ex, s_design, a.outdir / "psd_4_worst_voltage.png")
    fig_loop(a.outdir / "psd_5_decision_loop.png")
    print("wrote", sorted(str(p) for p in a.outdir.glob("psd_*.png")))


if __name__ == "__main__":
    main()
