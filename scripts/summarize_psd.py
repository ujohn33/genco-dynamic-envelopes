"""Summarise phase_switching_run outputs into results/phase_switching_fluvius.md.

    python scripts/summarize_psd.py ~/gencodoe_work/psd/2024-06-28.json ~/gencodoe_work/psd/2024-12-15.json

Decision question: how often does the sweep-guided search clearly beat the no-power-flow balance
heuristic (larger envelope, or, where the envelope stays 0, more design scenarios within limits)?
A large, frequent gap is what a learned (GENCO-style) switching policy would have to capture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

LEVEL_GAP_KW = 0.10      # "clearly larger envelope"
SHARE_GAP = 0.05         # "clearly more scenarios within limits" (when the envelope is 0 for both)


def main(files, out="results/phase_switching_fluvius.md"):
    rows, metas = [], []
    for f in files:
        d = json.loads(Path(f).expanduser().read_text())
        if d["meta"]["synthetic_profiles"]:
            raise SystemExit(f"{f}: synthetic profiles; this summary is for Fluvius runs only")
        metas.append(d["meta"])
        for r in d["rows"]:
            rows.append({**r, "day": d["meta"]["day"]})
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["day", "hour", "direction"], columns="method",
                         values=["level_kw", "design_share_opendss", "held_out_ok", "switches", "time_s"])

    def lv(m):
        return piv["level_kw"][m]

    def sh(m):
        return piv["design_share_opendss"][m]

    def better(a, b):            # a clearly better than b, per interval
        return (lv(a) > lv(b) + LEVEL_GAP_KW) | ((lv(a) == 0) & (lv(b) == 0) & (sh(a) > sh(b) + SHARE_GAP))

    n = len(piv)
    m = metas[0]
    out_lines = [
        "# Real-time phase switching on ENWL1 with Fluvius 2024 households",
        "",
        f"Exploratory. {n} interval-directions on {', '.join(sorted({x['day'] for x in metas}))} "
        f"(hours {', '.join(str(h) for x in metas for h in x['hours'])}). {m['psd_customers']} heat-pump / EV homes "
        f"have a phase-switching device; {m['scenarios']} design and {m['verify']} held-out scenarios per interval; "
        f"design rule: >= {m['threshold']:.0%} of design scenarios within all four-wire limits (OpenDSS). "
        f"Uniform envelope per home, caps {m['import_cap_kw']} kW import / {m['export_cap_kw']} kW export.",
        "",
        "## Headline", "",
        "| | Fixed phases | Balance heuristic | Sweep-guided search |", "|---|---|---|---|",
    ]
    for d in ("import", "export"):
        sel = piv.index.get_level_values("direction") == d
        out_lines.append(f"| Mean {d} envelope (kW per home) | " + " | ".join(
            f"{lv(mm)[sel].mean():.2f}" for mm in ("fixed", "balance", "search")) + " |")
    out_lines += [
        "| Intervals with envelope > 0 | " + " | ".join(f"{(lv(mm) > 0).sum()} / {n}" for mm in ("fixed", "balance", "search")) + " |",
        "| Mean design share within limits (OpenDSS) | " + " | ".join(f"{sh(mm).mean():.1%}" for mm in ("fixed", "balance", "search")) + " |",
        "| Mean held-out share within limits | " + " | ".join(f"{piv['held_out_ok'][mm].mean():.1%}" for mm in ("fixed", "balance", "search")) + " |",
        "| Mean switches per interval | 0 | " + " | ".join(f"{piv['switches'][mm].mean():.0f}" for mm in ("balance", "search")) + " |",
        "| Mean decision time per interval-direction (s, laptop CPU) | " + " | ".join(f"{piv['time_s'][mm].mean():.1f}" for mm in ("fixed", "balance", "search")) + " |",
        "",
        "## Does the search beat the heuristic?", "",
        f"Clearly better = envelope larger by > {LEVEL_GAP_KW} kW, or, where both envelopes are 0, "
        f"design share higher by > {SHARE_GAP:.0%} points.", "",
        "| Comparison | Intervals clearly better | clearly worse |", "|---|---|---|",
        f"| balance vs fixed | {better('balance', 'fixed').sum()} / {n} | {better('fixed', 'balance').sum()} / {n} |",
        f"| search vs fixed | {better('search', 'fixed').sum()} / {n} | {better('fixed', 'search').sum()} / {n} |",
        f"| **search vs balance** | **{better('search', 'balance').sum()} / {n}** | {better('balance', 'search').sum()} / {n} |",
        "", "## Per interval", "",
        "| Day | Hour | Direction | Fixed kW (design share) | Balance kW (share, switches) | Search kW (share, switches) |",
        "|---|---|---|---|---|---|",
    ]
    for (day, hour, d), r in piv.iterrows():
        cell = lambda mm: f"{r[('level_kw', mm)]:.2f} ({r[('design_share_opendss', mm)]:.1%}" + (
            f", {r[('switches', mm)]:.0f})" if mm != "fixed" else ")")
        out_lines.append(f"| {day} | {hour:02d}:00 | {d} | {cell('fixed')} | {cell('balance')} | {cell('search')} |")
    text = "\n".join(out_lines) + "\n"
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
