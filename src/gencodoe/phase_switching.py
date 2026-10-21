"""Real-time phase switching on ENWL1: phase-switching devices (PSDs) that reconnect a
single-phase customer to phase a, b or c every interval, chosen to enlarge the envelope.

Building blocks (this module):
  * PhaseSweep: the iterated four-wire backward/forward sweep (gencodoe.fourwire_net) with a
    phase assignment *per row*, so many candidate assignments x scenarios are scored in one batch;
    feasibility per row with the four-wire rules of gencodoe.unbalanced (customer phase-neutral
    voltage within gencodoe.limits, cable current <= NormAmps, transformer |S| <= 800 kVA)
  * fit_thevenin_opendss: the source + transformer seen from the LV root, fitted from OpenDSS states
  * opendss_with_phases: an exact OpenDSS model (UnbalancedENWL1) with customers moved to new phases
  * uniform_level: the largest uniform per-household envelope that passes the design-sample rule
    for a given assignment (same rule as doe_day: >= threshold share of the design scenarios)

The assignment search itself is in phase_switching_run.py.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .enwl import ENWL1_DIR, TR_KVA
from .fourwire_net import FourWireNet, Thevenin, V_PN_NOM, fit_thevenin, load_fourwire
from .limits import VMAX_PU, VMIN_PU

PHASES = 3


@dataclass
class PhaseSweep:
    net: FourWireNet
    th: Thevenin
    iterations: int = 30
    tol: float = 1e-7

    def _currents(self, v, s_kva, phase):
        """Customer injections for per-row phases `phase` (B, n_c) in 0..2."""
        net = self.net
        b = v.shape[0]
        vb = v[:, net.customer_bus]                                    # (B, n_c, 4)
        vph = np.take_along_axis(vb, phase[..., None], axis=2)[..., 0]
        vpn = vph - vb[..., 3]
        i_c = np.conj(s_kva / 1e3 / np.where(np.abs(vpn) > 1e-9, vpn, 1e-9))
        out = np.zeros((b, net.n_bus, 4), complex)
        rows = np.broadcast_to(np.arange(b)[:, None], phase.shape)
        bus = np.broadcast_to(net.customer_bus[None, :], phase.shape)
        np.add.at(out, (rows, bus, phase), i_c)
        np.add.at(out, (rows, bus, np.full_like(phase, 3)), -i_c)
        return out, vpn

    def _branch(self, inj):
        cur = inj.copy()
        for i in self.net.order[::-1][:-1]:
            cur[:, self.net.parent[i]] += cur[:, i]
        return cur

    def solve(self, s_kva: np.ndarray, phase: np.ndarray) -> dict:
        """s_kva (B, n_c) complex consumption (kVA, + = import), phase (B, n_c) or (n_c,) in 0..2."""
        net, th = self.net, self.th
        s_kva = np.atleast_2d(s_kva)
        phase = np.broadcast_to(np.asarray(phase), s_kva.shape)
        b = s_kva.shape[0]
        v = np.zeros((b, net.n_bus, 4), complex)
        v[:, :, :3] = th.e[None, None, :]
        it, delta = 0, np.inf
        for it in range(1, self.iterations + 1):
            inj, _ = self._currents(v, s_kva, phase)
            cur = self._branch(inj)
            vn = np.empty_like(v)
            vn[:, 0, :3] = th.e[None, :] - cur[:, 0, :3] @ th.zth.T
            vn[:, 0, 3] = 0.0
            for i in net.order[1:]:
                vn[:, i] = vn[:, net.parent[i]] - cur[:, i] @ net.z[i].T
            delta = np.abs(vn - v).max(axis=(1, 2)) / V_PN_NOM
            v = vn
            if delta.max() < self.tol:
                break
        inj, vpn = self._currents(v, s_kva, phase)
        cur = self._branch(inj)
        v_pn = np.abs(vpn) / V_PN_NOM
        rating = net.rating_a[1:] / 1e3                                   # kA
        line = (np.abs(cur[:, 1:, :3]) / rating[None, :, None]).max(axis=(1, 2))
        s_root = v[:, 0, :3] * np.conj(cur[:, 0, :3]) * 1e3               # kVA per phase at the LV root
        trafo = np.abs(s_root).sum(axis=1) / TR_KVA
        return {"v_pn_min": v_pn.min(axis=1), "v_pn_max": v_pn.max(axis=1), "line_loading_max": line,
                "transformer_loading": trafo, "converged": delta < 1e-5, "iterations": it}


def feasible(res: dict, direction: str) -> np.ndarray:
    """Direction-specific four-wire feasibility per row (as gencodoe.four_wire.four_wire_ok)."""
    ok = res["converged"] & (res["line_loading_max"] <= 1.0) & (res["transformer_loading"] <= 1.0)
    if direction == "import":
        ok &= res["v_pn_min"] >= VMIN_PU
    else:
        ok &= res["v_pn_max"] <= VMAX_PU
    return ok


def fit_thevenin_opendss(n_states: int = 40, seed: int = 0) -> Thevenin:
    """Fit E, Zth at the LV root from OpenDSS states with random customer loads (and exports)."""
    from .fourwire_labels import Extractor
    ex = Extractor()
    rng = np.random.default_rng(seed)
    V, I = [], []
    for _ in range(n_states):
        p = rng.uniform(-4.0, 6.0, len(ex.net.customer_names))
        st = ex.state(p, 0.33 * np.clip(p, 0, None))
        V.append(st["V"][0, :3]), I.append(st["I"][0, :3])
    return fit_thevenin(np.array(V), np.array(I))


def opendss_with_phases(phase: np.ndarray, base: Path = ENWL1_DIR):
    """UnbalancedENWL1 with every customer on phase[i] (0..2): a copy of the feeder folder with
    Loads.dss rewritten (Bus1=<bus>.<phase>.4), so the engine's voltage monitoring follows."""
    from .unbalanced import UnbalancedENWL1
    tmp = Path(tempfile.mkdtemp(prefix="enwl1_phases_"))
    for f in base.iterdir():
        shutil.copy(f, tmp / f.name)
    lines = (base / "Loads.dss").read_text().splitlines()
    out, k = [], 0
    for line in lines:
        m = re.search(r"(Bus1=)(\w+)\.([123])\.4", line, re.I)
        if m and line.lower().startswith("new load"):
            line = line[:m.start()] + f"{m.group(1)}{m.group(2)}.{int(phase[k]) + 1}.4" + line[m.end():]
            k += 1
        out.append(line)
    if k != len(phase):
        raise ValueError(f"{k} customers in Loads.dss, {len(phase)} phases given")
    (tmp / "Loads.dss").write_text("\n".join(out) + "\n")
    return UnbalancedENWL1(folder=tmp)


def uniform_level(score, s_design: np.ndarray, direction: str, cap_kw: float, threshold: float,
                  tol_kw: float = 0.05, levels_per_round: int = 8) -> tuple[float, int]:
    """Largest uniform per-household level L in [0, cap] (kW) with share feasible >= threshold, where
    score(s_kva) -> bool per row. Rows: design scenarios with every customer at +L (import) or -L (export)
    kW on top of its passive consumption. Returns (L, rows evaluated). k-section search as doe_day."""
    sign = 1.0 if direction == "import" else -1.0
    n_s = len(s_design)
    calls = 0

    def share(levels):
        nonlocal calls
        rows = (s_design[None, :, :] + sign * np.asarray(levels)[:, None, None]).reshape(-1, s_design.shape[1])
        calls += len(rows)
        return score(rows).reshape(len(levels), n_s).mean(axis=1)

    hi_ok, lo_ok = share([cap_kw, 0.0])
    if hi_ok >= threshold:
        return cap_kw, calls
    if lo_ok < threshold:
        return 0.0, calls
    lo, hi = 0.0, cap_kw
    while hi - lo > tol_kw:
        levels = np.linspace(lo, hi, levels_per_round + 2)[1:-1]
        ok = share(levels) >= threshold
        if ok.all():
            lo = levels[-1]
        else:
            i = int(np.argmin(ok))
            hi = levels[i]
            lo = levels[i - 1] if i > 0 else lo
    return float(lo), calls
