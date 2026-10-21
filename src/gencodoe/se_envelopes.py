"""Real-time envelopes from sparse measurements: GENCO state estimation vs classical WLS.

Setting (case33bw, balanced): the operator sees only a few live meters and must update the
envelope for the current interval. Home loads, including the flexible EV / heat-pump part, are
not measured.

  meters (fixed placement, 1% relative noise), two placements (set_placement):
    telemetry    substation voltage and P/Q, feeder-head flow, and voltage + P/Q at a random share of
                 load buses (the connection-point telemetry of participating homes, as CSIP-AUS
                 inverters report it; the practice-aligned case)
    feeder_ends  substation voltage and P/Q, feeder-head flow, voltage at the ends of the main
                 feeder and laterals (dedicated network monitors)

Estimators of each true state:
  forecast  the hour's point forecast of passive load and PV (gencodoe.demo), no meters
  wls       classical load-based WLS state estimation: unknown P/Q at every load bus, meters plus
            the forecast as pseudo-measurements (relative std PSEUDO_REL); Gauss-Newton with the
            exact sweep for h(x) and LinDistFlow sensitivities as the Jacobian
  genco     GENCO in StateEstimation mode (gridfm-graphkit): the meters as masked, noisy inputs,
            no forecast; outputs Vm, Va and the net injections implied by them

Each estimate gives net injections per bus; the real-time envelope is the largest uniform export
(or import) per flexible connection point that keeps that estimated state within the limits
(exact sweep). It is then applied to the TRUE state: safe or not, and how it compares with the
envelope computed from the true state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import demo, envelopes, feeder

METER_VM_BUSES = (0, 17, 21, 24, 32)       # substation + ends of the main feeder and the laterals
METER_INJ_BUSES = (0,)                      # substation P/Q (slack generator)
METER_FLOW_BRANCHES = ((0, 1),)             # feeder-head flow, measured at the substation end
NOISE_REL = 0.01
PSEUDO_REL = 0.5
PLACEMENT = "feeder_ends"


def set_placement(name: str, n_bus: int = 33, share: float = 0.3, seed: int = 0):
    """Select the meter placement used by measure(), wls() and GencoSE (module-level, fixed)."""
    global METER_VM_BUSES, METER_INJ_BUSES, PLACEMENT
    if name == "feeder_ends":
        METER_VM_BUSES, METER_INJ_BUSES = (0, 17, 21, 24, 32), (0,)
    elif name == "telemetry":
        rng = np.random.default_rng(seed)
        k = int(round(share * (n_bus - 1)))
        homes = tuple(int(b) for b in np.sort(rng.choice(np.arange(1, n_bus), k, replace=False)))
        METER_VM_BUSES, METER_INJ_BUSES = (0,) + homes, (0,) + homes
    else:
        raise ValueError(name)
    PLACEMENT = name
    return METER_VM_BUSES, METER_INJ_BUSES


@dataclass
class States:
    hour: np.ndarray        # (n,)
    p: np.ndarray           # (n, n_bus) true net injection, pu (+ = export)
    q: np.ndarray
    p_fc: np.ndarray        # forecast (passive part only)
    q_fc: np.ndarray
    truth: feeder.PFResult


def make_states(fd, pd, qd, cust, n, rng, batch=25) -> States:
    """Envelope-like true states (as in training) with the hour kept, plus each hour's point forecast."""
    C = cust.incidence(fd.n_bus)
    tan_phi = np.divide(qd, pd, out=np.zeros_like(pd), where=pd > 0)
    H, P, Q, PF, QF = [], [], [], [], []
    while sum(len(x) for x in P) < n:
        h = int(rng.integers(24))
        p, q = demo.scenario_set(fd, pd, qd, h, batch, rng)
        u = rng.uniform(size=batch)
        sign = np.where(u < 0.45, 1.0, np.where(u < 0.8, -1.0, 0.0))
        level = rng.uniform(0.0, 1.2, batch)
        lim = np.where(sign[:, None] > 0, envelopes.allocate(cust, "proportional", level, "export"),
                       envelopes.allocate(cust, "proportional", level, "import"))
        P.append(p + sign[:, None] * lim @ C), Q.append(q), H.append(np.full(batch, h))
        load = pd * demo.LOAD_SHAPE[h]
        pv = 0.5 * pd * demo.PV_SHAPE[h]
        PF.append(np.tile(pv - load, (batch, 1))), QF.append(np.tile(-load * tan_phi, (batch, 1)))
    cut = lambda xs: np.vstack(xs)[:n] if xs[0].ndim > 1 else np.concatenate(xs)[:n]
    p, q = cut(P), cut(Q)
    return States(cut(H), p, q, cut(PF), cut(QF), feeder.sweep_pf(fd, p, q))


def measure(fd, truth: feeder.PFResult, p, q, rng) -> dict:
    """Noisy meter readings of the true states (pu)."""
    n = len(p)
    v = truth.vm * np.exp(1j * truth.va)
    i_br = (v[:, [f for f, _ in METER_FLOW_BRANCHES]] - v[:, [t for _, t in METER_FLOW_BRANCHES]]) / \
        fd.z[[t for _, t in METER_FLOW_BRANCHES]][None, :]
    s_flow = v[:, [f for f, _ in METER_FLOW_BRANCHES]] * np.conj(i_br)
    s_slack = s_flow.sum(axis=1)                                     # substation injection = head flow here
    noise = lambda x: x * (1 + NOISE_REL * rng.standard_normal(np.shape(x)))
    homes = [b for b in METER_INJ_BUSES if b != 0]
    return {"vm": noise(truth.vm[:, list(METER_VM_BUSES)]),
            "p_flow": noise(s_flow.real), "q_flow": noise(s_flow.imag),
            "p_slack": noise(s_slack.real), "q_slack": noise(s_slack.imag),
            "p_inj": noise(p[:, homes]), "q_inj": noise(q[:, homes])}


def prior_covariance(states: States, ridge: float = 1e-8) -> np.ndarray:
    """Covariance of forecast errors [p - p_fc, q - q_fc] at non-slack buses, from historical states."""
    d = np.hstack([states.p[:, 1:] - states.p_fc[:, 1:], states.q[:, 1:] - states.q_fc[:, 1:]])
    return np.cov(d, rowvar=False) + ridge * np.eye(d.shape[1])


def wls(fd, meas, p_fc, q_fc, iters=8, vm_prior=None, vm_prior_sd=None, prior_cov=None):
    """Load-based WLS: x = [p, q] at non-slack buses; z = meters + pseudo-measurements x_fc.
    prior_cov: full covariance of forecast errors (prior_covariance); replaces the diagonal
    pseudo-measurement weights, so correlated behaviour across homes is used (Bayesian WLS).
    vm_prior (n, n_bus) / vm_prior_sd: optional voltage pseudo-measurements at every unmetered bus
    (e.g. GENCO's state estimate), added to the meters with that standard deviation."""
    n, nb = p_fc.shape
    R, X = feeder.sensitivity(fd)
    buses = np.arange(1, nb)
    vm_b = list(METER_VM_BUSES)
    if vm_prior is not None:
        extra = [b for b in range(1, nb) if b not in vm_b]
        meas = {**meas, "vm": np.hstack([meas["vm"], vm_prior[:, extra]])}
        vm_sd = np.concatenate([np.full(len(vm_b), np.nan), np.full(len(extra), vm_prior_sd)])
        vm_b = vm_b + extra
    else:
        vm_sd = np.full(len(vm_b), np.nan)
    head = [t for _, t in METER_FLOW_BRANCHES]
    # Jacobian of meters wrt x (LinDistFlow): dV = R p + X q; head flow (parent->child) = -sum of subtree injections
    Jv = np.hstack([R[np.ix_(vm_b, buses)], X[np.ix_(vm_b, buses)]])
    Tsub = fd.T[np.ix_(buses, head)].T                                 # (n_head, nb-1)
    Jp = np.hstack([-Tsub, np.zeros_like(Tsub)])
    Jq = np.hstack([np.zeros_like(Tsub), -Tsub])
    homes = [b for b in METER_INJ_BUSES if b != 0]                    # direct P/Q telemetry rows
    idx = np.array([b - 1 for b in homes], int)
    Ji = np.zeros((2 * len(homes), 2 * (nb - 1)))
    Ji[np.arange(len(homes)), idx] = 1.0
    Ji[len(homes) + np.arange(len(homes)), (nb - 1) + idx] = 1.0
    J = np.vstack([Jv, Jp, Jq, Ji])
    out_p, out_q = np.empty_like(p_fc), np.empty_like(q_fc)
    for s in range(n):
        x_fc = np.concatenate([p_fc[s, 1:], q_fc[s, 1:]])
        sd_ps = np.maximum(PSEUDO_REL * np.abs(x_fc), 0.02 * np.abs(x_fc).max() + 1e-4)
        z = np.concatenate([meas["vm"][s], meas["p_flow"][s], meas["q_flow"][s], meas["p_inj"][s], meas["q_inj"][s]])
        sd_v = np.where(np.isnan(vm_sd), NOISE_REL * np.abs(meas["vm"][s]) + 1e-4, vm_sd)
        sd_z = np.concatenate([sd_v,
                               NOISE_REL * np.abs(meas["p_flow"][s]) + 1e-4, NOISE_REL * np.abs(meas["q_flow"][s]) + 1e-4,
                               NOISE_REL * np.abs(meas["p_inj"][s]) + 1e-4, NOISE_REL * np.abs(meas["q_inj"][s]) + 1e-4])
        Wz = 1 / sd_z ** 2
        P_inv = np.linalg.inv(prior_cov) if prior_cov is not None else np.diag(1 / sd_ps ** 2)
        x = x_fc.copy()
        for _ in range(iters):
            pp = np.concatenate([[0.0], x[: nb - 1]])
            qq = np.concatenate([[0.0], x[nb - 1:]])
            r = feeder.sweep_pf(fd, pp[None], qq[None])
            v = r.vm[0] * np.exp(1j * r.va[0])
            i_h = (v[[f for f, _ in METER_FLOW_BRANCHES]] - v[head]) / fd.z[head]
            sf = v[[f for f, _ in METER_FLOW_BRANCHES]] * np.conj(i_h)
            h = np.concatenate([r.vm[0, vm_b], sf.real, sf.imag, x[idx], x[(nb - 1) + idx]])
            G = J.T @ (Wz[:, None] * J) + P_inv
            dx = np.linalg.solve(G, J.T @ (Wz * (z - h)) - P_inv @ (x - x_fc))
            x = x + dx
            if np.abs(dx).max() < 1e-7:
                break
        out_p[s, 1:], out_q[s, 1:] = x[: nb - 1], x[nb - 1:]
        out_p[s, 0] = out_q[s, 0] = 0.0
    return out_p, out_q


def envelope_level(fd, cust, p, q, direction, cap_scale=1.0, tol=1e-3) -> np.ndarray:
    """Per state: largest proportional level (fraction of each cap) keeping that state feasible."""
    C = cust.incidence(fd.n_bus)
    sign = envelopes.DIRECTIONS[direction]
    cap = cust.cap(direction)
    lo, hi = np.zeros(len(p)), np.full(len(p), cap_scale)
    ok0 = envelopes.is_feasible(fd, feeder.sweep_pf(fd, p, q), direction)
    for _ in range(int(np.ceil(np.log2(cap_scale / tol)))):
        mid = (lo + hi) / 2
        ok = envelopes.is_feasible(fd, feeder.sweep_pf(fd, p + sign * (mid[:, None] * cap[None, :]) @ C, q), direction)
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    return np.where(ok0, lo, 0.0)


def apply_level(fd, cust, p, q, level, direction) -> np.ndarray:
    C = cust.incidence(fd.n_bus)
    sign = envelopes.DIRECTIONS[direction]
    inj = sign * (level[:, None] * cust.cap(direction)[None, :]) @ C
    return envelopes.is_feasible(fd, feeder.sweep_pf(fd, p + inj, q), direction)


class SEGraphBuilder:
    """Graphs for GENCO state estimation, shared by training (scripts/train_genco_se_forecast.py) and
    inference (GencoSE), so their inputs cannot drift apart.

    One true state -> datakit-style graph (as scripts/make_training_data writes it) -> normalised ->
    graphkit's cleanup transforms -> metering in the format of graphkit's SimulateMeasurements:
      metered buses   voltage and P/Q measured (1% relative noise)
      flow branches   P/Q flow measured (1% relative noise)
      other buses     voltage hidden; P/Q hidden, or, with forecast inputs, set to the day-ahead
                      forecast (unmasked) and the bus flagged as unmetered in the BS column (always 0
                      on case33bw, so free to carry the metered flag: 1 = metered, 0 = forecast)
    Voltage angles are never measured.
    """

    def __init__(self, normalizer, raw_dir):
        from gridfm_graphkit.datasets.transforms import RemoveInactiveBranches, RemoveInactiveGenerators
        self.normalizer = normalizer
        raw = Path(raw_dir)
        first = lambda name: pd.read_parquet(raw / name).query("scenario == 0").drop(
            columns=["scenario_partition"], errors="ignore").reset_index(drop=True)
        self.bus_t = first("bus_data.parquet").sort_values("bus").reset_index(drop=True)
        self.gen_t, self.br_t = first("gen_data.parquet"), first("branch_data.parquet")
        if (self.bus_t["BS"] != 0).any():
            raise ValueError("BS column is used as the metered flag but is non-zero in this grid")
        self._clean_br, self._clean = RemoveInactiveBranches(), RemoveInactiveGenerators()

    def _graph(self, fd, v, p, q, base_mva):
        from gridfm_graphkit.datasets.graph_builder import build_hetero_data
        f, t = self.br_t.from_bus.to_numpy(), self.br_t.to_bus.to_numpy()
        i = (v[f] - v[t]) / fd.z[t]
        s_f, s_t = v[f] * np.conj(i), v[t] * np.conj(-i)
        s_slack = s_f[f == 0].sum()
        bus = self.bus_t.copy()
        bus["Pd"], bus["Qd"] = -p * base_mva, -q * base_mva
        bus["Pg"], bus["Qg"] = 0.0, 0.0
        bus.loc[0, ["Pg", "Qg"]] = s_slack.real * base_mva, s_slack.imag * base_mva
        bus["Vm"], bus["Va"] = np.abs(v), np.degrees(np.angle(v))
        gen = self.gen_t.copy()
        gen["p_mw"], gen["q_mvar"] = s_slack.real * base_mva, s_slack.imag * base_mva
        br = self.br_t.copy()
        br["pf"], br["qf"] = s_f.real * base_mva, s_f.imag * base_mva
        br["pt"], br["qt"] = s_t.real * base_mva, s_t.imag * base_mva
        bus = bus.merge(gen.groupby("bus")[["min_q_mvar", "max_q_mvar"]].sum().reset_index(), on="bus", how="left").fillna(0)
        return build_hetero_data(bus, gen, br)

    def build(self, fd, v, p, q, rng, vm_buses, inj_buses, flow_branches, p_fc=None, q_fc=None):
        import torch
        from gridfm_graphkit.datasets.globals import BS, PD_H, P_E, Q_E, QD_H, QG_H, VM_H
        data = self._graph(fd, v, p, q, fd.base_mva)
        self.normalizer.transform(data)
        data = self._clean(self._clean_br(data))
        nb, ny = data["bus"].y.shape
        std_bus = torch.full((nb, ny), float("inf"))
        std_bus[list(vm_buses), VM_H] = NOISE_REL
        for b in inj_buses:
            std_bus[b, [PD_H, QD_H, QG_H]] = NOISE_REL
        ei = data["bus", "connects", "bus"].edge_index
        std_br = torch.full((ei.shape[1], 2), float("inf"))
        for f, t in flow_branches:
            std_br[((ei[0] == f) & (ei[1] == t)).nonzero().ravel(), :] = NOISE_REL
        gen_bus = data["gen", "connected_to", "bus"].edge_index[1]
        std_gen = std_bus[gen_bus][:, [PD_H]]
        mask_bus, mask_br, mask_gen = torch.isinf(std_bus), torch.isinf(std_br), torch.isinf(std_gen)
        std_bus = torch.where(mask_bus, std_bus, std_bus * data["bus"].y.abs())
        ea = data["bus", "connects", "bus"].edge_attr
        std_br = torch.where(mask_br, std_br, std_br * ea[:, [P_E, Q_E]].abs())
        g = torch.Generator().manual_seed(int(rng.integers(2 ** 31)))
        x = data["bus"].x
        x[:, :ny] = torch.where(mask_bus, x[:, :ny], x[:, :ny] + std_bus * torch.randn(std_bus.shape, generator=g))
        ea[:, [P_E, Q_E]] = torch.where(mask_br, ea[:, [P_E, Q_E]], ea[:, [P_E, Q_E]] + std_br * torch.randn(std_br.shape, generator=g))
        data["gen"].x[:, :1] = torch.where(mask_gen, data["gen"].x[:, :1], data["gen"].x[:, :1] + std_gen * torch.randn(std_gen.shape, generator=g))
        if p_fc is not None:                                   # forecast inputs at unmetered buses
            un = np.array([b for b in range(1, nb) if b not in set(inj_buses)], int)
            scale = fd.base_mva / self.normalizer.baseMVA
            x[un, PD_H] = torch.as_tensor(-p_fc[un] * scale, dtype=x.dtype)
            x[un, QD_H] = torch.as_tensor(-q_fc[un] * scale, dtype=x.dtype)
            mask_bus[un, PD_H] = False
            mask_bus[un, QD_H] = False
            x[:, BS] = 0.0
            x[list(inj_buses), BS] = 1.0
        pad = lambda m, total: torch.nn.functional.pad(m, (0, total - m.shape[1]))
        data.mask_dict = {"bus": pad(mask_bus, x.shape[1]), "std_bus": std_bus, "outliers_bus": torch.zeros_like(mask_bus),
                          "gen": pad(mask_gen, data["gen"].x.shape[1]), "std_gen": std_gen,
                          "branch": pad(mask_br, ea.shape[1]), "std_branch": std_br,
                          "outliers_branch": torch.zeros_like(mask_br)}
        x[data.mask_dict["bus"]] = 0.0
        data["gen"].x[data.mask_dict["gen"]] = 0.0
        ea[data.mask_dict["branch"]] = 0.0
        return data


def random_placement(rng, n_bus, share):
    k = int(round(share * (n_bus - 1)))
    homes = tuple(int(b) for b in np.sort(rng.choice(np.arange(1, n_bus), k, replace=False)))
    return (0,) + homes, (0,) + homes


class GencoSE:
    """In-process GENCO state estimation with the module's fixed meter placement (gridfm-graphkit 0.9).
    forecast_inputs=True for models trained with day-ahead forecasts at unmetered buses."""

    def __init__(self, config, run_dir, raw_dir, network="case33bw", device=None, forecast_inputs=False):
        import torch
        import yaml
        from gridfm_graphkit.io.param_handler import NestedNamespace, load_model, load_normalizer
        from .genco import default_device
        self.torch = torch
        with open(config) as f:
            self.args = NestedNamespace(**yaml.safe_load(f))
        run_dir = Path(run_dir)
        self.device = device or default_device()
        self.model = load_model(self.args)
        state = torch.load(run_dir / "artifacts/model/best_model_state_dict.pt", map_location="cpu")
        self.model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()})
        self.model.eval().to(self.device)
        self.normalizer = load_normalizer(self.args)
        self.normalizer.fit_from_dict(torch.load(run_dir / "artifacts/stats/normalizer_stats.pt", map_location="cpu")[network])
        self.builder = SEGraphBuilder(self.normalizer, raw_dir)
        self.forecast_inputs = forecast_inputs

    def __call__(self, fd, states: States, rng, batch=512):
        """Estimated (vm, va[rad], p_inj, q_inj) per state; injections in pu of the feeder base."""
        torch = self.torch
        from torch_geometric.data import Batch
        from gridfm_graphkit.datasets.globals import PG_OUT, QG_OUT, VA_OUT, VM_OUT
        v_true = states.truth.vm * np.exp(1j * states.truth.va)
        graphs = [self.builder.build(fd, v_true[s], states.p[s], states.q[s], rng, METER_VM_BUSES, METER_INJ_BUSES,
                                     METER_FLOW_BRANCHES,
                                     *((states.p_fc[s], states.q_fc[s]) if self.forecast_inputs else (None, None)))
                  for s in range(len(states.p))]
        vm, va, pin, qin = [], [], [], []
        with torch.inference_mode():
            for k in range(0, len(graphs), batch):
                b = Batch.from_data_list(graphs[k:k + batch]).to(self.device)
                out = self.model(b)["bus"]
                o = out.float().cpu().numpy().reshape(-1, fd.n_bus, out.shape[1])
                vm.append(o[..., VM_OUT]), va.append(o[..., VA_OUT])
                pin.append(o[..., PG_OUT] * self.normalizer.baseMVA), qin.append(o[..., QG_OUT] * self.normalizer.baseMVA)
        return np.vstack(vm), np.vstack(va), np.vstack(pin) / fd.base_mva, np.vstack(qin) / fd.base_mva
