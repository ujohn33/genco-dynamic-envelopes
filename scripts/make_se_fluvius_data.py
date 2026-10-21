"""States for the state-estimation study: case33bw with real Fluvius homes and simulated EV sessions.

    python scripts/make_se_fluvius_data.py [--work ~/gencodoe_work]

Writes
  <work>/se_fluvius/states.npz          every 15-min step of 2024: true p, q (pu, base 1 MVA), the day-
                                        ahead forecast p_fc, q_fc (mean of the same step over the previous
                                        7 days), day, step, split (train / val / test by day)
  <work>/data_se/case33bw/raw/*.parquet the TRAINING-day states in gridfm-datakit's schema (exact AC
                                        sweep labels), for training GENCO in StateEstimation mode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gencodoe import demo, feeder, fluvius_feeder as ff

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_training_data import write  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path.home() / "gencodoe_work")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    fd, pd, qd, cust = demo.build(case33bw=True)
    pop = ff.build_population(pd * 1e3, seed=a.seed)
    net, ev = ff.bus_load_kw(pop, fd.n_bus, seed=a.seed + 1)
    fc = ff.day_ahead_forecast(net)
    tan = np.divide(qd, pd, out=np.zeros_like(pd), where=pd > 0)
    to_pq = lambda x: (-(x.T) / 1e3, -np.clip(x.T, 0, None) / 1e3 * tan)       # kW (+ import) -> pu injection
    p, q = to_pq(net)
    p_fc, q_fc = to_pq(fc)
    n = p.shape[0]
    day, step = np.arange(n) // 96, np.arange(n) % 96
    split = np.array([ff.split_of(d) for d in day])
    out = a.work / "se_fluvius"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "states.npz", p=p, q=q, p_fc=p_fc, q_fc=q_fc, day=day, step=step, split=split,
                        ev_kw=ev.T, homes=len(pop.bus), ev_owners=int(pop.ev.sum()))
    tr = split == "train"
    r = feeder.sweep_pf(fd, p[tr], q[tr])
    v = r.vm * np.exp(1j * r.va)
    raw = write(str(a.work / "data_out/case33bw/raw"), str(a.work / "data_se"), fd, v, p[tr], q[tr])
    print(f"{n} states ({tr.sum()} train / {(split == 'val').sum()} val / {(split == 'test').sum()} test); "
          f"{len(pop.bus)} homes, {int(pop.ev.sum())} EVs; training data in {raw}")


if __name__ == "__main__":
    main()
