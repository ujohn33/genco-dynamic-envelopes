"""Train GENCO in StateEstimation mode WITH day-ahead forecasts as inputs at unmetered buses.

The fair rematch against WLS (which uses the same forecast): graphkit's SE pipeline can only show or
hide a quantity, so this script builds the graphs itself with gencodoe.se_envelopes.SEGraphBuilder
(identical to inference) and trains graphkit's own StateEstimation task (same model, loss, optimizer
and scheduler as the config) with Lightning.

Per training state: a random monitored set (share of load buses, plus the substation) with voltage
and P/Q measured, feeder-head flow measured, day-ahead forecast P/Q at the other buses, flagged.
Training days only; validation days for early stopping and checkpointing.

    python scripts/train_genco_se_forecast.py --config configs/genco_se_forecast_case33bw.yaml \
        --states ~/gencodoe_work/se_fluvius/states.npz --raw ~/gencodoe_work/data_se/case33bw/raw \
        --out ~/gencodoe_work/runs/genco-se-forecast
Output: <out>/artifacts/model/best_model_state_dict.pt and <out>/artifacts/stats/normalizer_stats.pt
(the layout gencodoe.se_envelopes.GencoSE loads).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from gridfm_graphkit.io.param_handler import NestedNamespace, get_task, load_normalizer

from gencodoe import demo, feeder, se_envelopes as se


class SaveBest(Callback):
    def __init__(self, path: Path, monitor: str = "Validation loss"):
        self.path, self.monitor, self.best = path, monitor, float("inf")

    def on_validation_end(self, trainer, pl_module):
        cur = trainer.callback_metrics.get(self.monitor)
        if cur is None or trainer.sanity_checking:
            return
        if float(cur) < self.best:
            self.best = float(cur)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pl_module.state_dict(), self.path)


def graphs_for(builder, fd, d, idx, rng, share):
    p, q, p_fc, q_fc = d["p"][idx], d["q"][idx], d["p_fc"][idx], d["q_fc"][idx]
    r = feeder.sweep_pf(fd, p, q)
    v = r.vm * np.exp(1j * r.va)
    out = []
    for s in range(len(idx)):
        vm_b, inj_b = se.random_placement(rng, fd.n_bus, share)
        out.append(builder.build(fd, v[s], p[s], q[s], rng, vm_b, inj_b, se.METER_FLOW_BRANCHES, p_fc[s], q_fc[s]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--share", type=float, default=0.3)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use only this many training states (smoke tests)")
    ap.add_argument("--epochs", type=int, default=0, help="override the config's epochs")
    a = ap.parse_args(argv)
    L.seed_everything(a.seed)
    with open(a.config) as f:
        args = NestedNamespace(**yaml.safe_load(f))
    fd, _, _, _ = demo.build(case33bw=True)
    d = np.load(a.states)
    tr = np.flatnonzero(d["split"] == "train")
    va = np.flatnonzero(d["split"] == "val")
    rng = np.random.default_rng(a.seed)
    n_raw = len(tr)
    if a.limit:
        tr = tr[: a.limit]
    va = np.sort(rng.choice(va, min(a.n_val, len(va)), replace=False))

    normalizer = load_normalizer(args)
    normalizer.fit(str(a.raw), list(range(n_raw)))            # the raw data holds exactly the training days
    builder = se.SEGraphBuilder(normalizer, a.raw)
    t = time.time()
    g_tr = graphs_for(builder, fd, d, tr, rng, a.share)
    g_va = graphs_for(builder, fd, d, va, np.random.default_rng(a.seed + 1), a.share)
    print(f"built {len(g_tr)} train / {len(g_va)} val graphs in {time.time() - t:.0f}s", flush=True)

    task = get_task(args, [normalizer])
    bs = int(args.training.batch_size)
    train_loader = DataLoader(g_tr, batch_size=bs, shuffle=True, num_workers=a.workers, persistent_workers=a.workers > 0)
    val_loader = DataLoader(g_va, batch_size=bs, shuffle=False, num_workers=a.workers, persistent_workers=a.workers > 0)
    saver = SaveBest(a.out / "artifacts/model/best_model_state_dict.pt")
    trainer = L.Trainer(max_epochs=a.epochs or int(args.training.epochs), accelerator="auto", devices=1,
                        logger=CSVLogger(str(a.out), name="logs"),
                        callbacks=[EarlyStopping(monitor="Validation loss", patience=int(args.callbacks.patience)),
                                   saver],
                        log_every_n_steps=50, enable_progress_bar=False)
    trainer.fit(task, train_loader, val_loader)
    stats = a.out / "artifacts/stats"
    stats.mkdir(parents=True, exist_ok=True)
    torch.save({args.data.networks[0]: normalizer.get_stats()}, stats / "normalizer_stats.pt")
    meta = {"best_val_loss": saver.best, "epochs_run": trainer.current_epoch,
            "share": a.share, "train_states": len(tr), "val_states": len(va), "seed": a.seed}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("done", meta, flush=True)


if __name__ == "__main__":
    main()
