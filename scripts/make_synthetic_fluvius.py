"""Write SYNTHETIC Fluvius-shaped meter caches (data_out/fluvius/<group>.npz) for development
without the (non-redistributable) Fluvius 2024 data. Shapes and magnitudes are plausible, not real:
base load with morning/evening peaks, heat pumps scaled by season, EV sessions of 7.4 kW in the
evening, PV with a seasonal clear-sky shape and random clouds.

Every file carries synthetic=True; gencodoe.phase_switching_run refuses to call such results Fluvius.
Existing caches are never overwritten (a real cache wins).

    python scripts/make_synthetic_fluvius.py
"""

from __future__ import annotations

import numpy as np

from gencodoe import fluvius


def main(n_meters: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = fluvius.STEPS
    tod = (np.arange(s) % 96) / 4
    doy = np.arange(s) // 96
    fluvius.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t = (np.datetime64("2024-01-01T00:00") + np.arange(s) * np.timedelta64(15, "m")).astype("datetime64[m]")
    for name, (hp, ev, pv) in fluvius.GROUPS.items():
        out = fluvius.CACHE_DIR / f"{name}.npz"
        if out.exists():
            print(f"keep existing {out}")
            continue
        base = 0.25 + 0.5 * np.exp(-((tod - 19) / 2.5) ** 2) + 0.2 * np.exp(-((tod - 8) / 1.5) ** 2)
        off = np.clip(base[None] * rng.lognormal(0, 0.4, (n_meters, 1)) + 0.15 * rng.standard_normal((n_meters, s)), 0, None)
        if hp:
            off += (1.5 * (1 + np.cos(2 * np.pi * doy / 366)) / 2)[None] * rng.uniform(0.5, 1.5, (n_meters, 1))
        if ev:
            for i in range(n_meters):
                for d in range(366):
                    if rng.uniform() < 0.4:
                        a = d * 96 + int(rng.normal(72, 6)) % 96
                        off[i, a:a + int(rng.uniform(4, 24))] += 7.4
        inj = np.zeros((n_meters, s))
        if pv:
            sun = np.clip(np.sin(np.pi * (tod - 6) / 14), 0, None) * (0.6 + 0.4 * np.cos(2 * np.pi * (doy - 172) / 366))
            gen = sun[None] * rng.uniform(3, 6, (n_meters, 1)) * rng.uniform(0.3, 1, (n_meters, s))
            net = off - gen
            off, inj = np.clip(net, 0, None), np.clip(-net, 0, None)
        np.savez(out, ean=np.arange(n_meters), offtake_kw=off.astype(np.float32),
                 injection_kw=inj.astype(np.float32), time=t, synthetic=True)
        print(f"wrote {out} (synthetic)")


if __name__ == "__main__":
    main()
