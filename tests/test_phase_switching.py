import numpy as np
import pytest

from gencodoe.phase_switching import PHASES, uniform_level
from gencodoe.phase_switching_run import balance


def test_balance_moves_only_switchable_and_reduces_imbalance():
    rng = np.random.default_rng(0)
    n = 60
    base = rng.integers(0, PHASES, n)
    base[:30] = 0                                     # deliberately unbalanced: half the load on phase a
    value = rng.uniform(0.5, 3.0, n)
    switchable = rng.uniform(size=n) < 0.5
    feeder = np.repeat([0, 1], n // 2)
    ph = balance(base, switchable, feeder, value, "import")
    assert (ph[~switchable] == base[~switchable]).all()

    def spread(p):
        return sum(np.ptp(np.bincount(p[feeder == f], value[feeder == f], PHASES)) for f in (0, 1))
    assert spread(ph) < spread(base)


def test_uniform_level_finds_the_threshold_level():
    s = np.zeros((100, 5), complex)
    limit = np.linspace(1.0, 6.0, 100)                # scenario k is feasible up to limit[k] kW

    def score(rows):
        lv = rows.real[:, 0]
        return lv <= np.tile(limit, len(lv) // 100)

    level, _ = uniform_level(score, s, "import", cap_kw=7.4, threshold=0.98, tol_kw=0.01)
    assert abs(level - limit[2]) < 0.02               # 98% of scenarios allow at most limit[2]


def test_phase_sweep_matches_opendss_after_moving_customers():
    pytest.importorskip("opendssdirect")
    from gencodoe.fourwire_net import load_fourwire
    from gencodoe.phase_switching import PhaseSweep, fit_thevenin_opendss, opendss_with_phases
    net = load_fourwire()
    sw = PhaseSweep(net, fit_thevenin_opendss(20))
    rng = np.random.default_rng(1)
    n_c = len(net.customer_names)
    ph = net.customer_phase.copy()
    idx = rng.choice(n_c, 40, replace=False)
    ph[idx] = (ph[idx] + rng.integers(1, 3, 40)) % 3
    p = rng.uniform(-3, 6, (4, n_c))
    q = 0.33 * np.clip(p, 0, None)
    ub = opendss_with_phases(ph)
    exact = [ub.solve(p[i], q[i]) for i in range(len(p))]
    res = sw.solve(p + 1j * q, ph)
    for key, tol in (("v_pn_min", 2e-4), ("v_pn_max", 2e-4), ("line_loading_max", 2e-3)):
        assert np.abs(res[key] - np.array([e[key] for e in exact])).max() < tol


def test_balance_handles_mixed_signs():
    # importing fixed homes on phase a, exporting switchable homes (PV at noon / export envelope):
    # the old signed min-max sent them all to the most loaded phase
    n = 30
    base = np.zeros(n, int)
    switchable = np.arange(n) >= 10
    value = np.where(switchable, -2.0, 3.0)
    feeder = np.zeros(n, int)
    ph = balance(base, switchable, feeder, value, "export")
    before = np.abs(np.bincount(base, value, PHASES)).max()
    after = np.abs(np.bincount(ph, value, PHASES)).max()
    assert after < before
