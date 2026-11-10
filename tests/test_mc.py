"""蒙特卡洛模块测试（小样本）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.mc.tolerance import perturb, run_rc_mc, yield_curve


def test_perturb_uniform():
    rng = np.random.default_rng(0)
    vals = [perturb(100.0, 0.05, rng, "uniform") for _ in range(1000)]
    vals = np.array(vals)
    assert vals.min() >= 95.0 - 1e-9
    assert vals.max() <= 105.0 + 1e-9
    assert abs(vals.mean() - 100.0) < 1.0


def test_perturb_normal():
    rng = np.random.default_rng(0)
    vals = np.array([perturb(100.0, 0.05, rng, "normal") for _ in range(1000)])
    assert vals.min() >= 95.0 - 1e-9
    assert vals.max() <= 105.0 + 1e-9


def test_run_rc_mc_small():
    rng = np.random.default_rng(42)
    res = run_rc_mc(1e3, 1e-6, tol=0.05, n_trials=12, fstart=10, fstop=1e5,
                    f_spec=159.155, rng=rng, points_per_decade=5)
    assert res["n_trials"] == 12
    assert len(res["fc_Hz"]) == 12
    assert np.all(np.isfinite(res["fc_Hz"]))
    # fc 均值应在理论值 ±10% 内
    assert abs(res["fc_mean"] - res["fc_theory"]) / res["fc_theory"] < 0.1
    assert 0.0 <= res["yield"] <= 1.0


def test_yield_curve():
    rng = np.random.default_rng(0)
    res = run_rc_mc(1e3, 1e-6, tol=0.02, n_trials=8, fstart=10, fstop=1e5,
                    f_spec=159.155, rng=rng, points_per_decade=5)
    f_grid = np.logspace(1, 4, 20)
    yc = yield_curve({"RC LP": res}, f_grid, spec_db=1.0)
    assert len(yc["f_grid"]) == 20
    assert len(yc["curves"]["RC LP"]) == 20
    assert all(0.0 <= v <= 1.0 for v in yc["curves"]["RC LP"])
