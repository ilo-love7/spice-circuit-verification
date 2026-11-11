"""v1.2 公共 API 与网表模板健全性（防止 numpy.float64 毁网表等历史坑复发）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.theory import mfb, noise, distortion
from src.circuits import v12_builders as vb
from src.theory.nonideal import design_macromodel


def test_v12_modules_define_all():
    for mod in (mfb, noise, distortion, vb):
        assert hasattr(mod, "__all__") and len(mod.__all__) >= 4
        for name in mod.__all__:
            assert hasattr(mod, name), f"{mod.__name__} 缺少 {name}"


def test_netlists_are_numpy_scalar_safe():
    """v1.0 §4：`np.float64(...)` 进网表会让 ngspice 报 unknown parameter。"""
    p = design_macromodel(10e6, 1e5, 10e6)
    d = mfb.design_mfb_bp(np.float64(1591.5494309189535),
                          np.float64(5.0), np.float64(1e-8))
    a = tuple(d[k] for k in mfb.MFB_COMPONENTS)
    cases = [
        vb.mfb_bp_netlist(*a, np.float64(1.0), np.float64(1e6)),
        vb.mfb_bp_noise_netlist(*a, np.float64(1.0), np.float64(1e6)),
        vb.mfb_bp_gbw_netlist(*a, p["gm"], p["Rlag"], p["Cc"], p["Imax"],
                              np.float64(1.0), np.float64(1e6)),
        vb.rc_noise_netlist(np.float64(1e3), np.float64(1e-6),
                            np.float64(1.0), np.float64(1e6)),
        vb.sk_lp_noise_netlist(np.float64(1e4), np.float64(1e4),
                               np.float64(2.25e-8), np.float64(1.125e-8),
                               np.float64(1.0), np.float64(1e6)),
        vb.sine_slew_netlist(1e3, 1e4, p["gm"], p["Rlag"], p["Cc"], p["Imax"],
                             np.float64(1.0), np.float64(1e5)),
        vb.sk_lp_slew_step_netlist(1e4, 1e4, 2.25e-8, 1.125e-8,
                                   p["gm"], p["Rlag"], p["Cc"], p["Imax"],
                                   np.float64(5.0), np.float64(5e-3)),
    ]
    for nl in cases:
        assert "np.float64" not in nl
        assert ".end" in nl
        assert any(k in nl for k in (".ac ", ".noise ", ".tran "))
