"""v1.3 E20 单测：双极点开环 / 交越 / 相位裕度 / 过冲反推 / SPICE 对照。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.theory import stability as st
from src.circuits import v13_builders as vb
from src.engine.runner import run_netlist, DEFAULT_NGSPICE

NG_OK = Path(DEFAULT_NGSPICE).exists()

A0, F1, F2 = 1e5, 10.0, 2e5  # w1=2π·10Hz, w2=2π·200kHz
W1, W2 = 2 * np.pi * F1, 2 * np.pi * F2


def test_openloop_two_pole_dc_and_poles():
    assert st.open_loop_two_pole(0.0, A0, W1, W2) == pytest.approx(A0)
    # 在 w1 处幅度降 3.0103 dB（单极点主）
    a1 = abs(st.open_loop_two_pole(W1, A0, W1, W2))
    assert 20 * np.log10(a1 / A0) == pytest.approx(-3.01029995664, abs=0.01)


def test_gain_crossover_satisfies_unity():
    for ng in (1, 2, 11):
        wc = st.gain_crossover_rad(A0, W1, W2, 1.0 / ng)
        assert abs(st.loop_gain(wc, A0, W1, W2, 1.0 / ng)) == pytest.approx(1.0, rel=1e-6)


def test_ng_increase_improves_phase_margin():
    """NG 越大（β 越小）→ wc 越低 → PM 越高：同相单位缓冲 NG=1 最差。"""
    rows = st.pm_sweep([1, 2, 5, 11, 21], A0, W1, W2)
    pm = [r["PM_deg"] for r in rows]
    wc = [r["wc_rad"] for r in rows]
    assert all(pm[i] < pm[i + 1] for i in range(len(pm) - 1))
    assert all(wc[i] > wc[i + 1] for i in range(len(wc) - 1))
    assert rows[0]["PM_deg"] < 60.0  # 单位缓冲接近不稳定


def test_second_pole_lower_degrades_pm():
    hi = st.phase_margin_deg(A0, W1, 2e6, 1.0)
    lo = st.phase_margin_deg(A0, W1, 2e4, 1.0)
    assert lo < hi
    # w2→∞ 接近单极点 PM 上限
    inf = st.phase_margin_deg(A0, W1, 1e12, 1.0)
    assert inf > hi


def test_overshoot_pm_roundtrip():
    for pm in (30, 45, 60, 75):
        mp = st.overshoot_from_pm(pm)
        assert 0.0 <= mp <= 1.0
        if mp > 1e-6:
            assert st.pm_from_overshoot(mp) == pytest.approx(float(pm), abs=0.5)


def test_stability_pm_threshold():
    assert not st.is_stable_pm(30)
    assert st.is_stable_pm(60)


def test_two_pole_netlist_wellformed():
    net = vb.two_pole_openloop_netlist(1e-3, 1e8, 1.59e-10, 1e3, 7.96e-10,
                                       10, 1e8)
    assert "Ebuf1" in net and "Ebuf2" in net
    assert "R2 m1 nc2" in net
    assert net.rstrip().endswith(".end")


@pytest.mark.skipif(not NG_OK, reason="ngspice not available")
def test_two_pole_openloop_spice_matches_analytic():
    """SPICE 双极点开环 |A| vs 解析，全频段 <0.3 dB。"""
    gm, Rlag = 1e-3, 1e8
    Cc = 1.0 / (2 * np.pi * F1 * Rlag)
    R2, C2 = 1e3, 1.0 / (2 * np.pi * F2 * 1e3)
    net = vb.two_pole_openloop_netlist(gm, Rlag, Cc, R2, C2, 1, 1e8, ppd=40)
    tr = run_netlist(net).plot("AC Analysis")
    f = np.real(tr["frequency"])
    a = np.abs(np.asarray(tr["v(out)"]).flatten())
    pred = np.abs(st.open_loop_two_pole(2 * np.pi * f, gm * Rlag,
                                        2 * np.pi * F1, 2 * np.pi * F2))
    err = 20 * np.log10(a / pred)
    assert np.max(np.abs(err)) < 0.3
