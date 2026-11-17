"""v1.3 E19 单测：PSRR 解析 / 噪声增益放大 / 一阶滚降 / SPICE 对照。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.theory import psrr
from src.circuits import v13_builders as vb
from src.engine.runner import run_netlist, DEFAULT_NGSPICE

NG_OK = Path(DEFAULT_NGSPICE).exists()


def test_noise_gain_inverting():
    assert psrr.noise_gain_inverting(1e3, 1e4) == pytest.approx(11.0)
    assert psrr.noise_gain_inverting(1e3, 0.0) == pytest.approx(1.0)


def test_psrr_openloop_dc_and_rolloff():
    p0, fps = 1e4, 1e3
    assert psrr.psrr_openloop_ratio(p0, 0.0, fps) == pytest.approx(p0)
    # f=f_ps 时幅度降 3.0103 dB
    assert psrr.psrr_openloop_db(p0, fps, fps) == pytest.approx(
        20 * np.log10(p0) - 3.01029995664, rel=1e-9)


def test_psrr_closedloop_ng_penalty():
    p0, fps = 1e4, 1e3
    ng_list = [1.0, 2.0, 11.0]
    db = [psrr.psrr_closedloop_db(p0, ng, 1.0, fps) for ng in ng_list]
    # NG 越大 PSRR 越差，且低频恰好差 20log10(NG 比)
    assert db[0] > db[1] > db[2]
    assert db[0] - db[1] == pytest.approx(20 * np.log10(2.0), rel=1e-12)
    assert db[0] - db[2] == pytest.approx(20 * np.log10(11.0), rel=1e-12)


def test_psrr_slope_is_minus20_db_per_dec():
    f = np.logspace(5, 9, 50)  # 需 f≫f_ps 才进入渐近区，端点偏差 <0.002 dB
    assert psrr.psrr_slope_db_per_dec(f, 1e3) == pytest.approx(-20.0, abs=0.01)


def test_psrr_rc_for_fps_roundtrip():
    r, c = psrr.psrr_rc_for_fps(1e3, 1e4)
    assert 1.0 / (2 * np.pi * r * c) == pytest.approx(1e3)


def test_psrr_inverting_netlist_is_wellformed():
    net = vb.psrr_inverting_netlist(1e3, 1e4, 1e-3, 1e6, 1.59e-11, 1e-3,
                                    1e4, 1.59e-8, 1e4, 10, 1e6)
    assert "VCC vcc 0 DC 0 AC 1" in net
    assert ".ac dec" in net
    assert "V(vcc)" in net
    assert net.rstrip().endswith(".end")


@pytest.mark.skipif(not NG_OK, reason="ngspice not available")
def test_psrr_spice_matches_analytic():
    """SPICE 扫 V(out)/V(vcc) vs 解析闭环电源纹波增益（AC 逐点 <0.5 dB）。"""
    p0, fps = 2e4, 5e3
    rps, cps = psrr.psrr_rc_for_fps(fps, 1e4)
    R1, Rf = 1e3, 1e4
    ng = psrr.noise_gain_inverting(R1, Rf)
    net = vb.psrr_inverting_netlist(R1, Rf, 1e-3, 1e6, 1.5915e-9,
                                    1e-3, rps, cps, p0, 10, 1e6, ppd=30)
    tr = run_netlist(net).plot("AC Analysis")
    f = np.real(tr["frequency"])
    vout = np.asarray(tr["v(out)"]).flatten()
    gain = np.abs(vout)  # vcc ac=1 → V(out)/V(vcc)
    # 网表 gm=1e-3/Rlag=1e6/Cc=1.5915e-9 → A0=1000, f1≈100 Hz
    # 闭环带宽 ≈ A0·f1/NG = GBW/NG，只在 A≫NG 的低频区严格对照
    A0, f1 = 1000.0, 1.0 / (2 * np.pi * 1e6 * 1.5915e-9)
    pred = np.abs(psrr.psrr_closedloop_ratio_exact(p0, ng, f, fps, A0, f1))
    f_cl = A0 * f1 / ng
    m = (f > 20) & (f < 0.1 * f_cl)
    err_db = 20 * np.log10(gain[m] / pred[m])
    assert np.max(np.abs(err_db)) < 0.2


@pytest.mark.skipif(not NG_OK, reason="ngspice not available")
def test_psrr_spice_ng_penalty():
    """SPICE：Rf 加倍（NG 从 11→21）应使低频 PSRR 恶化约 20log10(21/11) dB。"""
    p0, fps = 2e4, 5e3
    rps, cps = psrr.psrr_rc_for_fps(fps, 1e4)
    gains = []
    for rf in (1e4, 2e4):
        net = vb.psrr_inverting_netlist(1e3, rf, 1e-3, 1e6, 1.5915e-9,
                                        1e-3, rps, cps, p0, 10, 1e5, ppd=30)
        tr = run_netlist(net).plot("AC Analysis")
        f = np.real(tr["frequency"])
        vout = np.abs(np.asarray(tr["v(out)"]).flatten())
        gains.append(vout[(f > 20) & (f < 500)].mean())
    # NG 越大 → 电源纹波增益越大 → PSRR 越差，故 gains[NG=21] > gains[NG=11]，
    # 比值 gains[11]/gains[21] < 1，dB 为负。含有限 A0 修正的解析期望：
    A0, f1 = 1000.0, 1.0 / (2 * np.pi * 1e6 * 1.5915e-9)
    ff = np.array([100.0, 200.0, 300.0])  # 测试取 f∈(20,500) 的均值段
    g11 = np.abs(psrr.psrr_closedloop_ratio_exact(p0, 11.0, ff, fps, A0, f1)).mean()
    g21 = np.abs(psrr.psrr_closedloop_ratio_exact(p0, 21.0, ff, fps, A0, f1)).mean()
    expected_db = 20 * np.log10(g11 / g21)
    ratio = gains[0] / gains[1]
    assert 20 * np.log10(ratio) == pytest.approx(expected_db, abs=0.6)
