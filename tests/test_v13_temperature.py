"""v1.3 E21 单测：电阻/电容 TC / RC 截止温漂 / 增益比抵消 / 失调温漂 / SPICE .temp。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.theory import temperature as tp
from src.circuits import v13_builders as vb
from src.engine.runner import run_netlist, DEFAULT_NGSPICE

NG_OK = Path(DEFAULT_NGSPICE).exists()
R0, C0, F0 = 1e4, 1e-8, 1.0 / (2 * np.pi * 1e4 * 1e-8)


def test_ppm_to_tc1():
    assert tp.ppm_to_tc1(100) == pytest.approx(1e-4)
    assert tp.ppm_to_tc1(0) == 0.0


def test_resistor_capacitor_at_T():
    assert tp.resistor_at_T(1e3, 1e-3, 25) == pytest.approx(1e3)
    assert tp.resistor_at_T(1e3, 1e-3, 125) == pytest.approx(1.1e3)
    assert tp.capacitor_at_T(1e-8, 2e-3, 25) == pytest.approx(1e-8)


def test_cutoff_drift_opposite_tc_cancels():
    # R +100 ppm/C、C −100 ppm/C → f0 一阶零温漂
    tr = tp.ppm_to_tc1(100)
    tc = tp.ppm_to_tc1(-100)
    assert tp.cutoff_drift_ratio(tr, tc, 50) == pytest.approx(0.0, abs=1e-15)
    # 精确值非零（二阶项 (tc·ΔT)²=2.5e-5），远小于同号叠加的 1e-2
    exact = tp.cutoff_drift_exact(R0, C0, tr, tc, 50)
    assert 0 < exact < 1e-4
    assert exact < abs(tp.cutoff_drift_exact(R0, C0, tp.ppm_to_tc1(100),
                                              tp.ppm_to_tc1(100), 50)) / 100


def test_cutoff_drift_same_tc_adds():
    tc = tp.ppm_to_tc1(100)
    assert tp.cutoff_drift_ratio(tc, tc, 50) == pytest.approx(-0.01)
    # 一阶 −0.01；精确 −0.0099255，二阶项 (tc·ΔT)² 使偏差约 7.5e-5
    assert tp.cutoff_drift_exact(R0, C0, tc, tc, 50) == pytest.approx(
        -0.01, abs=1e-4)


def test_gain_ratio_drift_same_tc_zero():
    tc = tp.ppm_to_tc1(100)
    assert tp.gain_ratio_drift(tc, tc, 80) == pytest.approx(0.0, abs=1e-15)
    diff = tp.gain_ratio_drift(tp.ppm_to_tc1(200), tc, 80)
    assert diff > 0  # Rf TC 更大 → 增益上升


def test_vos_and_output_offset():
    assert tp.vos_at_T(1e-3, 2e-6, 25) == pytest.approx(1e-3)
    assert tp.vos_at_T(1e-3, 2e-6, 75) == pytest.approx(1.1e-3)
    assert tp.output_offset_from_vos(1e-3, 2e-6, 11, 75) == pytest.approx(12.1e-3)


def test_temp_netlist_wellformed():
    net = vb.rc_temp_netlist(R0, C0, 1e-4, -1e-4, 85, 10, 1e5)
    assert ".temp 85" in net
    assert "TC=" in net
    assert net.rstrip().endswith(".end")


@pytest.mark.skipif(not NG_OK, reason="ngspice not available")
def test_rc_cutoff_temperature_spice_matches_analytic():
    """SPICE .temp 扫 RC 截止 vs 解析（同号 TC，两温度点）。"""
    tc = tp.ppm_to_tc1(2000)  # 放大效应便于测量
    preds = {}
    for temp in (25, 125):
        net = vb.rc_temp_netlist(R0, C0, tc, 0.0, temp, 10, 1e5, ppd=60)
        tr = run_netlist(net).plot("AC Analysis")
        f = np.real(tr["frequency"])
        a = np.abs(np.asarray(tr["v(out)"]).flatten())
        i = int(np.argmin(np.abs(a - 1 / np.sqrt(2))))
        preds[temp] = f[i]
        exact = tp.rc_cutoff_at_T(R0, C0, tc, 0.0, temp)
        assert preds[temp] == pytest.approx(exact, rel=0.02)
    drift = preds[125] / preds[25] - 1
    assert drift == pytest.approx(tp.cutoff_drift_exact(R0, C0, tc, 0.0, 100),
                                  abs=0.03)


@pytest.mark.skipif(not NG_OK, reason="ngspice not available")
def test_inv_gain_same_tc_no_drift():
    """两电阻同 TC → 反相增益温漂≈0（SPICE 两温度点比值）。"""
    tc = tp.ppm_to_tc1(2000)
    gains = {}
    for temp in (25, 125):
        net = vb.inv_gain_temp_netlist(1e3, 1e4, tc, tc, temp, 10, 1e5, ppd=20)
        tr = run_netlist(net).plot("AC Analysis")
        f = np.real(tr["frequency"])
        a = np.abs(np.asarray(tr["v(out)"]).flatten())
        gains[temp] = a[0]  # 低频增益
    assert gains[125] / gains[25] == pytest.approx(1.0, abs=0.01)
