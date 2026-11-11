"""v1.2 E16 单测：MFB 带通解析设计、灵敏度、Q 自由度与 ngspice 对照。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.theory import mfb, noise
from src.theory.transfer import sk_bandpass_params
from src.theory.nonideal import design_macromodel
from src.circuits import v12_builders as vb
from src.engine.runner import run_netlist, DEFAULT_NGSPICE

NG_OK = Path(DEFAULT_NGSPICE).exists()
F0 = 1591.5494309189535  # v1.0 SK 等元件带通的 f0 = 1/(2π·10k·10n)
C = 1e-8


def _d(Q=5.0, rho=1.0):
    return mfb.design_mfb_bp(F0, Q, C, rho=rho)


def _args(d):
    return tuple(d[k] for k in mfb.MFB_COMPONENTS)


# ---------- 解析设计 ----------
@pytest.mark.parametrize("Q,rho", [(0.3333333333, 1.0), (1.0, 1.0),
                                   (2.0, 1.0), (5.0, 1.0), (5.0, 4.0),
                                   (10.0, 0.25)])
def test_design_recovers_f0_and_Q(Q, rho):
    d = _d(Q, rho)
    assert d["f0_Hz"] == pytest.approx(F0, rel=1e-12)
    assert d["Q"] == pytest.approx(Q, rel=1e-12)
    assert d["R1"] / d["R2"] == pytest.approx(rho, rel=1e-12)
    assert all(d[k] > 0 for k in mfb.MFB_COMPONENTS)


@pytest.mark.parametrize("Q,rho", [(1.0, 1.0), (2.0, 3.0), (5.0, 4.0), (10.0, 0.25)])
def test_peak_gain_closed_form(Q, rho):
    """|H(jw0)| = Q/(C2 R1 w0) = 2Q²/(1+ρ)。"""
    d = _d(Q, rho)
    w0 = 2 * np.pi * F0
    H = mfb.mfb_bp_H(*_args(d), w0)
    assert abs(H) == pytest.approx(2 * Q * Q / (1 + rho), rel=1e-10)
    assert d["H0"] == pytest.approx(abs(H), rel=1e-12)
    # 峰值处相位为 ±180°（反相）
    assert abs(abs(np.angle(H)) - np.pi) < 1e-9


def test_transfer_function_matches_closed_form():
    d = _d(3.0, 2.0)
    tf = mfb.mfb_bp_tf(*_args(d))
    w = np.logspace(1, 6, 37)
    s = 1j * w
    Htf = np.polyval(tf.num, s) / np.polyval(tf.den, s)
    assert np.allclose(Htf, mfb.mfb_bp_H(*_args(d), w), rtol=1e-12, atol=1e-18)


def test_mfb_breaks_sk_bandpass_Q_constraint():
    """v1.0 §6 遗留：SK 等元件带通 Q ≡ 1/3；MFB 的 Q 自由。"""
    f_sk, q_sk = sk_bandpass_params(1e4, C)
    assert q_sk == pytest.approx(1 / 3, rel=1e-12)
    assert f_sk == pytest.approx(F0, rel=1e-9)
    for Q in (1, 2, 5, 10):
        assert _d(Q)["Q"] == pytest.approx(Q, rel=1e-12)


def test_sensitivity_analytic_values():
    """ρ=1 时 f0 对 R1/R2 各 −1/4、对 R3/C1/C2 −1/2；
    Q 对 C1/C2 精确 0（(C1+C2) 与 C1C2 的弹性对消）；H0 对 R2 精确 0。"""
    d = _d(5.0, 1.0)
    s = mfb.mfb_bp_sensitivity(*_args(d))
    for k in ("C1", "C2", "R3"):
        assert s["f0_Hz"][k] == pytest.approx(-0.5, abs=2e-6)
    for k in ("R1", "R2"):
        assert s["f0_Hz"][k] == pytest.approx(-0.25, abs=2e-6)
    for k in ("C1", "C2"):
        assert s["Q"][k] == pytest.approx(0.0, abs=2e-6)
    assert s["H0"]["R2"] == pytest.approx(0.0, abs=2e-6)
    assert s["H0"]["R1"] == pytest.approx(-1.0, abs=2e-6)


def test_noise_gain_formula():
    assert mfb.mfb_bp_noise_gain(25.0) == 26.0
    assert mfb.mfb_bp_noise_gain(-3.0) == 4.0


def test_params_array_matches_scalar_and_vectorizes():
    d = _d(5.0, 2.0)
    a = tuple(d[k] for k in mfb.MFB_COMPONENTS)
    scalar = mfb.mfb_bp_params(*a)
    vec = mfb.mfb_bp_params_array(*[np.full(7, x) for x in a])
    for key in ("f0_Hz", "Q", "H0", "w0_rad_per_s"):
        assert vec[key].shape == (7,)
        assert np.allclose(vec[key], scalar[key], rtol=1e-12)


def test_3db_bandwidth_equals_f0_over_Q():
    d = _d(4.0)
    a = _args(d)
    H0 = d["H0"]

    def g(f):
        return abs(mfb.mfb_bp_H(*a, 2 * np.pi * f)) ** 2 - H0 ** 2 / 2

    lo = brentq(g, F0 / 20, F0, xtol=1e-9)
    hi = brentq(g, F0, F0 * 20, xtol=1e-9)
    assert hi - lo == pytest.approx(d["bw_3db_Hz"], rel=1e-8)
    assert np.sqrt(lo * hi) == pytest.approx(F0, rel=1e-8)


def test_impedance_scaling_keeps_response():
    """α 标度（R→αR, C→C/α）严格保持 f0/Q/H0 —— E17 噪声标度实验的前提。"""
    d = _d(5.0, 2.0)
    base = {k: d[k] for k in mfb.MFB_COMPONENTS}
    for alpha in (0.1, 0.5, 2.0, 10.0):
        sc = noise.scale_passive_network(base, alpha)
        for key in ("f0_Hz", "Q", "H0"):
            assert mfb.mfb_bp_params(**sc)[key] == pytest.approx(d[key], rel=1e-10)
        w = np.logspace(2, 5, 31)
        assert np.allclose(mfb.mfb_bp_H(**sc, w=w),
                           mfb.mfb_bp_H(**base, w=w), rtol=1e-10)


# ---------- 网表字符串 ----------
def test_mfb_netlist_is_well_formed():
    d = _d()
    nl = vb.mfb_bp_netlist(*_args(d), 1.0, 1e6, 20)
    assert ".ac dec 20" in nl and ".end" in nl
    assert "np.float64" not in nl
    assert nl.count("n2") >= 2  # C1 与 VCVS 反相输入共用节点
    nln = vb.mfb_bp_noise_netlist(*_args(d), 1.0, 1e6, 20)
    assert ".noise v(out) V1" in nln


# ---------- ngspice 对照 ----------
@pytest.mark.skipif(not NG_OK, reason="ngspice engine not available")
def test_ngspice_ac_matches_theory():
    """AC 扫频 vs 手推解析式：仅剩有限 A_OL=1e6 效应（<1e-3 dB）。"""
    d = _d(5.0)
    a = _args(d)
    res = run_netlist(vb.mfb_bp_netlist(*a, 1.0, 1e6, 20))
    ac = res.plot("AC Analysis")
    f = np.real(ac["frequency"])
    vout = np.asarray(ac["v(out)"], dtype=complex)
    H = mfb.mfb_bp_H(*a, 2 * np.pi * f)
    err = 20 * np.log10(np.abs(vout)) - 20 * np.log10(np.abs(H))
    assert np.max(np.abs(err)) < 1e-3


@pytest.mark.skipif(not NG_OK, reason="ngspice engine not available")
def test_ngspice_peak_gain_tracks_H0():
    d = _d(3.0)
    res = run_netlist(vb.mfb_bp_netlist(*_args(d), 100.0, 1e5, 60))
    ac = res.plot("AC Analysis")
    f = np.real(ac["frequency"])
    vout = np.abs(np.asarray(ac["v(out)"], dtype=complex))
    # 60 点/dec 网格上峰值与理论 H0 差 <0.05%；峰位与 f0 差 <0.5%
    assert vout.max() == pytest.approx(d["H0"], rel=5e-4)
    assert f[np.argmax(vout)] == pytest.approx(F0, rel=5e-3)


@pytest.mark.skipif(not NG_OK, reason="ngspice engine not available")
def test_finite_gbw_degrades_high_Q_more():
    """同样 GBW 下 Q 越高退化越严重（E16 的核心负结果，定性锁定）。"""
    p = design_macromodel(2e6, 1e5, 10e6)  # 故意用较低 GBW
    dev = {}
    for Q in (1.0, 10.0):
        d = _d(Q)
        res = run_netlist(vb.mfb_bp_gbw_netlist(
            *_args(d), p["gm"], p["Rlag"], p["Cc"], p["Imax"], 100.0, 1e5, 40))
        ac = res.plot("AC Analysis")
        f = np.real(ac["frequency"])
        vout = np.abs(np.asarray(ac["v(out)"], dtype=complex))
        dev[Q] = abs(vout.max() / d["H0"] - 1.0)
    assert dev[10.0] > dev[1.0]
