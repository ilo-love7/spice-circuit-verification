"""v1.2 E18 单测：FPBW / 压摆判据 / THD / 波形指标与 ngspice 大信号对照。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.theory import distortion as dz
from src.theory.nonideal import design_macromodel
from src.circuits import v12_builders as vb
from src.engine.runner import run_netlist, DEFAULT_NGSPICE

NG_OK = Path(DEFAULT_NGSPICE).exists()
SR = 10e6  # 10 V/us


# ---------- FPBW / 斜率 ----------
def test_sine_slew_and_fpbw_inverse():
    f, vp = 1e5, 4.0
    assert dz.sine_max_slew_v_per_s(f, vp) == pytest.approx(2 * np.pi * f * vp, rel=1e-14)
    assert dz.required_sr_v_per_s(f, vp) == pytest.approx(2 * np.pi * f * vp, rel=1e-14)
    fp = dz.full_power_bandwidth_hz(SR, vp)
    assert fp == pytest.approx(SR / (2 * np.pi * vp), rel=1e-14)
    assert dz.slew_ratio(fp, vp, SR) == pytest.approx(1.0, rel=1e-14)
    assert dz.slew_ratio(2 * fp, vp, SR) == pytest.approx(2.0, rel=1e-14)


def test_fpbw_below_small_signal_bandwidth():
    """大信号 FPBW 可远低于小信号闭环带宽 GBW/NG —— E18 的核心分离。"""
    p = design_macromodel(10e6, 1e5, SR)
    gbw_over_ng = 10e6 / 11.0
    fpbw = dz.full_power_bandwidth_hz(p["SR_V_per_s"], 10.0)
    assert fpbw == pytest.approx(159154.943, rel=1e-6)
    assert fpbw < gbw_over_ng / 5


def test_slew_limited_rise_time():
    assert dz.slew_limited_rise_time_s(10.0, SR) == pytest.approx(1e-6, rel=1e-14)
    assert dz.slew_limited_rise_time_s(-10.0, SR) == pytest.approx(1e-6, rel=1e-14)


# ---------- THD ----------
def _wave(f=1000.0, n=4001, tstop=0.02, fn=None):
    t = np.linspace(0, tstop, n)
    y = np.sin(2 * np.pi * f * t) if fn is None else fn(t)
    return t, y


def test_thd_pure_sine_near_zero():
    t, y = _wave()
    r = dz.thd_percent(t, y, f_fund=1000.0)
    assert r["thd_pct"] < 0.01
    assert r["f_fund_Hz"] == pytest.approx(1000.0, rel=1e-2)


def test_thd_square_wave_matches_analytic():
    """方波前 10 次谐波（只含 3/5/7/9 次）THD = √(Σ1/h²)。"""
    t, y = _wave(fn=lambda x: np.sign(np.sin(2 * np.pi * 1000 * x)))
    r = dz.thd_percent(t, y, f_fund=1000.0)
    expected = 100 * np.sqrt(sum(1 / h ** 2 for h in (3, 5, 7, 9)))
    assert r["thd_pct"] == pytest.approx(expected, rel=0.03)
    assert r["thd_pct"] > 40


def test_thd_rises_with_clipping():
    t = np.linspace(0, 0.02, 4001)
    s = np.sin(2 * np.pi * 1000 * t)
    vals = [dz.thd_percent(t, np.clip(s, -k, k), f_fund=1000.0)["thd_pct"]
            for k in (1.0, 0.8, 0.5, 0.3)]
    assert vals[0] < 0.01
    assert all(a < b for a, b in zip(vals, vals[1:]))
    assert vals[-1] > 10


def test_harmonic_peak_tolerates_detuning():
    """N5 踩过的坑：频率略偏时精确 bin 会漏掉谐波，邻域峰值必须检出。"""
    t = np.linspace(0, 0.02, 4001)
    f0 = 1007.3  # 故意非整数 bin
    y = np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * 3 * f0 * t)
    r = dz.thd_percent(t, y, f_fund=f0)
    assert r["harmonic_mag"]["3"] / max(
        r["harmonic_mag"][k] for k in ("5", "7", "9")) > 5
    assert r["thd_pct"] == pytest.approx(10.0, rel=0.05)


def test_waveform_metrics():
    t, y = _wave()
    m = dz.waveform_metrics(t, y)
    assert m["vpp"] == pytest.approx(2.0, rel=1e-3)
    assert m["peak"] == pytest.approx(1.0, rel=1e-3)
    assert m["max_abs_slope_v_per_s"] == pytest.approx(2 * np.pi * 1000, rel=2e-3)


def test_resample_uniform_preserves_amplitude():
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.8, 1.2, 4000)) * 5e-6
    y = np.sin(2 * np.pi * 1000 * t)
    tu, yu = dz.resample_uniform(t, y)
    assert np.all(np.diff(tu) > 0)
    assert np.max(np.abs(yu)) == pytest.approx(1.0, rel=2e-3)


# ---------- ngspice 大信号对照 ----------
@pytest.mark.skipif(not NG_OK, reason="ngspice engine not available")
def test_ngspice_thd_onset_brackets_fpbw():
    """f = 0.5·FPBW 时线性、f = 3·FPBW 时压摆限幅且 THD 陡升。"""
    p = design_macromodel(10e6, 1e5, SR)
    v_amp = 1.0                 # 输出峰值 ≈ 10 V
    fpbw = dz.full_power_bandwidth_hz(SR, 10.0)
    out = {}
    for fac in (0.5, 3.0):
        f = fac * fpbw
        res = run_netlist(vb.sine_slew_netlist(
            1e3, 1e4, p["gm"], p["Rlag"], p["Cc"], p["Imax"], v_amp, f))
        tr = res.plot("Transient")
        t = np.real(tr["time"])
        y = np.real(tr["v(out)"])
        out[fac] = (dz.thd_percent(t, y, f_fund=f)["thd_pct"],
                    dz.waveform_metrics(t, y))
    assert out[0.5][0] < 0.1
    assert out[3.0][0] > 5.0
    assert out[3.0][1]["vpp"] < 0.8 * out[0.5][1]["vpp"]
    # 压摆区最大斜率钉在 SR 附近
    assert out[3.0][1]["max_abs_slope_v_per_us"] == pytest.approx(10.0, rel=0.08)


@pytest.mark.skipif(not NG_OK, reason="ngspice engine not available")
def test_small_signal_stays_linear_above_fpbw():
    """同一频率、幅度缩小 100× ⇒ 所需斜率下降 100×，失真消失（幅度而非频率决定）。"""
    p = design_macromodel(10e6, 1e5, SR)
    f = 3 * dz.full_power_bandwidth_hz(SR, 10.0)
    thds = []
    for amp in (1.0, 0.01):
        res = run_netlist(vb.sine_slew_netlist(
            1e3, 1e4, p["gm"], p["Rlag"], p["Cc"], p["Imax"], amp, f))
        tr = res.plot("Transient")
        thds.append(dz.thd_percent(np.real(tr["time"]),
                                   np.real(tr["v(out)"]), f_fund=f)["thd_pct"])
    assert thds[1] < 0.1 < thds[0]
