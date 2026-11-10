"""v1.1 深化单测：非理想运放 / 级联滤波器 / WCA 与灵敏度。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.theory.nonideal import (design_macromodel, macromodel_params,
                                 inverting_amp_H_finiteGBW,
                                 closed_loop_pole_approx_Hz, f3dB_from_pole,
                                 group_delay)
from src.theory.cascade import (BUTTER4_Q, design_sk_lp_equalR, butter4_lp_tf,
                                cascade_sk_H, step_response_metrics)
from src.theory.transfer import sk_lowpass_H, cutoff_precise, compare_response
from src.circuits.nonideal_builders import (inverting_amp_gbw_netlist,
                                            inverting_amp_slew_netlist,
                                            noninv_buffer_gbw_netlist,
                                            sk_cascade4_netlist)
from src.mc.wca import (sk_fc_corners, sk_fc_propagation_std,
                        sk_mag_sensitivity, sk_yield_analytic_design,
                        rc_fc_propagation_std)


# ---------- 深化 A：非理想运放 ----------
def test_macromodel_design_consistency():
    p = design_macromodel(GBW_Hz=10e6, A0=1e5, SR_V_per_s=10e6)
    assert p["GBW_Hz"] == pytest.approx(10e6, rel=1e-9)
    assert p["A0"] == pytest.approx(1e5, rel=1e-9)
    assert p["SR_V_per_s"] == pytest.approx(10e6, rel=1e-9)
    # GBW = gm/(2πCc)；SR = Imax/Cc
    assert p["GBW_Hz"] == pytest.approx(p["gm"] / (2 * np.pi * p["Cc"]), rel=1e-12)
    assert p["SR_V_per_s"] == pytest.approx(p["Imax"] / p["Cc"], rel=1e-12)


def test_f3dB_convention_factor():
    # 半功率点（-3.0103 dB）与 -3.00 dB 的换算因子
    fp = 1e6
    assert f3dB_from_pole(fp, -3.0103) == pytest.approx(fp, rel=1e-4)
    assert f3dB_from_pole(fp, -3.0) == pytest.approx(fp * np.sqrt(10 ** 0.3 - 1), rel=1e-9)


def test_closed_loop_gain_and_pole():
    p = design_macromodel(10e6, 1e5, 10e6)
    G, NG = 10.0, 11.0
    w = np.array([0.0, 2 * np.pi * 1e3])
    H = inverting_amp_H_finiteGBW(1e3, 1e4, w, p["A0"], p["wb_rad_per_s"])
    # 低频增益 → −G·A0/(A0+NG)
    assert H[0].real == pytest.approx(-G * p["A0"] / (p["A0"] + NG), rel=1e-9)
    # 极点频率 = wb·(A0+NG)/NG
    fp = p["wb_rad_per_s"] * (p["A0"] + NG) / NG / (2 * np.pi)
    fc3 = f3dB_from_pole(fp, -3.0)
    # 在 -3.0dB 点幅度应为 DC 的 1/√2·(10^0.03)…直接验证闭式
    w3 = 2 * np.pi * fc3
    H3 = inverting_amp_H_finiteGBW(1e3, 1e4, np.array([w3]), p["A0"], p["wb_rad_per_s"])[0]
    assert abs(H3) == pytest.approx(abs(H[0]) * 10 ** (-3 / 20), rel=1e-9)


def test_gbw_vs_ng_rule():
    p = design_macromodel(10e6, 1e5, 10e6)
    for G in (1.0, 10.0, 100.0):
        NG = 1 + G
        classic = closed_loop_pole_approx_Hz(p["GBW_Hz"], NG)
        exact = p["wb_rad_per_s"] * (p["A0"] + NG) / NG / (2 * np.pi)
        # 经典规则相对偏差 = NG/A0（A0→∞ 时消失）
        assert classic == pytest.approx(exact, rel=NG / p["A0"] * 1.01)


def test_gbw_netlist_ac_matches_theory():
    p = design_macromodel(10e6, 1e5, 10e6)
    net = inverting_amp_gbw_netlist(1e3, 1e4, p["gm"], p["Rlag"], p["Cc"], p["Imax"],
                                    1e2, 1e8, points_per_decade=5)
    from src.engine.runner import run_netlist
    pl = run_netlist(net).plot("AC Analysis")
    f = np.array(pl["frequency"]).real
    Hs = np.array(pl["v(out)"])
    Ht = inverting_amp_H_finiteGBW(1e3, 1e4, 2 * np.pi * f, p["A0"], p["wb_rad_per_s"])
    cmp = compare_response(f, Hs, Ht)
    assert cmp["max_abs_err_db"] < 1e-9


def test_slew_rate_measurement():
    p = design_macromodel(10e6, 1e5, 10e6)
    net = inverting_amp_slew_netlist(1e3, 1e4, p["gm"], p["Rlag"], p["Cc"], p["Imax"],
                                     v_amp=0.5, f_sq=1e4, tstop=2e-4)
    from src.engine.runner import run_netlist
    pl = run_netlist(net).plot("Transient Analysis")
    t = np.array(pl["time"]).real
    v = np.array(pl["v(out)"])
    dv = np.abs(np.diff(v) / np.diff(t))
    slew = float(np.max(dv[int(len(t) * 0.1):int(len(t) * 0.9)]))
    assert slew == pytest.approx(p["SR_V_per_s"], rel=0.02)


# ---------- 深化 B：级联滤波器 ----------
def test_butter4_q_values():
    assert BUTTER4_Q[0] == pytest.approx(0.5411961, rel=1e-6)
    assert BUTTER4_Q[1] == pytest.approx(1.3065630, rel=1e-6)


def test_design_sk_lp_equalR_hits_targets():
    f0, Q = 1e3, 0.5412
    R1, R2, C1, C2 = design_sk_lp_equalR(f0, Q, C2=1e-8)
    from src.theory.transfer import sk_lowpass_params
    f0b, Qb, _ = sk_lowpass_params(R1, R2, C1, C2)
    assert f0b == pytest.approx(f0, rel=1e-9)
    assert Qb == pytest.approx(Q, rel=1e-9)


def test_cascade_matches_scipy_butter4():
    f0 = 1e3
    w = 2 * np.pi * np.logspace(1, 5, 200)
    H1 = cascade_sk_H(f0, w)
    tf = butter4_lp_tf(f0)
    _, H2 = tf.freqresp(w)
    cmp = compare_response(w / (2 * np.pi), H1, H2)
    assert cmp["max_abs_err_db"] < 1e-9


def test_cascade_netlist_runs():
    f0 = 1e3
    Ra, R2a, C1a, C2a = design_sk_lp_equalR(f0, BUTTER4_Q[0], C2=1e-8)
    Rb, R2b, C1b, C2b = design_sk_lp_equalR(f0, BUTTER4_Q[1], C2=1e-8)
    net = sk_cascade4_netlist(Ra, R2a, C1a, C2a, Rb, R2b, C1b, C2b,
                              10, 1e5, buffered=True, points_per_decade=5)
    from src.engine.runner import run_netlist
    pl = run_netlist(net).plot("AC Analysis")
    f = np.array(pl["frequency"]).real
    H = np.array(pl["v(out)"])
    assert len(f) > 10
    # DC 增益 ≈ 1
    assert abs(H[0]) == pytest.approx(1.0, abs=2e-3)


def test_group_delay_theory():
    # 一阶 RC：群延迟 = RC/(1+(wRC)^2)
    R, C = 1e3, 1e-6
    w = 2 * np.pi * np.logspace(0, 5, 500)
    H = 1 / (1 + 1j * w * R * C)
    tg = group_delay(H, w)
    tg_th = R * C / (1 + (w * R * C) ** 2)
    # 中段（避开边界）应吻合
    m = slice(50, 450)
    assert np.max(np.abs(tg[m] - tg_th[m])) / np.max(tg_th[m]) < 0.01


def test_step_metrics_reference():
    # 一阶阶跃 1−e^{−t/τ}：t_r≈2.197τ，无过冲
    tau = 1e-3
    t = np.linspace(0, 10 * tau, 2000)
    y = 1 - np.exp(-t / tau)
    m = step_response_metrics(t, y)
    assert m["overshoot_pct"] == pytest.approx(0.0, abs=1e-6)
    assert m["rise_time_s"] == pytest.approx(2.1972 * tau, rel=0.02)


# ---------- 深化 C：WCA / 灵敏度 ----------
def test_wca_corners_extremes():
    R1 = R2 = 1e4
    C2 = 1e-8
    C1 = 4 * 0.5412 ** 2 * C2
    wca = sk_fc_corners(R1, R2, C1, C2, tol=0.05)
    # 最坏高 fc = nom·(0.95)^{-2}
    assert wca["fc_max"] == pytest.approx(wca["fc_nominal"] * 0.95 ** -2, rel=1e-9)
    assert wca["fc_min"] == pytest.approx(wca["fc_nominal"] * 1.05 ** -2, rel=1e-9)
    assert len(wca["fc_corners"]) == 16


def test_propagation_matches_mc():
    tol = 0.05
    ana = sk_fc_propagation_std(tol, "uniform")
    rng = np.random.default_rng(0)
    n = 100000
    s = rng.uniform(1 - tol, 1 + tol, size=(n, 4))
    fc = 1.0 / np.sqrt(s.prod(axis=1))  # 相对值
    assert ana == pytest.approx(float(np.std(fc) / np.mean(fc)), rel=0.02)
    ana_rc = rc_fc_propagation_std(1.0, 1.0, tol, "uniform")
    s2 = rng.uniform(1 - tol, 1 + tol, size=(n, 2))
    fc2 = 1.0 / s2.prod(axis=1)
    assert ana_rc == pytest.approx(float(np.std(fc2) / np.mean(fc2)), rel=0.02)


def test_sensitivity_c1_zero_at_f0():
    """|H(f0)|=Q 对 C1 一阶灵敏度为 0（w0 与 Q 随 C1 同步移动的精确对消）。"""
    f0 = 1e3
    R1, R2, C1, C2 = design_sk_lp_equalR(f0, BUTTER4_Q[0], C2=1e-8)
    sens = sk_mag_sensitivity(R1, R2, C1, C2, f0)
    assert abs(sens["sensitivity"]["C1"]) < 1e-3
    # C2 应主导
    assert abs(sens["sensitivity"]["C2"]) > abs(sens["sensitivity"]["C1"])


def test_yield_analytic_tighten_worst_component():
    R1 = R2 = 1e4
    C2 = 1e-8
    C1 = 4 * 0.5412 ** 2 * C2
    f_spec = 0.7e3
    H_nom = abs(sk_lowpass_H(R1, R2, C1, C2, 2 * np.pi * f_spec))
    full = {k: 0.05 for k in ("R1", "R2", "C1", "C2")}
    y0 = sk_yield_analytic_design(R1, R2, C1, C2, f_spec, H_nom, 5.0, full, n_samples=5000, seed=2)
    tm = dict(full)
    tm["C2"] = 0.01
    y1 = sk_yield_analytic_design(R1, R2, C1, C2, f_spec, H_nom, 5.0, tm, n_samples=5000, seed=2)
    tm2 = dict(full)
    tm2["C1"] = 0.01
    y2 = sk_yield_analytic_design(R1, R2, C1, C2, f_spec, H_nom, 5.0, tm2, n_samples=5000, seed=2)
    assert y1["yield"] >= y0["yield"]
    # 加严最不敏感的 C1 应不优于加严 C2
    assert y1["yield"] >= y2["yield"] - 1e-9
