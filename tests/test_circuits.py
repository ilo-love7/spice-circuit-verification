"""电路仿真 vs 理论一致性测试（含 PySpice 网表链路）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.engine.runner import run_netlist
from src.circuits.builders import (rc_lowpass_netlist, sk_lowpass_netlist,
                                   sk_bandpass_netlist, inverting_amp_netlist,
                                   rc_lowpass_tran, A_OL)
from src.circuits.pyspice_build import (pyspice_rc_lowpass, pyspice_sk_lowpass,
                                        pyspice_inverting_amp)
from src.theory.transfer import (rc_lowpass_H, sk_lowpass_H, sk_bandpass_H,
                                 inverting_amp_H, compare_response)


def _ac(net):
    p = run_netlist(net).plot("AC Analysis")
    f = np.array(p["frequency"]).real
    return f, np.array(p["v(out)"])


def test_rc_vs_theory():
    R, C = 1e3, 1e-6
    f, Hs = _ac(rc_lowpass_netlist(R, C, 1, 1e5, points_per_decade=10))
    Ht = rc_lowpass_H(R, C, 2 * np.pi * f)
    cmp = compare_response(f, Hs, Ht)
    assert cmp["max_abs_err_db"] < 1e-9


def test_sk_lp_vs_theory():
    R1 = R2 = 1e4
    C2 = 11.25e-9
    C1 = 2 * C2
    f, Hs = _ac(sk_lowpass_netlist(R1, R2, C1, C2, 10, 1e5, points_per_decade=10))
    Ht = sk_lowpass_H(R1, R2, C1, C2, 2 * np.pi * f)
    cmp = compare_response(f, Hs, Ht)
    # 有限开环增益带来 ~1e-5 dB 偏差
    assert cmp["max_abs_err_db"] < 1e-3


def test_sk_bp_vs_theory():
    f, Hs = _ac(sk_bandpass_netlist(1e4, 1e-8, 1.0, 1, 1e5, points_per_decade=10))
    Ht = sk_bandpass_H(1e4, 1e-8, 1.0, 2 * np.pi * f)
    cmp = compare_response(f, Hs, Ht)
    assert cmp["max_abs_err_db"] < 1e-9


def test_inverting_amp_gain():
    f, Hs = _ac(inverting_amp_netlist(1e3, 1e4, 1, 1e6, points_per_decade=5))
    G_th = -10.0 / (1 + 11 / A_OL)
    assert abs(abs(Hs[0]) - abs(G_th)) / abs(G_th) < 1e-6


def test_pyspice_netlist_matches_string():
    R, C = 1e3, 1e-6
    f1, H1 = _ac(rc_lowpass_netlist(R, C, 1, 1e5, points_per_decade=20))
    f2, H2 = _ac(pyspice_rc_lowpass(R, C, 1, 1e5))
    n = min(len(f1), len(f2))
    assert np.allclose(f1[:n], f2[:n], rtol=1e-6)
    assert np.allclose(H1[:n], H2[:n], rtol=1e-9, atol=1e-12)


def test_pyspice_sk_and_amp():
    R1 = R2 = 1e4
    C2 = 11.25e-9
    C1 = 2 * C2
    f, Hs = _ac(pyspice_sk_lowpass(R1, R2, C1, C2, 10, 1e5))
    Ht = sk_lowpass_H(R1, R2, C1, C2, 2 * np.pi * f)
    assert compare_response(f, Hs, Ht)["max_abs_err_db"] < 1e-3
    f, Hs = _ac(pyspice_inverting_amp(1e3, 1e4, 1, 1e6))
    assert abs(abs(Hs[0]) - 10.0 / (1 + 11 / A_OL)) < 1e-5


def test_transient_step_tau():
    R, C = 1e3, 1e-6
    tau = R * C
    net = rc_lowpass_tran(R, C, 1 / (2 * np.pi * tau), 10 * tau)
    p = run_netlist(net).plot("Transient")
    t = np.array(p["time"]).real
    v = np.array(p["v(out)"]).real
    # 用正弦稳态幅度检验：|H(fc)| = 1/sqrt(2)
    mask = t > t[len(t) // 2]
    amp = (v[mask].max() - v[mask].min()) / 2
    assert abs(amp - 1 / np.sqrt(2)) / (1 / np.sqrt(2)) < 0.02
