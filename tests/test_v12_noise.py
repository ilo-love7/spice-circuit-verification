"""v1.2 E17 单测：热噪声 / kT/C / ENBW / 积分噪声 / 阻抗标度与 .noise 对照。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.theory import noise, mfb
from src.circuits import v12_builders as vb
from src.engine.runner import run_netlist, DEFAULT_NGSPICE

NG_OK = Path(DEFAULT_NGSPICE).exists()
F0 = 1591.5494309189535
C = 1e-8
R = 1000.0


# ---------- 解析公式 ----------
def test_thermal_density_value_and_scaling():
    e = noise.thermal_density_v_per_sqrthz(R)
    assert e == pytest.approx(np.sqrt(4 * noise.BOLTZMANN * noise.T_NGSPICE_K * R), rel=1e-14)
    assert noise.thermal_density_v_per_sqrthz(4 * R) == pytest.approx(2 * e, rel=1e-14)
    assert noise.thermal_psd_v2_per_hz(R) == pytest.approx(e ** 2, rel=1e-14)


def test_kT_over_C_identity():
    """√(4kTR·1/(4RC)) = √(kT/C)：RC 输出噪声与 R 无关。"""
    for Cx in (1e-9, 1e-8, 1e-6):
        assert noise.rc_output_noise_rms(R, Cx) == pytest.approx(
            noise.kT_over_C_rms(Cx), rel=1e-14)
    assert noise.rc_output_noise_rms(10 * R, 1e-6) == pytest.approx(
        noise.rc_output_noise_rms(R, 1e-6), rel=1e-14)


@pytest.mark.parametrize("Q", [0.7071067811865476, 1.0, 2.0, 5.0, 10.0])
def test_bandpass_enbw_analytic_vs_numeric(Q):
    a = noise.bandpass_enbw_hz(1000.0, Q)
    n = noise.bandpass_enbw_numeric(1000.0, Q)
    assert n == pytest.approx(a, rel=2e-4)
    assert a == pytest.approx(np.pi * 1000.0 / (2 * Q), rel=1e-14)


def test_enbw_is_pi_over_2_times_3db_bandwidth():
    d = mfb.design_mfb_bp(F0, 5.0, C)
    assert noise.bandpass_enbw_hz(d["f0_Hz"], d["Q"]) == pytest.approx(
        np.pi / 2 * d["bw_3db_Hz"], rel=1e-12)


def test_bandpass_shape_peak_and_3db():
    f = np.array([F0])
    assert noise.bandpass_shape_sq(f, F0, 5.0)[0] == pytest.approx(1.0, rel=1e-14)
    # -3dB 点 x - 1/x = ±1/Q
    Q = 5.0
    x = (1 / Q + np.sqrt(1 / Q ** 2 + 4)) / 2
    assert noise.bandpass_shape_sq(np.array([x * F0]), F0, Q)[0] == pytest.approx(0.5, rel=1e-12)


def test_integrated_rms_flat_spectrum():
    f = np.linspace(0, 1000.0, 10001)
    d = np.full_like(f, 2e-9)  # 2 nV/√Hz
    assert noise.integrated_rms(f, d) == pytest.approx(2e-9 * np.sqrt(1000.0), rel=1e-8)


def test_integrated_rms_log_spacing_matches_uniform():
    f = np.linspace(1.0, 1000.0, 100001)
    d = np.full_like(f, 2e-9)
    assert noise.integrated_rms(f, d, log_spacing=True) == pytest.approx(
        noise.integrated_rms(f, d), rel=1e-8)


def test_opamp_noise_combines_in_quadrature():
    e_n, i_n, Rs = 10e-9, 1e-12, 1e4
    expected = np.sqrt(e_n ** 2 + (i_n * Rs) ** 2
                       + noise.thermal_psd_v2_per_hz(Rs))
    assert noise.opamp_input_noise_density(e_n, i_n, Rs) == pytest.approx(expected, rel=1e-14)
    # 电流噪声项 ∝ R²，高阻时主导
    assert noise.opamp_input_noise_density(e_n, i_n, 1e7) > expected * 100


def test_combine_and_snr():
    assert noise.combine_noise_densities(3e-9, 4e-9) == pytest.approx(5e-9, rel=1e-14)
    assert noise.snr_db(1.0, 1e-3) == pytest.approx(60.0, rel=1e-12)
    assert noise.snr_db(2.0, 1e-3) - noise.snr_db(1.0, 1e-3) == pytest.approx(6.0206, abs=1e-3)


def test_output_noise_from_input_density_matches_integral():
    Q, H0, e_in = 4.0, 9.0, 5e-9
    f = np.logspace(np.log10(F0 / 1e4), np.log10(F0 * 1e4), 200000)
    dens = e_in * H0 * np.sqrt(noise.bandpass_shape_sq(f, F0, Q))
    assert noise.integrated_rms(f, dens) == pytest.approx(
        noise.output_noise_from_input_density(e_in, F0, Q, H0), rel=2e-4)


def test_impedance_scaling_changes_noise_as_sqrt_alpha():
    """α 标度保持响应、噪声 ∝ √α —— 「减小电阻必然降噪」的一阶结论。"""
    d = mfb.design_mfb_bp(F0, 5.0, C)
    base = {k: d[k] for k in mfb.MFB_COMPONENTS}
    for alpha in (0.1, 10.0):
        sc = noise.scale_passive_network(base, alpha)
        e0 = noise.thermal_density_v_per_sqrthz(base["R1"])
        e1 = noise.thermal_density_v_per_sqrthz(sc["R1"])
        assert e1 / e0 == pytest.approx(np.sqrt(alpha), rel=1e-14)
        for key in ("f0_Hz", "Q", "H0"):
            assert mfb.mfb_bp_params(**sc)[key] == pytest.approx(d[key], rel=1e-10)


# ---------- ngspice .noise 对照 ----------
@pytest.mark.skipif(not NG_OK, reason="ngspice engine not available")
def test_ngspice_thermal_density_matches_4kTR():
    res = run_netlist(vb.rc_noise_netlist(R, 1e-6, 1.0, 1e6, 20))
    sd = res.plot("Noise Spectral Density")
    # 低频 |H|≈1，onoise_spectrum → √(4kTR)
    assert sd["onoise_spectrum"][0] == pytest.approx(
        noise.thermal_density_v_per_sqrthz(R), rel=2e-4)
    assert sd["inoise_spectrum"][0] == pytest.approx(
        noise.thermal_density_v_per_sqrthz(R), rel=2e-4)


@pytest.mark.skipif(not NG_OK, reason="ngspice engine not available")
def test_ngspice_rc_total_noise_matches_kT_over_C():
    res = run_netlist(vb.rc_noise_netlist(R, 1e-6, 0.01, 1e9, 20))
    integ = res.plot("Integrated Noise")
    got = float(integ["v(onoise_total)"][0])
    # 1 Hz..1 GHz 扫频覆盖了 250 Hz ENBW 的绝大部分（高频截断 <0.1%）
    assert got == pytest.approx(noise.kT_over_C_rms(1e-6), rel=3e-3)


@pytest.mark.skipif(not NG_OK, reason="ngspice engine not available")
def test_en_resistor_leaves_ac_response_unchanged():
    """e_n 等效电阻串在同相输入：VCVS 不取电流 ⇒ 传递函数严格不变。"""
    d = mfb.design_mfb_bp(F0, 5.0, C)
    a = tuple(d[k] for k in mfb.MFB_COMPONENTS)
    e_n = 10e-9
    R_eq = e_n ** 2 / noise.thermal_psd_v2_per_hz(1.0)
    plain = vb.mfb_bp_netlist(*a, 100.0, 1e5, 40)
    with_en = vb.mfb_bp_en_netlist(*a, R_eq, 100.0, 1e5, 40, analysis="ac")
    v1 = np.asarray(run_netlist(plain).plot("AC Analysis")["v(out)"], dtype=complex)
    v2 = np.asarray(run_netlist(with_en).plot("AC Analysis")["v(out)"], dtype=complex)
    assert np.allclose(v1, v2, rtol=1e-9, atol=1e-12)


@pytest.mark.skipif(not NG_OK, reason="ngspice engine not available")
def test_en_contribution_is_alpha_independent():
    """e_n 等效电阻不随阻抗标度 α 变化 ⇒ 其输出噪声功率贡献应恒定。

    分解用功率平方差：P_en = onoise_total²(with e_n) − onoise_total²(passive)。
    """
    d = mfb.design_mfb_bp(F0, 5.0, C)
    base = {k: d[k] for k in mfb.MFB_COMPONENTS}
    e_n = 10e-9
    R_eq = e_n ** 2 / noise.thermal_psd_v2_per_hz(1.0)
    powers = []
    for alpha in (0.5, 2.0):
        sc = noise.scale_passive_network(base, alpha)
        a = tuple(sc[k] for k in mfb.MFB_COMPONENTS)

        def total(nl):
            return float(run_netlist(nl).plot("Integrated Noise")["v(onoise_total)"][0])

        p_passive = total(vb.mfb_bp_noise_netlist(*a, 1.0, 1e6, 40))
        p_with = total(vb.mfb_bp_en_netlist(*a, R_eq, 1.0, 1e6, 40))
        assert p_with > p_passive
        powers.append(p_with ** 2 - p_passive ** 2)
    assert powers[1] == pytest.approx(powers[0], rel=0.02)


@pytest.mark.skipif(not NG_OK, reason="ngspice engine not available")
def test_ngspice_mfb_noise_peak_tracks_f0():
    d = mfb.design_mfb_bp(F0, 5.0, C)
    a = tuple(d[k] for k in mfb.MFB_COMPONENTS)
    res = run_netlist(vb.mfb_bp_noise_netlist(*a, 1.0, 1e6, 40))
    sd = res.plot("Noise Spectral Density")
    f, on = sd["frequency"], sd["onoise_spectrum"]
    assert f[np.argmax(on)] == pytest.approx(F0, rel=0.02)
    integ = res.plot("Integrated Noise")
    assert float(integ["v(onoise_total)"][0]) > 0
    # 输入参考噪声谱 = 输出谱 / |H|
    H = np.abs(mfb.mfb_bp_H(*a, 2 * np.pi * f))
    assert np.allclose(sd["inoise_spectrum"] * H, on, rtol=1e-3)
