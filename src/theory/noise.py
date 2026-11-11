"""噪声分析理论（v1.2）：热噪声、等效噪声带宽、噪声积分与 SNR。

与 ngspice `.noise` 三重对照：
  * 电阻电压噪声密度  √(4kTR)  [V/√Hz]
  * 一阶 RC 输出噪声  √(kT/C)  [V rms]   ← kT/C 噪声，与 R 无关
  * 二阶带通 ENBW     π f0/(2Q)（= 1.571 × −3dB 带宽）

ngspice 的 `.noise` 在 T = 27 °C = 300.15 K 下计算，raw 输出两段：
  'Noise Spectral Density Curves' → frequency / onoise_spectrum / inoise_spectrum
  'Integrated Noise'              → v(onoise_total) / v(inoise_total)
其中 inoise_spectrum = onoise_spectrum / |H|（输入参考噪声密度）。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "BOLTZMANN", "T_NGSPICE_K", "thermal_psd_v2_per_hz",
    "thermal_density_v_per_sqrthz", "kT_over_C_rms", "rc_enbw_hz",
    "rc_output_noise_rms", "bandpass_enbw_hz", "bandpass_shape_sq",
    "bandpass_enbw_numeric", "integrated_rms", "opamp_input_noise_density",
    "combine_noise_densities", "snr_db", "scale_passive_network",
    "output_noise_from_input_density",
]

BOLTZMANN = 1.380649e-23
T_NGSPICE_K = 300.15  # 27 °C，ngspice 默认


def thermal_psd_v2_per_hz(R: float, T: float = T_NGSPICE_K) -> float:
    """电阻热噪声电压功率谱密度 4kTR [V²/Hz]。"""
    return float(4.0 * BOLTZMANN * T * R)


def thermal_density_v_per_sqrthz(R: float, T: float = T_NGSPICE_K) -> float:
    """电阻热噪声电压密度 √(4kTR) [V/√Hz]。"""
    return float(np.sqrt(thermal_psd_v2_per_hz(R, T)))


def rc_enbw_hz(R: float, C: float) -> float:
    """一阶 RC 低通等效噪声带宽 = 1/(4RC) [Hz]。"""
    return 1.0 / (4.0 * R * C)


def kT_over_C_rms(C: float, T: float = T_NGSPICE_K) -> float:
    """kT/C 噪声：一阶 RC 输出总 rms 噪声 √(kT/C)，与 R 无关。"""
    return float(np.sqrt(BOLTZMANN * T / C))


def rc_output_noise_rms(R: float, C: float, T: float = T_NGSPICE_K) -> float:
    """√(4kTR · ENBW) = √(kT/C)。"""
    return float(np.sqrt(thermal_psd_v2_per_hz(R, T) * rc_enbw_hz(R, C)))


def bandpass_enbw_hz(f0: float, Q: float, H0: float = 1.0) -> float:
    """二阶带通等效噪声带宽 [Hz]：∫|H|²df = H0² · π f0/(2Q)。

    归一化到峰值增益时（H0=1）ENBW = (π/2)·BW_3dB。
    """
    return float(H0 ** 2 * np.pi * f0 / (2.0 * Q))


def bandpass_shape_sq(f, f0: float, Q: float):
    """归一化带通形状 |H(f)/H0|² = 1/[1 + Q²(x − 1/x)²]。"""
    x = np.asarray(f, dtype=float) / f0
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / (1.0 + Q ** 2 * (x - 1.0 / x) ** 2)


def bandpass_enbw_numeric(f0: float, Q: float, H0: float = 1.0,
                          ratio: float = 1e4, n: int = 200000) -> float:
    """对数网格数值积分 ∫|H/H0|²·H0² df，用于校验解析 ENBW。"""
    lo, hi = np.log(f0 / ratio), np.log(f0 * ratio)
    lf = np.linspace(lo, hi, n)
    f = np.exp(lf)
    y = H0 ** 2 * bandpass_shape_sq(f, f0, Q) * f  # df = f d(ln f)
    return float(np.trapezoid(y, lf))


def integrated_rms(f, density_v_per_sqrthz, log_spacing: bool = False) -> float:
    """由噪声密度谱 [V/√Hz] 积分得总 rms 噪声 [V]。

    ngspice `.noise` 输出的是对数扫频网格，线性梯形在大间隔上会显著高估
    1/f² 尾部；log_spacing=True 时作 df = f·d(ln f) 变量替换。
    """
    f = np.asarray(f, dtype=float)
    d = np.asarray(density_v_per_sqrthz, dtype=float)
    if log_spacing:
        return float(np.sqrt(np.trapezoid(d ** 2 * f, np.log(f))))
    return float(np.sqrt(np.trapezoid(d ** 2, f)))


def opamp_input_noise_density(e_n: float, i_n: float, r_source: float,
                              T: float = T_NGSPICE_K) -> float:
    """运放输入参考噪声密度 [V/√Hz]：e_n² + (i_n·R_s)² + 4kT·R_s 的平方和根。"""
    return float(np.sqrt(e_n ** 2 + (i_n * r_source) ** 2
                         + thermal_psd_v2_per_hz(r_source, T)))


def combine_noise_densities(*densities) -> float:
    """独立噪声源按功率相加（平方和根）。"""
    return float(np.sqrt(np.sum(np.square(np.asarray(densities, dtype=float)))))


def output_noise_from_input_density(e_in: float, f0: float, Q: float,
                                    H0: float = 1.0) -> float:
    """输入参考白噪声 e_in [V/√Hz] 经二阶带通后的输出 rms 噪声。"""
    return float(abs(e_in) * H0 * np.sqrt(np.pi * f0 / (2.0 * Q)))


def snr_db(signal_vrms: float, noise_vrms: float) -> float:
    """信噪比 [dB]。"""
    return float(20.0 * np.log10(abs(signal_vrms) / max(abs(noise_vrms), 1e-30)))


def scale_passive_network(values: dict, alpha: float) -> dict:
    """阻抗标度：R → αR、C → C/α，保持 f0/Q/H0 不变，仅改变热噪声。

    对 MFB：w0² = (1/R1+1/R2)/(C1C2R3)，α 缩放后 (1/α)/(1/α²·α) = 1，
    故频率响应严格不变；但 4kTR ∝ α、i_n·R ∝ α ⇒ 存在最优 α。
    """
    out = dict(values)
    for k, v in values.items():
        if k.startswith("R"):
            out[k] = float(v) * alpha
        elif k.startswith("C"):
            out[k] = float(v) / alpha
    return out
