"""大信号压摆 / 失真理论（v1.2）：FPBW、THD、斜率与建立时间。

v1.1 §7.5 只验证了方波边沿斜率（SR 实测 +0.028%）；本轮把「小信号带宽」
与「全功率带宽」分离，定量给出大信号正弦的失真起始点：

    正弦最大斜率   2π f Vp  [V/s]
    FPBW           SR/(2π Vp)  [Hz]      f > FPBW ⇒ 压摆限幅、波形三角化、THD 陡升
    小信号闭环带宽  GBW/NG  [Hz]          与幅度无关

THD 用稳态段 + Hanning 窗 + 均匀重采样 FFT；谐波在 h·f_fund 的 ±tol 邻域
取峰值（N5 已证明「精确 2×bin 单 bin」在谱泄漏下会系统性低估）。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "sine_max_slew_v_per_s", "full_power_bandwidth_hz", "slew_ratio",
    "required_sr_v_per_s", "slew_limited_rise_time_s", "resample_uniform",
    "thd_percent", "waveform_metrics", "harmonic_peak_at",
]


def sine_max_slew_v_per_s(f: float, v_peak: float) -> float:
    """正弦 v(t)=Vp sin(2πft) 的最大斜率 = 2πfVp [V/s]。"""
    return float(2.0 * np.pi * f * v_peak)


def full_power_bandwidth_hz(sr_v_per_s: float, v_peak: float) -> float:
    """全功率带宽：保持满幅正弦不失真的最高频率 SR/(2πVp) [Hz]。"""
    return float(sr_v_per_s / (2.0 * np.pi * v_peak))


def slew_ratio(f: float, v_peak: float, sr_v_per_s: float) -> float:
    """所需斜率 / 可用斜率；>1 即进入压摆限幅区。"""
    return float(sine_max_slew_v_per_s(f, v_peak) / sr_v_per_s)


def required_sr_v_per_s(f: float, v_peak: float) -> float:
    return sine_max_slew_v_per_s(f, v_peak)


def slew_limited_rise_time_s(v_step: float, sr_v_per_s: float) -> float:
    """纯压摆限制下的阶跃上升时间下限 ΔV/SR [s]。"""
    return float(abs(v_step) / sr_v_per_s)


def resample_uniform(t, y, n: int | None = None):
    """把 ngspice 变步长瞬态结果线性重采样到均匀时间网格。"""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if n is None:
        n = len(t)
    tu = np.linspace(float(t[0]), float(t[-1]), n)
    return tu, np.interp(tu, t, y)


def harmonic_peak_at(mag, freqs, f_target: float, tol_bins: int = 3) -> float:
    """在 f_target 附近 ±tol_bins 个 FFT bin 内取幅值峰值。"""
    j = int(np.argmin(np.abs(freqs - f_target)))
    lo = max(0, j - tol_bins)
    hi = min(mag.size, j + tol_bins + 1)
    return float(np.max(mag[lo:hi]))


def thd_percent(t, y, f_fund: float | None = None, n_harm: int = 10,
                steady_frac: float = 0.5, n_uniform: int | None = None,
                tol_bins: int = 3) -> dict:
    """总谐波失真 THD [%]（前 n_harm 次谐波，含 2..n_harm）。

    返回 dict(thd_pct, f_fund_Hz, harmonic_mag, vrms, n_points)。
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t0 = t[0] + steady_frac * (t[-1] - t[0])
    m = t >= t0
    tt, yy = t[m], y[m]
    if tt.size < 64:
        raise ValueError("too few steady-state samples for THD")
    tu, yu = resample_uniform(tt, yy, n_uniform)
    n = tu.size
    dt = float(tu[1] - tu[0])
    win = np.hanning(n)
    mag = np.abs(np.fft.rfft(yu * win))
    freqs = np.fft.rfftfreq(n, dt)
    if f_fund is None:
        j0 = int(np.argmax(mag[1:])) + 1
    else:
        j0 = int(np.argmin(np.abs(freqs - f_fund)))
    f0 = float(freqs[j0])
    fund = harmonic_peak_at(mag, freqs, f0, tol_bins)
    harm = {}
    for h in range(2, n_harm + 1):
        harm[str(h)] = harmonic_peak_at(mag, freqs, h * f0, tol_bins)
    thd = 100.0 * float(np.sqrt(np.sum(np.square(list(harm.values()))))) / max(fund, 1e-30)
    return {"thd_pct": thd, "f_fund_Hz": f0, "harmonic_mag": harm,
            "vrms": float(np.sqrt(np.mean(yu ** 2))), "n_points": n}


def waveform_metrics(t, y) -> dict:
    """波形指标：峰峰值、rms、峰值、最大绝对斜率 [V/s]。"""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    dt = np.diff(t)
    ok = dt > 0
    slope = float(np.max(np.abs(dy[ok] / dt[ok]))) if np.any(ok) else 0.0
    return {
        "vpp": float(np.ptp(y)),
        "vrms": float(np.sqrt(np.mean(y ** 2))),
        "peak": float(np.max(np.abs(y))),
        "max_abs_slope_v_per_s": slope,
        "max_abs_slope_v_per_us": slope / 1e6,
        "n_points": int(y.size),
    }
