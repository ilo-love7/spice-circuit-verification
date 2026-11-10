"""理论传递函数（由元件值解析推导）与误差指标。

全部传递函数为 s 域解析式，系数直接由元件值得到，
与 ngspice 仿真对比时不做任何拟合。
"""
from __future__ import annotations

import numpy as np
from scipy import signal


# ---------- 一阶 RC 低通 ----------
def rc_lowpass_tf(R: float, C: float) -> signal.TransferFunction:
    """H(s) = 1/(1 + sRC)。"""
    return signal.TransferFunction([1.0], [R * C, 1.0])


def rc_lowpass_H(R: float, C: float, w):
    """频率响应 H(jw)。"""
    return 1.0 / (1.0 + 1j * w * R * C)


# ---------- Sallen-Key 低通（单位增益） ----------
def sk_lowpass_tf(R1: float, R2: float, C1: float, C2: float) -> signal.TransferFunction:
    """单位增益 Sallen-Key 低通（正反馈 C1 拓扑）：

    Vin-R1-A-R2-B(=V+=Vout)，C1: A→Vout，C2: B→GND。
    H = 1/(1 + s(R1+R2)C2 + s^2 R1R2C1C2)
    """
    return signal.TransferFunction([1.0], [R1 * R2 * C1 * C2, (R1 + R2) * C2, 1.0])


def sk_lowpass_H(R1: float, R2: float, C1: float, C2: float, w):
    s = 1j * w
    return 1.0 / (1.0 + s * (R1 + R2) * C2 + s**2 * R1 * R2 * C1 * C2)


def sk_lowpass_params(R1: float, R2: float, C1: float, C2: float):
    """返回 (f0, Q, DC增益)。"""
    a2 = R1 * R2 * C1 * C2
    a1 = (R1 + R2) * C2
    w0 = 1.0 / np.sqrt(a2)
    Q = np.sqrt(a2) / a1
    return w0 / (2 * np.pi), Q, 1.0


# ---------- Sallen-Key 带通（等元件、接地 R 版本，Q=1/3） ----------
def sk_bandpass_tf(R: float, C: float, K: float = 1.0) -> signal.TransferFunction:
    """Sallen-Key 带通（Z1=C, Z2=R, Z3=R 接地, Z4=C 接地, 放大倍数 K）：

    H(s) = K·s/(RC) / (s² + 3s/(RC) + 1/(R²C²))

    f0 = 1/(2πRC)，Q = 1/3（接地 R 版本的结构约束，v1.1 深化突破它）。
    """
    wc = 1.0 / (R * C)
    # 分子 [K/RC, 0]，分母 [1, 3/RC, 1/(RC)^2]
    return signal.TransferFunction([K * wc, 0.0], [1.0, 3.0 * wc, wc * wc])


def sk_bandpass_H(R: float, C: float, K: float, w):
    wc = 1.0 / (R * C)
    s = 1j * w
    return K * s * wc / (s**2 + 3.0 * s * wc + wc * wc)


def sk_bandpass_params(R: float, C: float):
    w0 = 1.0 / (R * C)
    Q = 1.0 / 3.0
    return w0 / (2 * np.pi), Q


# ---------- 反相放大器（理想运放） ----------
def inverting_amp_H(R1: float, Rf: float, w, A_ol: float = np.inf):
    """H = -Rf/R1（理想）；有限开环增益 A_ol 时一阶修正。"""
    G = Rf / R1
    if not np.isfinite(A_ol):
        return np.full_like(np.asarray(w, dtype=float), -G, dtype=complex)
    # 单极点近似忽略（A_ol 为常数）：H = -G / (1 + (1+G)/A_ol)
    return np.full_like(np.asarray(w, dtype=float), -G / (1.0 + (1.0 + G) / A_ol), dtype=complex)


# ---------- 误差指标 ----------
def mag_db(H):
    """20log10|H|。"""
    return 20 * np.log10(np.maximum(np.abs(H), 1e-30))


def phase_deg(H):
    """相位（度），连续化处理。"""
    ph = np.unwrap(np.angle(H))
    return np.degrees(ph)


def compare_response(f, H_sim, H_th) -> dict:
    """对比仿真与理论：幅值误差(dB)、相位误差(deg)、最大/均方误差。"""
    err_db = mag_db(H_sim) - mag_db(H_th)
    err_deg = phase_deg(H_sim) - phase_deg(H_th)
    return {
        "max_abs_err_db": float(np.max(np.abs(err_db))),
        "rms_err_db": float(np.sqrt(np.mean(err_db**2))),
        "max_abs_err_deg": float(np.max(np.abs(err_deg))),
        "rms_err_deg": float(np.sqrt(np.mean(err_deg**2))),
        "err_db": err_db,
        "err_deg": err_deg,
    }


def cutoff_from_response(f, mag, level_db: float = -3.0) -> float:
    """由幅频曲线插值求指定衰减点频率（如 -3dB 截止）。"""
    mag_db_v = 20 * np.log10(np.maximum(mag, 1e-30))
    peak = np.max(mag_db_v)
    target = peak + level_db
    # 从低频向高频找第一个下穿
    for i in range(1, len(f)):
        if mag_db_v[i - 1] >= target > mag_db_v[i]:
            # 线性插值（log f）
            f1, f2 = f[i - 1], f[i]
            m1, m2 = mag_db_v[i - 1], mag_db_v[i]
            t = (target - m1) / (m2 - m1)
            return float(np.exp(np.log(f1) + t * (np.log(f2) - np.log(f1))))
    return float("nan")


def cutoff_precise(f_grid, H_vals, level_db: float = -3.0) -> float:
    """精确求 −level_db 截止频率（log-f 三次样条 + brentq）。

    相比 cutoff_from_response 的分段线性插值，消除网格曲率系统偏差
    （15 点/decade 时线性插值约有 −0.5% 偏差，见 v1.1 报告）。
    """
    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq

    f = np.asarray(f_grid, dtype=float)
    H = np.asarray(H_vals, dtype=complex) if np.iscomplexobj(H_vals) else np.asarray(H_vals, dtype=float)
    mag = np.abs(H)
    mag_db_v = 20 * np.log10(np.maximum(mag, 1e-30))
    peak = np.max(mag_db_v)
    target = peak + level_db
    spline = CubicSpline(np.log(f), mag_db_v)

    def g(logf):
        return float(spline(logf) - target)

    # 找首个下穿区间
    for i in range(1, len(f)):
        if mag_db_v[i - 1] >= target > mag_db_v[i]:
            lo, hi = np.log(f[i - 1]), np.log(f[i])
            root = brentq(g, lo, hi, xtol=1e-12)
            return float(np.exp(root))
    return float("nan")
