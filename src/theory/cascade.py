"""高阶滤波器级联理论（4 阶 Butterworth = 两级 Sallen-Key）。

归一化 4 阶 Butterworth 极点 Q：
    Q1 = 1/(2 sin(π/8)) = 0.5411961
    Q2 = 1/(2 cos(π/8)) = 1.3065630
两节 f0 相同（= 截止频率 fc），级联传递函数 H = H1·H2。
"""
from __future__ import annotations

import numpy as np
from scipy import signal

BUTTER4_Q = (0.5411961001461970, 1.3065629648763766)


def design_sk_lp_equalR(f0: float, Q: float, C2: float = 1e-8) -> tuple:
    """由 (f0, Q) 设计单位增益 SK 低通（R1=R2=R）。

    Q = ½√(C1/C2)  →  C1 = 4Q²·C2；w0 = 1/(R√(C1C2))  →  R = 1/(2Q·w0·C2)。
    返回 (R1, R2, C1, C2)。
    """
    w0 = 2 * np.pi * f0
    C1 = 4.0 * Q * Q * C2
    R = 1.0 / (2.0 * Q * w0 * C2)
    return R, R, C1, C2


def butter4_lp_tf(f0: float) -> signal.TransferFunction:
    """4 阶 Butterworth 低通（fc=f0），分子增益 1。"""
    wn = 2 * np.pi * f0
    num, den = signal.butter(4, wn, btype="low", analog=True, output="ba")
    return signal.TransferFunction(num, den)


def cascade_sk_H(f0: float, w, Q1: float = BUTTER4_Q[0], Q2: float = BUTTER4_Q[1]):
    """两级 SK 级联（级间理想缓冲）理论响应：H = H1·H2。"""
    s = 1j * np.asarray(w, dtype=float)
    w0 = 2 * np.pi * f0

    def sect(Q):
        return 1.0 / (1.0 + s / (w0 * Q) + (s / w0) ** 2)

    return sect(Q1) * sect(Q2)


def cascade_transfer(f0: float, Q1: float = BUTTER4_Q[0], Q2: float = BUTTER4_Q[1]):
    """级联传递函数（scipy，分子/分母多项式）。"""
    w0 = 2 * np.pi * f0
    # 每节: w0^2 / (s^2 + (w0/Q)s + w0^2) —— 与 unity-gain SK 相同（分子 w0^2）
    b1 = [w0 * w0]
    a1 = [1.0, w0 / Q1, w0 * w0]
    b2 = [w0 * w0]
    a2 = [1.0, w0 / Q2, w0 * w0]
    num = np.convolve(b1, b2)
    den = np.convolve(a1, a2)
    return signal.TransferFunction(num, den)


def step_response_metrics(t, y) -> dict:
    """阶跃响应指标：过冲 %、t_r（10→90%）、t_s（2% 稳定时间）。"""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    final = float(y[-1])
    peak = float(np.max(y)) if final >= 0 else float(np.min(y))
    overshoot = max(0.0, (abs(peak) - abs(final)) / abs(final) * 100.0) if final != 0 else 0.0

    def crossing(level):
        idx = np.where(np.abs(y - level) >= 0)[0]
        for i in range(1, len(y)):
            lo, hi = y[i - 1], y[i]
            if (lo - level) * (hi - level) <= 0 and lo != hi:
                tt = t[i - 1] + (level - lo) * (t[i] - t[i - 1]) / (hi - lo)
                return float(tt)
        return float("nan")

    t10 = crossing(final * 0.1)
    t90 = crossing(final * 0.9)
    tr = t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else float("nan")
    # 2% 稳定时间：最后一次进入 ±2% 带后不再离开
    band = 0.02 * abs(final)
    settle = float("nan")
    outside = np.where(np.abs(y - final) > band)[0]
    if len(outside) == 0:
        settle = float(t[0])
    else:
        last = outside[-1]
        if last + 1 < len(t):
            settle = float(t[last + 1])
    return {"overshoot_pct": overshoot, "rise_time_s": tr, "settling_2pct_s": settle,
            "final": final, "peak": peak}
