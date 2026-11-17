"""相位裕度与闭环稳定性：双极点运放模型（v1.3）。

单极点 A(s)=A0/(1+s/w1) 的闭环永远稳定（PM→90°），无法研究稳定性；
加入第二极点 w2 后：

    A(s) = A0 / [(1 + s/w1)(1 + s/w2)]        w2 = k·GBW

环路增益 L(s) = A(s)·β，β = 1/NG。增益交越频率 wc 满足 |L(jwc)| = 1。
相位裕度：

    PM = 180° + arg L(jwc) = 180° − arctan(wc/w1) − arctan(wc/w2)

关键结论（可证伪）：
  1. 噪声增益 NG 越大 → β 越小 → wc 越低 → PM 越高（同相单位缓冲 NG=1 最差）。
  2. 第二极点 w2 越低（k 越小），PM 越差；w2→∞ 时 PM→arctan 项→近 90°。
  3. 欠阻尼闭环的阶跃过冲 Mp 与 PM 一一对应（二阶主导）：
     ζ 由 PM 反解，Mp = exp(−πζ/√(1−ζ²))。

过冲→PM 的反解用于 SPICE 瞬态与解析的独立对照（不依赖读 AC 相位）。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "open_loop_two_pole", "loop_gain", "gain_crossover_rad",
    "phase_margin_deg", "pm_sweep", "damping_from_pm",
    "overshoot_from_pm", "pm_from_overshoot", "is_stable_pm",
    "step_overshoot",
]


def step_overshoot(t, y, y_final: float | None = None,
                   settle_frac: float = 0.9) -> float:
    """从阶跃响应测过冲 Mp=(y_max−y_final)/(y_final−y_initial)。

    取后 settle_frac 段的稳定值作 y_final（避开上升沿），y_initial 取前 5%。
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 20:
        return float("nan")
    y0 = float(np.mean(y[:max(2, n // 20)]))
    if y_final is None:
        y_final = float(np.mean(y[int(settle_frac * n):]))
    span = y_final - y0
    if abs(span) < 1e-15:
        return float("nan")
    return float(max(0.0, (np.max(y) - y_final) / abs(span)))


def open_loop_two_pole(w, A0: float, w1: float, w2: float):
    """双极点开环 A(jw)。"""
    s = 1j * np.asarray(w, dtype=float)
    return A0 / ((1.0 + s / w1) * (1.0 + s / w2))


def loop_gain(w, A0: float, w1: float, w2: float, beta: float):
    """环路增益 L(jw) = A(jw)·β。"""
    return open_loop_two_pole(w, A0, w1, w2) * float(beta)


def gain_crossover_rad(A0: float, w1: float, w2: float, beta: float) -> float:
    """|L(jwc)|=1 的交越角频率（解析求根，log 域二分）。

    双极点系统低频 |L|→A0·β（常数），高频 |L|→0，故 g(lo)>0、g(hi)<0。
    若 A0·β≤1 则无交越（返回 nan），不可用「降 lo」循环——低频 g 恒为正会死循环。
    """
    from scipy.optimize import brentq

    if A0 * beta <= 1.0:
        return float("nan")

    def g(logw):
        w = np.exp(logw)
        return np.abs(loop_gain(w, A0, w1, w2, beta)) - 1.0

    lo = np.log(w1) - 10.0
    hi = np.log(max(w1, w2)) + 30.0
    while g(hi) >= 0:  # 高频 |L|→0，此循环必然终止
        hi += 10.0
    return float(np.exp(brentq(g, lo, hi, xtol=1e-12)))


def phase_margin_deg(A0: float, w1: float, w2: float, beta: float) -> float:
    """相位裕度（度）。"""
    wc = gain_crossover_rad(A0, w1, w2, beta)
    phase = np.angle(loop_gain(wc, A0, w1, w2, beta), deg=True)
    return float(180.0 + phase)


def pm_sweep(ng_list, A0: float, w1: float, w2: float):
    """按 NG 扫描 PM（NG 越大越稳）。"""
    out = []
    for ng in ng_list:
        beta = 1.0 / float(ng)
        wc = gain_crossover_rad(A0, w1, w2, beta)
        out.append({"NG": float(ng), "wc_rad": wc,
                    "PM_deg": phase_margin_deg(A0, w1, w2, beta)})
    return out


def damping_from_pm(pm_deg: float) -> float:
    """由相位裕度反解二阶阻尼比 ζ（标准近似，PM∈(0,90)）。

    对标准二阶 L(s)=w0²/[s(s+2ζw0)]，PM≈atan(2ζ/√(√(1+4ζ⁴)−2ζ²))。
    这里用数值反解避免闭式歧义。
    """
    from scipy.optimize import brentq

    pm = np.deg2rad(float(pm_deg))

    def zeta_of(p):
        # 由 ζ 算 PM 的正向（标准二阶近似）
        z = float(p)
        m = 2.0 * z / np.sqrt(np.sqrt(1.0 + 4.0 * z ** 4) - 2.0 * z ** 2)
        return np.arctan(m)

    f = lambda z: zeta_of(z) - pm
    # ζ→∞ 时 PM→90°；高 PM（>86°）对应 ζ 很大、过冲≈0，扩上界避免无根
    if pm >= zeta_of(2.0):
        return 2.0
    return float(brentq(f, 1e-4, 2.0, xtol=1e-12))


def overshoot_from_pm(pm_deg: float) -> float:
    """由相位裕度预测阶跃过冲 Mp（0–1，ζ<1 时；ζ≥1 返回 0）。"""
    zeta = damping_from_pm(pm_deg)
    if zeta >= 1.0:
        return 0.0
    return float(np.exp(-np.pi * zeta / np.sqrt(1.0 - zeta ** 2)))


def pm_from_overshoot(mp: float) -> float:
    """由测得过冲反推 PM（度），用于 SPICE 瞬态独立验证。"""
    mp = float(mp)
    if mp <= 0.0:
        return 90.0
    mp = min(mp, 0.999999)
    zeta = -np.log(mp) / np.sqrt(np.pi ** 2 + np.log(mp) ** 2)
    pm = np.arctan(2.0 * zeta / np.sqrt(np.sqrt(1.0 + 4.0 * zeta ** 4)
                                        - 2.0 * zeta ** 2))
    return float(np.rad2deg(pm))


def is_stable_pm(pm_deg: float) -> bool:
    """相位裕度判据（工程常用 PM>45° 视为有足够稳定裕度）。"""
    return float(pm_deg) > 45.0
