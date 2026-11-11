"""多路反馈（MFB）带通滤波器：节点法解析传递函数、设计与灵敏度（v1.2）。

拓扑（本项目定义，全部由节点法手推，报告 §11.1）：

    R1: Vin -> n1       R2: n1 -> GND
    C1: n1  -> n2       C2: n1 -> Vout
    R3: n2  -> Vout     运放：V+ = GND，V− = n2，Vout = A_OL·(0 − V(n2))

Vout 经 C2（回 n1）与 R3（回 n2）两条路径反馈，故称 multiple feedback。
理想运放下 V(n2)=0、输入电流为 0：

    KCL@n2 :  −s C1 V(n1) − Vout/R3 = 0                 → V(n1) = −Vout/(s C1 R3)
    KCL@n1 :  (V(n1)−Vin)/R1 + V(n1)/R2 + s C1 V(n1) + s C2 (V(n1)−Vout) = 0

    H(s) = −(s/(C2 R1)) / [ s² + s·(C1+C2)/(C1 C2 R3)
                              + (1/R1 + 1/R2)/(C1 C2 R3) ]

与 v1.0 的 Sallen-Key 等元件带通（Q=1/3 为拓扑硬约束）不同，
本拓扑的 Q 由元件比自由设定，是报告 §6 遗留问题的正面回应。
"""
from __future__ import annotations

import numpy as np
from scipy import signal

__all__ = [
    "mfb_bp_H", "mfb_bp_tf", "mfb_bp_params", "mfb_bp_params_array",
    "design_mfb_bp", "mfb_bp_sensitivity", "mfb_bp_noise_gain",
    "MFB_COMPONENTS",
]

MFB_COMPONENTS = ("R1", "R2", "R3", "C1", "C2")


def mfb_bp_H(R1: float, R2: float, R3: float, C1: float, C2: float, w):
    """MFB 带通频率响应 H(jw)（理想运放）。"""
    s = 1j * np.asarray(w, dtype=float)
    a1 = (C1 + C2) / (C1 * C2 * R3)
    a0 = (1.0 / R1 + 1.0 / R2) / (C1 * C2 * R3)
    return -(s / (C2 * R1)) / (s ** 2 + s * a1 + a0)


def mfb_bp_tf(R1: float, R2: float, R3: float, C1: float,
              C2: float) -> signal.TransferFunction:
    """MFB 带通传递函数（scipy 对象，分子/分母按 s 降幂）。"""
    a1 = (C1 + C2) / (C1 * C2 * R3)
    a0 = (1.0 / R1 + 1.0 / R2) / (C1 * C2 * R3)
    return signal.TransferFunction([-1.0 / (C2 * R1), 0.0], [1.0, a1, a0])


def mfb_bp_params(R1: float, R2: float, R3: float, C1: float, C2: float) -> dict:
    """由元件值给出 (f0, Q, H0, 3dB 带宽)。

    w0²   = (1/R1 + 1/R2)/(C1 C2 R3)
    w0/Q  = (C1 + C2)/(C1 C2 R3)
    |H(jw0)| = (w0/(C2 R1))/(w0²/Q) = Q/(C2 R1 w0)
    """
    a1 = (C1 + C2) / (C1 * C2 * R3)
    a0 = (1.0 / R1 + 1.0 / R2) / (C1 * C2 * R3)
    w0 = float(np.sqrt(a0))
    Q = w0 / a1
    H0 = Q / (C2 * R1 * w0)
    return {
        "w0_rad_per_s": w0,
        "f0_Hz": w0 / (2 * np.pi),
        "Q": float(Q),
        "H0": float(H0),
        "bw_3db_Hz": w0 / (2 * np.pi) / Q,
        "a1": float(a1),
        "a0": float(a0),
    }


def mfb_bp_params_array(R1, R2, R3, C1, C2) -> dict:
    """`mfb_bp_params` 的向量化版本（供蒙特卡洛批量抽样，输入可为 ndarray）。"""
    R1, R2, R3, C1, C2 = (np.asarray(x, dtype=float)
                          for x in (R1, R2, R3, C1, C2))
    a1 = (C1 + C2) / (C1 * C2 * R3)
    a0 = (1.0 / R1 + 1.0 / R2) / (C1 * C2 * R3)
    w0 = np.sqrt(a0)
    Q = w0 / a1
    return {"w0_rad_per_s": w0, "f0_Hz": w0 / (2 * np.pi),
            "Q": Q, "H0": Q / (C2 * R1 * w0)}


def design_mfb_bp(f0: float, Q: float, C1: float, C2: float | None = None,
                  rho: float = 1.0) -> dict:
    """由 (f0, Q, C1, C2, ρ=R1/R2) 反解元件值。

    R3 = Q (C1+C2)/(C1 C2 w0)
    1/R1 + 1/R2 = w0 Q (C1+C2)
    R2 = (1 + 1/ρ)/(w0 Q (C1+C2)),  R1 = ρ R2
    C1=C2=C、ρ=1 时 H0 = Q²（增益随 Q 二次增长，高 Q 需留输出摆幅预算）。
    """
    if C2 is None:
        C2 = C1
    if rho <= 0 or Q <= 0 or f0 <= 0:
        raise ValueError("f0/Q/rho must be positive")
    w0 = 2 * np.pi * f0
    R3 = Q * (C1 + C2) / (C1 * C2 * w0)
    R2 = (1.0 + 1.0 / rho) / (w0 * Q * (C1 + C2))
    R1 = rho * R2
    out = {"R1": float(R1), "R2": float(R2), "R3": float(R3),
           "C1": float(C1), "C2": float(C2), "rho": float(rho),
           "f0_target_Hz": float(f0), "Q_target": float(Q)}
    out.update(mfb_bp_params(R1, R2, R3, C1, C2))
    return out


def mfb_bp_sensitivity(R1: float, R2: float, R3: float, C1: float, C2: float,
                       eps: float = 1e-6) -> dict:
    """f0/Q/H0 对 5 个元件的相对灵敏度 d(ln Y)/d(ln X)（中心差分）。

    解析可预期：f0 对 C1、C2、R3 均为 −0.5；ρ=1 时对 R1、R2 各 −0.25。
    """
    base = dict(R1=R1, R2=R2, R3=R3, C1=C1, C2=C2)
    p0 = mfb_bp_params(**base)
    keys = ("f0_Hz", "Q", "H0")
    out = {k: {} for k in keys}
    for name in MFB_COMPONENTS:
        up = base.copy()
        dn = base.copy()
        up[name] = base[name] * (1.0 + eps)
        dn[name] = base[name] * (1.0 - eps)
        pu = mfb_bp_params(**up)
        pd = mfb_bp_params(**dn)
        for k in keys:
            out[k][name] = float(
                (np.log(pu[k]) - np.log(pd[k])) / (2.0 * eps))
    return out


def mfb_bp_noise_gain(H0: float) -> float:
    """反相组态一阶噪声增益 NG ≈ 1 + |闭环峰值增益|。

    运放开环增益在 f0 处须远大于 NG，否则 Q 与 f0 都会退化（实验 E16 扫描验证）。
    """
    return 1.0 + abs(H0)
