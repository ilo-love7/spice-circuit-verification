"""非理想运放（有限 GBW / 压摆率）理论与宏模型常数。

宏模型（单极点 + 限流）：
    B1 0 nc I = max(-Imax, min(Imax, gm*(V(inp)-V(inn))))
    R1 nc 0 Rlag          C1 nc 0 Cc          E1 out 0 nc 0 1

小信号：A(s) = A0/(1 + s/wb)，A0 = gm·Rlag，wb = 1/(Rlag·Cc)
    GBW = A0·wb/(2π) = gm/(2π·Cc)          [Hz]
大信号：|i| ≤ Imax  →  SR = Imax/Cc        [V/s]
"""
from __future__ import annotations

import numpy as np


def macromodel_params(gm: float, Rlag: float, Cc: float, Imax: float) -> dict:
    """由宏模型元件值给出运放规格。"""
    A0 = gm * Rlag
    wb = 1.0 / (Rlag * Cc)
    gbw_hz = gm / (2.0 * np.pi * Cc)
    sr_v_per_s = Imax / Cc
    return {
        "A0": A0,
        "wb_rad_per_s": wb,
        "fb_Hz": wb / (2.0 * np.pi),
        "GBW_Hz": gbw_hz,
        "SR_V_per_s": sr_v_per_s,
    }


def design_macromodel(GBW_Hz: float, A0: float, SR_V_per_s: float,
                      Rlag: float = 1e6) -> dict:
    """按目标 GBW/A0/SR 反算宏模型元件。"""
    Cc = 1.0 / (2.0 * np.pi * GBW_Hz) * 1.0  # 由 GBW=gm/(2πCc) 与 gm 定标
    # 选 gm = 1e-3 S（标准跨导），则 Cc = gm/(2π·GBW)
    gm = 1e-3
    Cc = gm / (2.0 * np.pi * GBW_Hz)
    Rlag = A0 / gm
    Imax = SR_V_per_s * Cc
    return {"gm": gm, "Rlag": Rlag, "Cc": Cc, "Imax": Imax,
            **macromodel_params(gm, Rlag, Cc, Imax)}


def opamp_A(s, A0: float, wb: float):
    """开环 A(s) = A0/(1 + s/wb)。"""
    return A0 / (1.0 + s / wb)


def inverting_amp_H_finiteGBW(R1: float, Rf: float, w, A0: float, wb: float):
    """反相放大闭环频率响应（有限 GBW）。

    H(jw) = −G·A(s)/(A(s) + NG)，G = Rf/R1，NG = 1 + G（噪声增益）。
    """
    G = Rf / R1
    NG = 1.0 + G
    A = opamp_A(1j * np.asarray(w, dtype=float), A0, wb)
    return -G * A / (A + NG)


def closed_loop_pole_approx_Hz(GBW_Hz: float, NG: float) -> float:
    """经典规则：闭环 −3dB ≈ GBW / 噪声增益（NG ≪ A0 时）。

    注：该式给出的是极点频率（对应 −3.0103 dB 半功率点）；
    若按 −3.00 dB 判据提取，应乘 √(10^0.3 − 1) ≈ 0.99763。
    """
    return GBW_Hz / NG


def f3dB_from_pole(pole_Hz: float, level_db: float = -3.0) -> float:
    """一阶系统：由极点频率换算任意 −level_db 判据的截止频率。

    f_{-LdB} = fp·√(10^(L/10) − 1)，L=3.0103 时因子为 1。
    """
    return float(pole_Hz * np.sqrt(10.0 ** (abs(level_db) / 10.0) - 1.0))


def group_delay(H, w):
    """群延迟 τg(ω) = −dφ/dω（秒）。w 为 rad/s，需近似均匀 log 网格。"""
    H = np.asarray(H, dtype=complex)
    w = np.asarray(w, dtype=float)
    phi = np.unwrap(np.angle(H))
    dphi = np.gradient(phi, w)
    return -dphi


def slew_expected_V_per_us(SR_V_per_s: float) -> float:
    return SR_V_per_s / 1e6
