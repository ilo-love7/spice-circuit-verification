"""温度漂移解析模型（v1.3）。

电阻线性温度系数：R(T) = R0·[1 + TC1·(T−T0)]，TC1 单位 1/°C（ppm/°C ÷1e6）。
电容同理：C(T) = C0·[1 + TC1_C·(T−T0)]。

一阶 RC 低通截止频率 f0 = 1/(2πRC) 的温度漂移：

    df0/f0 ≈ −(TC1_R + TC1_C)·ΔT       （一阶，TC·ΔT ≪ 1）

关键结论（可证伪）：
  1. 电阻/电容 TC 同号 → f0 温漂叠加；反号且等值 → f0 一阶零温漂。
  2. 一阶近似在 |TC·ΔT|≪1 时与精确值吻合，ΔT 增大时二阶项显现。
  3. 反相放大器增益 G=Rf/R1：若两电阻同 TC，增益温度无关（比值抵消）——
     这是「匹配电阻比」降低温漂的理论依据。

运放失调温漂 Vos(T)=Vos0+TCVos·(T−T0) 折算到输出：
    Vout_os = NG·Vos(T)，同相抵消不了，靠器件选型/斩波。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "ppm_to_tc1", "resistor_at_T", "capacitor_at_T", "rc_cutoff_at_T",
    "cutoff_drift_ratio", "cutoff_drift_exact", "gain_ratio_drift",
    "vos_at_T", "output_offset_from_vos",
]


def ppm_to_tc1(ppm_per_c: float) -> float:
    """ppm/°C → 1/°C。"""
    return float(ppm_per_c) / 1e6


def resistor_at_T(R0: float, tc1: float, T, T0: float = 25.0):
    """线性温度系数电阻 R(T)=R0[1+tc1(T−T0)]。"""
    return float(R0) * (1.0 + float(tc1) * (np.asarray(T, dtype=float) - T0))


def capacitor_at_T(C0: float, tc1: float, T, T0: float = 25.0):
    """线性温度系数电容 C(T)=C0[1+tc1(T−T0)]。"""
    return float(C0) * (1.0 + float(tc1) * (np.asarray(T, dtype=float) - T0))


def rc_cutoff_at_T(R0: float, C0: float, tc1_r: float, tc1_c: float,
                   T, T0: float = 25.0):
    """RC 截止频率 f0(T)=1/[2π R(T) C(T)]。"""
    r = resistor_at_T(R0, tc1_r, T, T0)
    c = capacitor_at_T(C0, tc1_c, T, T0)
    return 1.0 / (2.0 * np.pi * r * c)


def cutoff_drift_ratio(tc1_r: float, tc1_c: float, dT: float) -> float:
    """一阶相对漂移 df0/f0 = −(tc1_r + tc1_c)·ΔT。"""
    return -float(tc1_r + tc1_c) * float(dT)


def cutoff_drift_exact(R0: float, C0: float, tc1_r: float, tc1_c: float,
                       dT: float, T0: float = 25.0) -> float:
    """精确相对漂移 f0(T0+ΔT)/f0(T0) − 1。"""
    f0 = 1.0 / (2.0 * np.pi * R0 * C0)
    fT = rc_cutoff_at_T(R0, C0, tc1_r, tc1_c, T0 + dT, T0)
    return float(fT / f0 - 1.0)


def gain_ratio_drift(tc1_rf: float, tc1_r1: float, dT: float) -> float:
    """反相增益 G=Rf/R1 的相对漂移 = (tc1_rf−tc1_r1)·ΔT/(1+tc1_r1·ΔT)。

    两电阻同 TC 时精确为 0（比值完全抵消）。
    """
    return float((tc1_rf - tc1_r1) * dT / (1.0 + tc1_r1 * dT))


def vos_at_T(Vos0: float, tc_vos: float, T, T0: float = 25.0):
    """运放输入失调电压温漂 Vos(T)=Vos0+tc_vos·(T−T0)。"""
    return float(Vos0) + float(tc_vos) * (np.asarray(T, dtype=float) - T0)


def output_offset_from_vos(Vos0: float, tc_vos: float, ng: float,
                           T, T0: float = 25.0):
    """失调折算到输出 = NG·Vos(T)。"""
    return float(ng) * vos_at_T(Vos0, tc_vos, T, T0)
