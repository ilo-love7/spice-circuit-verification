"""电源抑制比 PSRR 理论与宏模型电源耦合（v1.3）。

电源纹波 Vcc 经有限 PSRR 折成输入等效失调 v_os = Vcc / PSRR_ol，
再被噪声增益 NG 放大到输出。开环 PSRR 一阶滚降（数据手册常见近似）：

    PSRR_ol(s) = PSRR0 / (1 + s/wp)          wp = 2π f_ps

反相放大器闭环（A(s) ≫ NG 时）：

    Vout/Vcc = NG / [PSRR0 · (1 + s/wp)]
    PSRR_cl(dB) = PSRR0(dB) − 20·log10(NG) − 20·log10|1 + j·f/f_ps|

两条可证伪判据：
  1. 噪声增益 NG = 1 + Rf/R1 越大，闭环 PSRR 越差（反直觉：信号增益与电源抑制反向）。
  2. f > f_ps 后以 −20 dB/dec 滚降，斜率与 NG 无关。

宏模型实现：Rps-Cps 低通产生 eos = Vcc/(1+s/wp)，由 B 源以 gm·eos/PSRR0
注入输入差，SPICE 与解析逐点对照。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "noise_gain_inverting", "psrr_openloop_ratio", "psrr_openloop_db",
    "psrr_closedloop_db", "psrr_closedloop_ratio", "psrr_closedloop_ratio_exact",
    "psrr_rc_for_fps", "psrr_slope_db_per_dec",
]


def noise_gain_inverting(R1: float, Rf: float) -> float:
    """反相放大器噪声增益 NG = 1 + Rf/R1（＝同相端看到的闭环增益）。"""
    return 1.0 + float(Rf) / float(R1)


def psrr_openloop_ratio(psrr0: float, f, f_ps: float):
    """开环 PSRR（V/V），一阶滚降 PSRR0/(1 + j f/f_ps)。"""
    fr = np.asarray(f, dtype=float) / float(f_ps)
    return float(psrr0) / (1.0 + 1j * fr)


def psrr_openloop_db(psrr0: float, f, f_ps: float):
    """开环 PSRR 分贝值 = 20 log10|PSRR_ol|。"""
    return 20.0 * np.log10(np.abs(psrr_openloop_ratio(psrr0, f, f_ps)))


def psrr_closedloop_ratio(psrr0: float, ng: float, f, f_ps: float):
    """闭环电源纹波增益 Vout/Vcc = NG / PSRR_ol。"""
    return float(ng) / psrr_openloop_ratio(psrr0, f, f_ps)


def psrr_closedloop_ratio_exact(psrr0: float, ng: float, f, f_ps: float,
                                A0: float, f_cl_pole: float):
    """宏模型常数耦合下的精确闭环电源纹波增益（对应 SPICE 直接注入 V(vcc)）。

    电源纹波经固定系数 1/PSRR0 折成输入等效失调，再经闭环抑制 A/(A+NG)：

        Vout/Vcc = NG · A(s)/[A(s)+NG] / PSRR0,   A(s)=A0/(1+j f/f1)

    f ≪ 闭环带宽时 A≫NG → Vout/Vcc ≈ NG/PSRR0（低频 PSRR=PSRR0/NG）。
    这是 SPICE 常数注入宏模型的严格解析式，用于逐点对照。
    f_ps 参数保留仅为接口一致（该模型下 PSRR 滚降由 A(s) 而非独立 RC 决定）。
    """
    f = np.asarray(f, dtype=float)
    A = float(A0) / (1.0 + 1j * f / float(f_cl_pole))
    return float(ng) * A / ((A + float(ng)) * float(psrr0))


def psrr_closedloop_db(psrr0: float, ng: float, f, f_ps: float):
    """闭环 PSRR(dB) = 20log10(PSRR0) − 20log10(NG) − 20log10|1+j f/f_ps|。"""
    return 20.0 * np.log10(float(psrr0)) - 20.0 * np.log10(float(ng)) \
        - 20.0 * np.log10(np.abs(1.0 + 1j * np.asarray(f, dtype=float) / float(f_ps)))


def psrr_slope_db_per_dec(f, f_ps: float) -> float:
    """高频段 PSRR 渐近斜率（dB/dec），应恒为 −20。"""
    f = np.asarray(f, dtype=float)
    hi = f > 20.0 * f_ps
    if not np.any(hi):
        return float("nan")
    ff = f[hi]
    y = -20.0 * np.log10(np.abs(1.0 + 1j * ff / float(f_ps)))
    return float((y[-1] - y[0]) / (np.log10(ff[-1]) - np.log10(ff[0])))


def psrr_rc_for_fps(f_ps: float, r_ps: float = 1e4) -> tuple[float, float]:
    """给定 PSRR 极点频率与电阻，返回 (Rps, Cps)。"""
    c_ps = 1.0 / (2.0 * np.pi * float(f_ps) * float(r_ps))
    return float(r_ps), float(c_ps)
