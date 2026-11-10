"""容差方法学：最坏情况分析（WCA）、解析误差传播、灵敏度排序、良率优化。"""
from __future__ import annotations

from itertools import product

import numpy as np


# ---------- 解析误差传播 ----------
def rc_fc_propagation_std(R: float, C: float, tol: float, dist: str = "uniform") -> float:
    """fc = 1/(2πRC)：δfc/fc = −(δR/R + δC/C)。返回 σ(fc)/fc。

    均匀 ±tol：σ(x̂)=tol/√3；正态 3σ=tol：σ(x̂)=tol/3。
    """
    sx = _single_std(tol, dist)
    return float(np.sqrt(2.0) * sx)


def sk_fc_propagation_std(tol: float, dist: str = "uniform") -> float:
    """fc = 1/(2π√(R1R2C1C2))：δfc/fc = −½Σδᵢ（4 元件）。返回 σ(fc)/fc。"""
    sx = _single_std(tol, dist)
    return float(0.5 * np.sqrt(4.0) * sx)


def sk_q_propagation_std(tol: float, dist: str = "uniform") -> float:
    """Q = ½√(C1/C2)（R1=R2 设计）：δQ/Q = ½(δC1/C1 − δC2/C2)。返回 σ(Q)/Q。"""
    sx = _single_std(tol, dist)
    return float(0.5 * np.sqrt(2.0) * sx)


def _single_std(tol: float, dist: str) -> float:
    if dist == "uniform":
        return tol / np.sqrt(3.0)
    if dist == "normal":
        return tol / 3.0
    raise ValueError(dist)


# ---------- 最坏情况分析 ----------
def sk_fc_corners(R1: float, R2: float, C1: float, C2: float, tol: float) -> dict:
    """4 元件 ±tol 角点（2^4=16）的 fc 分布（解析式，无仿真）。"""
    fc0 = 1.0 / (2 * np.pi * np.sqrt(R1 * R2 * C1 * C2))
    fcs, corners = [], []
    for sgn in product((1 - tol, 1 + tol), repeat=4):
        r1, r2, c1, c2 = R1 * sgn[0], R2 * sgn[1], C1 * sgn[2], C2 * sgn[3]
        fc = 1.0 / (2 * np.pi * np.sqrt(r1 * r2 * c1 * c2))
        fcs.append(fc)
        corners.append(sgn)
    fcs = np.array(fcs)
    return {
        "fc_nominal": fc0,
        "fc_corners": fcs,
        "corner_signs": corners,
        "fc_min": float(fcs.min()),
        "fc_max": float(fcs.max()),
        "fc_worst_dev_pct": float(max(abs(fcs.min() - fc0), abs(fcs.max() - fc0)) / fc0 * 100),
    }


def rc_fc_corners(R: float, C: float, tol: float) -> dict:
    fc0 = 1.0 / (2 * np.pi * R * C)
    fcs = []
    for sgn in product((1 - tol, 1 + tol), repeat=2):
        fcs.append(1.0 / (2 * np.pi * R * sgn[0] * C * sgn[1]))
    fcs = np.array(fcs)
    return {
        "fc_nominal": fc0,
        "fc_corners": fcs,
        "fc_min": float(fcs.min()),
        "fc_max": float(fcs.max()),
        "fc_worst_dev_pct": float(max(abs(fcs.min() - fc0), abs(fcs.max() - fc0)) / fc0 * 100),
    }


# ---------- 灵敏度排序（|H(f_spec)| 对各元件） ----------
def sk_mag_sensitivity(R1: float, R2: float, C1: float, C2: float, f_spec: float) -> dict:
    """数值微分灵敏度 S_x = d|H|/|H| ÷ dx/x 在 f_spec 处（±0.1% 扰动）。"""
    from ..theory.transfer import sk_lowpass_H

    w = 2 * np.pi * f_spec
    base = abs(sk_lowpass_H(R1, R2, C1, C2, w))
    names = ("R1", "R2", "C1", "C2")
    vals = dict(R1=R1, R2=R2, C1=C1, C2=C2)
    sens = {}
    for n in names:
        v = vals.copy()
        v[n] = vals[n] * 1.001
        h1 = abs(sk_lowpass_H(v["R1"], v["R2"], v["C1"], v["C2"], w))
        v[n] = vals[n] * 0.999
        h2 = abs(sk_lowpass_H(v["R1"], v["R2"], v["C1"], v["C2"], w))
        d_rel = (h1 - h2) / (2 * 0.001) / base  # d|H|/|H| ÷ dx/x
        sens[n] = float(d_rel)
    ranked = sorted(sens.items(), key=lambda kv: -abs(kv[1]))
    return {"sensitivity": sens, "ranked": ranked, "f_spec": f_spec}


# ---------- 良率优化（解析抽样，不跑 ngspice） ----------
def sk_yield_analytic_design(R1: float, R2: float, C1: float, C2: float,
                             f_spec: float, h_nominal: float, h_tol_pct: float,
                             tol_map: dict, n_samples: int = 20000,
                             seed: int = 0) -> dict:
    """按各元件独立均匀容差抽样（解析 |H|），统计 |H(f_spec)| 良率。

    tol_map: {"R1": tol, ...}；良率判据：|H − H_nom|/H_nom < h_tol_pct%。
    """
    from ..theory.transfer import sk_lowpass_H

    rng = np.random.default_rng(seed)
    w = 2 * np.pi * f_spec
    names = ("R1", "R2", "C1", "C2")
    base = dict(R1=R1, R2=R2, C1=C1, C2=C2)
    h = np.empty(n_samples)
    for i in range(n_samples):
        v = {n: base[n] * (1.0 + rng.uniform(-tol_map.get(n, 0.0), tol_map.get(n, 0.0)))
             for n in names}
        h[i] = abs(sk_lowpass_H(v["R1"], v["R2"], v["C1"], v["C2"], w))
    rel = np.abs(h - h_nominal) / h_nominal
    ok = rel < h_tol_pct / 100.0
    return {
        "yield": float(ok.mean()),
        "h_mean": float(h.mean()),
        "h_std": float(h.std()),
        "h_nominal": h_nominal,
        "h_tol_pct": h_tol_pct,
        "tol_map": tol_map,
        "n_samples": n_samples,
    }
