"""元件容差蒙特卡洛分析。

±5% 元件容差（均匀分布或正态截断），批量 ngspice AC 仿真，
输出关键指标分布与良率曲线。
"""
from __future__ import annotations

import numpy as np

from ..engine.runner import run_netlist
from ..theory.transfer import cutoff_from_response, mag_db


def perturb(value: float, tol: float, rng: np.random.Generator, dist: str = "uniform") -> float:
    """按容差扰动元件值。"""
    if dist == "uniform":
        return value * (1.0 + rng.uniform(-tol, tol))
    elif dist == "normal":
        # 3σ = tol，截断在 ±tol
        v = value * (1.0 + rng.normal(0.0, tol / 3.0))
        lo, hi = value * (1 - tol), value * (1 + tol)
        return float(np.clip(v, lo, hi))
    raise ValueError(dist)


def run_rc_mc(R0: float, C0: float, tol: float, n_trials: int,
              fstart: float, fstop: float, f_spec: float,
              rng: np.random.Generator, dist: str = "uniform",
              points_per_decade: int = 10) -> dict:
    """RC 低通蒙特卡洛：返回 fc 分布、|H(f_spec)| 分布、良率。"""
    from ..circuits.builders import rc_lowpass_netlist

    fc_list, h_list, ok = [], [], 0
    fc_th = 1.0 / (2 * np.pi * R0 * C0)
    h_spec_th = 1.0 / np.sqrt(1.0 + (f_spec / fc_th) ** 2)
    for i in range(n_trials):
        R = perturb(R0, tol, rng, dist)
        C = perturb(C0, tol, rng, dist)
        net = rc_lowpass_netlist(R, C, fstart, fstop, points_per_decade=points_per_decade)
        plot = run_netlist(net).plot("AC Analysis")
        f = np.array(plot["frequency"]).real
        H = np.array(plot["v(out)"])
        mag = np.abs(H)
        fc = cutoff_from_response(f, mag, -3.0)
        # |H| 在 f_spec 处插值（log f）
        h_spec = float(np.exp(np.interp(np.log(f_spec), np.log(f), np.log(np.maximum(mag, 1e-30)))))
        fc_list.append(fc)
        h_list.append(h_spec)
        # 良率判据：|H(f_spec)| 偏差相对理论 < 5%
        if abs(h_spec - h_spec_th) / h_spec_th < 0.05:
            ok += 1
    fc_arr = np.array(fc_list)
    h_arr = np.array(h_list)
    return {
        "fc_Hz": fc_arr,
        "h_spec": h_arr,
        "fc_mean": float(np.nanmean(fc_arr)),
        "fc_std": float(np.nanstd(fc_arr)),
        "fc_theory": fc_th,
        "h_spec_mean": float(np.nanmean(h_arr)),
        "h_spec_std": float(np.nanstd(h_arr)),
        "h_spec_theory": h_spec_th,
        "yield": ok / n_trials,
        "n_trials": n_trials,
        "tol": tol,
        "dist": dist,
        "f_spec": f_spec,
    }


def run_sk_lp_mc(R1_0: float, R2_0: float, C1_0: float, C2_0: float,
                 tol: float, n_trials: int, fstart: float, fstop: float,
                 f_spec: float, rng: np.random.Generator,
                 dist: str = "uniform", points_per_decade: int = 10) -> dict:
    """Sallen-Key 低通蒙特卡洛：fc 与 Q 的分布（由 -3dB 带宽估算 Q）。"""
    from ..circuits.builders import sk_lowpass_netlist

    fc_list, h_list, ok = [], [], 0
    from ..theory.transfer import sk_lowpass_params, sk_lowpass_H
    f0_th, Q_th, _ = sk_lowpass_params(R1_0, R2_0, C1_0, C2_0)
    h_spec_th = float(np.abs(sk_lowpass_H(R1_0, R2_0, C1_0, C2_0, 2 * np.pi * f_spec)))
    for i in range(n_trials):
        R1 = perturb(R1_0, tol, rng, dist)
        R2 = perturb(R2_0, tol, rng, dist)
        C1 = perturb(C1_0, tol, rng, dist)
        C2 = perturb(C2_0, tol, rng, dist)
        net = sk_lowpass_netlist(R1, R2, C1, C2, fstart, fstop,
                                 points_per_decade=points_per_decade)
        plot = run_netlist(net).plot("AC Analysis")
        f = np.array(plot["frequency"]).real
        H = np.array(plot["v(out)"])
        mag = np.abs(H)
        fc = cutoff_from_response(f, mag, -3.0)
        h_spec = float(np.exp(np.interp(np.log(f_spec), np.log(f), np.log(np.maximum(mag, 1e-30)))))
        fc_list.append(fc)
        h_list.append(h_spec)
        # 良率判据：|H(f_spec)| 偏离名义理论 < 15%（f_spec=f0 时名义值≈0.707）
        if abs(h_spec - h_spec_th) / max(h_spec_th, 1e-12) < 0.15:
            ok += 1
    fc_arr = np.array(fc_list)
    h_arr = np.array(h_list)
    return {
        "fc_Hz": fc_arr,
        "h_spec": h_arr,
        "fc_mean": float(np.nanmean(fc_arr)),
        "fc_std": float(np.nanstd(fc_arr)),
        "f0_theory": f0_th,
        "fc_theory": f0_th,  # yield_curve 统一用此键
        "Q_theory": Q_th,
        "h_spec_mean": float(np.nanmean(h_arr)),
        "h_spec_std": float(np.nanstd(h_arr)),
        "h_spec_theory": h_spec_th,
        "yield": ok / n_trials,
        "n_trials": n_trials,
        "tol": tol,
        "dist": dist,
        "f_spec": f_spec,
    }


def yield_curve(fc_arrays: dict, f_grid: np.ndarray, spec_db: float = -3.0) -> dict:
    """幅频特性良率曲线：每个频率点上 |H| 在 ±spec 窗内的比例。

    fc_arrays: {"label": {"fc_Hz": ndarray, ...}} —— 简化版用 fc 分布合成
    理论一阶响应的良率曲线（用于展示容差 → 性能分布的映射）。
    """
    curves = {}
    for label, d in fc_arrays.items():
        fc = d["fc_Hz"]
        fc_th = d.get("fc_theory", d.get("f0_theory"))
        # 一阶低通幅频 |H| = 1/sqrt(1+(f/fc)^2)；良率 = |20log10|H|| 在理论值 ±spec 内
        fr = f_grid[None, :] / fc[:, None]
        mag_db_v = -10 * np.log10(1 + fr**2)
        nominal = -10 * np.log10(1 + (f_grid / fc_th) ** 2)
        within = np.abs(mag_db_v - nominal[None, :]) <= spec_db
        curves[label] = within.mean(axis=0).tolist()
    return {"f_grid": f_grid.tolist(), "curves": curves, "spec_db": spec_db}
