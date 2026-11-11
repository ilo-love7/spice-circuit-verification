# -*- coding: utf-8 -*-
"""E16 · MFB 多路反馈带通：突破 Sallen-Key Q=1/3 约束的代价（v1.2）。

五组量化：
  1. Q 自由度：Q=1/3..10 的设计、AC vs 手推解析式、3dB 带宽实测
  2. ρ=R1/R2 解耦峰值增益 H0 = 2Q²/(1+ρ)
  3. 有限 GBW 下高 Q 退化：扫描 6 档 GBW × 3 档 Q，找 1% 误差阈值
  4. ±5% 容差 MC（解析向量化）vs 误差传播解析式，延续 v1.1 §9 方法学
  5. f0/Q/H0 对 5 元件的解析灵敏度排序
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import load_config, save_json, save_fig

from src.theory import mfb
from src.theory.transfer import sk_bandpass_H, sk_bandpass_params
from src.theory.nonideal import design_macromodel
from src.circuits import v12_builders as vb
from src.circuits.builders import sk_bandpass_netlist
from src.mc.wca import sk_fc_propagation_std
from src.engine.runner import run_netlist

CFG = load_config()["v12"]["mfb_bandpass"]
F0 = CFG["f0_Hz"]
C = CFG["C_F"]


def ac_of(netlist: str):
    ac = run_netlist(netlist).plot("AC Analysis")
    f = np.real(ac["frequency"])
    v = np.asarray(ac["v(out)"], dtype=complex)
    return f, v


def bp_3db(f, mag):
    """log-f 线性插值求带通两个半功率点。"""
    peak = int(np.argmax(mag))
    target = mag[peak] / np.sqrt(2.0)
    lo, hi = float("nan"), float("nan")
    for k in range(peak, 0, -1):
        if mag[k - 1] < target <= mag[k]:
            t = (target - mag[k - 1]) / (mag[k] - mag[k - 1])
            lo = float(np.exp(np.log(f[k - 1]) + t * (np.log(f[k]) - np.log(f[k - 1]))))
            break
    for k in range(peak, len(f) - 1):
        if mag[k] >= target > mag[k + 1]:
            t = (target - mag[k]) / (mag[k + 1] - mag[k])
            hi = float(np.exp(np.log(f[k]) + t * (np.log(f[k + 1]) - np.log(f[k]))))
            break
    return lo, hi


def peak_refined(f, mag):
    """log-f 三次样条精化峰值 —— 网格 argmax 会引入 ~0.1% 的伪影与固定峰位偏移。

    v1.1 §7.4 已系统记录过 fc 提取的插值陷阱；带通峰值同理。
    """
    j = int(np.argmax(mag))
    lo, hi = max(1, j - 4), min(len(f) - 2, j + 4)
    sp = CubicSpline(np.log(f[lo:hi + 1]), mag[lo:hi + 1])
    opt = minimize_scalar(lambda x: -sp(x),
                          bounds=(np.log(f[lo]), np.log(f[hi])),
                          method="bounded")
    return float(np.exp(opt.x)), float(-opt.fun)


def threshold_gbw(gbws, devs, thr_pct: float) -> float:
    """|dev| 随 GBW 单调下降，在 log-log 上插值求 |dev|=thr 所需 GBW。"""
    x = np.log(np.asarray(gbws, dtype=float))
    y = np.log(np.maximum(np.abs(np.asarray(devs, dtype=float)), 1e-12))
    if not (y.min() < np.log(thr_pct) < y.max()):
        return float("nan")
    order = np.argsort(y)
    return float(np.exp(np.interp(np.log(thr_pct), y[order], x[order])))


def design_args(Q, rho=1.0):
    d = mfb.design_mfb_bp(F0, Q, C, rho=rho)
    return d, tuple(d[k] for k in mfb.MFB_COMPONENTS)


def main():
    out = {"config": CFG}
    ppd, fstart, fstop = CFG["ac_ppd"], CFG["fstart"], CFG["fstop"]

    # ---------- 1. Q 自由度 ----------
    print("[E16] 1/5 Q freedom sweep")
    curves, rows = {}, []
    for Q in CFG["q_list"]:
        d, a = design_args(Q)
        f, v = ac_of(vb.mfb_bp_netlist(*a, fstart, fstop, ppd))
        mag = np.abs(v)
        H = mfb.mfb_bp_H(*a, 2 * np.pi * f)
        err_db = 20 * np.log10(mag) - 20 * np.log10(np.abs(H))
        lo, hi = bp_3db(f, mag)
        peak_f, peak_gain = peak_refined(f, mag)
        rec = {
            "Q_target": float(Q), "f0_Hz_theory": d["f0_Hz"],
            "H0_theory": d["H0"], "bw3db_Hz_theory": d["bw_3db_Hz"],
            "peak_gain_sim": peak_gain,
            "peak_f_sim_Hz": peak_f,
            "peak_gain_grid_max": float(mag.max()),
            "bw3db_Hz_sim": hi - lo,
            "Q_from_bw": F0 / (hi - lo),
            "max_abs_err_db": float(np.max(np.abs(err_db))),
            "rms_err_db": float(np.sqrt(np.mean(err_db ** 2))),
            "components": {k: d[k] for k in mfb.MFB_COMPONENTS},
        }
        rows.append(rec)
        curves[f"Q={Q:g}"] = (f, mag / d["H0"])
        print(f"   Q={Q:7.3f}  H0={d['H0']:8.3f}  peak={peak_gain:8.3f}  "
              f"Q_bw={rec['Q_from_bw']:7.3f}  err={rec['max_abs_err_db']:.2e} dB")
    out["q_freedom"] = rows

    # SK 带通对照（v1.0 拓扑，Q 恒 1/3）
    fsk, qsk = sk_bandpass_params(1e4, C)
    fs, vs = ac_of(sk_bandpass_netlist(1e4, C, 1.0, fstart, fstop, ppd))
    msk = np.abs(vs)
    out["sk_reference"] = {
        "f0_Hz": fsk, "Q": qsk,
        "peak_gain_sim": float(msk.max()),
        "max_abs_err_db": float(np.max(np.abs(
            20 * np.log10(msk) - 20 * np.log10(np.abs(sk_bandpass_H(1e4, C, 1.0, 2 * np.pi * fs)))))),
    }
    curves["SK BP (Q=1/3)"] = (fs, msk / msk.max())

    # ---------- 2. ρ 解耦增益 ----------
    print("[E16] 2/5 rho / H0 decoupling")
    rho_rows = []
    for rho in CFG["rho_list"]:
        d, a = design_args(CFG["rho_scan_Q"], rho)
        f, v = ac_of(vb.mfb_bp_netlist(*a, fstart, fstop, ppd))
        peak_f, peak_gain = peak_refined(f, np.abs(v))
        rho_rows.append({
            "rho": float(rho), "H0_theory": d["H0"],
            "H0_sim": peak_gain,
            "rel_err_pct": float((peak_gain / d["H0"] - 1) * 100),
            "noise_gain": mfb.mfb_bp_noise_gain(d["H0"]),
        })
    out["rho_scan"] = rho_rows
    print(f"   H0 range {rho_rows[-1]['H0_theory']:.2f} -> "
          f"{rho_rows[0]['H0_theory']:.2f} (max rel err "
          f"{max(abs(r['rel_err_pct']) for r in rho_rows):.4f}%)")

    # ---------- 3. 有限 GBW 退化 ----------
    print("[E16] 3/5 finite-GBW degradation")
    gbw_rows, required = {}, {}
    thr = CFG["gbw_threshold_pct"]
    for Q in CFG["gbw_scan_q_list"]:
        d, a = design_args(Q)
        series = []
        for gbw in CFG["gbw_list_Hz"]:
            p = design_macromodel(gbw, CFG["A0"], CFG["SR_V_per_s"])
            f, v = ac_of(vb.mfb_bp_gbw_netlist(
                *a, p["gm"], p["Rlag"], p["Cc"], p["Imax"], fstart, fstop, ppd))
            peak_f, peak_gain = peak_refined(f, np.abs(v))
            series.append({
                "gbw_Hz": float(gbw),
                "peak_gain": peak_gain,
                "gain_dev_pct": float((peak_gain / d["H0"] - 1) * 100),
                "peak_f_Hz": peak_f,
                "f_dev_pct": float((peak_f / F0 - 1) * 100),
                "gbw_over_f0": float(gbw / F0),
            })
        gbw_rows[str(Q)] = series
        gbws = [s["gbw_Hz"] for s in series]
        req_gain = threshold_gbw(gbws, [s["gain_dev_pct"] for s in series], thr)
        req_f0 = threshold_gbw(gbws, [s["f_dev_pct"] for s in series], thr)
        required[str(Q)] = {
            "gbw_for_1pct_gain_error_Hz": req_gain,
            "gbw_for_1pct_f0_error_Hz": req_f0,
            "gbw_over_f0_for_f0_1pct": req_f0 / F0 if np.isfinite(req_f0) else None,
            "noise_gain": mfb.mfb_bp_noise_gain(d["H0"]),
            "heuristic_100_f0_Q2_Hz": 100.0 * F0 * Q * Q,
            "heuristic_over_required": (100.0 * F0 * Q * Q / req_f0
                                        if np.isfinite(req_f0) and req_f0 > 0 else None),
        }
        print(f"   Q={Q:5.1f}  NG={mfb.mfb_bp_noise_gain(d['H0']):6.1f}  "
              f"GBW@1% f0 = {req_f0/1e6 if np.isfinite(req_f0) else float('nan'):7.3f} MHz "
              f"({req_f0/F0 if np.isfinite(req_f0) else float('nan'):.0f}x f0)  "
              f"heuristic 100*f0*Q^2 / req = "
              f"{required[str(Q)]['heuristic_over_required'] or float('nan'):.1f}x")
    out["gbw_degradation"] = gbw_rows
    out["gbw_required_at_1pct"] = required

    # ---------- 4. 容差 MC vs 解析误差传播 ----------
    print("[E16] 4/5 tolerance MC vs analytic propagation")
    rng = np.random.default_rng(CFG["mc_seed"])
    n, tol = CFG["mc_n"], CFG["mc_tol"]
    mc_rows = []
    for Q in (1.0, 5.0, 10.0):
        d, _ = design_args(Q)
        base = {k: d[k] for k in mfb.MFB_COMPONENTS}
        arr = {k: base[k] * (1.0 + rng.uniform(-tol, tol, n)) for k in base}
        p = mfb.mfb_bp_params_array(**arr)
        s = mfb.mfb_bp_sensitivity(**base)
        sigma = tol / np.sqrt(3.0)
        rec = {"Q_target": float(Q), "n_samples": n, "tol": tol}
        for tag, skey in (("f0", "f0_Hz"), ("Q", "Q"), ("H0", "H0")):
            mc = float(np.std(p[skey]) / d[skey] * 100)
            analytic = float(np.sqrt(np.sum(np.square(list(s[skey].values()))))
                             * sigma * 100)
            rec[f"{tag}_rel_std_pct"] = mc
            rec[f"{tag}_rel_std_pct_analytic_pct"] = analytic
            rec[f"{tag}_rel_std_pct_ratio"] = mc / analytic
        mc_rows.append(rec)
        print(f"   Q={Q:5.1f}  sigma(f0) MC={rec['f0_rel_std_pct']:.4f}% "
              f"analytic={rec['f0_rel_std_pct_analytic_pct']:.4f}% "
              f"ratio={rec['f0_rel_std_pct_ratio']:.5f}")
    out["tolerance_mc"] = mc_rows
    out["sk_lp_fc_analytic_std_pct"] = float(sk_fc_propagation_std(tol) * 100)

    # ---------- 5. 灵敏度 ----------
    d, a = design_args(5.0)
    out["sensitivity_Q5"] = mfb.mfb_bp_sensitivity(*a)
    top_q = sorted(out["sensitivity_Q5"]["Q"].items(), key=lambda kv: -abs(kv[1]))
    print("[E16] 5/5 Q sensitivity ranking:", top_q)

    save_json("v12_mfb_bandpass.json", out)

    # ---------- 图 1：Q 自由度 ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, (f, m) in curves.items():
        ax.semilogx(f, 20 * np.log10(np.maximum(m, 1e-12)),
                    lw=1.8 if "SK" not in label else 2.4,
                    ls="--" if "SK" in label else "-", label=label)
    ax.axvline(F0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized gain (dB)")
    ax.set_title("E16: MFB bandpass breaks the Sallen-Key Q = 1/3 limit")
    ax.set_ylim(-45, 3)
    ax.legend(fontsize=7.5, ncol=2)
    save_fig(fig, "v12_mfb_q_freedom.png")

    # ---------- 图 2：GBW 退化（幅值 / 中心频率双面板） ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for Q, series in gbw_rows.items():
        g = [s["gbw_Hz"] / 1e6 for s in series]
        axes[0].loglog(g, [abs(s["gain_dev_pct"]) for s in series], "o-",
                       lw=1.7, label=f"Q={float(Q):g}")
        axes[1].loglog(g, [abs(s["f_dev_pct"]) for s in series], "o-",
                       lw=1.7, label=f"Q={float(Q):g}")
    for ax, ttl, ylab in zip(axes,
                             ("Peak-gain error", "Center-frequency error"),
                             ("|gain deviation| (%)", "|f0 deviation| (%)")):
        ax.axhline(thr, color="crimson", ls="--", lw=1.2, label="1% limit")
        ax.set_xlabel("Op-amp GBW (MHz)")
        ax.set_ylabel(ylab)
        ax.set_title("E16: " + ttl + " vs finite GBW")
        ax.legend(fontsize=8)
    save_fig(fig, "v12_mfb_gbw_degradation.png")

    # ---------- 图 3：容差 ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    d5, _ = design_args(5.0)
    base5 = {k: d5[k] for k in mfb.MFB_COMPONENTS}
    arr5 = {k: base5[k] * (1.0 + np.random.default_rng(CFG["mc_seed"]).uniform(-tol, tol, n))
            for k in base5}
    p5 = mfb.mfb_bp_params_array(**arr5)
    axes[0].hist((p5["Q"] / d5["Q"] - 1) * 100, bins=60, color="steelblue", alpha=0.85)
    axes[0].set_xlabel("Q deviation (%)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Q=5, +/-{tol*100:.0f}% component tolerance")
    labels = ["f0", "Q", "H0"]
    mc_vals = [mc_rows[1][f"{k}_rel_std_pct"] for k in labels]
    an_vals = [mc_rows[1][f"{k}_rel_std_pct_analytic_pct"] for k in labels]
    x = np.arange(3)
    axes[1].bar(x - 0.18, mc_vals, 0.36, label="Monte Carlo")
    axes[1].bar(x + 0.18, an_vals, 0.36, label="analytic propagation")
    axes[1].axhline(out["sk_lp_fc_analytic_std_pct"], color="darkorange", ls="--",
                    label="v1.1 SK LP fc (2.887%)")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("relative std (%)")
    axes[1].set_title("MC vs analytic error propagation")
    axes[1].legend(fontsize=7.5)
    save_fig(fig, "v12_mfb_tolerance.png")

    print("[E16] done")


if __name__ == "__main__":
    main()
