# -*- coding: utf-8 -*-
"""E19 · 电源抑制比 PSRR：噪声增益惩罚 + PSRR0 缩放 + SPICE 对照（v1.3）。

回应报告 §15「运放非理想参数只做了 GBW/SR，未做 PSRR」。三条线：
  1. 低频 PSRR vs NG：Rf 从 1k→40k（NG 2→41），验证 PSRR 随 NG 恶化 20log10(NG)
  2. PSRR0 缩放：开环 PSRR0 加倍，闭环 PSRR 加倍（+6.02 dB）
  3. SPICE 电源耦合宏模型 vs 解析（常数注入 + 有限 A0 闭环抑制）

方法学发现：PSRR 的一阶滚降与开环增益 A(s) 共享主极点（PSRR≈A/k），
不能用独立 RC 低通模拟电源耦合——那会让高频 PSRR「变好」，方向反了。
宏模型改为把 vcc/PSRR0 直接注入输入差，滚降由 A(s) 自然给出。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from common import load_config, save_json, save_fig

from src.theory import psrr as ps
from src.circuits import v13_builders as vb
from src.engine.runner import run_netlist

CFG = load_config()["v13"]["psrr"]
FSTART, FSTOP, PPD = CFG["fstart"], CFG["fstop"], CFG["ac_ppd"]
GBW, A0 = CFG["GBW_Hz"], CFG["A0"]
GM = 1e-3
CC = GM / (2 * np.pi * GBW)
RLAG = A0 / GM
F1 = GBW / A0  # 开环主极点 = PSRR 极点（共享）
RPS, CPS = 1e4, 1.0 / (2 * np.pi * CFG["f_ps_Hz"] * 1e4)  # 仅占位，模型不用


def sim_psrr(R1, Rf, psrr0):
    net = vb.psrr_inverting_netlist(R1, Rf, GM, RLAG, CC, 1e-3,
                                    RPS, CPS, psrr0, FSTART, FSTOP, PPD)
    tr = run_netlist(net).plot("AC Analysis")
    f = np.real(tr["frequency"])
    gain = np.abs(np.asarray(tr["v(out)"]).flatten())
    return f, gain


def main():
    out = {"config": CFG, "A0": A0, "f1_Hz": F1, "Cc_F": CC, "Rlag": RLAG,
           "note": "PSRR 与开环共享主极点；常数电源耦合经 A(s) 闭环抑制"}
    R1 = CFG["R1"]
    p0 = CFG["psrr0_list"][0]
    f_cl = GBW / ps.noise_gain_inverting(R1, CFG["Rf_list_ohm"][0])

    # ---------- 1. NG 扫描 ----------
    print("[E19] 1/3 NG sweep (PSRR0=%.0f)" % p0)
    rows = []
    for rf in CFG["Rf_list_ohm"]:
        ng = ps.noise_gain_inverting(R1, rf)
        f, g = sim_psrr(R1, rf, p0)
        m = (f > 20) & (f < 0.1 * GBW / ng)  # A≫NG 低频区
        g_dc = float(np.mean(g[m]))
        db = -20 * np.log10(g_dc)
        pred_db = 20 * np.log10(p0) - 20 * np.log10(ng)
        rows.append({"Rf": rf, "NG": ng, "psrr_db_sim": db,
                     "psrr_db_analytic": pred_db, "gain_dc": g_dc,
                     "err_db": db - pred_db})
        print(f"  Rf={rf:>7.0f} NG={ng:>5.1f}  PSRR={db:6.2f} dB "
              f"(解析 {pred_db:6.2f}, 误差 {db - pred_db:+.3f})")
    out["ng_sweep"] = rows
    out["ng_penalty"] = [
        {"NG_from": rows[i - 1]["NG"], "NG_to": rows[i]["NG"],
         "delta_db_sim": rows[i]["psrr_db_sim"] - rows[i - 1]["psrr_db_sim"],
         "delta_db_expected": -20 * np.log10(rows[i]["NG"] / rows[i - 1]["NG"])}
        for i in range(1, len(rows))]
    print("  NG 惩罚：")
    for x in out["ng_penalty"]:
        print(f"    NG {x['NG_from']:.0f}→{x['NG_to']:.0f}: "
              f"{x['delta_db_sim']:+.2f} dB (期望 {x['delta_db_expected']:+.2f})")

    # ---------- 2. PSRR0 缩放 ----------
    print("[E19] 2/3 PSRR0 scaling")
    rf = CFG["ng_spice_pair"][0]
    ng = ps.noise_gain_inverting(R1, rf)
    scaling = []
    for p0x in CFG["psrr0_list"]:
        f, g = sim_psrr(R1, rf, p0x)
        m = (f > 20) & (f < 0.1 * GBW / ng)
        db = -20 * np.log10(float(np.mean(g[m])))
        pred = 20 * np.log10(p0x / ng)
        scaling.append({"psrr0": p0x, "psrr_db_sim": db,
                        "psrr_db_analytic": pred, "err_db": db - pred})
        print(f"  PSRR0={p0x:>8.0f}: {db:6.2f} dB (解析 {pred:6.2f})")
    out["psrr0_scaling"] = scaling
    d_psrr0 = scaling[1]["psrr_db_sim"] - scaling[0]["psrr_db_sim"]
    ratio_psrr0 = CFG["psrr0_list"][1] / CFG["psrr0_list"][0]
    expected = 20 * np.log10(ratio_psrr0)
    out["psrr0_scale_factor"] = ratio_psrr0
    out["psrr0_scaling_db"] = d_psrr0
    out["psrr0_scaling_expected_db"] = expected
    print(f"  PSRR0 ×{ratio_psrr0:g} → PSRR {d_psrr0:+.2f} dB (期望 {expected:+.2f})")

    # ---------- 3. 全频段 SPICE vs 解析（含有限 A0） ----------
    print("[E19] 3/3 full-band analytic check")
    f, g = sim_psrr(R1, rf, p0)
    pred = np.abs(ps.psrr_closedloop_ratio_exact(p0, ng, f, F1, A0, F1))
    err_db = 20 * np.log10(g / pred)
    m = (f > 20) & (f < 0.5 * GBW)  # A(s) 单极点适用区
    out["fullband"] = {
        "f": f.tolist(), "gain_sim": g.tolist(), "gain_analytic": pred.tolist(),
        "max_abs_err_db": float(np.max(np.abs(err_db[m]))),
    }
    print(f"  全带最大误差 {out['fullband']['max_abs_err_db']:.3f} dB")

    save_json("v13_psrr.json", out)

    # ---------- 图 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = [r["NG"] for r in rows]
    ax[0].plot(x, [r["psrr_db_sim"] for r in rows], "o-", label="SPICE")
    ax[0].plot(x, [r["psrr_db_analytic"] for r in rows], "s--", label="analytic")
    ax[0].set_xlabel("Noise gain NG = 1 + Rf/R1")
    ax[0].set_ylabel("Low-f PSRR (dB)")
    ax[0].set_title("PSRR degrades with noise gain")
    ax[0].legend()
    ax[1].loglog(f, g, label="SPICE")
    ax[1].loglog(f, pred, "--", label="analytic (finite A0)")
    ax[1].axvspan(F1, GBW, alpha=0.1, color="r")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("V(out)/V(CC)")
    ax[1].set_title("PSRR frequency response")
    ax[1].legend()
    save_fig(fig, "v13_psrr.png")
    print("E19 完成")


if __name__ == "__main__":
    main()
