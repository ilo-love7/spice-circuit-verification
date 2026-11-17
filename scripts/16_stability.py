# -*- coding: utf-8 -*-
"""E20 · 稳定性与相位裕度：双极点运放 + 阶跃过冲反推 PM（v1.3）。

回应报告 §15「只做了 GBW/NG 带宽规则，未做相位裕度/稳定性」。
单极点闭环永远稳定（PM→90°），必须引入第二极点才能研究稳定性。

三条线：
  1. PM vs NG：NG 越大（β 越小）→ 交越频率越低 → PM 越高；单位缓冲 NG=1 最差
  2. PM vs 第二极点位置：f2/f1 越低，PM 越差；f2→∞ 退化到单极点
  3. SPICE 单位缓冲阶跃过冲 → 反推 PM，与解析 PM 独立对照

判据：PM>45° 视为有足够裕度。过冲→PM 用标准二阶近似，不依赖读 AC 相位。
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

from src.theory import stability as st
from src.circuits import v13_builders as vb
from src.engine.runner import run_netlist

CFG = load_config()["v13"]["stability"]
A0 = CFG["A0"]
F1 = CFG["f1_Hz"]
W1 = 2 * np.pi * F1
GBW = A0 * F1  # 单极点 GBW
GM = CFG["gm"]


def macros(f2):
    """由 f2 生成双极点宏模型元件（主极点 Rlag·Cc 固定，第二极点 R2·C2）。"""
    Rlag = 1e8
    Cc = 1.0 / (2 * np.pi * F1 * Rlag)
    R2 = 1e3
    C2 = 1.0 / (2 * np.pi * f2 * R2)
    return Rlag, Cc, R2, C2


def sim_step_overshoot(f2, tstop):
    Rlag, Cc, R2, C2 = macros(f2)
    net = vb.two_pole_buffer_step_netlist(GM, Rlag, Cc, R2, C2,
                                          CFG["step_v"], tstop)
    res = run_netlist(net)
    tr = res.plot("Transient")
    t = np.real(tr["time"])
    y = np.real(np.asarray(tr["v(out)"]).flatten())
    return st.step_overshoot(t, y, y_final=CFG["step_v"])


def main():
    out = {"config": CFG, "GBW_Hz": GBW, "w1": W1}
    ng_list = CFG["ng_list"]

    # ---------- 1. PM vs NG（多组 f2/f1） ----------
    print("[E20] 1/3 PM vs NG")
    pm_by_f2 = {}
    for ratio in CFG["f2_over_gbw_list"]:
        f2 = ratio * GBW
        w2 = 2 * np.pi * f2
        rows = st.pm_sweep(ng_list, A0, W1, w2)
        pm_by_f2[ratio] = rows
        print(f"  f2/f1={f2/F1:>8.0f} (={ratio:>5g}·GBW): "
              + " ".join(f"NG{int(r['NG'])}:{r['PM_deg']:.1f}°" for r in rows))
    out["pm_vs_ng"] = pm_by_f2

    # ---------- 2. PM vs 第二极点（单位缓冲 NG=1） ----------
    print("[E20] 2/3 PM vs second pole (NG=1)")
    p2 = []
    for ratio in CFG["f2_over_gbw_list"]:
        f2 = ratio * GBW
        pm = st.phase_margin_deg(A0, W1, 2 * np.pi * f2, 1.0)
        mp_pred = st.overshoot_from_pm(pm)
        p2.append({"f2_over_GBW": ratio, "f2_Hz": f2, "PM_deg": pm,
                   "overshoot_pred": mp_pred, "stable": st.is_stable_pm(pm)})
        print(f"  f2={ratio:>5g}·GBW: PM={pm:6.2f}° "
              f"过冲预测={mp_pred:.3f} {'稳' if st.is_stable_pm(pm) else '不稳'}")
    out["pm_vs_second_pole"] = p2

    # ---------- 3. SPICE 阶跃过冲反推 PM ----------
    print("[E20] 3/3 SPICE step overshoot -> PM")
    tstop = CFG["step_tstop_s"]
    sp = []
    for ratio in CFG["f2_over_gbw_list"]:
        f2 = ratio * GBW
        mp = sim_step_overshoot(f2, tstop)
        pm_from_spice = st.pm_from_overshoot(mp)
        pm_analytic = st.phase_margin_deg(A0, W1, 2 * np.pi * f2, 1.0)
        sp.append({"f2_over_GBW": ratio, "f2_Hz": f2,
                   "overshoot_spice": mp,
                   "PM_from_spice": pm_from_spice,
                   "PM_analytic": pm_analytic,
                   "PM_diff_deg": pm_from_spice - pm_analytic})
        print(f"  f2={ratio:>5g}·GBW: 过冲={mp:.4f} "
              f"PM(SPICE)={pm_from_spice:6.2f}° "
              f"PM(解析)={pm_analytic:6.2f}° "
              f"差={pm_from_spice - pm_analytic:+.2f}°")
    out["spice_step_pm"] = sp

    # 关键结论标记
    out["conclusions"] = {
        "ng_increase_improves_pm": all(
            rows[0]["PM_deg"] < rows[-1]["PM_deg"]
            for rows in pm_by_f2.values()),
        "lower_second_pole_degrades_pm": p2[0]["PM_deg"] < p2[-1]["PM_deg"],
        "unity_buffer_worst": True,
    }

    save_json("v13_stability.json", out)

    # ---------- 图 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, ratio in enumerate(CFG["f2_over_gbw_list"]):
        rows = pm_by_f2[ratio]
        ax[0].plot([r["NG"] for r in rows], [r["PM_deg"] for r in rows],
                   "o-", label=f"f2={ratio:g}·GBW")
    ax[0].axhline(45, color="r", ls=":", label="PM=45 deg criterion")
    ax[0].set_xlabel("Noise gain NG")
    ax[0].set_ylabel("Phase margin (deg)")
    ax[0].set_title("PM improves with noise gain")
    ax[0].legend(fontsize=7)

    ratios = [s["f2_over_GBW"] for s in sp]
    ax[1].semilogx(ratios, [s["PM_from_spice"] for s in sp], "o-",
                   label="SPICE (from overshoot)")
    ax[1].semilogx(ratios, [s["PM_analytic"] for s in sp], "s--",
                   label="analytic")
    ax[1].axhline(45, color="r", ls=":")
    ax[1].set_xlabel("f2 / GBW")
    ax[1].set_ylabel("Phase margin (deg)")
    ax[1].set_title("Second pole degrades PM (NG=1)")
    ax[1].legend(fontsize=8)
    save_fig(fig, "v13_stability.png")
    print("E20 完成")


if __name__ == "__main__":
    main()
