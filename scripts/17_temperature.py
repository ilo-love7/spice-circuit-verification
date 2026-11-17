# -*- coding: utf-8 -*-
"""E21 · 温度漂移：RC 截止温漂 / 匹配电阻增益抵消 / 失调温漂（v1.3）。

回应报告 §15「未做温度漂移/工艺角」。三条线：
  1. RC 低通 fc 温漂：R/C 同号 TC 叠加 −(αR+αC)·ΔT；反号等值一阶抵消
  2. 反相增益温漂：两电阻同 TC 时比值精确抵消（增益零温漂）
  3. 运放失调温漂：Vos(T)=Vos0+TCVos·ΔT，输出 = NG·Vos（匹配救不了）

判据：SPICE .temp 扫描 vs 解析一阶/精确式，标注二阶项边界。
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

from src.theory import temperature as tp
from src.circuits import v13_builders as vb
from src.engine.runner import run_netlist

CFG = load_config()["v13"]["temperature"]
R0, C0 = CFG["R0"], CFG["C0"]
TR = tp.ppm_to_tc1(CFG["tc_r_ppm"])
TC_SAME = tp.ppm_to_tc1(CFG["tc_c_ppm_same"])
TC_OPP = tp.ppm_to_tc1(CFG["tc_c_ppm_opp"])
TEMPS = CFG["temp_list_C"]


def sim_rc_cutoff(tc_c, temp):
    net = vb.rc_temp_netlist(R0, C0, TR, tc_c, temp,
                             CFG["fstart"], CFG["fstop"], CFG["ac_ppd"])
    tr = run_netlist(net).plot("AC Analysis")
    f = np.real(tr["frequency"])
    a = np.abs(np.asarray(tr["v(out)"]).flatten())
    # −3.0103 dB 点（幅值 1/√2），log-f 线性插值
    i = int(np.argmin(np.abs(a - 1 / np.sqrt(2))))
    if 0 < i < len(f) - 1:
        a0, a1 = a[i - 1], a[i + 1]
        if a0 > a1:
            t = (a0 - 1 / np.sqrt(2)) / (a0 - a1)
            return float(f[i - 1] ** (1 - t) * f[i + 1] ** t)
    return float(f[i])


def sim_inv_gain(temp, tc_r1, tc_rf):
    net = vb.inv_gain_temp_netlist(CFG["gain_R1"], CFG["gain_Rf"],
                                   tc_r1, tc_rf, temp,
                                   CFG["fstart"], CFG["fstop"], 20)
    tr = run_netlist(net).plot("AC Analysis")
    a = np.abs(np.asarray(tr["v(out)"]).flatten())
    return float(np.mean(a[:3]))  # 低频增益


def main():
    out = {"config": CFG, "tc_R": TR, "tc_C_same": TC_SAME, "tc_C_opp": TC_OPP,
           "T0_C": 25.0}

    # ---------- 1. RC fc 温漂（同号/反号） ----------
    print("[E21] 1/3 RC cutoff temp drift")
    f0_ref = sim_rc_cutoff(TC_SAME, 25.0)
    rc_rows = []
    for tt in TEMPS:
        dT = tt - 25.0
        fc_same = sim_rc_cutoff(TC_SAME, tt)
        fc_opp = sim_rc_cutoff(TC_OPP, tt)
        pred_same = tp.cutoff_drift_exact(R0, C0, TR, TC_SAME, dT)
        pred_opp = tp.cutoff_drift_exact(R0, C0, TR, TC_OPP, dT)
        row = {"temp_C": tt, "dT": dT,
               "fc_same_sim": fc_same, "drift_same_sim": fc_same / f0_ref - 1,
               "drift_same_analytic": pred_same,
               "fc_opp_sim": fc_opp, "drift_opp_sim": fc_opp / sim_rc_cutoff(TC_OPP, 25.0) - 1,
               "drift_opp_analytic": pred_opp}
        rc_rows.append(row)
        print(f"  T={tt:>5.0f}°C 同号漂移={row['drift_same_sim']:+.5f} "
              f"(解析 {pred_same:+.5f})  反号={row['drift_opp_sim']:+.6f} "
              f"(解析 {pred_opp:+.6f})")
    out["rc_cutoff_drift"] = rc_rows

    # ---------- 2. 反相增益温漂（匹配 vs 不匹配） ----------
    print("[E21] 2/3 inv gain temp drift")
    g_rows = []
    for tt in TEMPS:
        dT = tt - 25.0
        g_match = sim_inv_gain(tt, TR, TR)
        g_mismatch = sim_inv_gain(tt, TR, 2 * TR)
        pred_match = tp.gain_ratio_drift(TR, TR, dT)
        pred_mismatch = tp.gain_ratio_drift(2 * TR, TR, dT)
        g_rows.append({"temp_C": tt, "dT": dT,
                       "gain_match_sim": g_match,
                       "gain_match_drift": g_match / sim_inv_gain(25.0, TR, TR) - 1,
                       "gain_match_analytic": pred_match,
                       "gain_mismatch_drift": g_mismatch / sim_inv_gain(25.0, TR, 2 * TR) - 1,
                       "gain_mismatch_analytic": pred_mismatch})
        print(f"  T={tt:>5.0f}°C 匹配漂移={g_rows[-1]['gain_match_drift']:+.6f} "
              f"不匹配={g_rows[-1]['gain_mismatch_drift']:+.5f} "
              f"(解析 {pred_mismatch:+.5f})")
    out["gain_drift"] = g_rows

    # ---------- 3. 失调温漂（解析，匹配救不了） ----------
    print("[E21] 3/3 offset temp drift")
    vos0 = CFG["vos0"]
    tcvos = tp.ppm_to_tc1(CFG["tc_vos_ppm_per_c"])  # 注：此处 ppm 实为 µV/°C 量纲
    # Vos 温漂系数直接给 µV/°C：2 µV/°C = 2e-6 V/°C
    tcvos_v = CFG["tc_vos_ppm_per_c"] * 1e-6
    ng = CFG["ng_vos"]
    vos_rows = []
    for tt in TEMPS:
        dT = tt - 25.0
        vos = tp.vos_at_T(vos0, tcvos_v, tt)
        vout = tp.output_offset_from_vos(vos0, tcvos_v, ng, tt)
        vos_rows.append({"temp_C": tt, "Vos_V": float(vos),
                         "Vout_offset_V": float(vout), "NG": ng})
        print(f"  T={tt:>5.0f}°C Vos={float(vos)*1e3:.3f} mV "
              f"→ 输出失调={float(vout)*1e3:.3f} mV")
    out["offset_drift"] = vos_rows
    out["conclusions"] = {
        "same_tc_rc_drift_adds": abs(rc_rows[-1]["drift_same_sim"]) > 0.1,
        "opposite_tc_cancels_first_order": abs(rc_rows[-1]["drift_opp_sim"]) < 1e-3,
        "matched_resistors_zero_gain_drift": abs(g_rows[-1]["gain_match_drift"]) < 1e-4,
        "offset_drift_not_canceled_by_matching": abs(vos_rows[-1]["Vout_offset_V"]) > 0,
    }

    save_json("v13_temperature.json", out)

    # ---------- 图 ----------
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    dTs = [r["dT"] for r in rc_rows]
    ax[0].plot(dTs, [r["drift_same_sim"] for r in rc_rows], "o-", label="same sign SPICE")
    ax[0].plot(dTs, [r["drift_same_analytic"] for r in rc_rows], "s--", label="same sign analytic")
    ax[0].plot(dTs, [r["drift_opp_sim"] for r in rc_rows], "^-", label="opposite sign SPICE")
    ax[0].set_xlabel("ΔT (°C)")
    ax[0].set_ylabel("Δfc/fc")
    ax[0].set_title("RC cutoff temp drift")
    ax[0].legend(fontsize=7)

    ax[1].plot(dTs, [r["gain_mismatch_drift"] for r in g_rows], "o-", label="mismatch SPICE")
    ax[1].plot(dTs, [r["gain_mismatch_analytic"] for r in g_rows], "s--", label="mismatch analytic")
    ax[1].plot(dTs, [r["gain_match_drift"] for r in g_rows], "^-", label="matched SPICE")
    ax[1].set_xlabel("ΔT (°C)")
    ax[1].set_ylabel("ΔG/G")
    ax[1].set_title("Gain drift (matching cancels)")
    ax[1].legend(fontsize=7)

    ax[2].plot([r["temp_C"] for r in vos_rows],
               [r["Vout_offset_V"] * 1e3 for r in vos_rows], "o-")
    ax[2].set_xlabel("T (°C)")
    ax[2].set_ylabel("Output offset (mV)")
    ax[2].set_title("Vos drift × NG (no cancel)")
    save_fig(fig, "v13_temperature.png")
    print("E21 完成")


if __name__ == "__main__":
    main()
