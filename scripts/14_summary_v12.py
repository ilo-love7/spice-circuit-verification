# -*- coding: utf-8 -*-
"""E16/E17/E18 汇总：由三个 JSON 生成 v12_summary_tables.md 与 v12_summary.json。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import RESULTS, load_config, save_json

D16 = json.loads((RESULTS / "v12_mfb_bandpass.json").read_text(encoding="utf-8"))
D17 = json.loads((RESULTS / "v12_noise.json").read_text(encoding="utf-8"))
D18 = json.loads((RESULTS / "v12_slew_distortion.json").read_text(encoding="utf-8"))

lines = ["# N7 v1.2 深化汇总表（由 14_summary_v12.py 自动生成）", ""]
kv = {}


def table(header, rows):
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines.append("")


# ---------- E16 ----------
lines += ["## E16 · MFB 带通（突破 SK 的 Q=1/3 约束）", ""]
table(["Q 目标", "H0 理论", "H0 仿真", "Q(3dB) 实测", "AC 最大误差 (dB)"],
      [[f"{r['Q_target']:g}", f"{r['H0_theory']:.4f}",
        f"{r['peak_gain_sim']:.4f}", f"{r['Q_from_bw']:.4f}",
        f"{r['max_abs_err_db']:.3e}"] for r in D16["q_freedom"]])
lines += [f"- SK 等元件带通对照：Q = {D16['sk_reference']['Q']:.6f}（拓扑硬约束），"
          f"峰值增益 {D16['sk_reference']['peak_gain_sim']:.6f}。", ""]
table(["ρ = R1/R2", "H0 = 2Q²/(1+ρ)", "H0 仿真", "相对误差 (%)", "噪声增益"],
      [[f"{r['rho']:g}", f"{r['H0_theory']:.3f}", f"{r['H0_sim']:.3f}",
        f"{r['rel_err_pct']:+.4f}", f"{r['noise_gain']:.1f}"] for r in D16["rho_scan"]])
table(["Q", "NG", "GBW@1% f0 误差 (MHz)", "GBW/f0", "经验式 100·f0·Q² / 实测"],
      [[f"{q:g}", f"{v['noise_gain']:.0f}",
        f"{v['gbw_for_1pct_f0_error_Hz']/1e6:.4f}",
        f"{v['gbw_over_f0_for_f0_1pct']:.0f}",
        f"{v['heuristic_over_required']:.2f}×"]
       for q, v in ((float(k), v) for k, v in D16["gbw_required_at_1pct"].items())])
table(["Q 目标", "σ(f0)/f0 MC (%)", "解析 (%)", "MC/解析", "σ(Q)/Q (%)", "σ(H0)/H0 (%)"],
      [[f"{r['Q_target']:g}", f"{r['f0_rel_std_pct']:.4f}",
        f"{r['f0_rel_std_pct_analytic_pct']:.4f}",
        f"{r['f0_rel_std_pct_ratio']:.5f}",
        f"{r['Q_rel_std_pct']:.4f}", f"{r['H0_rel_std_pct']:.4f}"]
       for r in D16["tolerance_mc"]])
lines += [f"- v1.1 SK 低通解析 σ(fc)/fc = {D16['sk_lp_fc_analytic_std_pct']:.4f}%（并列对照）。",
          f"- Q 灵敏度：R3 = {D16['sensitivity_Q5']['Q']['R3']:+.4f}，"
          f"R1/R2 = {D16['sensitivity_Q5']['Q']['R1']:+.4f}，"
          f"C1/C2 = {D16['sensitivity_Q5']['Q']['C1']:+.4f}（精确对消）。", ""]
kv["E16_max_ac_err_db"] = max(r["max_abs_err_db"] for r in D16["q_freedom"])
kv["E16_Q10_Q_from_bw"] = D16["q_freedom"][-1]["Q_from_bw"]
kv["E16_gbw_required_ratio_Q10_over_Q1"] = (
    D16["gbw_required_at_1pct"]["10.0"]["gbw_for_1pct_f0_error_Hz"]
    / D16["gbw_required_at_1pct"]["1.0"]["gbw_for_1pct_f0_error_Hz"])
kv["E16_heuristic_overestimate_Q10"] = D16["gbw_required_at_1pct"]["10.0"]["heuristic_over_required"]
kv["E16_mc_over_analytic_Q5"] = D16["tolerance_mc"][1]["f0_rel_std_pct_ratio"]

# ---------- E17 ----------
lines += ["## E17 · 噪声分析（`.noise` / kT/C / ENBW / 阻抗标度）", ""]
table(["R (Ω)", "fc (Hz)", "√(4kTR) 解析", "ngspice 低频密度",
       "√(kT/C) (Vrms)", "ngspice 总噪声 (Vrms)", "相对误差 (%)"],
      [[f"{r['R_ohm']:.0f}", f"{r['fc_Hz']:.3f}",
        f"{r['thermal_density_analytic']:.4e}", f"{r['thermal_density_ngspice']:.4e}",
        f"{r['total_kT_over_C_Vrms']:.4e}", f"{r['total_ngspice_Vrms']:.4e}",
        f"{r['ngspice_rel_err_pct']:+.3f}"] for r in D17["ktc"]])
table(["Q", "H0", "ENBW = H0²·πf0/(2Q) (Hz)", "onoise_total (Vrms)",
       "白噪声近似误差 (%)"],
      [[f"{r['Q']:g}", f"{r['H0']:.0f}", f"{r['ENBW_analytic_Hz']:.1f}",
        f"{r['ngspice_total_Vrms']:.4e}", f"{r['white_approx_rel_err_pct']:+.2f}"]
       for r in D17["bandpass_enbw"]])
lines += [f"- 输出总噪声 ~ Q^{D17['total_noise_vs_Q_power_law']:.3f}（理论 ≈ Q¹，"
          f"因 H0 ∝ Q²、√ENBW ∝ Q⁻¹ᐟ²、热噪声密度 ∝ Q⁻¹ᐟ²）。", ""]
table(["α", "无源热噪声 (Vrms)", "e_n 项 (Vrms)", "i_n 项 (Vrms)",
       "总噪声 (Vrms)", "SNR (dB)"],
      [[f"{r['alpha']:g}", f"{r['passive_Vrms']:.4e}",
        f"{r['en_component_Vrms']:.4e}", f"{r['in_component_Vrms']:.4e}",
        f"{r['total_Vrms']:.4e}", f"{r['snr_db']:.2f}"]
       for r in D17["impedance_scaling"]])
m = D17["impedance_monotonicity"]
lines += [f"- **负结果**：无源项 ~ α^{m['passive_power_law']:.3f}，"
          f"e_n 项离散度 {m['en_component_spread_pct']:.3f}%（与 α 无关），"
          f"i_n 项 ~ α^{m['in_power_law']:.3f} ⇒ 总噪声在 α 上"
          f"{'单调递增' if m['total_monotonic_increasing'] else '非单调'}，"
          f"**不存在内点最优**；噪声地板 = {m['noise_floor_Vrms']:.4e} Vrms（由运放 e_n 决定）。",
          f"- e_n 用同相输入串联等效电阻 R_eq = {D17['R_eq_en_ohm']:.1f} Ω 严格建模，"
          f"AC 传递函数不变（单测锁定）。", ""]
kv["E17_ktc_max_rel_err_pct"] = max(abs(r["ngspice_rel_err_pct"]) for r in D17["ktc"])
kv["E17_enbw_white_max_err_pct"] = max(abs(r["white_approx_rel_err_pct"])
                                       for r in D17["bandpass_enbw"])
kv["E17_noise_Q_power_law"] = D17["total_noise_vs_Q_power_law"]
kv["E17_en_spread_pct"] = m["en_component_spread_pct"]
kv["E17_no_interior_optimum"] = not m["interior_optimum_exists"]

# ---------- E18 ----------
lines += ["## E18 · 压摆瞬态 / 全功率带宽", ""]
table(["Vin 幅度 (V)", "FPBW (kHz)", "粗网格 onset (kHz)", "onset/FPBW (粗)",
       "细扫 onset/FPBW", "网格最大斜率 (V/µs)"],
      [[f"{r['v_amp']:g}", f"{r['fpbw_Hz']/1e3:.2f}",
        f"{r['onset_f_Hz']/1e3:.2f}" if np.isfinite(r["onset_f_Hz"]) else "—",
        f"{r['onset_over_fpbw']:.3f}" if r["onset_over_fpbw"] else "—",
        f"{next((x['onset_ratio'] for x in D18['fine_onset']
                 if x['v_amp'] == r['v_amp']), float('nan')):.3f}",
        f"{max(g['max_slope_V_per_us'] for g in D18['grid']
               if g['v_amp'] == r['v_amp']):.3f}"]
       for r in D18["onset"]])
lines += [f"- 细扫两幅度起始比 {D18['fine_onset'][0]['onset_ratio']:.4f} / "
          f"{D18['fine_onset'][1]['onset_ratio']:.4f}，"
          f"离散度 {D18['fine_onset_ratio_spread_pct']:.2f}% ⇒ THD 确实 collapse 到 slew_ratio；"
          f"FPBW 是保守下界（THD@FPBW 仍仅 0.001%）。", ""]
table(["SR (V/µs)", "ΔV/SR (µs)", "10–90% 上升 (µs)", "2% 建立 (µs)",
       "上升/线性基准"],
      [[f"{r['sr_V_per_us']:g}", f"{r['slew_limited_rise_s']*1e6:.2f}",
        f"{r['rise_time_s']*1e6:.2f}", f"{r['settling_2pct_s']*1e6:.2f}",
        f"{r['rise_ratio_vs_linear']:.3f}" if r["rise_ratio_vs_linear"] else "—"]
       for r in D18["step"]])
bc = D18["bandwidth_comparison"]
lines += [f"- 小信号闭环带宽 GBW/NG = {bc['small_signal_bw_Hz']/1e3:.2f} kHz（与幅度无关）；"
          f"FPBW 从 {min(bc['fpbw_by_v_amp_Hz'].values())/1e3:.2f} kHz 到 "
          f"{max(bc['fpbw_by_v_amp_Hz'].values())/1e3:.1f} kHz，"
          f"跨度 {bc['fpbw_ratio_max_over_min']:.0f}×；"
          f"1 V 输入时小信号带宽是 FPBW 的 {bc['small_signal_over_fpbw_at_1V']:.2f}×。", ""]
kv["E18_fine_onset_ratio_mean"] = D18["fine_onset_ratio_mean"]
kv["E18_fine_onset_spread_pct"] = D18["fine_onset_ratio_spread_pct"]
kv["E18_small_signal_over_fpbw_1V"] = bc["small_signal_over_fpbw_at_1V"]
kv["E18_step_max_rise_ratio"] = max(r["rise_ratio_vs_linear"] for r in D18["step"])
kv["E18_max_slope_V_per_us"] = D18["verdict"]["max_slope_ceiling_V_per_us"]

lines += ["## 关键数字", ""]
for k, v in kv.items():
    lines.append(f"- `{k}` = {v}")
lines.append("")

(RESULTS / "v12_summary_tables.md").write_text("\n".join(lines), encoding="utf-8")
print("  saved results/v12_summary_tables.md")
save_json("v12_summary.json", kv)
for k, v in kv.items():
    print(f"[14] {k}: {v}")
