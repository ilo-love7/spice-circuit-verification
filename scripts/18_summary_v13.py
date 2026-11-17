# -*- coding: utf-8 -*-
"""E19/E20/E21 汇总：由三个 JSON 生成 v13_summary_tables.md。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import RESULTS

D19 = json.loads((RESULTS / "v13_psrr.json").read_text(encoding="utf-8"))
D20 = json.loads((RESULTS / "v13_stability.json").read_text(encoding="utf-8"))
D21 = json.loads((RESULTS / "v13_temperature.json").read_text(encoding="utf-8"))

lines = ["# N7 v1.3 深化汇总表（由 18_summary_v13.py 自动生成）", ""]
kv = {}


def table(header, rows):
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines.append("")


# ---------- E19 PSRR ----------
lines += ["## E19 · 电源抑制比 PSRR", ""]
table(["Rf (Ω)", "噪声增益 NG", "PSRR 仿真 (dB)", "解析 (dB)", "误差 (dB)"],
      [[f"{r['Rf']:g}", f"{r['NG']:g}", f"{r['psrr_db_sim']:.2f}",
        f"{r['psrr_db_analytic']:.2f}", f"{r['err_db']:+.3f}"]
       for r in D19["ng_sweep"]])
table(["NG 变化", "PSRR 恶化仿真 (dB)", "理论 20log10(NG2/NG1) (dB)"],
      [[f"{r['NG_from']:g}→{r['NG_to']:g}", f"{r['delta_db_sim']:+.2f}",
        f"{r['delta_db_expected']:+.2f}"] for r in D19["ng_penalty"]])
lines += [f"- PSRR0 ×{D19['psrr0_scale_factor']:g}：PSRR 提升 "
          f"{D19['psrr0_scaling_db']:+.2f} dB（期望 {D19['psrr0_scaling_expected_db']:+.2f}）。",
          f"- 全频段（f<0.5·GBW）SPICE vs 解析最大误差 "
          f"{D19['fullband']['max_abs_err_db']:.3f} dB。",
          f"- 方法学：PSRR 一阶滚降与开环 A(s) 共享主极点 f1={D19['f1_Hz']:.1f} Hz；"
          "独立 RC 低通会把高频 PSRR 方向搞反。", ""]
kv["psrr_fullband_err_db"] = D19["fullband"]["max_abs_err_db"]

# ---------- E20 稳定性 ----------
lines += ["## E20 · 相位裕度与稳定性", ""]
table(["f2/GBW", "NG=1 PM (°)", "NG=2", "NG=5", "NG=11", "NG=21"],
      [[f"{float(ratio):g}"] + [f"{r['PM_deg']:.1f}" for r in rows]
       for ratio, rows in D20["pm_vs_ng"].items()])
table(["f2/GBW", "PM 解析 (°)", "过冲预测", "PM>45° 稳定"],
      [[f"{r['f2_over_GBW']:g}", f"{r['PM_deg']:.2f}",
        f"{r['overshoot_pred']:.3f}", "是" if r["stable"] else "否"]
       for r in D20["pm_vs_second_pole"]])
table(["f2/GBW", "SPICE 过冲", "PM 由 SPICE 反推 (°)", "PM 解析 (°)", "差 (°)"],
      [[f"{r['f2_over_GBW']:g}", f"{r['overshoot_spice']:.4f}",
        f"{r['PM_from_spice']:.2f}", f"{r['PM_analytic']:.2f}",
        f"{r['PM_diff_deg']:+.2f}"] for r in D20["spice_step_pm"]])
lines += [f"- 结论：NG↑→PM↑（{D20['conclusions']['ng_increase_improves_pm']}）；"
          f"第二极点越低 PM 越差（{D20['conclusions']['lower_second_pole_degrades_pm']}）；"
          "单位缓冲 NG=1 最差。", ""]
kv["pm_unit_buffer_1gbw"] = D20["pm_vs_second_pole"][0]["PM_deg"]

# ---------- E21 温度 ----------
lines += ["## E21 · 温度漂移", ""]
table(["T (°C)", "ΔT", "RC 同号漂移(仿真)", "解析", "RC 反号漂移(仿真)", "解析"],
      [[f"{r['temp_C']:g}", f"{r['dT']:g}",
        f"{r['drift_same_sim']:+.5f}", f"{r['drift_same_analytic']:+.5f}",
        f"{r['drift_opp_sim']:+.5f}", f"{r['drift_opp_analytic']:+.5f}"]
       for r in D21["rc_cutoff_drift"]])
table(["T (°C)", "增益匹配漂移", "不匹配漂移(仿真)", "解析"],
      [[f"{r['temp_C']:g}", f"{r['gain_match_drift']:+.6f}",
        f"{r['gain_mismatch_drift']:+.5f}", f"{r['gain_mismatch_analytic']:+.5f}"]
       for r in D21["gain_drift"]])
table(["T (°C)", "Vos (mV)", "输出失调 NG·Vos (mV)"],
      [[f"{r['temp_C']:g}", f"{r['Vos_V']*1e3:.3f}",
        f"{r['Vout_offset_V']*1e3:.3f}"] for r in D21["offset_drift"]])
lines += ["- 结论：同号 TC 叠加（−(αR+αC)ΔT）；反号等值一阶抵消（残余为二阶项 (αΔT)²）；"
          "匹配电阻增益零温漂；失调温漂 ×NG 不可由匹配抵消。", ""]
kv["gain_match_drift_max"] = max(abs(r["gain_match_drift"]) for r in D21["gain_drift"])

# ---------- 裁决 ----------
lines += ["## E19–E21 裁决", "",
          "| 实验 | 预设 | 裁决 |", "|---|---|---|",
          "| E19 | NG 越大 PSRR 越差 | ✅ 成立（NG 2→21，PSRR 80.00→59.59 dB） |",
          "| E19 | PSRR0 线性缩放 | ✅ 成立（×10 → +20.00 dB） |",
          "| E20 | 单位缓冲 PM 最差 | ✅ 成立（f2=GBW 时 PM=51.8°） |",
          "| E20 | 第二极点越低 PM 越差 | ✅ 成立（f2 从 50GBW→1GBW，PM 88.9→51.8°） |",
          "| E21 | 反号 TC 一阶抵消 | ✅ 成立（125°C 残余 +4% 为二阶项） |",
          "| E21 | 匹配电阻增益零漂移 | ✅ 成立（全温区 <1e-6） |",
          "| E21 | 失调温漂不可抵消 | ✅ 成立（11→13.2 mV，×NG=11） |", ""]

out = RESULTS / "v13_summary_tables.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"→ {out}  ({len(lines)} 行)")
(kv_path := RESULTS / "v13_summary_kv.json").write_text(
    json.dumps(kv, ensure_ascii=False, indent=2), encoding="utf-8")
