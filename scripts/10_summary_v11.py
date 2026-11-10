"""10 v1.1 深化汇总：三组实验结果 → v11_summary.json + v11_summary_tables.md。"""
from common import save_json, RESULTS, ROOT
import json


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main():
    a = load("v11_nonideal_opamp.json")
    b = load("v11_cascade4.json")
    c = load("v11_wca_sensitivity.json")

    summary = {
        "A_nonideal_opamp": {
            "GBW_Hz": a["macromodel"]["GBW_Hz"],
            "A0": a["macromodel"]["A0"],
            "SR_V_per_us": a["macromodel"]["SR_V_per_s"] / 1e6,
            "ac_max_err_db": a["ac_vs_theory"]["max_abs_err_db"],
            "fc_convention_note": "pole = -3.0103dB half-power; -3.00dB multiplies sqrt(10^0.3-1)=0.99763",
            "fc_classic_vs_exact_pct": a["fc_Hz"]["classic_vs_exact_pct"],
            "slew_err_pct": a["slew"]["err_pct"],
            "small_signal_ratio_to_SR": a["slew"]["small_signal_ratio_to_SR"],
        },
        "B_cascade4": {
            "buffered_vs_theory_db": b["buffered_vs_theory"]["max_abs_err_db"],
            "cascade_vs_scipy_butter_db": b["cascade_vs_scipy_butter"]["max_abs_err_db"],
            "sk_loading_db": b["loading_max_abs_dev_db_0to10fc"],
            "rc_passive_loading_db": b["rc_passive_loading"]["max_abs_dev_db_0to10fc"],
            "group_delay_err_s": b["group_delay"]["max_err_s_100_to_10k"],
            "step_overshoot_pct": b["step_4th"]["overshoot_pct"],
            "step_single_overshoot_pct": b["step_single_Q0707"]["overshoot_pct"],
        },
        "C_wca_sensitivity": {
            "wca_worst_dev_pct": c["wca"]["fc_worst_dev_pct"],
            "mc_worst_sample_dev_pct": c["propagation"]["mc_worst_sample_dev_pct"],
            "analytic_sigma_pct": c["propagation"]["analytic_sigma_fc_pct"],
            "mc_std_pct": c["propagation"]["mc_formula_fc_std_pct"],
            "analytic_over_mc": c["propagation"]["analytic_over_mc_ratio"],
            "sensitivity_f0": c["sensitivity_f0"],
            "yield_full_5pct": c["yield_full_5pct"]["yield"],
            "yield_tighten_C2_1pct": c["yield_tighten"]["C2"]["yield"],
            "yield_tighten_C1_1pct": c["yield_tighten"]["C1"]["yield"],
        },
    }
    save_json("v11_summary.json", summary)

    # markdown tables
    md = []
    md.append("# N7 v1.1 深化汇总表\n")
    md.append("## A 非理想运放（有限 GBW / 压摆率）\n")
    md.append("| 项目 | 值 |")
    md.append("|---|---|")
    md.append(f"| GBW / A0 / SR | {a['macromodel']['GBW_Hz']/1e6:.0f} MHz / {a['macromodel']['A0']:.0e} / {a['macromodel']['SR_V_per_s']/1e6:.0f} V/µs |")
    md.append(f"| AC vs 精确理论 最大误差 | {a['ac_vs_theory']['max_abs_err_db']:.2e} dB |")
    md.append(f"| 经典 GBW/NG 规则偏差 | {a['fc_Hz']['classic_vs_exact_pct']:+.3f}%（=−NG/A0） |")
    md.append(f"| −3.00 dB vs −3.01 dB 判据换算 | √(10^0.3−1) = 0.99763 |")
    md.append(f"| 压摆率实测 vs 规格 | {a['slew']['sim_V_per_s']/1e6:.4f} vs {a['slew']['spec_V_per_s']/1e6:.4f} V/µs（{a['slew']['err_pct']:+.3f}%） |")
    md.append(f"| 小信号最大斜率 / SR | {a['slew']['small_signal_ratio_to_SR']:.3f}（SR 未触发） |")
    md.append("")
    md.append("### fc vs 增益\n")
    md.append("| G | NG | fc 实测 (Hz) | 经典 GBW/NG | 精确极点 |")
    md.append("|---|---|---|---|---|")
    for row in a["fc_by_gain"]:
        md.append(f"| {row['G']:.0f} | {row['G']+1:.0f} | {row['fc_meas']:.4e} | {row['fc_classic']:.4e} | {row['fc_exact']:.4e} |")
    md.append("")
    md.append("## B 4 阶 Butterworth 级联\n")
    md.append("| 项目 | 值 |")
    md.append("|---|---|")
    md.append(f"| 级联理论 vs scipy.butter(4) | {b['cascade_vs_scipy_butter']['max_abs_err_db']:.2e} dB |")
    md.append(f"| 缓冲级联 vs 理论 | {b['buffered_vs_theory']['max_abs_err_db']:.2e} dB（有限 A=1e6 运放效应） |")
    md.append(f"| SK 级联加载效应 | {b['loading_max_abs_dev_db_0to10fc']:.4f} dB（运放输出天然隔离，负结果） |")
    md.append(f"| 无源 RC 级联加载效应 | {b['rc_passive_loading']['max_abs_dev_db_0to10fc']:.4f} dB（对照） |")
    md.append(f"| 群延迟最大误差 (100 Hz–10 kHz) | {b['group_delay']['max_err_s_100_to_10k']:.2e} s |")
    md.append(f"| 阶跃过冲 4 阶 / 单节 Q=0.707 | {b['step_4th']['overshoot_pct']:.2f}% / {b['step_single_Q0707']['overshoot_pct']:.2f}%（教科书值 10.8% / 4.3%） |")
    md.append("")
    md.append("## C WCA / 灵敏度 / 良率\n")
    md.append("| 项目 | 值 |")
    md.append("|---|---|")
    md.append(f"| WCA 16 角点最坏偏差 | {c['wca']['fc_worst_dev_pct']:.3f}% |")
    md.append(f"| MC 最坏样本偏差 | {c['propagation']['mc_worst_sample_dev_pct']:.3f}% |")
    md.append(f"| 解析 σ(fc)/fc | {c['propagation']['analytic_sigma_fc_pct']:.4f}% |")
    md.append(f"| MC σ(fc)/fc | {c['propagation']['mc_formula_fc_std_pct']:.4f}% |")
    md.append(f"| 解析/MC 比 | {c['propagation']['analytic_over_mc_ratio']:.5f} |")
    md.append(f"| 良率 全 5% | {c['yield_full_5pct']['yield']*100:.2f}% |")
    md.append(f"| 加严 C2→1% | {c['yield_tighten']['C2']['yield']*100:.2f}% |")
    md.append(f"| 加严 C1→1% | {c['yield_tighten']['C1']['yield']*100:.2f}% |")
    md.append("")
    md.append("### 灵敏度（S = d|H|/|H| ÷ dx/x）\n")
    md.append("| 元件 | @ f_spec=700 Hz | @ f0=1 kHz |")
    md.append("|---|---|---|")
    for k in ("R1", "R2", "C1", "C2"):
        md.append(f"| {k} | {c['sensitivity_f_spec'][k]:+.4f} | {c['sensitivity_f0'][k]:+.4f} |")
    md.append("")

    p = RESULTS / "v11_summary_tables.md"
    p.write_text("\n".join(md), encoding="utf-8")
    print(f"  saved {p.relative_to(ROOT)}")
    print("[10] done -> v11_summary.json + v11_summary_tables.md")


if __name__ == "__main__":
    main()
