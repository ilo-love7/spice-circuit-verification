"""06 汇总：聚合所有结果 JSON → summary.json + summary_tables.md。"""
from common import load_config, save_json, ROOT, RESULTS
import json


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main():
    smoke = load("smoke.json")
    rc = load("rc_lp.json")
    sk = load("sk_lp.json")
    inv = load("inv_amp.json")
    bp = load("sk_bp.json")
    mc = load("mc_results.json")

    summary = {
        "version": "v1.0.0",
        "smoke": smoke,
        "rc_lp": {k: v for k, v in rc.items()},
        "sk_lp": {k: v for k, v in sk.items()},
        "inverting_amp": {k: v for k, v in inv.items()},
        "sk_bp": {k: v for k, v in bp.items()},
        "monte_carlo": {
            "tol_pct": mc["tol"] * 100,
            "n_trials": mc["n_trials"],
            "rc_yield": mc["rc_lp"]["yield"],
            "rc_fc_mean": mc["rc_lp"]["fc_mean"],
            "rc_fc_std": mc["rc_lp"]["fc_std"],
            "sk_yield": mc["sk_lp"]["yield"],
            "sk_fc_mean": mc["sk_lp"]["fc_mean"],
            "sk_fc_std": mc["sk_lp"]["fc_std"],
        },
    }
    save_json("summary.json", summary)

    # ---- 表格 ----
    lines = []
    lines.append("# N7 SPICE 电路自动化仿真验证 — 汇总表\n")
    lines.append(f"版本 {summary['version']} | 引擎 {smoke['engine']} | PySpice 网表 {smoke['pyspice']}\n")
    lines.append("\n## 理论 vs 仿真误差（AC）\n")
    lines.append("| 电路 | 最大幅值误差 (dB) | 均方误差 (dB) | 最大相位误差 (deg) |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 一阶 RC 低通 | {rc['ac_max_err_db']:.3e} | {rc['ac_rms_err_db']:.3e} | {rc['ac_max_err_deg']:.3e} |")
    lines.append(f"| Sallen-Key 低通 | {sk['ac_max_err_db']:.3e} | {sk['ac_rms_err_db']:.3e} | {sk['ac_max_err_deg']:.3e} |")
    lines.append(f"| 反相放大器（vs 理想） | {inv['ac_max_err_db_vs_ideal']:.3e} | — | {inv['ac_max_err_deg_vs_ideal']:.3e} |")
    lines.append(f"| Sallen-Key 带通 | {bp['ac_max_err_db']:.3e} | {bp['ac_rms_err_db']:.3e} | {bp['ac_max_err_deg']:.3e} |")

    lines.append("\n## 关键参数\n")
    lines.append("| 项 | 理论 | 仿真 | 误差 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| RC fc (Hz) | {rc['fc_theory_Hz']:.4f} | {rc['fc_sim_Hz']:.4f} | {rc['fc_err_pct']:.4f}% |")
    lines.append(f"| SK LP f0 (Hz) / Q | {sk['f0_theory_Hz']:.3f} / {sk['Q_theory']:.4f} | peak {sk['peak_sim']:.6f} | — |")
    lines.append(f"| 反相增益 | {inv['G_finite_A_theory']:.6f} | {inv['G_sim_dc']:.6f} | {inv['G_sim_err_vs_finite_pct']:.4f}% |")
    lines.append(f"| SK BP 峰值 / f0 | {bp['peak_theory']:.6f} @ {bp['f_peak_th_Hz']:.2f} | {bp['peak_sim']:.6f} @ {bp['f_peak_sim_Hz']:.2f} | — |")

    lines.append("\n## 瞬态验证\n")
    lines.append("| 项 | 理论 | 仿真 | 误差 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| RC 正弦@fc 幅度 | {rc['tran_sine_amp_th']:.6f} | {rc['tran_sine_amp_sim']:.6f} | {rc['tran_sine_amp_err_pct']:.4f}% |")
    lines.append(f"| RC 阶跃 t63 (ms) | {rc['step_tau_theory_s']*1e3:.4f} | {rc['step_t63_sim_s']*1e3:.4f} | — |")
    lines.append(f"| SK LP 正弦@f0 幅度 | {sk['tran_amp_th']:.6f} | {sk['tran_amp_sim']:.6f} | {sk['tran_amp_err_pct']:.4f}% |")
    lines.append(f"| 反相 1kHz 幅度 | {abs(inv['G_finite_A_theory']):.6f} | {inv['tran_amp_sim']:.6f} | {inv['tran_amp_err_pct']:.4f}% |")
    lines.append(f"| SK BP 正弦@f0 幅度 | {bp['tran_amp_th']:.6f} | {bp['tran_amp_sim']:.6f} | {bp['tran_amp_err_pct']:.4f}% |")

    mc = summary["monte_carlo"]
    lines.append("\n## 蒙特卡洛（±5% 容差）\n")
    lines.append("| 电路 | fc 理论 (Hz) | fc 均值±std (Hz) | 良率 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| RC 低通 | {summary['rc_lp']['fc_theory_Hz']:.2f} | {mc['rc_fc_mean']:.2f} ± {mc['rc_fc_std']:.2f} | {mc['rc_yield']*100:.1f}% |")
    lines.append(f"| SK 低通 | {summary['sk_lp']['f0_theory_Hz']:.2f} | {mc['sk_fc_mean']:.2f} ± {mc['sk_fc_std']:.2f} | {mc['sk_yield']*100:.1f}% |")
    lines.append(f"\n试验数 n={mc['n_trials']}，均匀分布 ±{mc['tol_pct']:.0f}%，种子 20261110。\n")

    out = RESULTS / "summary_tables.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
