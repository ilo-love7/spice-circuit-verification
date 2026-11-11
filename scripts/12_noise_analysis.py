# -*- coding: utf-8 -*-
"""E17 · 噪声分析（补报告 §6 的 `.noise` 未做项，v1.2）。

四组量化：
  1. kT/C 三重验证：RC 低通的 √(4kTR) 密度、√(kT/C) 总噪声（解析 / ngspice 积分 / 谱数值积分）
  2. MFB 带通 ENBW：解析 π f0/(2Q)、谱积分、ngspice `onoise_total` 三者对照
  3. 运放 e_n 的严格 SPICE 建模：同相输入串等效噪声电阻（不改变传递函数）
  4. 阻抗标度 α：无源热噪声 ∝ √α、e_n 项恒定、i_n·R3 ∝ α ⇒ 存在最优 α 与 SNR 峰值
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import load_config, save_json, save_fig

from src.theory import mfb, noise
from src.circuits import v12_builders as vb
from src.engine.runner import run_netlist

CFG = load_config()["v12"]["noise"]
F0, C = CFG["f0_Hz"], CFG["C_F"]
PPD = CFG["noise_ppd"]
E_N = CFG["e_n_V_per_sqrtHz"]
I_N = CFG["i_n_A_per_sqrtHz"]
R_EQ_EN = E_N ** 2 / noise.thermal_psd_v2_per_hz(1.0)


def noise_of(netlist: str):
    res = run_netlist(netlist)
    sd = res.plot("Noise Spectral Density")
    integ = res.plot("Integrated Noise")
    return (np.asarray(sd["frequency"], dtype=float),
            np.asarray(sd["onoise_spectrum"], dtype=float),
            np.asarray(sd["inoise_spectrum"], dtype=float),
            float(integ["v(onoise_total)"][0]))


def design_args(Q, alpha=1.0):
    d = mfb.design_mfb_bp(F0, Q, C)
    base = {k: d[k] for k in mfb.MFB_COMPONENTS}
    sc = noise.scale_passive_network(base, alpha)
    return d, tuple(sc[k] for k in mfb.MFB_COMPONENTS), sc


def main():
    out = {"config": CFG, "e_n": E_N, "i_n": I_N,
           "R_eq_en_ohm": R_EQ_EN}

    # ---------- 1. kT/C 三重验证 ----------
    print("[E17] 1/4 kT/C triple check")
    ktc = []
    for R in CFG["r_list_ohm"]:
        f, on, inn, total = noise_of(vb.rc_noise_netlist(
            R, CFG["rc_C_F"], CFG["fstart"], CFG["fstop"], PPD))
        num = noise.integrated_rms(f, on, log_spacing=True)
        kt = noise.kT_over_C_rms(CFG["rc_C_F"])
        rec = {
            "R_ohm": float(R),
            "fc_Hz": 1.0 / (2 * np.pi * R * CFG["rc_C_F"]),
            "ENBW_Hz": noise.rc_enbw_hz(R, CFG["rc_C_F"]),
            "thermal_density_analytic": noise.thermal_density_v_per_sqrthz(R),
            "thermal_density_ngspice": float(on[0]),
            "total_kT_over_C_Vrms": kt,
            "total_ngspice_Vrms": total,
            "total_spectral_integral_Vrms": num,
            "ngspice_rel_err_pct": float((total / kt - 1) * 100),
            "integral_rel_err_pct": float((num / kt - 1) * 100),
        }
        ktc.append(rec)
        print(f"   R={R:8.0f}  e_n={rec['thermal_density_analytic']:.4e}  "
              f"ngspice={rec['thermal_density_ngspice']:.4e}  "
              f"rms: kT/C={kt:.4e} ngspice={total:.4e} ({rec['ngspice_rel_err_pct']:+.3f}%)")
    out["ktc"] = ktc

    # ---------- 2. MFB 带通 ENBW / 噪声谱 ----------
    print("[E17] 2/4 bandpass ENBW")
    bp, spectra = [], {}
    for Q in CFG["q_list"]:
        d, a, _ = design_args(Q)
        f, on, inn, total = noise_of(vb.mfb_bp_noise_netlist(
            *a, CFG["fstart_bp"], CFG["fstop_bp"], PPD))
        spectra[f"Q={Q:g}"] = (f, on)
        j = int(np.argmin(np.abs(f - F0)))
        e_in = float(inn[j])
        enbw = noise.bandpass_enbw_hz(F0, Q, d["H0"])
        rec = {
            "Q": float(Q), "H0": d["H0"], "bw3db_Hz": d["bw_3db_Hz"],
            "ENBW_analytic_Hz": enbw,
            "onoise_peak_V_per_sqrtHz": float(on.max()),
            "onoise_peak_f_Hz": float(f[int(np.argmax(on))]),
            "input_referred_density_at_f0": e_in,
            "ngspice_total_Vrms": total,
            "spectral_integral_Vrms": noise.integrated_rms(f, on, log_spacing=True),
            "white_noise_approx_Vrms": noise.output_noise_from_input_density(
                e_in, F0, Q, d["H0"]),
            "ENBW_pi_over_2_BW_ratio": enbw / d["H0"] ** 2 / d["bw_3db_Hz"],
        }
        rec["white_approx_rel_err_pct"] = float(
            (rec["white_noise_approx_Vrms"] / rec["spectral_integral_Vrms"] - 1) * 100)
        bp.append(rec)
        print(f"   Q={Q:5.1f}  ENBW={enbw:12.1f} Hz  "
              f"total={total:.4e} Vrms  white-approx err={rec['white_approx_rel_err_pct']:+.2f}%")
    out["bandpass_enbw"] = bp
    qs = np.array([r["Q"] for r in bp])
    tots = np.array([r["ngspice_total_Vrms"] for r in bp])
    slope = float(np.polyfit(np.log(qs), np.log(tots), 1)[0])
    out["total_noise_vs_Q_power_law"] = slope
    print(f"   onoise_total ~ Q^{slope:.3f}")

    # ---------- 3+4. 阻抗标度与最优噪声 ----------
    print("[E17] 3/4 op-amp e_n (SPICE) + 4/4 impedance scaling")
    Qopt = CFG["optimal_alpha_Q"]
    d, _, _ = design_args(Qopt)
    scale_rows = []
    for alpha in CFG["alpha_list"]:
        _, a, sc = design_args(Qopt, alpha)
        _, _, _, passive = noise_of(vb.mfb_bp_noise_netlist(
            *a, CFG["fstart_bp"], CFG["fstop_bp"], PPD))
        _, _, _, with_en = noise_of(vb.mfb_bp_en_netlist(
            *a, R_EQ_EN, CFG["fstart_bp"], CFG["fstop_bp"], PPD))
        p_en = float(np.sqrt(max(with_en ** 2 - passive ** 2, 0.0)))
        K = p_en / E_N  # 单位输入噪声电压 -> 输出 rms 的传递系数（由 SPICE 标定）
        p_i = float(I_N * sc["R3"] * K)  # 一阶工程模型：i_n 流过反馈电阻 R3
        total = float(np.sqrt(passive ** 2 + p_en ** 2 + p_i ** 2))
        scale_rows.append({
            "alpha": float(alpha), "R1_ohm": sc["R1"], "R3_ohm": sc["R3"],
            "passive_Vrms": passive, "with_en_Vrms": with_en,
            "en_component_Vrms": p_en, "in_component_Vrms": p_i,
            "K_noise_transfer": K, "total_Vrms": total,
            "snr_db": noise.snr_db(CFG["signal_vrms"], total),
        })
        print(f"   alpha={alpha:6.2f}  passive={passive:.3e}  e_n={p_en:.3e}  "
              f"i_n={p_i:.3e}  total={total:.3e}  SNR={scale_rows[-1]['snr_db']:.2f} dB")
    out["impedance_scaling"] = scale_rows
    # 预设「存在最优阻抗标度」的检验：三项噪声中两项随 alpha 单调增、
    # e_n 项恒定 ⇒ 总噪声必单调，最优在边界（负结果，如实保留）。
    totals = np.array([r["total_Vrms"] for r in scale_rows])
    passive = np.array([r["passive_Vrms"] for r in scale_rows])
    en_term = np.array([r["en_component_Vrms"] for r in scale_rows])
    in_term = np.array([r["in_component_Vrms"] for r in scale_rows])
    alphas = np.array([r["alpha"] for r in scale_rows])
    p_passive = float(np.polyfit(np.log(alphas), np.log(passive), 1)[0])
    p_in = float(np.polyfit(np.log(alphas), np.log(in_term), 1)[0])
    row1 = scale_rows[0]
    alpha_passive_eq_en = float((row1["en_component_Vrms"] / row1["passive_Vrms"]) ** 2
                                * row1["alpha"])
    alpha_in_eq_en = float(row1["en_component_Vrms"] / row1["in_component_Vrms"]
                           * row1["alpha"])
    out["impedance_monotonicity"] = {
        "total_monotonic_increasing": bool(np.all(np.diff(totals) > 0)),
        "interior_optimum_exists": bool(np.any(np.diff(totals) < 0)),
        "minimum_at_alpha": float(alphas[int(np.argmin(totals))]),
        "passive_power_law": p_passive, "in_power_law": p_in,
        "en_component_spread_pct": float((en_term.max() / en_term.min() - 1) * 100),
        "noise_floor_Vrms": float(np.mean(en_term)),
        "alpha_where_passive_equals_en": alpha_passive_eq_en,
        "alpha_where_in_equals_en": alpha_in_eq_en,
        "verdict": "no interior optimum; noise floor set by op-amp e_n",
    }
    # i_n 量级敏感性：结论是否与电流噪声假设无关（纯解析，K 已由 SPICE 标定）
    sens = []
    for mult in (0.01, 0.1, 1.0, 10.0):
        tt = np.sqrt(passive ** 2 + en_term ** 2 + (mult * in_term) ** 2)
        sens.append({"i_n_multiplier": mult,
                     "total_monotonic": bool(np.all(np.diff(tt) > 0)),
                     "total_at_min_alpha_Vrms": float(tt[0]),
                     "total_at_max_alpha_Vrms": float(tt[-1])})
    out["i_n_sensitivity"] = sens
    print(f"   passive ~ alpha^{p_passive:.3f}, i_n ~ alpha^{p_in:.3f}; "
          f"monotonic={out['impedance_monotonicity']['total_monotonic_increasing']}, "
          f"e_n spread={out['impedance_monotonicity']['en_component_spread_pct']:.3f}%")
    print("   NEGATIVE RESULT: no interior optimum; floor = "
          f"{out['impedance_monotonicity']['noise_floor_Vrms']:.3e} Vrms (e_n)")

    save_json("v12_noise.json", out)

    # ---------- 图 1：kT/C ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    Rs = [r["R_ohm"] for r in ktc]
    axes[0].loglog(Rs, [r["thermal_density_analytic"] for r in ktc], "k-",
                   label="sqrt(4kTR) analytic")
    axes[0].loglog(Rs, [r["thermal_density_ngspice"] for r in ktc], "o",
                   label="ngspice low-f density")
    axes[0].set_xlabel("R (ohm)"); axes[0].set_ylabel("noise density (V/sqrtHz)")
    axes[0].set_title("E17: resistor thermal noise"); axes[0].legend(fontsize=8)
    kt_line = ktc[0]["total_kT_over_C_Vrms"]
    axes[1].axhline(kt_line, color="k", ls="-", label="sqrt(kT/C) analytic")
    axes[1].semilogx(Rs, [r["total_ngspice_Vrms"] for r in ktc], "s-",
                     label="ngspice onoise_total")
    axes[1].semilogx(Rs, [r["total_spectral_integral_Vrms"] for r in ktc], "^--",
                     label="spectral integral")
    axes[1].set_xlabel("R (ohm)"); axes[1].set_ylabel("output rms noise (V)")
    axes[1].set_title("E17: kT/C noise is independent of R")
    axes[1].legend(fontsize=8)
    save_fig(fig, "v12_noise_ktc.png")

    # ---------- 图 2：带通噪声谱 + ENBW ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, (f, on) in spectra.items():
        ax.loglog(f, on, lw=1.6, label=label)
    ax.axvline(F0, color="gray", ls=":", lw=1.0, label="f0")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("output noise density (V/sqrtHz)")
    ax.set_title(f"E17: bandpass output noise, total ~ Q^{slope:.2f}")
    ax.legend(fontsize=8)
    save_fig(fig, "v12_noise_spectrum_enbw.png")

    # ---------- 图 3：最优阻抗标度 ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    al = [r["alpha"] for r in scale_rows]
    for key, lab in (("passive_Vrms", "resistor thermal"),
                     ("en_component_Vrms", "op-amp e_n"),
                     ("in_component_Vrms", "op-amp i_n (analytic)"),
                     ("total_Vrms", "total")):
        axes[0].loglog(al, [r[key] for r in scale_rows], "o-", lw=1.7, label=lab)
    axes[0].axhline(out["impedance_monotonicity"]["noise_floor_Vrms"],
                    color="crimson", ls="--", lw=1.2, label="e_n noise floor")
    axes[0].set_xlabel("impedance scale alpha")
    axes[0].set_ylabel("output rms noise (V)")
    axes[0].set_title("E17: noise is monotonic in alpha (negative result)")
    axes[0].legend(fontsize=7.5)
    axes[1].semilogx(al, [r["snr_db"] for r in scale_rows], "o-", color="darkgreen")
    axes[1].axhline(noise.snr_db(CFG["signal_vrms"],
                                 out["impedance_monotonicity"]["noise_floor_Vrms"]),
                    color="crimson", ls="--", lw=1.2, label="SNR ceiling (e_n)")
    axes[1].set_xlabel("impedance scale alpha")
    axes[1].set_ylabel("SNR (dB, 1 Vrms signal)")
    axes[1].set_title("E17: SNR has no interior maximum")
    axes[1].legend(fontsize=8)
    save_fig(fig, "v12_noise_optimal_scale.png")

    print("[E17] done")


if __name__ == "__main__":
    main()
