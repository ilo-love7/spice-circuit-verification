"""02 Sallen-Key 低通（Butterworth 二阶）：AC + 瞬态 vs 理论。"""
from common import load_config, save_json, save_fig
import numpy as np
import matplotlib.pyplot as plt

from src.engine.runner import run_netlist
from src.circuits.builders import sk_lowpass_netlist, sk_lowpass_tran
from src.theory.transfer import (sk_lowpass_H, sk_lowpass_params, compare_response,
                                 mag_db, phase_deg, cutoff_from_response)


def main():
    cfg = load_config()["sk_lowpass"]
    R1, R2, C1, C2 = cfg["R1"], cfg["R2"], cfg["C1"], cfg["C2"]
    f0, Q, dcg = sk_lowpass_params(R1, R2, C1, C2)
    print(f"[SK LP] R1={R1} R2={R2} C1={C1:.4e} C2={C2:.4e} → f0={f0:.3f} Hz Q={Q:.4f}")

    net = sk_lowpass_netlist(R1, R2, C1, C2, cfg["fstart"], cfg["fstop"], points_per_decade=20)
    p = run_netlist(net).plot("AC Analysis")
    f = np.array(p["frequency"]).real
    Hs = np.array(p["v(out)"])
    Ht = sk_lowpass_H(R1, R2, C1, C2, 2 * np.pi * f)
    cmp = compare_response(f, Hs, Ht)

    # 峰值（Q>0.707 时有谐振峰；Butterworth 无峰）
    peak_sim = float(np.max(np.abs(Hs)))
    peak_th = float(np.max(np.abs(Ht)))
    fc_sim = cutoff_from_response(f, np.abs(Hs), -3.0)
    # -3dB 带宽 → Q 估计：BW = f0/Q
    f_lo = cutoff_from_response(f[f <= f0][::-1] if np.any(f <= f0) else f, np.abs(Hs)[f <= f0][::-1] if np.any(f <= f0) else np.array([1]), -3.0)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    ax = axes[0, 0]
    ax.semilogx(f, mag_db(Hs), label="ngspice", lw=1.5)
    ax.semilogx(f, mag_db(Ht), "--", label="theory", lw=1.2)
    ax.set_ylabel("|H| (dB)")
    ax.set_title(f"SK LP (f0={f0:.1f} Hz, Q={Q:.3f}, peak={20*np.log10(peak_sim):.3f} dB)")
    ax.legend()
    ax = axes[0, 1]
    ax.semilogx(f, phase_deg(Hs), label="ngspice", lw=1.5)
    ax.semilogx(f, phase_deg(Ht), "--", label="theory", lw=1.2)
    ax.set_ylabel("Phase (deg)")
    ax.legend()
    ax = axes[1, 0]
    ax.semilogx(f, cmp["err_db"], lw=1.2)
    ax.set_ylabel("Magnitude err (dB)")
    ax.set_title(f"max |err| = {cmp['max_abs_err_db']:.2e} dB")
    ax = axes[1, 1]
    ax.semilogx(f, cmp["err_deg"], lw=1.2, color="C1")
    ax.set_ylabel("Phase err (deg)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(f"max |err| = {cmp['max_abs_err_deg']:.2e} deg")
    save_fig(fig, "sk_lp_ac_error.png")

    # ---- 瞬态：f0 正弦 ----
    fsine = f0
    tstop = 30 / fsine
    net = sk_lowpass_tran(R1, R2, C1, C2, fsine, tstop)
    p = run_netlist(net).plot("Transient")
    t = np.array(p["time"]).real
    vin = np.array(p["v(in)"]).real
    vout = np.array(p["v(out)"]).real
    # 理论：|H(jw0)| 与相位
    H_at = sk_lowpass_H(R1, R2, C1, C2, 2 * np.pi * fsine)
    amp_th = float(np.abs(H_at))
    ph_th = float(np.angle(H_at))
    mask = t > t[len(t) // 2]
    v_ss = vout[mask]
    amp_sim = (v_ss.max() - v_ss.min()) / 2

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t * 1e3, vin, lw=0.9, label="Vin")
    ax.plot(t * 1e3, vout, lw=1.1, label="Vout ngspice")
    ax.plot(t * 1e3, amp_th * np.sin(2 * np.pi * fsine * t + ph_th), "--", lw=1, label="Vout theory")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V")
    ax.set_title(f"SK LP transient @ f0 (|H|th={amp_th:.4f}, |H|sim={amp_sim:.4f})")
    ax.legend()
    save_fig(fig, "sk_lp_transient.png")

    res = {
        "f0_theory_Hz": f0,
        "Q_theory": Q,
        "peak_sim": peak_sim,
        "peak_theory": peak_th,
        "fc_3dB_sim_Hz": fc_sim,
        "ac_max_err_db": cmp["max_abs_err_db"],
        "ac_rms_err_db": cmp["rms_err_db"],
        "ac_max_err_deg": cmp["max_abs_err_deg"],
        "tran_amp_sim": float(amp_sim),
        "tran_amp_th": amp_th,
        "tran_amp_err_pct": abs(amp_sim - amp_th) / max(amp_th, 1e-12) * 100,
    }
    save_json("sk_lp.json", res)
    print(f"  AC max err {res['ac_max_err_db']:.2e} dB | peak sim {peak_sim:.6f} vs th {peak_th:.6f} | "
          f"fc(-3dB) {fc_sim:.2f} Hz | tran amp err {res['tran_amp_err_pct']:.3f}%")


if __name__ == "__main__":
    main()
