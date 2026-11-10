"""04 Sallen-Key 带通（等元件、Q=1/3）：AC + 瞬态 vs 理论。"""
from common import load_config, save_json, save_fig
import numpy as np
import matplotlib.pyplot as plt

from src.engine.runner import run_netlist
from src.circuits.builders import sk_bandpass_netlist, sk_bandpass_tran
from src.theory.transfer import (sk_bandpass_H, sk_bandpass_params,
                                 compare_response, mag_db, phase_deg)


def main():
    cfg = load_config()["sk_bandpass"]
    R, C, K = cfg["R"], cfg["C"], cfg["K"]
    f0, Q = sk_bandpass_params(R, C)
    print(f"[SK BP] R={R} C={C:.4e} K={K} → f0={f0:.3f} Hz Q={Q:.4f}")

    net = sk_bandpass_netlist(R, C, K, cfg["fstart"], cfg["fstop"], points_per_decade=20)
    p = run_netlist(net).plot("AC Analysis")
    f = np.array(p["frequency"]).real
    Hs = np.array(p["v(out)"])
    Ht = sk_bandpass_H(R, C, K, 2 * np.pi * f)
    cmp = compare_response(f, Hs, Ht)

    peak_sim = float(np.max(np.abs(Hs)))
    f_peak_sim = float(f[np.argmax(np.abs(Hs))])
    peak_th = float(np.max(np.abs(Ht)))
    f_peak_th = float(f[np.argmax(np.abs(Ht))])

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    ax = axes[0, 0]
    ax.semilogx(f, mag_db(Hs), label="ngspice", lw=1.5)
    ax.semilogx(f, mag_db(Ht), "--", label="theory", lw=1.2)
    ax.set_ylabel("|H| (dB)")
    ax.set_title(f"SK BP (f0={f0:.1f} Hz, Q={Q:.3f}, peak={20*np.log10(peak_sim):.3f} dB)")
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
    save_fig(fig, "sk_bp_ac_error.png")

    # ---- 瞬态：f0 正弦 ----
    fsine = f0
    tstop = 30 / fsine
    net = sk_bandpass_tran(R, C, K, fsine, tstop)
    p = run_netlist(net).plot("Transient")
    t = np.array(p["time"]).real
    vin = np.array(p["v(in)"]).real
    vout = np.array(p["v(out)"]).real
    H_at = sk_bandpass_H(R, C, K, 2 * np.pi * fsine)
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
    ax.set_title(f"SK BP transient @ f0 (|H|th={amp_th:.4f}, |H|sim={amp_sim:.4f})")
    ax.legend()
    save_fig(fig, "sk_bp_transient.png")

    res = {
        "f0_theory_Hz": f0,
        "Q_theory": Q,
        "K": K,
        "peak_sim": peak_sim,
        "peak_theory": peak_th,
        "f_peak_sim_Hz": f_peak_sim,
        "f_peak_th_Hz": f_peak_th,
        "ac_max_err_db": cmp["max_abs_err_db"],
        "ac_rms_err_db": cmp["rms_err_db"],
        "ac_max_err_deg": cmp["max_abs_err_deg"],
        "tran_amp_sim": float(amp_sim),
        "tran_amp_th": amp_th,
        "tran_amp_err_pct": abs(amp_sim - amp_th) / max(amp_th, 1e-12) * 100,
    }
    save_json("sk_bp.json", res)
    print(f"  AC max err {res['ac_max_err_db']:.2e} dB | peak {peak_sim:.6f} @ {f_peak_sim:.1f} Hz "
          f"(theory {peak_th:.6f} @ {f_peak_th:.1f}) | tran amp err {res['tran_amp_err_pct']:.3f}%")


if __name__ == "__main__":
    main()
