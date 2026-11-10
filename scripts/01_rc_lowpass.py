"""01 一阶 RC 低通：AC 扫频 + 瞬态正弦/阶跃 vs 理论。"""
from common import load_config, save_json, save_fig, loglog_axes
import numpy as np
import matplotlib.pyplot as plt

from src.engine.runner import run_netlist
from src.circuits.builders import (rc_lowpass_netlist, rc_lowpass_tran,
                                   rc_lowpass_step_tran)
from src.theory.transfer import (rc_lowpass_H, compare_response, mag_db,
                                 cutoff_from_response, phase_deg)


def main():
    cfg = load_config()["rc_lowpass"]
    R, C = cfg["R"], cfg["C"]
    fc_th = 1.0 / (2 * np.pi * R * C)
    print(f"[RC LP] R={R} C={C} fc_theory={fc_th:.4f} Hz")

    # ---- AC ----
    net = rc_lowpass_netlist(R, C, cfg["fstart"], cfg["fstop"], points_per_decade=20)
    p = run_netlist(net).plot("AC Analysis")
    f = np.array(p["frequency"]).real
    Hs = np.array(p["v(out)"])
    Ht = rc_lowpass_H(R, C, 2 * np.pi * f)
    cmp = compare_response(f, Hs, Ht)
    fc_sim = cutoff_from_response(f, np.abs(Hs), -3.0)
    fc_thn = cutoff_from_response(f, np.abs(Ht), -3.0)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    ax = axes[0, 0]
    ax.semilogx(f, mag_db(Hs), label="ngspice", lw=1.5)
    ax.semilogx(f, mag_db(Ht), "--", label="theory", lw=1.2)
    ax.set_ylabel("|H| (dB)")
    ax.set_title(f"RC LP magnitude (fc={fc_th:.1f} Hz)")
    ax.legend()
    ax = axes[0, 1]
    ax.semilogx(f, phase_deg(Hs), label="ngspice", lw=1.5)
    ax.semilogx(f, phase_deg(Ht), "--", label="theory", lw=1.2)
    ax.set_ylabel("Phase (deg)")
    ax.set_title("RC LP phase")
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
    save_fig(fig, "rc_lp_ac_error.png")

    # ---- 瞬态：f = fc 正弦 ----
    fsine = fc_th
    tstop = 20 / fsine
    net = rc_lowpass_tran(R, C, fsine, tstop)
    p = run_netlist(net).plot("Transient")
    t = np.array(p["time"]).real
    vin = np.array(p["v(in)"])
    vout = np.array(p["v(out)"])
    # 理论稳态：幅度 1/sqrt(2)，相位 -45deg
    amp_th = 1 / np.sqrt(2)
    ph_th = -np.pi / 4
    # 取后 50% 稳态段拟合幅度
    mask = t > t[len(t) // 2]
    v_ss = vout[mask].real
    amp_sim = (v_ss.max() - v_ss.min()) / 2
    # 相位：过零比较
    def first_zero_up(x, tt):
        for i in range(1, len(x)):
            if x[i - 1] < 0 <= x[i]:
                return tt[i]
        return np.nan
    t0_in = first_zero_up(vin[mask].real, t[mask])
    t0_out = first_zero_up(v_ss, t[mask])
    ph_sim = -2 * np.pi * fsine * (t0_out - t0_in)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(t * 1e3, vin.real, lw=1, label="Vin")
    axes[0].plot(t * 1e3, vout.real, lw=1, label="Vout (ngspice)")
    axes[0].plot(t * 1e3, amp_th * np.sin(2 * np.pi * fsine * t + ph_th),
                 "--", lw=1, label="Vout theory (steady)")
    axes[0].set_ylabel("V")
    axes[0].set_title(f"RC LP transient @ f=fc ({fsine:.2f} Hz)")
    axes[0].legend()
    axes[1].plot(t * 1e3, vout.real - amp_th * np.sin(2 * np.pi * fsine * t + ph_th),
                 lw= 1, color="C3")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Residual (V)")
    save_fig(fig, "rc_lp_transient_sine.png")

    # ---- 瞬态：阶跃（时间常数核对）----
    tau = R * C
    net = rc_lowpass_step_tran(R, C, 10 * tau)
    p = run_netlist(net).plot("Transient")
    t = np.array(p["time"]).real
    vout = np.array(p["v(out)"]).real
    # 数值求 63.2% 上升时间
    idx = np.searchsorted(vout, 1 - np.exp(-1))
    t_sim = t[idx] if idx < len(t) else np.nan

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t * 1e3, vout, lw=1.5, label="ngspice")
    ax.plot(t * 1e3, 1 - np.exp(-t / tau), "--", lw=1.2, label="theory 1-e^{-t/τ}")
    ax.axhline(1 - np.exp(-1), color="gray", lw=0.7, ls=":")
    ax.axvline(tau * 1e3, color="gray", lw=0.7, ls=":")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Vout (V)")
    ax.set_title(f"RC LP step response (τ = {tau*1e3:.3f} ms, t_63 sim={t_sim*1e3:.4f} ms)")
    ax.legend()
    save_fig(fig, "rc_lp_step.png")

    res = {
        "fc_theory_Hz": fc_th,
        "fc_sim_Hz": fc_sim,
        "fc_err_pct": abs(fc_sim - fc_th) / fc_th * 100,
        "ac_max_err_db": cmp["max_abs_err_db"],
        "ac_rms_err_db": cmp["rms_err_db"],
        "ac_max_err_deg": cmp["max_abs_err_deg"],
        "tran_sine_amp_sim": float(amp_sim),
        "tran_sine_amp_th": amp_th,
        "tran_sine_amp_err_pct": abs(amp_sim - amp_th) / amp_th * 100,
        "tran_sine_phase_sim_deg": float(np.degrees(ph_sim)),
        "tran_sine_phase_th_deg": -45.0,
        "step_tau_theory_s": tau,
        "step_t63_sim_s": float(t_sim),
    }
    save_json("rc_lp.json", res)
    print(f"  fc err {res['fc_err_pct']:.4f}% | AC max err {res['ac_max_err_db']:.2e} dB | "
          f"sine amp err {res['tran_sine_amp_err_pct']:.3f}% | t63 sim {t_sim*1e3:.4f} ms vs τ {tau*1e3:.4f} ms")


if __name__ == "__main__":
    main()
