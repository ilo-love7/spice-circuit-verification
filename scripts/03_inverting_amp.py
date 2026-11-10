"""03 反相放大器：AC 增益/带宽 + 瞬态正弦 vs 理论（含有限开环增益修正）。"""
from common import load_config, save_json, save_fig
import numpy as np
import matplotlib.pyplot as plt

from src.engine.runner import run_netlist
from src.circuits.builders import inverting_amp_netlist, inverting_amp_tran
from src.theory.transfer import (inverting_amp_H, compare_response, mag_db,
                                 phase_deg)
from src.circuits.builders import A_OL


def main():
    cfg = load_config()["inverting_amp"]
    R1, Rf = cfg["R1"], cfg["Rf"]
    G_ideal = -Rf / R1
    G_finite = G_ideal / (1 + (1 + Rf / R1) / A_OL)
    print(f"[Inv amp] R1={R1} Rf={Rf} → G_ideal={G_ideal} G_finite_A={G_finite:.6f}")

    net = inverting_amp_netlist(R1, Rf, cfg["fstart"], cfg["fstop"], points_per_decade=20)
    p = run_netlist(net).plot("AC Analysis")
    f = np.array(p["frequency"]).real
    Hs = np.array(p["v(out)"])
    Ht_ideal = inverting_amp_H(R1, Rf, 2 * np.pi * f, A_ol=np.inf)
    Ht_finite = inverting_amp_H(R1, Rf, 2 * np.pi * f, A_ol=A_OL)

    cmp_ideal = compare_response(f, Hs, Ht_ideal)
    cmp_finite = compare_response(f, Hs, Ht_finite)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    ax = axes[0]
    ax.semilogx(f, mag_db(Hs), label="ngspice", lw=1.5)
    ax.semilogx(f, mag_db(Ht_ideal), "--", label="ideal -Rf/R1", lw=1.2)
    ax.semilogx(f, mag_db(Ht_finite), ":", label=f"finite A={A_OL:.0e}", lw=1.4)
    ax.set_ylabel("|H| (dB)")
    ax.set_title("Inverting amp magnitude")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.semilogx(f, cmp_ideal["err_db"], lw=1.2, label="vs ideal")
    ax.semilogx(f, cmp_finite["err_db"], lw=1.2, label="vs finite-A")
    ax.set_ylabel("Magnitude err (dB)")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.semilogx(f, cmp_ideal["err_deg"], lw=1.2)
    ax.set_ylabel("Phase err vs ideal (deg)")
    ax.set_xlabel("Frequency (Hz)")
    save_fig(fig, "inv_amp_ac.png")

    # ---- 瞬态：1 kHz 正弦 ----
    fsine = 1000.0
    tstop = 10 / fsine
    net = inverting_amp_tran(R1, Rf, fsine, tstop)
    p = run_netlist(net).plot("Transient")
    t = np.array(p["time"]).real
    vin = np.array(p["v(in)"]).real
    vout = np.array(p["v(out)"]).real
    mask = t > t[len(t) // 2]
    v_ss = vout[mask]
    amp_sim = (v_ss.max() - v_ss.min()) / 2
    amp_th = abs(G_finite)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t * 1e3, vin, lw=0.9, label="Vin")
    ax.plot(t * 1e3, vout, lw=1.1, label="Vout ngspice")
    ax.plot(t * 1e3, amp_th * np.sin(2 * np.pi * fsine * t + np.pi), "--", lw=1,
            label="theory (inverted)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V")
    ax.set_title(f"Inverting amp transient @ 1 kHz (G={G_finite:.6f})")
    ax.legend()
    save_fig(fig, "inv_amp_transient.png")

    res = {
        "G_ideal": G_ideal,
        "G_finite_A_theory": G_finite,
        "G_sim_dc": float(np.abs(Hs[0])),
        "G_sim_err_vs_finite_pct": abs(float(np.abs(Hs[0])) - abs(G_finite)) / abs(G_finite) * 100,
        "ac_max_err_db_vs_ideal": cmp_ideal["max_abs_err_db"],
        "ac_max_err_db_vs_finite": cmp_finite["max_abs_err_db"],
        "ac_max_err_deg_vs_ideal": cmp_ideal["max_abs_err_deg"],
        "tran_amp_sim": float(amp_sim),
        "tran_amp_err_pct": abs(amp_sim - amp_th) / amp_th * 100,
    }
    save_json("inv_amp.json", res)
    print(f"  G sim {res['G_sim_dc']:.6f} vs finite-A theory {G_finite:.6f} "
          f"(err {res['G_sim_err_vs_finite_pct']:.4f}%) | tran amp err {res['tran_amp_err_pct']:.3f}%")


if __name__ == "__main__":
    main()
