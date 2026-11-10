"""07 深化 A：非理想运放（有限 GBW / 压摆率）。

- 宏模型：单极点限流跨导（B 源），A(s)=A0/(1+s/wb)，SR=Imax/Cc
- AC：反相放大闭环 H=−G·A/(A+NG) 理论 vs ngspice（机器精度）
- 经典规则 fc≈GBW/NG vs 精确极点的偏差量化
- 大信号方波：压摆率实测 vs 规格 Imax/Cc
- 方法学：cutoff_from_response（线性插值）vs cutoff_precise（样条）偏差
"""
from common import save_json, save_fig
import numpy as np
import matplotlib.pyplot as plt

from src.engine.runner import run_netlist
from src.theory.nonideal import (design_macromodel, inverting_amp_H_finiteGBW,
                                 closed_loop_pole_approx_Hz, macromodel_params,
                                 f3dB_from_pole)
from src.theory.transfer import (compare_response, mag_db, phase_deg,
                                 cutoff_from_response, cutoff_precise)
from src.circuits.nonideal_builders import (inverting_amp_gbw_netlist,
                                            inverting_amp_slew_netlist,
                                            noninv_buffer_gbw_netlist)


def main():
    # ---- 设计：GBW=10 MHz, A0=1e5, SR=10 V/us ----
    p = design_macromodel(GBW_Hz=10e6, A0=1e5, SR_V_per_s=10e6)
    print(f"[A] macro: gm={p['gm']:.3g} Rlag={p['Rlag']:.3g} Cc={p['Cc']:.4g} Imax={p['Imax']:.4g}")
    print(f"    A0={p['A0']:.3g} fb={p['fb_Hz']:.1f} Hz GBW={p['GBW_Hz']:.3g} Hz SR={p['SR_V_per_s']/1e6:.3g} V/us")

    R1, Rf = 1e3, 1e4          # G = -10, NG = 11
    G, NG = Rf / R1, 1 + Rf / R1

    # ---- AC：有限 GBW 反相放大 ----
    net = inverting_amp_gbw_netlist(R1, Rf, p['gm'], p['Rlag'], p['Cc'], p['Imax'],
                                    1e2, 1e8, points_per_decade=15)
    pl = run_netlist(net).plot("AC Analysis")
    f = np.array(pl['frequency']).real
    Hs = np.array(pl['v(out)'])
    w = 2 * np.pi * f
    Ht = inverting_amp_H_finiteGBW(R1, Rf, w, p['A0'], p['wb_rad_per_s'])
    cmp = compare_response(f, Hs, Ht)
    print(f"[A] AC vs exact-H theory: max|err|={cmp['max_abs_err_db']:.3e} dB, phase {cmp['max_abs_err_deg']:.3e} deg")

    fc_sim_prec = cutoff_precise(f, Hs, -3.0)
    fc_sim_lin = cutoff_from_response(f, np.abs(Hs), -3.0)
    fc_th_prec = cutoff_precise(f, Ht, -3.0)
    # 稠密网格收敛性核对（fc 提取数值误差）
    f_dense = np.logspace(2, 8, 60 * 6 + 1)
    Ht_dense = inverting_amp_H_finiteGBW(R1, Rf, 2 * np.pi * f_dense, p['A0'], p['wb_rad_per_s'])
    fc_dense = cutoff_precise(f_dense, Ht_dense, -3.0)
    wb = p['wb_rad_per_s']
    fc_pole_exact = wb * (p['A0'] + NG) / NG / (2 * np.pi)
    fc_classic = closed_loop_pole_approx_Hz(p['GBW_Hz'], NG)
    print(f"[A] fc: sim(spline)={fc_sim_prec:.1f}  sim(linear)={fc_sim_lin:.1f}  exact-pole={fc_pole_exact:.1f}  classic GBW/NG={fc_classic:.1f}")
    fc_pole_as_3db = f3dB_from_pole(fc_pole_exact, -3.0)
    print(f"    fc extraction: spline-15/dec={fc_th_prec:.1f}  spline-60/dec={fc_dense:.1f}  bias={(fc_th_prec/fc_pole_exact-1)*100:+.3f}% -> {(fc_dense/fc_pole_exact-1)*100:+.4f}%")
    print(f"    convention: pole is the -3.0103dB half-power point; -3.00dB criterion -> fp*sqrt(10^0.3-1)={fc_pole_as_3db:.1f} (residual {(fc_dense/fc_pole_as_3db-1)*100:+.4f}%)")

    # 缓冲器（NG=1）核对
    netb = noninv_buffer_gbw_netlist(p['gm'], p['Rlag'], p['Cc'], p['Imax'], 1e2, 1e8, points_per_decade=15)
    plb = run_netlist(netb).plot('AC Analysis')
    fb = np.array(plb['frequency']).real
    Hsb = np.array(plb['v(out)'])
    Ab = p['A0'] / (1 + 1j * 2 * np.pi * fb / p['wb_rad_per_s'])
    Htb = Ab / (Ab + 1)   # 同相缓冲闭环 = A/(1+A)
    cmpb = compare_response(fb, Hsb, Htb)
    print(f"[A] buffer AC: max|err|={cmpb['max_abs_err_db']:.3e} dB")

    # ---- 多增益点：fc_vs_gain 曲线 ----
    gains = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    fc_by_gain = []
    for g in gains:
        r1, rf = 1e3, 1e3 * g
        netg = inverting_amp_gbw_netlist(r1, rf, p['gm'], p['Rlag'], p['Cc'], p['Imax'],
                                         1e2, 1e8, points_per_decade=15)
        pg = run_netlist(netg).plot("AC Analysis")
        fg = np.array(pg['frequency']).real
        Hg = np.array(pg['v(out)'])
        ng = 1 + g
        fc_meas = cutoff_precise(fg, Hg, -3.0)
        fc_cl = closed_loop_pole_approx_Hz(p['GBW_Hz'], ng)
        fc_ex = p['wb_rad_per_s'] * (p['A0'] + ng) / ng / (2 * np.pi)
        fc_by_gain.append((g, fc_meas, fc_cl, fc_ex))
        print(f"    G={g:6.1f} NG={ng:6.1f}  fc_meas={fc_meas:.3e}  classic={fc_cl:.3e} ({(fc_cl/fc_ex-1)*100:+.2f}%)  exact={fc_ex:.3e}")

    # ---- 压摆率 ----
    v_amp, f_sq = 0.5, 1e4
    net2 = inverting_amp_slew_netlist(R1, Rf, p['gm'], p['Rlag'], p['Cc'], p['Imax'],
                                      v_amp=v_amp, f_sq=f_sq, tstop=2e-4)
    pl2 = run_netlist(net2).plot("Transient Analysis")
    t = np.array(pl2['time']).real
    v = np.array(pl2['v(out)'])
    dv = np.abs(np.diff(v) / np.diff(t))
    n = len(t)
    slew = float(np.max(dv[int(n * 0.1):int(n * 0.9)]))
    sr_spec = p['SR_V_per_s']
    print(f"[A] slew: sim={slew/1e6:.4f} V/us  spec={sr_spec/1e6:.4f} V/us  err={(slew/sr_spec-1)*100:+.3f}%")

    # 欠摆幅（小信号正弦不应压摆限幅）
    net3 = inverting_amp_slew_netlist(R1, Rf, p['gm'], p['Rlag'], p['Cc'], p['Imax'],
                                      v_amp=0.005, f_sq=1e4, tstop=2e-4)
    pl3 = run_netlist(net3).plot("Transient Analysis")
    t3 = np.array(pl3['time']).real
    v3 = np.array(pl3['v(out)'])
    dv3 = np.abs(np.diff(v3) / np.diff(t3))
    slew_small = float(np.max(dv3[int(len(t3) * 0.1):int(len(t3) * 0.9)]))
    # 小信号方波最大斜率由闭环带宽决定（一阶阶跃：ΔV·2πfc），而非 SR
    fc_cl_lo = closed_loop_pole_approx_Hz(p['GBW_Hz'], NG)
    slew_linear = 2 * np.pi * fc_cl_lo * (G * 2 * 0.005)   # ΔV_pp_out = G·10mV
    print(f"[A] small-signal slew: sim={slew_small/1e6:.4f} V/us  bandwidth-limited pred={slew_linear/1e6:.4f} V/us  ratio_to_SR={slew_small/sr_spec:.3f} (SR not engaged)")

    # ---- 图 ----
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    ax = axes[0, 0]
    ax.semilogx(f, mag_db(Hs), label='ngspice', lw=1.5)
    ax.semilogx(f, mag_db(Ht), '--', label='theory exact H', lw=1.2)
    ax.axvline(fc_classic, color='C2', ls=':', lw=1, label=f'GBW/NG={fc_classic/1e3:.0f}k')
    ax.set_ylabel('|H| (dB)')
    ax.set_title(f'Inverting amp G=-10 finite GBW ({p["GBW_Hz"]/1e6:.0f} MHz)')
    ax.legend()
    ax = axes[0, 1]
    ax.semilogx(f, phase_deg(Hs), label='ngspice', lw=1.5)
    ax.semilogx(f, phase_deg(Ht), '--', label='theory', lw=1.2)
    ax.set_ylabel('Phase (deg)')
    ax.set_title('Phase')
    ax.legend()
    ax = axes[1, 0]
    ax.semilogx(f, mag_db(Hs) - mag_db(Ht), lw=1.2)
    ax.set_ylabel('Mag err (dB)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title(f'max|err|={cmp["max_abs_err_db"]:.1e} dB')
    ax = axes[1, 1]
    ax.plot(t * 1e6, v, lw=1.2, label='V(out)')
    ax.set_xlabel('t (us)')
    ax.set_ylabel('V')
    ax.set_title(f'slew sim={slew/1e6:.3f} vs spec={sr_spec/1e6:.3f} V/us')
    ax.legend()
    save_fig(fig, 'v11_gbw_ac_slew.png')

    fig, ax = plt.subplots(figsize=(6, 4))
    g_arr = np.array([x[0] for x in fc_by_gain])
    fc_m = np.array([x[1] for x in fc_by_gain])
    fc_c = np.array([x[2] for x in fc_by_gain])
    fc_e = np.array([x[3] for x in fc_by_gain])
    ax.semilogx(g_arr, fc_m, 'o-', label='ngspice (spline cutoff)')
    ax.semilogx(g_arr, fc_e, '--', label='exact pole wb(A0+NG)/NG')
    ax.semilogx(g_arr, fc_c, ':', label='classic GBW/NG')
    ax.set_xlabel('|G|')
    ax.set_ylabel('closed-loop fc (Hz)')
    ax.set_title('Closed-loop bandwidth vs gain')
    ax.legend()
    save_fig(fig, 'v11_gbw_fc_vs_gain.png')

    fig, ax = plt.subplots(figsize=(6, 4))
    t_arr = np.array(pl2['time']).real
    ax.plot(t_arr * 1e6, np.array(pl2['v(out)']), lw=1.2, label='V(out) step')
    ax.set_xlabel('t (us)')
    ax.set_ylabel('V')
    ax.set_title(f'Slew-limited square response (SR={sr_spec/1e6:.1f} V/us)')
    save_fig(fig, 'v11_slew_transient.png')

    out = {
        "macromodel": {k: float(v) for k, v in p.items()},
        "gain": {"R1": R1, "Rf": Rf, "G": G, "NG": NG},
        "ac_vs_theory": {k: (float(v) if np.isscalar(v) else None)
                         for k, v in cmp.items() if k not in ("err_db", "err_deg")},
        "fc_Hz": {
            "sim_spline": fc_sim_prec,
            "sim_linear_interp": fc_sim_lin,
            "theory_spline": fc_th_prec,
            "theory_spline_dense_60perdec": fc_dense,
            "exact_pole": fc_pole_exact,
            "classic_GBW_over_NG": fc_classic,
            "classic_vs_exact_pct": (fc_classic / fc_pole_exact - 1) * 100,
            "linear_interp_bias_pct": (fc_sim_lin / fc_sim_prec - 1) * 100,
            "spline_15_bias_pct": (fc_th_prec / fc_pole_exact - 1) * 100,
            "spline_60_bias_pct": (fc_dense / fc_pole_exact - 1) * 100,
            "pole_as_minus3dB_convention_Hz": f3dB_from_pole(fc_pole_exact, -3.0),
            "spline_60_residual_vs_convention_pct": (fc_dense / f3dB_from_pole(fc_pole_exact, -3.0) - 1) * 100,
        },
        "fc_by_gain": [{"G": g, "fc_meas": m, "fc_classic": c, "fc_exact": e}
                       for g, m, c, e in fc_by_gain],
        "buffer_ac_max_err_db": cmpb["max_abs_err_db"],
        "slew": {"sim_V_per_s": slew, "spec_V_per_s": sr_spec,
                 "err_pct": (slew / sr_spec - 1) * 100,
                 "small_signal_max_slope_V_per_s": slew_small,
                 "small_signal_bandwidth_limited_pred_V_per_s": slew_linear,
                 "small_signal_ratio_to_SR": slew_small / sr_spec},
    }
    save_json("v11_nonideal_opamp.json", out)
    print("[A] done -> v11_nonideal_opamp.json")


if __name__ == "__main__":
    main()
