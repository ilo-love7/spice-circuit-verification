"""08 深化 B：4 阶 Butterworth = 两级 Sallen-Key 级联。

- Q1=0.5412, Q2=1.3066，f0=fc=1 kHz，等 R 设计
- 级间缓冲 vs 直接级联（负载效应）幅频偏差
- 群延迟理论 vs 仿真
- 阶跃响应过冲/上升/稳定时间（两级 vs 单级对照）
"""
from common import save_json, save_fig
import numpy as np
import matplotlib.pyplot as plt

from src.engine.runner import run_netlist
from src.theory.cascade import (BUTTER4_Q, design_sk_lp_equalR, butter4_lp_tf,
                                cascade_sk_H, step_response_metrics)
from src.theory.transfer import (compare_response, mag_db, phase_deg,
                                 sk_lowpass_H, cutoff_precise)
from src.theory.nonideal import group_delay
from src.circuits.nonideal_builders import sk_cascade4_netlist, sk_cascade4_tran_netlist


def main():
    f0 = 1e3
    Q1, Q2 = BUTTER4_Q
    Ra, R2a, C1a, C2a = design_sk_lp_equalR(f0, Q1, C2=1e-8)
    Rb, R2b, C1b, C2b = design_sk_lp_equalR(f0, Q2, C2=1e-8)
    print(f"[B] f0={f0} Hz  stageA(Q={Q1:.4f}): R={Ra:.4g} C1={C1a:.4g} C2={C2a:.4g}")
    print(f"            stageB(Q={Q2:.4f}): R={Rb:.4g} C1={C1b:.4g} C2={C2b:.4g}")

    fstart, fstop = 10.0, 1e6
    # ---- AC：缓冲级联 ----
    net_buf = sk_cascade4_netlist(Ra, R2a, C1a, C2a, Rb, R2b, C1b, C2b,
                                  fstart, fstop, buffered=True, points_per_decade=20)
    pb = run_netlist(net_buf).plot("AC Analysis")
    f = np.array(pb['frequency']).real
    H_buf = np.array(pb['v(out)'])

    # ---- AC：直接级联（无缓冲，级间负载） ----
    net_dir = sk_cascade4_netlist(Ra, R2a, C1a, C2a, Rb, R2b, C1b, C2b,
                                  fstart, fstop, buffered=False, points_per_decade=20)
    pd_ = run_netlist(net_dir).plot("AC Analysis")
    H_dir = np.array(pd_['v(out)'])

    # ---- 理论 ----
    w = 2 * np.pi * f
    H_th = cascade_sk_H(f0, w, Q1, Q2)
    # scipy butter4 校验（角频率）
    tf = butter4_lp_tf(f0)
    _, H_butter = tf.freqresp(w)
    cmp_buf = compare_response(f, H_buf, H_th)
    cmp_dir = compare_response(f, H_dir, H_th)
    cmp_butter = compare_response(f, H_th, H_butter)
    print(f"[B] buffered vs theory: max|err|={cmp_buf['max_abs_err_db']:.3e} dB")
    print(f"[B] direct   vs theory: max|err|={cmp_dir['max_abs_err_db']:.3e} dB  (loading)")
    print(f"[B] cascade-H vs scipy.butter4: max|err|={cmp_butter['max_abs_err_db']:.3e} dB")

    # 负载效应量化：幅值差（dB）统计
    load_err_db = mag_db(H_dir) - mag_db(H_buf)
    f_mask = f <= 1e4   # 0~10fc
    print(f"[B] SK loading deviation (0~10fc): max|Δ|={np.max(np.abs(load_err_db[f_mask])):.4f} dB")
    print("    note: SK stages are driven by ideal VCVS outputs -> inherent isolation (negative result)")

    # ---- 对照：无源 RC 两级级联的加载效应 ----
    Rc = 1.0 / (2 * np.pi * f0 * 1e-8)   # fc=1kHz, C=10nF
    net_rc_buf = f"""* RC cascade buffered
V1 in 0 DC 0 AC 1
R1 in mid {Rc!r}
C1 mid 0 1e-8
Ebuf mid2 0 mid 0 1
R2 mid2 out {Rc!r}
C2 out 0 1e-8
.ac dec 20 {fstart!r} {fstop!r}
.end
"""
    net_rc_dir = f"""* RC cascade direct
V1 in 0 DC 0 AC 1
R1 in mid {Rc!r}
C1 mid 0 1e-8
R2 mid out {Rc!r}
C2 out 0 1e-8
.ac dec 20 {fstart!r} {fstop!r}
.end
"""
    prc_b = run_netlist(net_rc_buf).plot("AC Analysis")
    prc_d = run_netlist(net_rc_dir).plot("AC Analysis")
    f_rc = np.array(prc_b['frequency']).real
    H_rc_buf = np.array(prc_b['v(out)'])
    H_rc_dir = np.array(prc_d['v(out)'])
    rc_load_err_db = mag_db(H_rc_dir) - mag_db(H_rc_buf)
    rc_mask = f_rc <= 1e4
    # 无源缓冲级联理论 = (1/(1+jwRC))^2
    w_rc = 2 * np.pi * f_rc
    H_rc_th = 1.0 / (1 + 1j * w_rc * Rc * 1e-8) ** 2
    cmp_rc = compare_response(f_rc, H_rc_buf, H_rc_th)
    print(f"[B] RC passive loading deviation (0~10fc): max|Δ|={np.max(np.abs(rc_load_err_db[rc_mask])):.4f} dB")
    print(f"[B] RC buffered vs theory: max|err|={cmp_rc['max_abs_err_db']:.3e} dB")

    # -3dB 带宽
    fc_buf = cutoff_precise(f, H_buf, -3.0)
    fc_dir = cutoff_precise(f, H_dir, -3.0)
    fc_th = cutoff_precise(f, H_th, -3.0)
    print(f"[B] fc(-3dB): th={fc_th:.2f}  buffered={fc_buf:.2f}  direct={fc_dir:.2f}")

    # ---- 群延迟 ----
    w_arr = 2 * np.pi * f
    tg_sim = group_delay(H_buf, w_arr)
    tg_th = group_delay(H_th, w_arr)
    f_mask2 = (f >= 100) & (f <= 1e4)
    tg_err = float(np.max(np.abs(tg_sim[f_mask2] - tg_th[f_mask2])))
    print(f"[B] group delay (100Hz~10kHz): max|err|={tg_err:.3e} s")
    # 群延迟在 f0 附近：理论值
    i0 = np.argmin(np.abs(f - f0))
    print(f"    tau_g(th, f0)={tg_th[i0]*1e3:.4f} ms  tau_g(sim, f0)={tg_sim[i0]*1e3:.4f} ms")

    # ---- 阶跃响应（缓冲级联 vs 单节 Q=0.707 对照） ----
    tstop = 8e-3
    net_step = sk_cascade4_tran_netlist(Ra, R2a, C1a, C2a, Rb, R2b, C1b, C2b,
                                        v_amp=1.0, tstop=tstop, buffered=True)
    ps = run_netlist(net_step).plot("Transient Analysis")
    t = np.array(ps['time']).real
    y = np.array(ps['v(out)'])
    metrics = step_response_metrics(t, y)
    print(f"[B] step: overshoot={metrics['overshoot_pct']:.2f}%  tr={metrics['rise_time_s']*1e6:.1f} us  ts(2%)={metrics['settling_2pct_s']*1e3:.3f} ms")

    # 单节 Q=0.707（Butterworth 2 阶）对照
    Rc1, Rc2, Cc1, Cc2 = design_sk_lp_equalR(f0, 0.7071, C2=1e-8)
    # 用级联构建器退化为单节：手工网表
    net_single = f"""* single SK LP Q=0.707
V1 in 0 DC 0 PULSE(0 1 0 {tstop/10000!r} {tstop/10000!r} {tstop!r} {2*tstop!r})
R1 in a1 {Rc1!r}
R2 a1 b1 {Rc2!r}
C1 a1 out {Cc1!r}
C2 b1 0 {Cc2!r}
E1 out 0 b1 out 1e6
.tran {tstop/4000!r} {tstop!r}
.end
"""
    psg = run_netlist(net_single).plot("Transient Analysis")
    tg_ = np.array(psg['time']).real
    yg = np.array(psg['v(out)'])
    metrics_single = step_response_metrics(tg_, yg)
    print(f"[B] single Q=0.707 step: overshoot={metrics_single['overshoot_pct']:.2f}%  tr={metrics_single['rise_time_s']*1e6:.1f} us")

    # ---- 图 ----
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    ax = axes[0, 0]
    ax.semilogx(f, mag_db(H_buf), label='buffered cascade', lw=1.5)
    ax.semilogx(f, mag_db(H_dir), '--', label='direct cascade', lw=1.2)
    ax.semilogx(f, mag_db(H_th), ':', label='theory (butter4)', lw=1.2)
    ax.set_ylabel('|H| (dB)')
    ax.set_title(f'4th-order Butterworth SK cascade (fc={f0} Hz)')
    ax.legend()
    ax = axes[0, 1]
    ax.semilogx(f, phase_deg(H_buf), label='buffered', lw=1.5)
    ax.semilogx(f, phase_deg(H_th), ':', label='theory', lw=1.2)
    ax.set_ylabel('Phase (deg)')
    ax.set_title('Phase')
    ax.legend()
    ax = axes[1, 0]
    ax.semilogx(f, load_err_db, lw=1.2, color='C3', label='SK cascade (isolated)')
    ax.semilogx(f_rc, rc_load_err_db, lw=1.2, color='C4', label='passive RC cascade')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('direct − buffered (dB)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title(f'Loading effect: SK max|Δ|={np.max(np.abs(load_err_db[f_mask])):.3f} vs RC {np.max(np.abs(rc_load_err_db[rc_mask])):.3f} dB')
    ax.legend()
    ax = axes[1, 1]
    ax.semilogx(f, tg_sim * 1e3, label='ngspice', lw=1.5)
    ax.semilogx(f, tg_th * 1e3, ':', label='theory', lw=1.2)
    ax.set_ylabel('Group delay (ms)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title(f'Group delay (max err {tg_err*1e6:.2f} us @100-10k)')
    ax.legend()
    save_fig(fig, 'v11_cascade4_ac.png')

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t * 1e3, y, lw=1.2, label=f'4th-order (Q={Q1:.2f}/{Q2:.2f})')
    ax.plot(tg_ * 1e3, yg, lw=1.2, label='single 2nd-order Q=0.707')
    ax.axhline(1, color='k', lw=0.5, ls=':')
    ax.set_xlabel('t (ms)')
    ax.set_ylabel('V')
    ax.set_title(f'Step response: overshoot {metrics["overshoot_pct"]:.1f}% vs {metrics_single["overshoot_pct"]:.1f}%')
    ax.legend()
    save_fig(fig, 'v11_cascade4_step.png')

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(f, mag_db(H_buf) - mag_db(H_th), lw=1.2, label='buffered − theory')
    ax.semilogx(f, mag_db(H_dir) - mag_db(H_th), lw=1.2, label='direct − theory')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude err (dB)')
    ax.set_title('Error vs theory')
    ax.legend()
    save_fig(fig, 'v11_cascade4_error.png')

    out = {
        "design": {"f0_Hz": f0, "Q1": Q1, "Q2": Q2,
                   "stageA": {"R": Ra, "R2": R2a, "C1": C1a, "C2": C2a},
                   "stageB": {"R": Rb, "R2": R2b, "C1": C1b, "C2": C2b}},
        "buffered_vs_theory": {k: float(v) for k, v in cmp_buf.items() if k not in ("err_db", "err_deg")},
        "direct_vs_theory": {k: float(v) for k, v in cmp_dir.items() if k not in ("err_db", "err_deg")},
        "cascade_vs_scipy_butter": {k: float(v) for k, v in cmp_butter.items() if k not in ("err_db", "err_deg")},
        "loading_max_abs_dev_db_0to10fc": float(np.max(np.abs(load_err_db[f_mask]))),
        "loading_note": "SK stages inherently isolated by op-amp output (VCVS); passive RC cascade shows real loading",
        "rc_passive_loading": {
            "max_abs_dev_db_0to10fc": float(np.max(np.abs(rc_load_err_db[rc_mask]))),
            "buffered_vs_theory_max_err_db": cmp_rc["max_abs_err_db"],
            "R": Rc, "C": 1e-8,
        },
        "fc_Hz": {"theory": fc_th, "buffered": fc_buf, "direct": fc_dir},
        "group_delay": {"max_err_s_100_to_10k": tg_err,
                        "tau_g_f0_theory_s": float(tg_th[i0]),
                        "tau_g_f0_sim_s": float(tg_sim[i0])},
        "step_4th": metrics,
        "step_single_Q0707": metrics_single,
    }
    save_json("v11_cascade4.json", out)
    print("[B] done -> v11_cascade4.json")


if __name__ == "__main__":
    main()
