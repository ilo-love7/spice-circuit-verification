# -*- coding: utf-8 -*-
"""E18 · 大信号压摆瞬态：小信号带宽 vs 全功率带宽的分离（v1.2）。

v1.1 §7.5 只测了方波斜率（SR 实测 +0.028%）；本轮补：
  1. 统一频率网格 × 幅度网格的 THD/Vpp/斜率，全部 collapse 到 slew_ratio
  2. 失真起始频率（THD=1%）vs 解析 FPBW = SR/(2πVp)
  3. 低 SR 阶跃扫描：建立时间如何被 ΔV/SR 下限拖长
  4. 小信号闭环带宽 GBW/NG 与 FPBW 的幅度依赖对照
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import load_config, save_json, save_fig

from src.theory import distortion as dz
from src.theory.cascade import step_response_metrics
from src.theory.nonideal import design_macromodel, closed_loop_pole_approx_Hz
from src.circuits import v12_builders as vb
from src.engine.runner import run_netlist

CFG = load_config()["v12"]["slew"]
GBW, A0, SR = CFG["GBW_Hz"], CFG["A0"], CFG["SR_V_per_s"]
R1, RF = CFG["R1"], CFG["Rf"]
GAIN = RF / R1
NG = 1.0 + GAIN
THR = CFG["thd_threshold_pct"]


def tran_of(v_amp, f, sr=SR):
    p = design_macromodel(GBW, A0, sr)
    res = run_netlist(vb.sine_slew_netlist(
        R1, RF, p["gm"], p["Rlag"], p["Cc"], p["Imax"],
        v_amp, f, CFG["n_cycles"], CFG["pts_per_cycle"]))
    tr = res.plot("Transient")
    return np.real(tr["time"]), np.real(tr["v(out)"])


def onset_frequency(fs, thds, thr):
    """log-log 插值求 THD 首次越过 thr 的频率。"""
    for i in range(1, len(fs)):
        a, b = max(thds[i - 1], 1e-12), max(thds[i], 1e-12)
        if a < thr <= b:
            t = (np.log(thr) - np.log(a)) / (np.log(b) - np.log(a))
            return float(np.exp(np.log(fs[i - 1])
                               + t * (np.log(fs[i]) - np.log(fs[i - 1]))))
    return float("nan")


def main():
    out = {"config": CFG, "gain": GAIN, "noise_gain": NG}
    fpbw = {a: dz.full_power_bandwidth_hz(SR, GAIN * a) for a in CFG["v_amp_list"]}
    f_cl = closed_loop_pole_approx_Hz(GBW, NG)
    out["small_signal_bw_Hz"] = f_cl

    # ---------- 1. 幅度 × 频率网格 ----------
    print("[E18] 1/3 THD grid")
    grid = []
    for v_amp in CFG["v_amp_list"]:
        for f in CFG["f_grid_Hz"]:
            t, y = tran_of(v_amp, f)
            thd = dz.thd_percent(t, y, f_fund=f, n_harm=CFG["n_harm"])
            wm = dz.waveform_metrics(t, y)
            grid.append({
                "v_amp": float(v_amp), "f_Hz": float(f),
                "fpbw_Hz": fpbw[v_amp],
                "f_over_fpbw": float(f / fpbw[v_amp]),
                "f_over_small_signal_bw": float(f / f_cl),
                "slew_ratio": dz.slew_ratio(f, GAIN * v_amp, SR),
                "thd_pct": thd["thd_pct"], "vpp": wm["vpp"], "vrms": wm["vrms"],
                "max_slope_V_per_us": wm["max_abs_slope_v_per_us"],
            })
        sel = [g for g in grid if g["v_amp"] == v_amp]
        print(f"   v_amp={v_amp:5.2f}  FPBW={fpbw[v_amp]/1e3:9.2f} kHz  "
              f"THD@3MHz={sel[-1]['thd_pct']:8.3f}%  "
              f"max|slope|={max(g['max_slope_V_per_us'] for g in sel):7.3f} V/us")
    out["grid"] = grid

    # ---------- 2. 失真起始 vs 解析 FPBW ----------
    print("[E18] 2/3 distortion onset")
    onset = []
    for v_amp in CFG["v_amp_list"]:
        sel = [g for g in grid if g["v_amp"] == v_amp]
        fs = [g["f_Hz"] for g in sel]
        ts = [g["thd_pct"] for g in sel]
        fo = onset_frequency(fs, ts, THR)
        onset.append({
            "v_amp": float(v_amp), "fpbw_Hz": fpbw[v_amp],
            "onset_f_Hz": fo,
            "onset_over_fpbw": fo / fpbw[v_amp] if np.isfinite(fo) else None,
            "max_thd_pct": float(max(ts)),
        })
        r = onset[-1]["onset_over_fpbw"]
        print(f"   v_amp={v_amp:5.2f}  onset={fo/1e3 if np.isfinite(fo) else float('nan'):9.2f} kHz  "
              f"FPBW={fpbw[v_amp]/1e3:9.2f} kHz  ratio="
              f"{r if r is not None else float('nan')}")
    out["onset"] = onset
    valid = [o["onset_over_fpbw"] for o in onset if o["onset_over_fpbw"] is not None]
    out["onset_over_fpbw_mean"] = float(np.mean(valid)) if valid else None

    # ---------- 2b. FPBW 附近细扫：验证 THD 是否真的 collapse ----------
    print("[E18] 2b fine onset scan near FPBW")
    fine_ratios = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5])
    fine = []
    for v_amp in (0.5, 1.0):
        fs = fpbw[v_amp] * fine_ratios
        thds = []
        for f in fs:
            t, y = tran_of(v_amp, f)
            thds.append(dz.thd_percent(t, y, f_fund=f,
                                       n_harm=CFG["n_harm"])["thd_pct"])
        fo = onset_frequency(list(fs), thds, THR)
        rec = {"v_amp": float(v_amp), "fpbw_Hz": fpbw[v_amp],
               "ratios": fine_ratios.tolist(), "thd_pct": thds,
               "onset_f_Hz": fo,
               "onset_ratio": fo / fpbw[v_amp] if np.isfinite(fo) else None}
        fine.append(rec)
        print(f"   v_amp={v_amp:4.2f}  THD@1.0x={thds[0]:6.3f}%  "
              f"@1.4x={thds[2]:6.3f}%  @2.5x={thds[-1]:6.3f}%  "
              f"onset ratio={rec['onset_ratio']}")
    out["fine_onset"] = fine
    fr = [r["onset_ratio"] for r in fine if r["onset_ratio"] is not None]
    out["fine_onset_ratio_mean"] = float(np.mean(fr)) if fr else None
    out["fine_onset_ratio_spread_pct"] = (
        float((max(fr) - min(fr)) / np.mean(fr) * 100) if len(fr) > 1 else None)

    # ---------- 3. 低 SR 阶跃：建立时间延长 ----------
    print("[E18] 3/3 step response vs SR")
    step_rows, step_curves = [], {}
    for sr_us in CFG["step_sr_list_V_per_us"]:
        sr = sr_us * 1e6
        p = design_macromodel(GBW, A0, sr)
        res = run_netlist(vb.sk_lp_slew_step_netlist(
            CFG["sk_R1"], CFG["sk_R2"], CFG["sk_C1"], CFG["sk_C2"],
            p["gm"], p["Rlag"], p["Cc"], p["Imax"],
            CFG["step_v_fixed"], CFG["step_tstop_s"]))
        tr = res.plot("Transient")
        t = np.real(tr["time"])
        y = np.real(tr["v(out)"])
        m = step_response_metrics(t, y)
        step_rows.append({
            "sr_V_per_us": float(sr_us),
            "slew_limited_rise_s": dz.slew_limited_rise_time_s(CFG["step_v_fixed"], sr),
            "rise_time_s": m["rise_time_s"],
            "settling_2pct_s": m["settling_2pct_s"],
            "overshoot_pct": m["overshoot_pct"],
            "final_V": m["final"],
        })
        if sr_us in (0.005, 0.05, 10.0):
            step_curves[sr_us] = (t, y)
        print(f"   SR={sr_us:7.3f} V/us  dV/SR={step_rows[-1]['slew_limited_rise_s']*1e6:8.2f} us  "
              f"t_r={m['rise_time_s']*1e6 if np.isfinite(m['rise_time_s']) else float('nan'):8.2f} us  "
              f"t_s={m['settling_2pct_s']*1e6 if np.isfinite(m['settling_2pct_s']) else float('nan'):8.2f} us")
    linear_rise = step_rows[-1]["rise_time_s"]
    for r in step_rows:
        r["rise_ratio_vs_linear"] = (r["rise_time_s"] / linear_rise
                                     if np.isfinite(r["rise_time_s"]) and np.isfinite(linear_rise)
                                     else None)
    out["step"] = step_rows
    out["bandwidth_comparison"] = {
        "small_signal_bw_Hz": f_cl,
        "fpbw_by_v_amp_Hz": {str(k): v for k, v in fpbw.items()},
        "fpbw_ratio_max_over_min": max(fpbw.values()) / min(fpbw.values()),
        "small_signal_over_fpbw_at_1V": f_cl / fpbw[max(fpbw)],
    }
    out["verdict"] = {
        "thd_collapse_onset_slew_ratio": 1.0,
        "SR_V_per_us": SR / 1e6,
        "max_slope_ceiling_V_per_us": max(g["max_slope_V_per_us"] for g in grid),
    }

    save_json("v12_slew_distortion.json", out)

    # ---------- 图 1：collapsed THD curve ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for v_amp in CFG["v_amp_list"]:
        sel = [g for g in grid if g["v_amp"] == v_amp]
        ax.loglog([g["slew_ratio"] for g in sel],
                  [max(g["thd_pct"], 1e-4) for g in sel], "o-", lw=1.5,
                  label=f"Vin={v_amp:g} V")
    for rec in fine:
        ax.loglog(rec["ratios"], [max(v, 1e-4) for v in rec["thd_pct"]],
                  "*--", ms=5, lw=1.0,
                  label=f"fine scan Vin={rec['v_amp']:g} V")
    ax.axvline(1.0, color="crimson", ls="--", lw=1.3, label="FPBW: 2*pi*f*Vp = SR")
    ax.axhline(THR, color="gray", ls=":", lw=1.1, label="THD = 1%")
    ax.set_xlabel("slew ratio  2*pi*f*Vpeak / SR")
    ax.set_ylabel("THD (%)")
    ax.set_title("E18: THD collapses onto the slew-ratio axis")
    ax.legend(fontsize=7.5, ncol=2)
    save_fig(fig, "v12_thd_vs_slew_ratio.png")

    # ---------- 图 2：THD 热图 + FPBW 理论线 ----------
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    amps = np.array(CFG["v_amp_list"])
    fs = np.array(CFG["f_grid_Hz"])
    Z = np.full((len(amps), len(fs)), np.nan)
    for i, a in enumerate(CFG["v_amp_list"]):
        sel = [g for g in grid if g["v_amp"] == a]
        Z[i, :] = [g["thd_pct"] for g in sel]
    pc = ax.pcolormesh(fs, amps, np.maximum(Z, 1e-3), shading="nearest",
                       norm=LogNorm(vmin=1e-3, vmax=100),
                       cmap="inferno")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot([fpbw[a] for a in amps], amps, "c--", lw=2.0,
            label="FPBW = SR/(2*pi*Vp)")
    ax.axvline(f_cl, color="lime", ls=":", lw=1.5, label="small-signal bw = GBW/NG")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("input amplitude (V)")
    ax.set_title("E18: THD map and full-power bandwidth")
    fig.colorbar(pc, ax=ax, label="THD (%)")
    ax.legend(fontsize=8, loc="lower left")
    save_fig(fig, "v12_fpbw_map.png")

    # ---------- 图 3：阶跃与建立时间 ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for sr_us, (t, y) in step_curves.items():
        axes[0].plot(t * 1e3, y, lw=1.5, label=f"SR={sr_us:g} V/us")
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("V(out) (V)")
    axes[0].set_title("E18: large-signal step waveforms")
    axes[0].legend(fontsize=8)
    srs = [r["sr_V_per_us"] for r in step_rows]
    axes[1].loglog(srs, [max(r["rise_time_s"], 1e-9) * 1e6 for r in step_rows],
                   "o-", label="measured 10-90% rise (us)")
    axes[1].loglog(srs, [r["slew_limited_rise_s"] * 1e6 for r in step_rows],
                   "s--", label="dV/SR lower bound (us)")
    axes[1].axhline(linear_rise * 1e6, color="gray", ls=":",
                    label="linear (high-SR) baseline")
    axes[1].set_xlabel("SR (V/us)")
    axes[1].set_ylabel("rise time (us)")
    axes[1].set_title("E18: settling stretched by slew limit")
    axes[1].legend(fontsize=7.5)
    save_fig(fig, "v12_step_slew.png")

    print("[E18] done")


if __name__ == "__main__":
    main()
