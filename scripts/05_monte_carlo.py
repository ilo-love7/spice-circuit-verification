"""05 ±5% 元件容差蒙特卡洛：RC 低通 + Sallen-Key 低通，良率曲线。"""
from common import load_config, save_json, save_fig
import numpy as np
import matplotlib.pyplot as plt

from src.mc.tolerance import run_rc_mc, run_sk_lp_mc, yield_curve


def main():
    cfg_all = load_config()
    mc = cfg_all["monte_carlo"]
    rng = np.random.default_rng(mc["seed"])
    tol = mc["tol"]
    n = mc["n_trials"]
    print(f"[MC] tol=±{tol*100:.0f}% n={n} dist={mc['dist']}")

    # ---- RC 低通 ----
    rc = cfg_all["rc_lowpass"]
    print("  RC lowpass Monte Carlo ...")
    rc_res = run_rc_mc(rc["R"], rc["C"], tol, n, mc["fstart"], mc["fstop"],
                       f_spec=rc["f_spec"], rng=rng, dist=mc["dist"])
    print(f"    fc: {rc_res['fc_mean']:.2f} ± {rc_res['fc_std']:.2f} Hz (theory {rc_res['fc_theory']:.2f})")
    print(f"    |H(f_spec)|: {rc_res['h_spec_mean']:.5f} ± {rc_res['h_spec_std']:.5f} (theory {rc_res['h_spec_theory']:.5f})")
    print(f"    yield: {rc_res['yield']*100:.1f}%")

    # ---- Sallen-Key 低通 ----
    sk = cfg_all["sk_lowpass"]
    print("  Sallen-Key LP Monte Carlo ...")
    sk_res = run_sk_lp_mc(sk["R1"], sk["R2"], sk["C1"], sk["C2"], tol, n,
                          mc["fstart"], mc["fstop"], f_spec=sk["f0_target"],
                          rng=rng, dist=mc["dist"])
    print(f"    fc: {sk_res['fc_mean']:.2f} ± {sk_res['fc_std']:.2f} Hz (f0 theory {sk_res['f0_theory']:.2f})")
    print(f"    yield: {sk_res['yield']*100:.1f}%")

    # ---- 图 1：fc 分布直方图 ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.hist(rc_res["fc_Hz"], bins=30, color="C0", alpha=0.75, edgecolor="white")
    ax.axvline(rc_res["fc_theory"], color="k", ls="--", lw=1.2, label=f"theory {rc_res['fc_theory']:.1f} Hz")
    ax.axvline(rc_res["fc_mean"], color="C3", ls="-", lw=1.2, label=f"mean {rc_res['fc_mean']:.1f} Hz")
    ax.set_xlabel("fc (Hz)")
    ax.set_ylabel("count")
    ax.set_title(f"RC LP fc distribution (±{tol*100:.0f}%, n={n})")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(sk_res["fc_Hz"], bins=30, color="C2", alpha=0.75, edgecolor="white")
    ax.axvline(sk_res["f0_theory"], color="k", ls="--", lw=1.2, label=f"f0 theory {sk_res['f0_theory']:.1f} Hz")
    ax.axvline(sk_res["fc_mean"], color="C3", ls="-", lw=1.2, label=f"mean {sk_res['fc_mean']:.1f} Hz")
    ax.set_xlabel("fc (Hz)")
    ax.set_title(f"SK LP fc distribution (±{tol*100:.0f}%, n={n})")
    ax.legend(fontsize=8)
    save_fig(fig, "mc_fc_histograms.png")

    # ---- 图 2：|H(f_spec)| 分布 ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.hist(rc_res["h_spec"], bins=30, color="C0", alpha=0.75, edgecolor="white")
    ax.axvline(rc_res["h_spec_theory"], color="k", ls="--", lw=1.2, label="theory")
    ax.set_xlabel(f"|H| @ {rc_res['f_spec']:.1f} Hz")
    ax.set_title(f"RC LP |H(f_spec)| (yield {rc_res['yield']*100:.1f}%)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(sk_res["h_spec"], bins=30, color="C2", alpha=0.75, edgecolor="white")
    ax.axvline(1.0, color="k", ls="--", lw=1.2, label="ideal passband 1.0")
    ax.set_xlabel(f"|H| @ {sk_res['f_spec']:.1f} Hz")
    ax.set_title(f"SK LP |H(f_spec)| (yield {sk_res['yield']*100:.1f}%)")
    ax.legend(fontsize=8)
    save_fig(fig, "mc_hspec_histograms.png")

    # ---- 图 3：幅频特性良率曲线 ----
    f_grid = np.logspace(np.log10(mc["fstart"]), np.log10(mc["fstop"]), 60)
    yc = yield_curve({"RC LP": rc_res, "SK LP": sk_res}, f_grid, spec_db=1.0)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for label, vals in yc["curves"].items():
        ax.semilogx(f_grid, np.array(vals) * 100, lw=1.8, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Yield (%)  — |H| within ±1 dB of nominal")
    ax.set_title(f"Amplitude-response yield curve (tolerance ±{tol*100:.0f}%, n={n})")
    ax.set_ylim(0, 105)
    ax.legend()
    save_fig(fig, "mc_yield_curve.png")

    # ---- 图 4：蒙特卡洛幅频曲线族（抽 60 条）----
    # 用 fc 分布合成一阶响应族
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, res, label, color in [
        (axes[0], rc_res, "RC LP", "C0"),
        (axes[1], sk_res, "SK LP", "C2"),
    ]:
        f_plot = np.logspace(np.log10(mc["fstart"]), np.log10(mc["fstop"]), 80)
        fr = f_plot[None, :] / res["fc_Hz"][:60, None]
        mag_db_v = -10 * np.log10(1 + fr**2)
        ax.semilogx(f_plot, mag_db_v.T, color=color, alpha=0.12, lw=0.8)
        nominal = -10 * np.log10(1 + (f_plot / res["fc_theory"]) ** 2) if label == "RC LP" else \
            -10 * np.log10(1 + (f_plot / res["f0_theory"]) ** 2)
        ax.semilogx(f_plot, nominal, "k--", lw=1.5, label="nominal")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("|H| (dB)")
        ax.set_title(f"{label} response family ({len(res['fc_Hz'])} trials)")
        ax.legend(fontsize=8)
    save_fig(fig, "mc_response_family.png")

    # JSON：数组转 list
    def slim(d):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    save_json("mc_results.json", {
        "tol": tol, "n_trials": n, "dist": mc["dist"], "seed": mc["seed"],
        "rc_lp": slim(rc_res), "sk_lp": slim(sk_res),
        "yield_curve": yc,
    })


if __name__ == "__main__":
    main()
