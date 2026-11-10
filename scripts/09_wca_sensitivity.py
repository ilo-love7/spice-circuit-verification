"""09 深化 C：容差方法学 —— WCA / 解析误差传播 / 灵敏度排序 / 良率优化。

- 2^4=16 角点最坏情况分析（解析 fc）vs 蒙特卡洛分布
- 解析 σ(fc)/fc = tol/√3（均匀）vs MC 实测（区分元件容差与测量噪声）
- |H(f_spec)| 对 R1/R2/C1/C2 的数值灵敏度排序
- 良率优化：全部 5% vs 关键元件 1% 加严，解析抽样良率曲线
"""
from common import save_json, save_fig
import numpy as np
import matplotlib.pyplot as plt

from src.theory.transfer import sk_lowpass_H, sk_lowpass_params
from src.theory.cascade import design_sk_lp_equalR, BUTTER4_Q
from src.mc.wca import (sk_fc_corners, sk_fc_propagation_std,
                        sk_mag_sensitivity, sk_yield_analytic_design,
                        rc_fc_propagation_std)


def main():
    # 标称：SK 低通 f0=1 kHz, Q=0.5412（Butterworth 第一节）
    f0 = 1e3
    R1, R2, C1, C2 = design_sk_lp_equalR(f0, BUTTER4_Q[0], C2=1e-8)
    f0_th, Q_th, _ = sk_lowpass_params(R1, R2, C1, C2)
    print(f"[C] nominal: R1={R1:.4g} R2={R2:.4g} C1={C1:.4g} C2={C2:.4g}  f0={f0_th:.3f} Q={Q_th:.4f}")

    tol = 0.05
    # ---- WCA 16 角点 ----
    wca = sk_fc_corners(R1, R2, C1, C2, tol)
    print(f"[C] WCA fc: nom={wca['fc_nominal']:.3f}  min={wca['fc_min']:.3f}  max={wca['fc_max']:.3f}  worst={wca['fc_worst_dev_pct']:.3f}%")

    # ---- 解析传播 ----
    sigma_sk = sk_fc_propagation_std(tol, "uniform")
    print(f"[C] analytic sigma(fc)/fc (uniform ±5%) = {sigma_sk*100:.4f}%")

    # ---- MC（解析式 fc，分离测量噪声）----
    rng = np.random.default_rng(42)
    n = 50000
    scales = rng.uniform(1 - tol, 1 + tol, size=(n, 4))
    fc_mc = 1.0 / (2 * np.pi * np.sqrt(R1 * scales[:, 0] * R2 * scales[:, 1]
                                       * C1 * scales[:, 2] * C2 * scales[:, 3]))
    fc_mc_std_pct = float(np.std(fc_mc) / np.mean(fc_mc) * 100)
    fc_mc_minmax_pct = float(max(abs(fc_mc.min() - wca['fc_nominal']),
                                 abs(fc_mc.max() - wca['fc_nominal'])) / wca['fc_nominal'] * 100)
    print(f"[C] MC (formula fc, n={n}): std={fc_mc_std_pct:.4f}%  worst-sample dev={fc_mc_minmax_pct:.3f}%")
    print(f"    analytic/MC ratio = {sigma_sk*100/fc_mc_std_pct:.4f}")

    # ---- RC 对照（一阶，两元件）----
    R0, C0 = 1e3, 1e-6
    sigma_rc = rc_fc_propagation_std(R0, C0, tol, "uniform")
    scales2 = rng.uniform(1 - tol, 1 + tol, size=(n, 2))
    fc_rc = 1.0 / (2 * np.pi * R0 * scales2[:, 0] * C0 * scales2[:, 1])
    fc_rc_std_pct = float(np.std(fc_rc) / np.mean(fc_rc) * 100)
    print(f"[C] RC: analytic={sigma_rc*100:.4f}%  MC={fc_rc_std_pct:.4f}%")

    # ---- 灵敏度排序（|H| 在 f=0.7·fc 处，过渡带）----
    f_spec = 0.7 * f0
    sens = sk_mag_sensitivity(R1, R2, C1, C2, f_spec)
    print(f"[C] |H| sensitivity @ f_spec={f_spec:.0f} Hz:")
    for name, s in sens["ranked"]:
        print(f"    {name}: S = {s:+.4f}")

    # f0 处（Q 敏感区）
    sens_f0 = sk_mag_sensitivity(R1, R2, C1, C2, f0)
    print(f"[C] sensitivity @ f0={f0} Hz:")
    for name, s in sens_f0["ranked"]:
        print(f"    {name}: S = {s:+.4f}")

    # ---- 良率优化：|H(f_spec)| 偏差 <5% 判据 ----
    H_nom = abs(sk_lowpass_H(R1, R2, C1, C2, 2 * np.pi * f_spec))
    tol_map_full = {k: tol for k in ("R1", "R2", "C1", "C2")}
    y_full = sk_yield_analytic_design(R1, R2, C1, C2, f_spec, H_nom, 5.0, tol_map_full, n_samples=20000, seed=1)
    print(f"[C] yield all-5%: {y_full['yield']*100:.2f}%  (h_std={y_full['h_std']:.5f})")

    # 加严 C1 至 1%（C1 是 |H| 过渡带最敏感之一）
    tol_maps = {}
    for tighten in ("C1", "R1", "C2", "R2"):
        tm = {k: tol for k in ("R1", "R2", "C1", "C2")}
        tm[tighten] = 0.01
        y = sk_yield_analytic_design(R1, R2, C1, C2, f_spec, H_nom, 5.0, tm, n_samples=20000, seed=1)
        tol_maps[tighten] = y
        print(f"    tighten {tighten} to 1%: yield {y['yield']*100:.2f}%")

    # 全部 1%
    tm1 = {k: 0.01 for k in ("R1", "R2", "C1", "C2")}
    y_all1 = sk_yield_analytic_design(R1, R2, C1, C2, f_spec, H_nom, 5.0, tm1, n_samples=20000, seed=1)
    print(f"    all 1%: yield {y_all1['yield']*100:.2f}%")

    # 良率 vs 容差扫描（全部同步）
    tol_sweep = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10])
    yield_sweep = []
    for tt in tol_sweep:
        tm = {k: float(tt) for k in ("R1", "R2", "C1", "C2")}
        y = sk_yield_analytic_design(R1, R2, C1, C2, f_spec, H_nom, 5.0, tm, n_samples=20000, seed=1)
        yield_sweep.append(y["yield"])
        print(f"    tol=±{tt*100:.0f}%: yield={y['yield']*100:.1f}%")

    # 判据加严扫描（固定 ±5%）
    spec_sweep = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 10.0])
    yield_spec = []
    for sp in spec_sweep:
        y = sk_yield_analytic_design(R1, R2, C1, C2, f_spec, H_nom, float(sp), tol_map_full,
                                     n_samples=20000, seed=1)
        yield_spec.append(y["yield"])
        print(f"    spec=±{sp}%: yield={y['yield']*100:.1f}%")

    # ---- 图 ----
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    ax = axes[0, 0]
    ax.hist(fc_mc / wca['fc_nominal'], bins=60, density=True, alpha=0.75, label=f'MC (n={n})')
    ax.axvline(wca['fc_min'] / wca['fc_nominal'], color='C3', ls='--', label='WCA min/max')
    ax.axvline(wca['fc_max'] / wca['fc_nominal'], color='C3', ls='--')
    ax.axvline(1, color='k', lw=0.8)
    ax.set_xlabel('fc / fc_nominal')
    ax.set_ylabel('density')
    ax.set_title(f'fc distribution ±5% (MC std={fc_mc_std_pct:.2f}%)')
    ax.legend()
    ax = axes[0, 1]
    names = [k for k, _ in sens["ranked"]]
    vals = [abs(v) for _, v in sens["ranked"]]
    names2 = [k for k, _ in sens_f0["ranked"]]
    vals2 = [abs(v) for _, v in sens_f0["ranked"]]
    x = np.arange(4)
    ax.bar(x - 0.18, vals, 0.36, label=f'|S| @ {f_spec:.0f} Hz')
    ax.bar(x + 0.18, vals2, 0.36, label=f'|S| @ f0')
    ax.set_xticks(x)
    ax.set_xticklabels(["R1", "R2", "C1", "C2"])
    ax.set_ylabel('|d|H|/|H| ÷ dx/x|')
    ax.set_title('Component sensitivity ranking')
    ax.legend()
    ax = axes[1, 0]
    ax.plot(tol_sweep * 100, np.array(yield_sweep) * 100, 'o-', lw=1.5)
    ax.axhline(95, color='C3', ls=':', lw=1, label='95% target')
    ax.set_xlabel('component tolerance (±%)')
    ax.set_ylabel('yield (%)')
    ax.set_title('Yield vs tolerance (spec: |H| err <5%)')
    ax.legend()
    ax = axes[1, 1]
    labels = ['all 5%', 'C1 1%', 'R1 1%', 'C2 1%', 'R2 1%', 'all 1%']
    ys = [y_full['yield']] + [tol_maps[k]['yield'] for k in ("C1", "R1", "C2", "R2")] + [y_all1['yield']]
    ax.bar(labels, np.array(ys) * 100, color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
    ax.set_ylabel('yield (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Yield improvement by selective tightening')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    save_fig(fig, 'v11_wca_yield.png')

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(fc_mc / wca['fc_nominal'], bins=80, density=True, alpha=0.6, label='SK fc (4 comps)')
    # 标准正态对照
    xs = np.linspace(0.9, 1.1, 400)
    ax.plot(xs, np.exp(-0.5 * ((xs - 1) / (sigma_sk)) ** 2) / (sigma_sk * np.sqrt(2 * np.pi)),
            'k--', lw=1, label=f'analytic σ={sigma_sk*100:.2f}%')
    ax.set_xlabel('fc / nominal')
    ax.set_ylabel('density')
    ax.set_title('MC vs analytic propagation (uniform ±5%)')
    ax.legend()
    save_fig(fig, 'v11_propagation_check.png')

    out = {
        "nominal": {"R1": R1, "R2": R2, "C1": C1, "C2": C2,
                    "f0_Hz": f0_th, "Q": Q_th, "f_spec_Hz": f_spec,
                    "H_f_spec_nominal": H_nom},
        "tol": tol,
        "wca": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                for k, v in wca.items()},
        "propagation": {
            "analytic_sigma_fc_pct": sigma_sk * 100,
            "mc_formula_fc_std_pct": fc_mc_std_pct,
            "mc_worst_sample_dev_pct": fc_mc_minmax_pct,
            "analytic_over_mc_ratio": sigma_sk * 100 / fc_mc_std_pct,
            "rc_analytic_sigma_pct": sigma_rc * 100,
            "rc_mc_std_pct": fc_rc_std_pct,
        },
        "sensitivity_f_spec": {n: s for n, s in sens["sensitivity"].items()},
        "sensitivity_f0": {n: s for n, s in sens_f0["sensitivity"].items()},
        "yield_full_5pct": y_full,
        "yield_tighten": {k: v for k, v in tol_maps.items()},
        "yield_all_1pct": y_all1,
        "yield_tol_sweep": {"tol_pct": (tol_sweep * 100).tolist(),
                            "yield": yield_sweep},
        "yield_spec_sweep": {"spec_pct": spec_sweep.tolist(), "yield": yield_spec},
    }
    save_json("v11_wca_sensitivity.json", out)
    print("[C] done -> v11_wca_sensitivity.json")


if __name__ == "__main__":
    main()
