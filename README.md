# N7 · SPICE 模拟电路自动化仿真验证

博远线上科研新项目提案 v2 · N7（电子 / EDA / 电路仿真 · ★★）。

Python 脚本化 EDA 验证流水线：**PySpice 建网表 → ngspice 仿真 → scipy 理论对比 → 容差蒙特卡洛**。
RC / Sallen-Key 低通·带通 / 反相放大器四类电路，AC + 瞬态全自动验证，±5% 元件容差良率曲线。

**当前版本 v1.2.0 / 2026-11-11**（v1.0 流水线 → v1.1 非理想运放·高阶滤波·容差方法学 → v1.2 MFB 带通·噪声·压摆瞬态）。

## 流水线

```
PySpice Circuit API ──┐
                      ├─→ SPICE 网表 ─→ ngspice_con.exe -b ─→ ASCII raw ─→ rawparse ─→ 对比/scipy 理论
字符串模板 builders ──┘                                                    └─→ 蒙特卡洛 ±5% → 良率曲线
```

## 核心结果（v1.0.0）

| 电路 | AC 最大幅值误差 | 说明 |
|---|---|---|
| 一阶 RC 低通 | 7.1e-15 dB | 机器精度 |
| Sallen-Key 低通（Butterworth f0=1 kHz） | 1.74e-05 dB | 有限开环增益 A=1e6 |
| 反相放大器（G=−10） | 0（vs 有限-A 理论） | −9.999890 精确命中 |
| Sallen-Key 带通（f0=1.59 kHz, Q=1/3） | 1.4e-14 dB | 机器精度 |

±5% 容差 → fc 散布 3.6–4.0%（与误差传播公式一致）；良率曲线见 `figures/mc_yield_curve.png`。
20 项单测全绿；`run_all.sh` 一键复现。（fc 提取的判据/插值陷阱见 v1.1 §7.4 与 `cutoff_precise`。）


## 核心结果（v1.1.0 深化）

| 深化组 | 关键结果 |
|---|---|
| A 非理想运放 | 单极点限流宏模型（GBW=10 MHz, SR=10 V/µs）：AC vs 精确理论 7.1e-15 dB；SR 实测 +0.028%；经典 GBW/NG 偏差精确 = −NG/A0；−3.00/−3.01 dB 判据换算 0.99763 解释全部 fc 残差 |
| B 4 阶 Butterworth 级联 | 两级 SK（Q=0.5412/1.3066）vs scipy.butter(4) 5.7e-14 dB；群延迟误差 1.4e-9 s；阶跃过冲 10.83%/4.32% 命中教科书值；SK 级联加载 0 dB（运放天然隔离，负结果）vs 无源 RC 3.52 dB |
| C 容差方法学 | WCA 16 角点 10.80% vs MC 最坏 10.28%；解析 σ(fc)/fc 2.8868% vs MC 2.8860%（比 1.00026）；f0 处 C1 灵敏度精确为 0（解析对消）；加严 C2→1% 良率 97.3%→100%（C1 几乎无效） |

36 项单测全绿（v1.0 20 + v1.1 16）；`run_all.sh` 覆盖 scripts/00–10。

## 核心结果（v1.2.0 深化）

| 深化组 | 关键结果 |
|---|---|
| E16 MFB 带通对比 | 多路反馈拓扑把带通 Q 从 SK 的**结构约束 1/3** 解放到 0.333–10；Q=10 时 AC vs 手推解析式 1.73e-3 dB、Q(3dB)=9.916；`H0 = 2Q²/(1+ρ)` 用 ρ 解耦增益（10→40，误差 0.0055%）。**高 Q 不免费**：f0 误差 <1% 所需 GBW 从 Q=1 的 0.159 MHz 升到 Q=10 的 1.561 MHz（9.84×），经典 `100·f0·Q²` 经验式在高 Q 下高估 10.2×；±5% 容差 MC / 解析误差传播 = 1.0024，Q 对 C1/C2 灵敏度精确为 0 |
| E17 噪声分析（`.noise`） | 补上报告 §6 遗留项：RC 的 √(4kTR) 与 √(kT/C) 被 ngspice 复现到 0.009–0.207%；二阶带通 ENBW = H0²·πf0/(2Q)，白噪声近似误差 ≤0.22%；输出总噪声 ~ Q^0.909（理论 Q¹）。运放 e_n 用同相输入串联等效电阻严格建模（AC 不变），阻抗标度下 e_n 项离散度 0.000%。**负结果**：无源 ∝ α^0.5、i_n ∝ α¹、e_n 恒定 ⇒ 总噪声在 α 上单调递增，**不存在内点最优**，噪声地板由 e_n 决定 |
| E18 压摆瞬态 / FPBW | 小信号带宽 GBW/NG = 909 kHz 与幅度无关，FPBW = SR/(2πVp) 跨幅度变化 100×；1 V 输入时前者是后者的 5.71×。THD 在统一 slew_ratio 轴上 collapse：THD@FPBW 仍仅 0.001%，1% 失真起始于 **1.183× / 1.165× FPBW**（离散度 1.51%）⇒ FPBW 是保守下界。低 SR 阶跃使 10–90% 上升时间从 341.75 µs 拉长到 1223.30 µs（3.58×） |

90 项单测全绿（v1.0 20 + v1.1 16 + v1.2 54）；`run_all.sh` 覆盖 scripts/00–14。
汇总表见 `results/v12_summary_tables.md`，9 张新图见 `figures/v12_*.png`。

## 环境（如实标注）

- **PySpice 1.5** 作网表前端；**ngspice-46 win64**（`ngspice_con.exe` 批处理）作引擎；自研 ASCII raw 解析。
- 社区 ngspice Windows 构建不含 `libngspice-0.dll`，PySpice 共享库模式不可用 → 批处理模式替代（详见报告 §1.2）。
- 引擎路径：`D:\项目\02_数据集\ngspice-46-win64\Spice64\bin\ngspice_con.exe`（可用环境变量 `NGSPICE_EXE` 覆盖）。

## 一键复现

```bash
bash run_all.sh                        # pytest（90 项）+ scripts/00–14 全流程
python -m pytest tests/ -q             # 90 项
python scripts/11_mfb_bandpass.py      # E16 MFB 带通 / GBW 退化 / 容差
python scripts/12_noise_analysis.py    # E17 .noise / kT/C / ENBW / 阻抗标度
python scripts/13_slew_distortion.py   # E18 THD / FPBW / 低 SR 阶跃
python scripts/14_summary_v12.py       # 由三个 JSON 重算汇总表
```

依赖见 `requirements.txt`（numpy / scipy / matplotlib / pytest / PySpice）。
`.noise` 输出为 ngspice 特有 raw 段（`Noise Spectral Density Curves` + `Integrated Noise`），解析见 `src/theory/noise.py`。

## 目录

```
configs/circuits.json     元件值与蒙特卡洛协议
src/engine/               ngspice 运行器 + raw 解析
src/circuits/             网表构建（字符串 + PySpice 双轨）
src/theory/transfer.py    解析传递函数与误差指标（+ cutoff_precise）
src/theory/nonideal.py    有限 GBW/压摆率运放理论（v1.1）
src/theory/cascade.py     4 阶级联与阶跃指标（v1.1）
src/theory/mfb.py         MFB 带通解析设计 / 灵敏度 / 向量化参数（v1.2）
src/theory/noise.py       4kTR / kT/C / ENBW / 噪声积分 / SNR（v1.2）
src/theory/distortion.py  FPBW / slew_ratio / THD / 波形指标（v1.2）
src/mc/tolerance.py       容差蒙特卡洛
src/mc/wca.py             WCA/解析传播/灵敏度/良率优化（v1.1）
src/circuits/nonideal_builders.py  非理想宏模型与级联网表（v1.1）
src/circuits/v12_builders.py       MFB / .noise / e_n 等效电阻 / 大信号网表（v1.2）
scripts/00–14             生产实验脚本（v1.0 00–06 + v1.1 07–10 + v1.2 11–14）
tests/                    90 项单测
reports/                  基准评测报告 + 工作记录
results/ figures/         结果 JSON 与 30 张图（v1.0 13 + v1.1 8 + v1.2 9）
```
