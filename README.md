# N7 · SPICE 模拟电路自动化仿真验证

博远线上科研新项目提案 v2 · N7（电子 / EDA / 电路仿真 · ★★）。

Python 脚本化 EDA 验证流水线：**PySpice 建网表 → ngspice 仿真 → scipy 理论对比 → 容差蒙特卡洛**。
RC / Sallen-Key 低通·带通 / 反相放大器四类电路，AC + 瞬态全自动验证，±5% 元件容差良率曲线。

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
20 项单测全绿；`run_all.sh` 一键复现。

## 环境（如实标注）

- **PySpice 1.5** 作网表前端；**ngspice-46 win64**（`ngspice_con.exe` 批处理）作引擎；自研 ASCII raw 解析。
- 社区 ngspice Windows 构建不含 `libngspice-0.dll`，PySpice 共享库模式不可用 → 批处理模式替代（详见报告 §1.2）。
- 引擎路径：`D:\项目\02_数据集\ngspice-46-win64\Spice64\bin\ngspice_con.exe`（可用环境变量 `NGSPICE_EXE` 覆盖）。

## 一键复现

```bash
bash run_all.sh   # pytest（20 项）+ scripts/00–06
```

依赖见 `requirements.txt`（numpy / scipy / matplotlib / pytest / PySpice）。

## 目录

```
configs/circuits.json     元件值与蒙特卡洛协议
src/engine/               ngspice 运行器 + raw 解析
src/circuits/             网表构建（字符串 + PySpice 双轨）
src/theory/transfer.py    解析传递函数与误差指标
src/mc/tolerance.py       容差蒙特卡洛
scripts/00–06             生产实验脚本
tests/                    20 项单测
reports/                  基准评测报告 + 工作记录
results/ figures/         结果 JSON 与 13 张图
```
