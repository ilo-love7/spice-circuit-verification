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
20 项单测全绿；`run_all.sh` 一键复现。（fc 提取的判据/插值陷阱见 v1.1 §7.4 与 `cutoff_precise`。）


## 核心结果（v1.1.0 深化）

| 深化组 | 关键结果 |
|---|---|
| A 非理想运放 | 单极点限流宏模型（GBW=10 MHz, SR=10 V/µs）：AC vs 精确理论 7.1e-15 dB；SR 实测 +0.028%；经典 GBW/NG 偏差精确 = −NG/A0；−3.00/−3.01 dB 判据换算 0.99763 解释全部 fc 残差 |
| B 4 阶 Butterworth 级联 | 两级 SK（Q=0.5412/1.3066）vs scipy.butter(4) 5.7e-14 dB；群延迟误差 1.4e-9 s；阶跃过冲 10.83%/4.32% 命中教科书值；SK 级联加载 0 dB（运放天然隔离，负结果）vs 无源 RC 3.52 dB |
| C 容差方法学 | WCA 16 角点 10.80% vs MC 最坏 10.28%；解析 σ(fc)/fc 2.8868% vs MC 2.8860%（比 1.00026）；f0 处 C1 灵敏度精确为 0（解析对消）；加严 C2→1% 良率 97.3%→100%（C1 几乎无效） |

36 项单测全绿（v1.0 20 + v1.1 16）；`run_all.sh` 覆盖 scripts/00–10。

## 环境（如实标注）

- **PySpice 1.5** 作网表前端；**ngspice-46 win64**（`ngspice_con.exe` 批处理）作引擎；自研 ASCII raw 解析。
- 社区 ngspice Windows 构建不含 `libngspice-0.dll`，PySpice 共享库模式不可用 → 批处理模式替代（详见报告 §1.2）。
- 引擎路径：`D:\项目\02_数据集\ngspice-46-win64\Spice64\bin\ngspice_con.exe`（可用环境变量 `NGSPICE_EXE` 覆盖）。

## 一键复现

```bash
bash run_all.sh   # pytest（36 项）+ scripts/00–10
```

依赖见 `requirements.txt`（numpy / scipy / matplotlib / pytest / PySpice）。

## 目录

```
configs/circuits.json     元件值与蒙特卡洛协议
src/engine/               ngspice 运行器 + raw 解析
src/circuits/             网表构建（字符串 + PySpice 双轨）
src/theory/transfer.py    解析传递函数与误差指标（+ cutoff_precise）
src/theory/nonideal.py    有限 GBW/压摆率运放理论（v1.1）
src/theory/cascade.py     4 阶级联与阶跃指标（v1.1）
src/mc/tolerance.py       容差蒙特卡洛
src/mc/wca.py             WCA/解析传播/灵敏度/良率优化（v1.1）
src/circuits/nonideal_builders.py  非理想宏模型与级联网表（v1.1）
scripts/00–10             生产实验脚本（v1.0 00–06 + v1.1 07–10）
tests/                    36 项单测
reports/                  基准评测报告 + 工作记录
results/ figures/         结果 JSON 与 21 张图（v1.0 13 + v1.1 8）
```
