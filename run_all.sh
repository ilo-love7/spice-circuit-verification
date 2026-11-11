#!/usr/bin/env bash
# N7 SPICE 电路自动化仿真验证 — 一键复现（v1.0 + v1.1 + v1.2）
# 依赖: D:/项目/venvs/flsoh-py312 (numpy scipy matplotlib pytest pyspice)
# 引擎: D:/项目/02_数据集/ngspice-46-win64/Spice64/bin/ngspice_con.exe
set -e
cd "$(dirname "$0")"
PY="D:/项目/venvs/flsoh-py312/Scripts/python.exe"

echo "== [1/3] pytest =="
"$PY" -m pytest tests/ -q

echo "== [2/3] scripts 00-14 =="
for s in scripts/00_smoke.py scripts/01_rc_lowpass.py scripts/02_sallen_key_lp.py \
         scripts/03_inverting_amp.py scripts/04_sallen_key_bp.py \
         scripts/05_monte_carlo.py scripts/06_summary.py \
         scripts/07_nonideal_opamp.py scripts/08_cascade_filter.py \
         scripts/09_wca_sensitivity.py scripts/10_summary_v11.py \
         scripts/11_mfb_bandpass.py scripts/12_noise_analysis.py \
         scripts/13_slew_distortion.py scripts/14_summary_v12.py; do
  echo "-- $s"
  "$PY" "$s"
done

echo "== [3/3] done =="
echo "results/ 与 figures/ 已更新。"
