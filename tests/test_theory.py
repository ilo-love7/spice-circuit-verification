"""理论传递函数测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.theory.transfer import (rc_lowpass_H, rc_lowpass_tf, sk_lowpass_H,
                                 sk_lowpass_params, sk_bandpass_H,
                                 sk_bandpass_params, inverting_amp_H,
                                 compare_response, cutoff_from_response,
                                 mag_db)


def test_rc_lowpass_tf():
    R, C = 1e3, 1e-6
    fc = 1 / (2 * np.pi * R * C)
    # -3dB 点
    H = rc_lowpass_H(R, C, 2 * np.pi * fc)
    assert abs(abs(H) - 1 / np.sqrt(2)) < 1e-12
    assert abs(np.angle(H) - (-np.pi / 4)) < 1e-12
    # TransferFunction 系数（scipy 按首项归一 → den = [1, 1/RC]）
    tf = rc_lowpass_tf(R, C)
    den = np.asarray(tf.den)
    assert np.allclose(den, [1.0, 1.0 / (R * C)])


def test_sk_lowpass_butterworth():
    R = 1e4
    C2 = 11.25e-9
    C1 = 2 * C2
    f0, Q, dcg = sk_lowpass_params(R, R, C1, C2)
    assert abs(Q - 1 / np.sqrt(2)) < 1e-9
    assert abs(f0 - 1000.0) < 2.0
    assert dcg == 1.0
    # f0 处 |H| = Q（二阶低通归一）→ Butterworth: |H(f0)| = 1/sqrt(2)·... 实际 |H(jw0)|=1/(2Q)·... 校验与解析式一致即可
    H = sk_lowpass_H(R, R, C1, C2, 2 * np.pi * f0)
    assert abs(abs(H) - 1 / np.sqrt(2)) < 0.02  # Butterworth -3dB at f0


def test_sk_bandpass():
    R, C = 1e4, 1e-8
    f0, Q = sk_bandpass_params(R, C)
    assert abs(f0 - 1591.549) < 0.1
    assert abs(Q - 1 / 3) < 1e-12
    # 中心频率处 |H| = K/3·... 解析峰值 = K·Q = K/3
    H = sk_bandpass_H(R, C, 1.0, 2 * np.pi * f0)
    assert abs(abs(H) - 1 / 3) < 0.01


def test_inverting_amp():
    w = np.array([1.0, 100.0])
    H = inverting_amp_H(1e3, 1e4, w, A_ol=np.inf)
    assert np.allclose(H, -10.0)
    H2 = inverting_amp_H(1e3, 1e4, w, A_ol=1e6)
    assert abs(H2[0] - (-10 / (1 + 11 / 1e6))) < 1e-9


def test_compare_response():
    f = np.logspace(0, 3, 20)
    w = 2 * np.pi * f
    H = rc_lowpass_H(1e3, 1e-6, w)
    cmp = compare_response(f, H, H)
    assert cmp["max_abs_err_db"] == 0.0
    assert cmp["max_abs_err_deg"] == 0.0


def test_cutoff_from_response():
    R, C = 1e3, 1e-6
    fc = 1 / (2 * np.pi * R * C)
    f = np.logspace(1, 4, 200)
    mag = np.abs(rc_lowpass_H(R, C, 2 * np.pi * f))
    fc_est = cutoff_from_response(f, mag, -3.0)
    assert abs(fc_est - fc) / fc < 0.01
