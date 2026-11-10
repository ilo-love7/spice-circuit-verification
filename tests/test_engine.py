"""引擎测试：raw 解析 + ngspice 运行器。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.engine.rawparse import parse_raw_text, find_plot
from src.engine.runner import run_netlist
from src.circuits.builders import rc_lowpass_netlist


RAW_SAMPLE = """Title: * test
Date: now
Command: ngspice-46
Plotname: AC Analysis
Flags: complex
No. Variables: 3
No. Points: 2
Variables:
	0	frequency	frequency	grid=3
	1	v(in)	voltage
	2	v(out)	voltage
Values:
0		1.000000000000000e+00,0.000000000000000e+00
	1.000000000000000e+00,0.000000000000000e+00
	5.000000000000000e-01,-1.000000000000000e-01
1		1.000000000000000e+01,0.000000000000000e+00
	1.000000000000000e+00,0.000000000000000e+00
	1.000000000000000e-01,-9.950000000000000e-01
"""


def test_parse_raw_text():
    plots = parse_raw_text(RAW_SAMPLE)
    assert len(plots) == 1
    p = plots[0]
    assert p.name == "AC Analysis"
    assert set(p.data.keys()) == {"frequency", "v(in)", "v(out)"}
    assert p.npoints == 2
    f = np.array(p["frequency"]).real
    assert np.allclose(f, [1.0, 10.0])
    vout = np.array(p["v(out)"])
    assert np.allclose(vout, [0.5 - 0.1j, 0.1 - 0.995j])


def test_find_plot():
    plots = parse_raw_text(RAW_SAMPLE)
    assert find_plot(plots, "ac analysis") is plots[0]
    assert find_plot(plots, "transient") is None


def test_run_netlist_rc():
    net = rc_lowpass_netlist(1e3, 1e-6, 1, 1e5, points_per_decade=5)
    res = run_netlist(net)
    p = res.plot("AC Analysis")
    f = np.array(p["frequency"]).real
    H = np.array(p["v(out)"])
    assert len(f) >= 10
    # DC 附近 |H|≈1
    assert abs(abs(H[0]) - 1.0) < 0.01
    # 单调衰减
    assert abs(H[-1]) < abs(H[0])
