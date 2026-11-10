"""PySpice 网表构建（提案要求的 EDA 流水线入口）。

用 PySpice 的 Circuit API 构建与 builders.py 等价的电路，
输出网表字符串后交给 ngspice 批处理引擎。
（PySpice 1.5 + ngspice-46 社区 win64 无共享库 DLL，
故不走 NgSpiceShared，只用 PySpice 做网表前端。
PySpice 1.5 的 VoltageSource 未实现 ac 参数，
生成网表后以文本方式补 "AC 1" 激励，已在 README 说明。）
"""
from __future__ import annotations

import re

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_kΩ, u_uF, u_nF, u_V, u_Hz


def _inject_ac(netlist: str, acmag: float = 1.0) -> str:
    """给 V 源行补 AC 激励（PySpice 1.5 无法直接生成）。"""
    def repl(m):
        line = m.group(0)
        if re.search(r"\bAC\b", line, re.I):
            return line
        return line + f" AC {acmag!r}"
    return re.sub(r"(?m)^V\d+\s+\S+\s+\S+\s+.*$", repl, netlist)


def pyspice_rc_lowpass(R_ohm: float, C_farad: float, fstart: float, fstop: float) -> str:
    c = Circuit("RC lowpass (PySpice)")
    c.V("1", "inp", c.gnd, 1 @ u_V)
    c.R("1", "inp", "out", R_ohm)
    c.C("1", "out", c.gnd, C_farad)
    netlist = _inject_ac(str(c))
    netlist += f".ac dec 20 {float(fstart)!r} {float(fstop)!r}\n.end\n"
    return netlist


def pyspice_sk_lowpass(R1: float, R2: float, C1: float, C2: float,
                       fstart: float, fstop: float) -> str:
    c = Circuit("Sallen-Key LP (PySpice)")
    c.V("1", "inp", c.gnd, 1 @ u_V)
    c.R("1", "inp", "a", R1)
    c.R("2", "a", "b", R2)
    c.C("1", "a", "out", C1)
    c.C("2", "b", c.gnd, C2)
    c.VCVS("1", "out", c.gnd, "b", "out", 1e6)
    netlist = _inject_ac(str(c))
    netlist += f".ac dec 20 {float(fstart)!r} {float(fstop)!r}\n.end\n"
    return netlist


def pyspice_inverting_amp(R1: float, Rf: float, fstart: float, fstop: float) -> str:
    c = Circuit("Inverting amp (PySpice)")
    c.V("1", "inp", c.gnd, 1 @ u_V)
    c.R("1", "inp", "inv", R1)
    c.R("f", "inv", "out", Rf)
    c.VCVS("1", "out", c.gnd, c.gnd, "inv", 1e6)
    netlist = _inject_ac(str(c))
    netlist += f".ac dec 20 {float(fstart)!r} {float(fstop)!r}\n.end\n"
    return netlist
