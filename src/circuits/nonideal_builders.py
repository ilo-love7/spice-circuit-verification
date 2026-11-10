"""非理想运放宏模型与级联滤波器网表（ngspice 批处理友好）。

宏模型（B 源限流跨导 + 补偿电容）：
    B1 0 nc I = max(-Imax,min(Imax, gm*(V(inp)-V(inn))))
    R1 nc 0 Rlag   C1 nc 0 Cc   E1 out 0 nc 0 1
"""
from __future__ import annotations


def opamp_subckt_lines(name: str = "XU", gm: float = 1e-3, Rlag: float = 1e8,
                       Cc: float = 1.5915e-11, Imax: float = 1e-3) -> str:
    """生成宏模型片段（内联，不用 .subckt 以简化 MC）。节点：inp inn out。

    用法：把返回的行放进网表，把 name 的节点接到 (inp inn out)。
    注意：这里用内联形式，B 源节点名带前缀避免冲突。
    """
    raise RuntimeError("use explicit inline netlists instead")


def inverting_amp_gbw_netlist(R1: float, Rf: float, gm: float, Rlag: float, Cc: float,
                              Imax: float, fstart: float, fstop: float,
                              points_per_decade: int = 20, vac: float = 1.0,
                              title: str = "Inverting amp finite-GBW") -> str:
    """反相放大 + 单极点限流运放 AC 扫频。"""
    return f"""* {title}
V1 in 0 DC 0 AC {float(vac)!r}
R1 in inv {float(R1)!r}
Rf inv out {float(Rf)!r}
B1 0 nc I = max(-{float(Imax)!r}, min({float(Imax)!r}, -{float(gm)!r}*V(inv)))
* 注：反相组态 V+ = 0，V- = inv，故 gm*(V(V+)-V(V-)) = -gm*V(inv)，受控电流灌入 nc。
Rlag nc 0 {float(Rlag)!r}
Cc nc 0 {float(Cc)!r}
E1 out 0 nc 0 1
.ac dec {points_per_decade} {float(fstart)!r} {float(fstop)!r}
.end
"""


def noninv_buffer_gbw_netlist(gm: float, Rlag: float, Cc: float, Imax: float,
                              fstart: float, fstop: float,
                              points_per_decade: int = 20, vac: float = 1.0,
                              title: str = "Buffer finite-GBW") -> str:
    """同相单位增益缓冲器（有限 GBW）：A(s)/(1+A(s))。"""
    return f"""* {title}
V1 in 0 DC 0 AC {float(vac)!r}
B1 0 nc I = max(-{float(Imax)!r}, min({float(Imax)!r}, {float(gm)!r}*(V(in)-V(out))))
Rlag nc 0 {float(Rlag)!r}
Cc nc 0 {float(Cc)!r}
E1 out 0 nc 0 1
.ac dec {points_per_decade} {float(fstart)!r} {float(fstop)!r}
.end
"""


def inverting_amp_slew_netlist(R1: float, Rf: float, gm: float, Rlag: float, Cc: float,
                               Imax: float, v_amp: float, f_sq: float, tstop: float,
                               tstep: float | None = None,
                               title: str = "Inverting amp slew") -> str:
    """大信号方波瞬态（压摆率测试）。方波边沿默认 tstop/1000。"""
    if tstep is None:
        tstep = tstop / 5000
    tr = tstop / 1000.0
    return f"""* {title}
V1 in 0 PULSE({-float(v_amp)!r} {float(v_amp)!r} 0 {float(tr)!r} {float(tr)!r} {float(0.5 / f_sq)!r} {float(1.0 / f_sq)!r})
R1 in inv {float(R1)!r}
Rf inv out {float(Rf)!r}
B1 0 nc I = max(-{float(Imax)!r}, min({float(Imax)!r}, -{float(gm)!r}*V(inv)))
Rlag nc 0 {float(Rlag)!r}
Cc nc 0 {float(Cc)!r}
E1 out 0 nc 0 1
.tran {float(tstep)!r} {float(tstop)!r}
.end
"""


def sk_cascade4_netlist(R1a: float, R2a: float, C1a: float, C2a: float,
                        R1b: float, R2b: float, C1b: float, C2b: float,
                        fstart: float, fstop: float, buffered: bool = True,
                        points_per_decade: int = 20, vac: float = 1.0,
                        title: str = "SK cascade 4th") -> str:
    """两级 Sallen-Key 级联 4 阶低通。buffered=True 级间加理想缓冲。"""
    buf = "Ebuf mid2 0 mid 0 1\n" if buffered else ""
    n2 = "mid2" if buffered else "mid"
    return f"""* {title}
V1 in 0 DC 0 AC {float(vac)!r}
* stage A
R1 in a1 {float(R1a)!r}
R2 a1 b1 {float(R2a)!r}
C1 a1 mid {float(C1a)!r}
C2 b1 0 {float(C2a)!r}
E1 mid 0 b1 mid 1e6
{buf}* stage B
R1b {n2} a2 {float(R1b)!r}
R2b a2 b2 {float(R2b)!r}
C1b a2 out {float(C1b)!r}
C2b b2 0 {float(C2b)!r}
E2 out 0 b2 out 1e6
.ac dec {points_per_decade} {float(fstart)!r} {float(fstop)!r}
.end
"""


def sk_cascade4_tran_netlist(R1a: float, R2a: float, C1a: float, C2a: float,
                             R1b: float, R2b: float, C1b: float, C2b: float,
                             v_amp: float, tstop: float, tstep: float | None = None,
                             buffered: bool = True,
                             title: str = "SK cascade step") -> str:
    if tstep is None:
        tstep = tstop / 4000
    buf = "Ebuf mid2 0 mid 0 1\n" if buffered else ""
    n2 = "mid2" if buffered else "mid"
    return f"""* {title}
V1 in 0 DC 0 PULSE(0 {float(v_amp)!r} 0 {float(tstop / 10000)!r} {float(tstop / 10000)!r} {float(tstop)!r} {float(2 * tstop)!r})
R1 in a1 {float(R1a)!r}
R2 a1 b1 {float(R2a)!r}
C1 a1 mid {float(C1a)!r}
C2 b1 0 {float(C2a)!r}
E1 mid 0 b1 mid 1e6
{buf}* stage B
R1b {n2} a2 {float(R1b)!r}
R2b a2 b2 {float(R2b)!r}
C1b a2 out {float(C1b)!r}
C2b b2 0 {float(C2b)!r}
E2 out 0 b2 out 1e6
.tran {float(tstep)!r} {float(tstop)!r}
.end
"""
