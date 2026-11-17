"""v1.3 网表：PSRR 电源耦合 / 双极点稳定性 / 温度扫描（ngspice 批处理友好）。

沿用项目纪律：所有元件值用 float(x)!r 锁定，避免 numpy.float64 破坏网表。
"""
from __future__ import annotations


def psrr_inverting_netlist(R1: float, Rf: float, gm: float, Rlag: float,
                           Cc: float, Imax: float, Rps: float, Cps: float,
                           psrr0: float, fstart: float, fstop: float,
                           ppd: int = 20, title: str = "PSRR inverting") -> str:
    """反相放大器电源抑制：信号源 ac=0，电源 vcc 加 ac=1 扫 V(out)。

    宏模型输入差 = (V+ − V−) + V(eos)/PSRR0，其中
    eos = vcc/(1+s·Rps·Cps)（PSRR 一阶滚降极点 f_ps=1/(2πRpsCps)）。
    V+=0，V−=inv，故 gm 电流灌入方向取 −gm·V(inv) + gm·V(eos)/PSRR0。
    """
    return f"""* {title}
V1 in 0 DC 0 AC 0
VCC vcc 0 DC 0 AC 1
R1 in inv {float(R1)!r}
Rf inv out {float(Rf)!r}
* PSRR 与开环共享主极点：电源耦合直接注入输入差，经 A(s) 自然滚降
B1 0 nc I = max(-{float(Imax)!r}, min({float(Imax)!r}, -{float(gm)!r}*V(inv) + {float(gm / psrr0)!r}*V(vcc)))
Rlag nc 0 {float(Rlag)!r}
Cc nc 0 {float(Cc)!r}
E1 out 0 nc 0 1
.ac dec {int(ppd)} {float(fstart)!r} {float(fstop)!r}
.end
"""


def two_pole_openloop_netlist(gm: float, Rlag: float, Cc: float,
                              R2: float, C2: float, fstart: float, fstop: float,
                              ppd: int = 20, title: str = "Two-pole open loop") -> str:
    """双极点运放开环扫频：主极点 Rlag·Cc，第二极点 R2·C2。

    节点链：B 源电流 → nc1（Rlag//Cc）→ E1 缓冲 → nc2（R2//C2）→ E2 输出。
    E 缓冲隔离两级，避免加载改变极点位置（对照 v1.1 的 SK 级联加载负结果）。
    """
    return f"""* {title}
Vac in 0 DC 0 AC 1
B1 0 nc1 I = {float(gm)!r}*V(in)
Rlag nc1 0 {float(Rlag)!r}
Cc nc1 0 {float(Cc)!r}
Ebuf1 m1 0 nc1 0 1
R2 m1 nc2 {float(R2)!r}
C2 nc2 0 {float(C2)!r}
Ebuf2 out 0 nc2 0 1
.ac dec {int(ppd)} {float(fstart)!r} {float(fstop)!r}
.end
"""


def two_pole_buffer_step_netlist(gm: float, Rlag: float, Cc: float,
                                 R2: float, C2: float, v_step: float,
                                 tstop: float, tstep: float | None = None,
                                 title: str = "Two-pole buffer step") -> str:
    """双极点同相单位缓冲器阶跃（NG=1，稳定性最差情形）。"""
    if tstep is None:
        tstep = tstop / 20000.0
    tr = tstop / 500.0
    return f"""* {title}
V1 in 0 PULSE(0 {float(v_step)!r} 0 {float(tr)!r} {float(tr)!r} {float(tstop)!r} {float(2 * tstop)!r})
B1 0 nc1 I = {float(gm)!r}*(V(in)-V(out))
Rlag nc1 0 {float(Rlag)!r}
Cc nc1 0 {float(Cc)!r}
Ebuf1 m1 0 nc1 0 1
R2 m1 nc2 {float(R2)!r}
C2 nc2 0 {float(C2)!r}
Ebuf2 out 0 nc2 0 1
.tran {float(tstep)!r} {float(tstop)!r}
.end
"""


def rc_temp_netlist(R0: float, C0: float, tc1_r: float, tc1_c: float,
                    temp_c: float, fstart: float, fstop: float,
                    ppd: int = 40, title: str = "RC temperature") -> str:
    """一阶 RC 低通，电阻电容带 TC1，指定 .temp 扫 AC 求 fc。"""
    return f"""* {title}
.temp {float(temp_c)!r}
V1 in 0 DC 0 AC 1
R1 in out {float(R0)!r} TC={float(tc1_r)!r}
C1 out 0 {float(C0)!r} TC1={float(tc1_c)!r}
.ac dec {int(ppd)} {float(fstart)!r} {float(fstop)!r}
.end
"""


def inv_gain_temp_netlist(R1: float, Rf: float, tc1_r1: float, tc1_rf: float,
                          temp_c: float, fstart: float, fstop: float,
                          ppd: int = 20, title: str = "Inv gain temperature") -> str:
    """反相放大器增益温漂：两电阻各自 TC1，理想运放（隔离电阻温漂与运放温漂）。"""
    return f"""* {title}
.temp {float(temp_c)!r}
V1 in 0 DC 0 AC 1
R1 in inv {float(R1)!r} TC={float(tc1_r1)!r}
Rf inv out {float(Rf)!r} TC={float(tc1_rf)!r}
E1 out 0 inv 0 -1e6
.ac dec {int(ppd)} {float(fstart)!r} {float(fstop)!r}
.end
"""
