"""电路网表构建器（字符串模板，蒙特卡洛友好）。

运放用 VCVS 理想模型（开环增益 1e6）：
  E1 out 0 inp inn 1e6  →  V(out) = 1e6·(V(inp) − V(inn))
"""
from __future__ import annotations

A_OL = 1e6

# ---------- 一阶 RC 低通 ----------
def rc_lowpass_netlist(R: float, C: float, fstart: float, fstop: float,
                       points_per_decade: int = 20, vac: float = 1.0,
                       title: str = "RC lowpass") -> str:
    return f"""* {title}
V1 in 0 DC 0 AC {vac}
R1 in out {float(R)!r}
C1 out 0 {float(C)!r}
.ac dec {points_per_decade} {float(fstart)!r} {float(fstop)!r}
.end
"""


# ---------- Sallen-Key 低通（单位增益跟随器） ----------
def sk_lowpass_netlist(R1: float, R2: float, C1: float, C2: float,
                       fstart: float, fstop: float, points_per_decade: int = 20,
                       vac: float = 1.0, title: str = "Sallen-Key LP") -> str:
    # Vin-R1-A-R2-B; C1: A->out; C2: B->GND; 跟随器 E1 out 0 B out A_OL
    return f"""* {title}
V1 in 0 DC 0 AC {vac}
R1 in a {float(R1)!r}
R2 a b {float(R2)!r}
C1 a out {float(C1)!r}
C2 b 0 {float(C2)!r}
E1 out 0 b out {float(A_OL)!r}
.ac dec {points_per_decade} {float(fstart)!r} {float(fstop)!r}
.end
"""


# ---------- Sallen-Key 带通（等元件、接地 R1、增益 K） ----------
def sk_bandpass_netlist(R: float, C: float, K: float = 1.0,
                        fstart: float = 1.0, fstop: float = 1e5,
                        points_per_decade: int = 20, vac: float = 1.0,
                        title: str = "Sallen-Key BP") -> str:
    # Vin-C1-a-R2-b(=V+); R1: a->GND; C2: b->GND; Vout = K·Vb
    return f"""* {title}
V1 in 0 DC 0 AC {vac}
C1 in a {float(C)!r}
R2 a b {float(R)!r}
R1 a 0 {float(R)!r}
C2 b 0 {float(C)!r}
E1 out 0 b 0 {float(K)!r}
.ac dec {points_per_decade} {float(fstart)!r} {float(fstop)!r}
.end
"""


# ---------- 反相放大器 ----------
def inverting_amp_netlist(R1: float, Rf: float,
                          fstart: float = 1.0, fstop: float = 1e6,
                          points_per_decade: int = 20, vac: float = 1.0,
                          title: str = "Inverting amp") -> str:
    return f"""* {title}
V1 in 0 DC 0 AC {vac}
R1 in inv {float(R1)!r}
Rf inv out {float(Rf)!r}
E1 out 0 0 inv {float(A_OL)!r}
.ac dec {points_per_decade} {float(fstart)!r} {float(fstop)!r}
.end
"""


# ---------- 瞬态（正弦 / 脉冲 / 阶跃） ----------
def rc_lowpass_tran(R: float, C: float, fsine: float, tstop: float,
                    tstep: float | None = None, title: str = "RC LP transient") -> str:
    if tstep is None:
        tstep = tstop / 2000
    return f"""* {title}
V1 in 0 DC 0 SIN(0 1 {float(fsine)!r})
R1 in out {float(R)!r}
C1 out 0 {float(C)!r}
.tran {float(tstep)!r} {float(tstop)!r}
.end
"""


def rc_lowpass_step_tran(R: float, C: float, tstop: float,
                         tstep: float | None = None,
                         title: str = "RC LP step") -> str:
    if tstep is None:
        tstep = tstop / 2000
    return f"""* {title}
V1 in 0 DC 1 PWL(0 0 1n 0 1.000001n 1)
R1 in out {float(R)!r}
C1 out 0 {float(C)!r}
.tran {float(tstep)!r} {float(tstop)!r}
.end
"""


def sk_lowpass_tran(R1: float, R2: float, C1: float, C2: float,
                    fsine: float, tstop: float, tstep: float | None = None,
                    title: str = "Sallen-Key LP transient") -> str:
    if tstep is None:
        tstep = tstop / 2000
    return f"""* {title}
V1 in 0 DC 0 SIN(0 1 {float(fsine)!r})
R1 in a {float(R1)!r}
R2 a b {float(R2)!r}
C1 a out {float(C1)!r}
C2 b 0 {float(C2)!r}
E1 out 0 b out {float(A_OL)!r}
.tran {float(tstep)!r} {float(tstop)!r}
.end
"""


def sk_bandpass_tran(R: float, C: float, K: float, fsine: float, tstop: float,
                     tstep: float | None = None,
                     title: str = "Sallen-Key BP transient") -> str:
    if tstep is None:
        tstep = tstop / 2000
    return f"""* {title}
V1 in 0 DC 0 SIN(0 1 {float(fsine)!r})
C1 in a {float(C)!r}
R2 a b {float(R)!r}
R1 a 0 {float(R)!r}
C2 b 0 {float(C)!r}
E1 out 0 b 0 {float(K)!r}
.tran {float(tstep)!r} {float(tstop)!r}
.end
"""


def inverting_amp_tran(R1: float, Rf: float, fsine: float, tstop: float,
                       tstep: float | None = None,
                       title: str = "Inverting amp transient") -> str:
    if tstep is None:
        tstep = tstop / 2000
    return f"""* {title}
V1 in 0 DC 0 SIN(0 1 {float(fsine)!r})
R1 in inv {float(R1)!r}
Rf inv out {float(Rf)!r}
E1 out 0 0 inv {float(A_OL)!r}
.tran {float(tstep)!r} {float(tstop)!r}
.end
"""
