"""v1.2 网表构建器：MFB 带通 / .noise / 大信号压摆瞬态（ngspice 批处理友好）。

所有 float 槽位一律 float(x)!r —— v1.0 §4 记录过 numpy.float64 进 f-string
会渲染成 np.float64(...) 直接毁网表。
"""
from __future__ import annotations

A_OL = 1e6

__all__ = [
    "mfb_bp_netlist", "mfb_bp_noise_netlist", "mfb_bp_gbw_netlist",
    "rc_noise_netlist", "sk_lp_noise_netlist", "sine_slew_netlist",
    "sk_lp_slew_step_netlist", "noise_analysis_line", "mfb_bp_en_netlist",
]


def _mfb_body(R1, R2, R3, C1, C2, opamp_line: str, vac: float = 1.0) -> str:
    """MFB 带通公共主体。opamp_line 由调用方给（理想 VCVS 或限流宏模型）。"""
    return f"""V1 in 0 DC 0 AC {float(vac)!r}
R1 in n1 {float(R1)!r}
R2 n1 0 {float(R2)!r}
C1 n1 n2 {float(C1)!r}
C2 n1 out {float(C2)!r}
R3 n2 out {float(R3)!r}
{opamp_line}
"""


def _ideal_opamp_line() -> str:
    """反相组态理想 VCVS：Vout = A_OL·(V(+) − V(−)) = −A_OL·V(n2)。"""
    return f"E1 out 0 0 n2 {float(A_OL)!r}"


def _macro_opamp_lines(gm, Rlag, Cc, Imax, noninv_node: str = "0",
                       inv_node: str = "n2") -> str:
    """单极点限流跨导宏模型（v1.1 §7.1），Vout = V(nc)。"""
    return (f"B1 0 nc I = max(-{float(Imax)!r}, min({float(Imax)!r}, "
            f"{float(gm)!r}*(V({noninv_node})-V({inv_node}))))\n"
            f"Rlag nc 0 {float(Rlag)!r}\n"
            f"Cc nc 0 {float(Cc)!r}\n"
            f"E1 out 0 nc 0 1")


def noise_analysis_line(ppd: int, fstart: float, fstop: float,
                        out_node: str = "out", src: str = "V1") -> str:
    return (f".noise v({out_node}) {src} dec {int(ppd)} "
            f"{float(fstart)!r} {float(fstop)!r}")


def mfb_bp_netlist(R1, R2, R3, C1, C2, fstart: float, fstop: float,
                   points_per_decade: int = 20, vac: float = 1.0,
                   title: str = "MFB bandpass AC") -> str:
    """MFB 带通 AC 扫频（理想运放）。"""
    return (f"* {title}\n"
            + _mfb_body(R1, R2, R3, C1, C2, _ideal_opamp_line(), vac)
            + f".ac dec {int(points_per_decade)} {float(fstart)!r} {float(fstop)!r}\n"
            + ".end\n")


def mfb_bp_gbw_netlist(R1, R2, R3, C1, C2, gm, Rlag, Cc, Imax,
                       fstart: float, fstop: float,
                       points_per_decade: int = 20, vac: float = 1.0,
                       title: str = "MFB bandpass finite-GBW") -> str:
    """MFB 带通 + 有限 GBW 限流运放 AC（E16 的 Q 退化扫描）。"""
    return (f"* {title}\n"
            + _mfb_body(R1, R2, R3, C1, C2,
                        _macro_opamp_lines(gm, Rlag, Cc, Imax), vac)
            + f".ac dec {int(points_per_decade)} {float(fstart)!r} {float(fstop)!r}\n"
            + ".end\n")


def mfb_bp_noise_netlist(R1, R2, R3, C1, C2, fstart: float, fstop: float,
                         points_per_decade: int = 20,
                         title: str = "MFB bandpass noise") -> str:
    """MFB 带通 `.noise`（电阻热噪声；VCVS 本身无噪声，运放噪声走解析叠加）。"""
    return (f"* {title}\n"
            + _mfb_body(R1, R2, R3, C1, C2, _ideal_opamp_line())
            + noise_analysis_line(points_per_decade, fstart, fstop) + "\n"
            + ".end\n")


def mfb_bp_en_netlist(R1, R2, R3, C1, C2, R_eq_en: float,
                      fstart: float, fstop: float,
                      points_per_decade: int = 20,
                      analysis: str = "noise",
                      title: str = "MFB bandpass with op-amp e_n") -> str:
    """把运放电压噪声 e_n 表示为同相输入端串联电阻的热噪声。

    取 R_eq_en = e_n²/(4kT)。VCVS 输入不取电流 ⇒ 该电阻上无 DC/AC 压降，
    传递函数**严格不变**；但 `.noise` 会把它的热噪声按噪声增益放大到输出，
    于是「运放电压噪声」也变成 ngspice 可严格计算的真实噪声源。
    analysis="ac" 时返回 AC 扫频网表，用于验证传递函数未被改变。
    """
    body = (f"V1 in 0 DC 0 AC 1\n"
            f"R1 in n1 {float(R1)!r}\n"
            f"R2 n1 0 {float(R2)!r}\n"
            f"C1 n1 n2 {float(C1)!r}\n"
            f"C2 n1 out {float(C2)!r}\n"
            f"R3 n2 out {float(R3)!r}\n"
            f"Ren 0 en {float(R_eq_en)!r}\n"
            f"E1 out 0 en n2 {float(A_OL)!r}\n")
    if analysis == "ac":
        line = f".ac dec {int(points_per_decade)} {float(fstart)!r} {float(fstop)!r}"
    elif analysis == "noise":
        line = noise_analysis_line(points_per_decade, fstart, fstop)
    else:
        raise ValueError("analysis must be 'ac' or 'noise'")
    return f"* {title}\n" + body + line + "\n.end\n"


def rc_noise_netlist(R: float, C: float, fstart: float, fstop: float,
                     points_per_decade: int = 20,
                     title: str = "RC lowpass noise") -> str:
    return (f"* {title}\n"
            f"V1 in 0 DC 0 AC 1\n"
            f"R1 in out {float(R)!r}\n"
            f"C1 out 0 {float(C)!r}\n"
            + noise_analysis_line(points_per_decade, fstart, fstop) + "\n"
            + ".end\n")


def sk_lp_noise_netlist(R1, R2, C1, C2, fstart: float, fstop: float,
                        points_per_decade: int = 20,
                        title: str = "Sallen-Key LP noise") -> str:
    return (f"* {title}\n"
            f"V1 in 0 DC 0 AC 1\n"
            f"R1 in a {float(R1)!r}\n"
            f"R2 a b {float(R2)!r}\n"
            f"C1 a out {float(C1)!r}\n"
            f"C2 b 0 {float(C2)!r}\n"
            f"E1 out 0 b out {float(A_OL)!r}\n"
            + noise_analysis_line(points_per_decade, fstart, fstop) + "\n"
            + ".end\n")


def sine_slew_netlist(R1: float, Rf: float, gm, Rlag, Cc, Imax,
                      v_amp: float, f_sine: float,
                      n_cycles: int = 24, pts_per_cycle: int = 200,
                      title: str = "Large-signal sine slew") -> str:
    """反相放大 + 有限 GBW/SR 宏模型的大信号正弦瞬态。

    固定每周期采样点数（而非固定 tstep），保证各频率 THD 的 FFT 口径一致。
    """
    tstop = n_cycles / float(f_sine)
    tstep = 1.0 / (float(f_sine) * pts_per_cycle)
    return (f"* {title}\n"
            f"V1 in 0 DC 0 SIN(0 {float(v_amp)!r} {float(f_sine)!r})\n"
            f"R1 in inv {float(R1)!r}\n"
            f"Rf inv out {float(Rf)!r}\n"
            + _macro_opamp_lines(gm, Rlag, Cc, Imax, "0", "inv") + "\n"
            f".tran {float(tstep)!r} {float(tstop)!r}\n"
            ".end\n")


def sk_lp_slew_step_netlist(R1, R2, C1, C2, gm, Rlag, Cc, Imax,
                            v_step: float, tstop: float,
                            pts: int = 4000,
                            title: str = "SK LP large-signal step") -> str:
    """SK 低通 + 跟随器宏模型的大信号阶跃（E18 建立时间延长对照）。"""
    tstep = tstop / pts
    tr = tstop / 10000.0
    return (f"* {title}\n"
            f"V1 in 0 DC 0 PULSE(0 {float(v_step)!r} 0 {float(tr)!r} "
            f"{float(tr)!r} {float(tstop)!r} {float(2 * tstop)!r})\n"
            f"R1 in a {float(R1)!r}\n"
            f"R2 a b {float(R2)!r}\n"
            f"C1 a out {float(C1)!r}\n"
            f"C2 b 0 {float(C2)!r}\n"
            + _macro_opamp_lines(gm, Rlag, Cc, Imax, "b", "out") + "\n"
            f".tran {float(tstep)!r} {float(tstop)!r}\n"
            ".end\n")
