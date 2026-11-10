"""ngspice 批处理运行器：网表 → ngspice_con.exe -b → raw 解析。"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .rawparse import RawPlot, parse_raw_file, find_plot

# 默认引擎路径（D:\项目\02_数据集\ngspice-46-win64）
DEFAULT_NGSPICE = r"D:\项目\02_数据集\ngspice-46-win64\Spice64\bin\ngspice_con.exe"


@dataclass
class SimResult:
    plots: list
    log: str
    raw_path: str

    def plot(self, name: str) -> RawPlot:
        p = find_plot(self.plots, name)
        if p is None:
            raise KeyError(f"plot {name!r} not found in {[x.name for x in self.plots]}")
        return p


def run_netlist(netlist: str, ngspice_exe: str | None = None,
                workdir: str | None = None, keep_files: bool = False) -> SimResult:
    """运行一个完整网表（含 .ac/.tran/.op 等），返回解析后的 plots。"""
    exe = ngspice_exe or os.environ.get("NGSPICE_EXE", DEFAULT_NGSPICE)
    if not Path(exe).exists():
        raise FileNotFoundError(f"ngspice executable not found: {exe}")

    if workdir is None:
        workdir = tempfile.mkdtemp(prefix="n7spice_")
    os.makedirs(workdir, exist_ok=True)
    cir = os.path.join(workdir, "circuit.cir")
    raw = os.path.join(workdir, "circuit.raw")
    log = os.path.join(workdir, "circuit.log")
    with open(cir, "w", encoding="utf-8") as f:
        f.write(netlist)

    r = subprocess.run([exe, "-b", "-o", log, "-r", raw, cir],
                       capture_output=True, text=True, timeout=120,
                       cwd=os.path.dirname(exe))
    log_text = ""
    if os.path.exists(log):
        log_text = open(log, "r", encoding="utf-8", errors="replace").read()
    if r.returncode != 0 and not os.path.exists(raw):
        raise RuntimeError(f"ngspice failed rc={r.returncode}\n{r.stdout}\n{r.stderr}\n{log_text}")
    if not os.path.exists(raw):
        raise RuntimeError(f"no raw file produced\n{log_text}")

    plots = parse_raw_file(raw)
    result = SimResult(plots=plots, log=log_text, raw_path=raw)
    if not keep_files:
        for p in (cir, raw, log):
            try:
                os.remove(p)
            except OSError:
                pass
    return result


def ac_sweep(netlist: str, **kw) -> RawPlot:
    """运行并返回 AC Analysis plot。"""
    res = run_netlist(netlist, **kw)
    return res.plot("AC Analysis")


def tran_sweep(netlist: str, **kw) -> RawPlot:
    """运行并返回 Transient Analysis plot。"""
    res = run_netlist(netlist, **kw)
    return res.plot("Transient")
