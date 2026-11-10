"""ngspice ASCII raw 文件解析器。

ngspice `set filetype=ascii` 输出格式：
    Header 行（Title/Date/Command/Plotname/Flags/No. Variables/No. Points）
    Variables:
        <idx>  <name>  <type>
    Values:
        <point_idx>  <re>,<im>   (每个变量一行；complex)
        <point_idx>  <val>       (real)

支持同一文件内多段 plot（.op + .ac 等依次写入）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RawPlot:
    name: str
    flags: str
    variables: list  # [(idx, name, type), ...]
    data: dict = field(default_factory=dict)  # name -> np.ndarray (complex or float)

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    @property
    def npoints(self):
        if not self.data:
            return 0
        return len(next(iter(self.data.values())))


def parse_raw_file(path) -> list:
    """解析 ASCII raw 文件，返回 RawPlot 列表。"""
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    return parse_raw_text(text)


def parse_raw_text(text: str) -> list:
    plots = []
    # 按 "Title:" 切段（每段一个 plot）
    segments = re.split(r"(?m)^(?=Title:)", text)
    for seg in segments:
        if not seg.strip().startswith("Title:"):
            continue
        m_flags = re.search(r"(?m)^Flags:\s*(.+)$", seg)
        m_nvars = re.search(r"(?m)^No\. Variables:\s*(\d+)", seg)
        m_npts = re.search(r"(?m)^No\. Points:\s*(\d+)", seg)
        m_plot = re.search(r"(?m)^Plotname:\s*(.+)$", seg)
        if not (m_flags and m_nvars and m_npts):
            continue
        flags = m_flags.group(1).strip()
        nvars = int(m_nvars.group(1))
        npts = int(m_npts.group(1))
        name = m_plot.group(1).strip() if m_plot else "unknown"

        m_var = re.search(r"(?m)^Variables:\s*\n((?:\s*\d+\s+\S+\s+\S+.*\n?)+)", seg)
        if not m_var:
            continue
        variables = []
        for line in m_var.group(1).strip().splitlines():
            parts = line.split()
            variables.append((int(parts[0]), parts[1], parts[2]))

        m_val = re.search(r"(?m)^Values:\s*\n", seg)
        if not m_val:
            continue
        body = seg[m_val.end():]
        complex_flag = "complex" in flags.lower()
        cols = [[] for _ in range(nvars)]
        # 每个 point：第一行 "idx  v"，后续行 "\tv"
        # 所有值是 "re,im" 或 "re"
        vals = re.findall(r"(-?[\d.]+(?:e[+-]?\d+)?(?:,-?[\d.]+(?:e[+-]?\d+)?)?)", body)
        # 按顺序 nvars*npts 个值（每个值可能带逗号）
        # 更稳：逐 token 匹配
        tokens = re.findall(r"[-+]?[\d.]+(?:[eE][-+]?\d+)?(?:,[-+]?[\d.]+(?:[eE][-+]?\d+)?)?", body)
        # 过滤掉 point 序号 token：序号是独立整数且计数为 npts 个
        # 直接按格式读取更稳
        cols = _parse_values_block(body, nvars, npts, complex_flag)
        data = {}
        for i, (_idx, vname, _vt) in enumerate(variables):
            arr = np.array(cols[i])
            data[vname] = arr
        plots.append(RawPlot(name=name, flags=flags, variables=variables, data=data))
    return plots


def _parse_values_block(body: str, nvars: int, npts: int, complex_flag: bool):
    """逐 point 解析 Values 块。"""
    cols = [[] for _ in range(nvars)]
    lines = body.splitlines()
    i = 0
    for p in range(npts):
        # point 首行: "<p>\t<val>"  或 "<p> <val>"
        if i >= len(lines):
            break
        first = lines[i].strip()
        i += 1
        if not first:
            continue
        # 拆出首个整数序号
        parts = first.split(None, 1)
        if len(parts) == 2:
            val_tokens = [parts[1]]
        else:
            val_tokens = []
        # 后续 nvars-1 行
        for _ in range(nvars - 1):
            if i >= len(lines):
                break
            val_tokens.append(lines[i].strip())
            i += 1
        for v in range(nvars):
            if v >= len(val_tokens):
                cols[v].append(np.nan)
                continue
            tok = val_tokens[v].strip()
            if complex_flag or ("," in tok):
                re_s, _, im_s = tok.partition(",")
                cols[v].append(complex(float(re_s), float(im_s) if im_s else 0.0))
            else:
                cols[v].append(float(tok))
    return cols


def find_plot(plots, name: str):
    """按 Plotname 查找（模糊匹配）。"""
    name_l = name.lower()
    for p in plots:
        if name_l in p.name.lower():
            return p
    return None
