# 公共工具：路径、配置加载、绘图风格
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CONFIGS = ROOT / "configs"
for d in (RESULTS, FIGURES):
    d.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.autolayout": True,
})


def load_config() -> dict:
    return json.loads((CONFIGS / "circuits.json").read_text(encoding="utf-8"))


def save_json(name: str, obj) -> Path:
    p = RESULTS / name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  saved {p.relative_to(ROOT)}")
    return p


def save_fig(fig, name: str) -> Path:
    p = FIGURES / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {p.relative_to(ROOT)}")
    return p


def loglog_axes(ax, f, xlabel="Frequency (Hz)"):
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
