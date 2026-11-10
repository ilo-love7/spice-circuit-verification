"""00 环境冒烟：ngspice 引擎 + PySpice 网表 + raw 解析链路。"""
from common import load_config, save_json, ROOT
import numpy as np

from src.engine.runner import run_netlist
from src.circuits.builders import rc_lowpass_netlist
from src.circuits.pyspice_build import pyspice_rc_lowpass
from src.theory.transfer import rc_lowpass_H, compare_response


def main():
    cfg = load_config()["rc_lowpass"]
    R, C = cfg["R"], cfg["C"]
    out = {"engine": "ngspice_con.exe (ngspice-46 win64)", "pyspice": True}

    # 字符串网表
    net = rc_lowpass_netlist(R, C, 1, 1e5, points_per_decade=10)
    p = run_netlist(net).plot("AC Analysis")
    f = np.array(p["frequency"]).real
    Hs = np.array(p["v(out)"])
    Ht = rc_lowpass_H(R, C, 2 * np.pi * f)
    cmp1 = compare_response(f, Hs, Ht)
    out["string_netlist_max_err_db"] = cmp1["max_abs_err_db"]

    # PySpice 网表
    net2 = pyspice_rc_lowpass(R, C, 1, 1e5)
    p2 = run_netlist(net2).plot("AC Analysis")
    f2 = np.array(p2["frequency"]).real
    Hs2 = np.array(p2["v(out)"])
    cmp2 = compare_response(f2, Hs2, rc_lowpass_H(R, C, 2 * np.pi * f2))
    out["pyspice_netlist_max_err_db"] = cmp2["max_abs_err_db"]
    out["status"] = "OK" if max(cmp1["max_abs_err_db"], cmp2["max_abs_err_db"]) < 1e-6 else "FAIL"

    print(f"string netlist max err {cmp1['max_abs_err_db']:.2e} dB")
    print(f"PySpice netlist max err {cmp2['max_abs_err_db']:.2e} dB")
    print(f"status: {out['status']}")
    save_json("smoke.json", out)


if __name__ == "__main__":
    main()
