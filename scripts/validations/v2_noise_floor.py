#!/usr/bin/env python3
"""
V2 -- Noise floor: the distribution of clean-vs-clean divergence.

REQUIREMENT SOURCE
  The benign/SDC decision compares a faulted run against the reference run.
  Two clean runs already differ in atomic mode (floating-point atomicAdd is
  order-nondeterministic). A threshold set from ONE clean-vs-clean pair (the
  original 10x rule) has no statistical basis; a reviewer will ask "why 10".
  This test measures the distribution of the floor and, for colored mode,
  proves that the floor is identically zero, which removes the threshold
  problem from the methodology entirely.

WHAT IS CHECKED
  K independent invocations of the twin-run driver, no injection, on a
  heterogeneous (lognormal) field, N given, for both scatter modes.
  From each: E_floor_max over the trajectory (fi_summary).
    atomic  : report max / mean / std / 99th pct of E_floor_max; if every
              invocation gives exactly 0, the hardware schedules identically
              and the "noise floor" premise is empty on this GPU (a finding).
    colored : every E_floor_max must be exactly 0.0.
  Also: the cleanA final fields across invocations, compared pairwise
  (should be bitwise identical in colored mode; that is the reproducibility
  guarantee used by V4/V5).

PASS
  colored: all E_floor_max == 0 and all cleanA dumps bitwise identical.
  atomic : no pass/fail on the magnitude (it is a measurement), but the
           report records thr_recommended = max(E_floor_max) * 3 and the
           full sample, which the ensemble scripts read.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import vlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--gen", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=32); ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--t", type=float, default=0.02); ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--kernel", default="fast")
    a = ap.parse_args()
    out = os.path.join(a.out, "v2"); os.makedirs(out, exist_ok=True); log = os.path.join(out, "run.log")
    caps = vlib.capabilities(a.bin)
    if "--scatter" not in caps:
        vlib.write_report(out, "v2", False, {"status": "UNSUPPORTED: no --scatter; only atomic floor measurable"}); return
    fp = os.path.join(out, f"ln{a.N}")
    vlib.gen_field(a.gen, a.N, "lognormal", fp, sigma=1.0, L=0.125, seed=7)
    det = {}
    ok = True
    for scatter in ["colored", "atomic"]:
        vals = []; fields = []
        for k in range(a.K):
            d = os.path.join(out, f"{scatter}_{k:02d}"); os.makedirs(d, exist_ok=True)
            vlib.run([a.bin, str(a.N), str(a.t), str(a.dt), fp, d, "--tag", "nf", "--snap", "5",
                      "--kernel", a.kernel, "--scatter", scatter], log=log)
            row = vlib.read_csv(os.path.join(d, "fi_summary_nf.csv"))[0]
            vals.append(float(row["E_floor_max"]))
            fields.append(np.fromfile(os.path.join(d, "T_final_nf_clean_rank0.bin")))
        vals = np.array(vals)
        bitwise = all(np.array_equal(fields[0], f) for f in fields[1:])
        det[scatter] = {"E_floor_max_samples": vals.tolist(), "max": float(vals.max()), "mean": float(vals.mean()),
                        "std": float(vals.std()), "p99": float(np.percentile(vals, 99)),
                        "all_zero": bool((vals == 0).all()), "cleanA_bitwise_identical_across_invocations": bitwise,
                        "thr_recommended": float(max(vals.max() * 3, 1e-13))}
        if scatter == "colored":
            ok &= bool((vals == 0).all()) and bitwise
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for i, s in enumerate(["atomic", "colored"]):
            v = np.array(det[s]["E_floor_max_samples"]); v = np.where(v > 0, v, 1e-18)
            ax.semilogy(np.full(len(v), i) + np.random.uniform(-0.1, 0.1, len(v)), v, "o", label=s)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["atomic", "colored (0 shown as 1e-18)"]); ax.set_ylabel("E_floor_max")
        ax.set_title(f"V2 clean-vs-clean floor, N={a.N}, K={a.K}"); fig.tight_layout()
        fig.savefig(os.path.join(out, "v2_noise_floor.png"), dpi=140)
    except Exception as ex:
        det["plot_error"] = str(ex)
    vlib.write_report(out, "v2", ok, det)

if __name__ == "__main__":
    main()
