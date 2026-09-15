#!/usr/bin/env python3
"""Layer 9: parse fi_summary_*.csv across R* folders -> time/step, speedup, efficiency plot."""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validations")); import vlib
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--mode", default="strong"); a = ap.parse_args()
rows = []
for d in sorted(glob.glob(os.path.join(a.root, "R*"))):
    f = glob.glob(os.path.join(d, "fi_summary_*.csv"))
    if not f: continue
    r = vlib.read_csv(f[0])[0]; R = int(r["nprocs"]); ms = float(r["solve_ms_A"]); n = int(r["n_steps"]); N = int(r["N"])
    rows.append((R, N, ms / n, float(r["mean_cg_iters_A"])))
rows.sort(); R0, N0, t0, _ = rows[0]
with open(os.path.join(a.root, "scaling.csv"), "w") as f:
    f.write("ranks,N,ms_per_step,mean_cg_iters,speedup,efficiency\n")
    for R, N, t, it in rows:
        sp = t0 / t if a.mode == "strong" else (t0 / t) * (N**3 / N0**3) / (R / R0) * (R / R0)  # weak: ideal is constant time
        eff = sp / (R / R0) if a.mode == "strong" else t0 / t
        f.write(f"{R},{N},{t:.3f},{it:.1f},{sp:.3f},{eff:.3f}\n")
print(open(os.path.join(a.root, "scaling.csv")).read())
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = [r[0] for r in rows]; t = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.4)); ax.loglog(R, t, "o-", label="measured")
    if a.mode == "strong": ax.loglog(R, [t0 * R0 / r for r in R], "k--", label="ideal")
    else: ax.axhline(t0, color="k", ls="--", label="ideal (constant)")
    ax.set_xlabel("MPI ranks (GPUs)"); ax.set_ylabel("ms / time step"); ax.set_title(f"{a.mode} scaling, MN5 ACC"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(a.root, "scaling.png"), dpi=140)
except Exception as ex: print("plot skipped:", ex)
