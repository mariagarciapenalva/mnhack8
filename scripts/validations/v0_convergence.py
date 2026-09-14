#!/usr/bin/env python3
"""
V0 -- Analytic eigenmode convergence (discretization + time stepping + BC).

REQUIREMENT SOURCE
  Every E(t) the fault study reports is a difference between two solver
  outputs. If the discretization or the boundary-condition treatment is wrong,
  E(t) partly measures the bug. This test is the only place where the solver is
  compared against a KNOWN answer, so nothing downstream is trustworthy until
  it passes. It cannot be replaced by a profiler.

WHAT IS CHECKED
  Homogeneous material, k = rhoc = 1 (alpha = 1), source off,
  T(x,0) = sin(pi x) sin(pi y) sin(pi z), exact T = exp(-3 pi^2 t) T(x,0).
  Two sweeps that isolate the two error sources:
    spatial : N = 8,16,32 at t = 0.02, dt = 1e-5 (temporal error ~1e-4, below
              the spatial error at N=32). Prediction for Q1 elements with
              consistent mass: rel err ~ pi^4 h^2 t / 4 -> 7.6e-3, 1.9e-3,
              4.8e-4. Ratio ~4 per doubling (order 2).
    temporal: N = 32 at t = 0.04, dt = 4e-3 ... 5e-4. Backward Euler:
              rel err ~ lambda^2 dt t / 2, lambda = 3 pi^2 -> 0.070, 0.035,
              0.0175, 0.0088. Ratio ~2 per halving (order 1).
  Both error sources have opposite sign (FEM decays too fast, BE too slowly),
  which is why they are measured separately.

PASS
  observed spatial order in [1.7, 2.3] on the first doubling, >= 1.6 on the
  second (temporal error starts to show at N=32); temporal order in
  [0.85, 1.15] for every halving; every magnitude within a factor 2 of the
  prediction (the prediction ignores the nodal-interpolation error of the
  initial condition and the cross term, so 20% is too tight, 2x is honest).

RUNS ON: any kernel/scatter combination; run all four to prove they agree.
"""
import argparse, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import vlib

SPATIAL  = [(8, 0.02, 1e-5), (16, 0.02, 1e-5), (32, 0.02, 1e-5)]
TEMPORAL = [(32, 0.04, 4e-3), (32, 0.04, 2e-3), (32, 0.04, 1e-3), (32, 0.04, 5e-4)]
LAM = 3 * math.pi**2

def predict_spatial(N, t): return math.pi**4 * (1.0 / N)**2 * t / 4
def predict_temporal(dt, t): return LAM**2 * dt * t / 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", default="fast"); ap.add_argument("--scatter", default="atomic")
    ap.add_argument("--np", type=int, default=1)
    a = ap.parse_args()
    out = os.path.join(a.out, f"v0_{a.kernel}_{a.scatter}_np{a.np}")
    os.makedirs(out, exist_ok=True); log = os.path.join(out, "run.log")
    caps = vlib.capabilities(a.bin)
    if "--verify" not in caps:
        vlib.write_report(out, "v0", False, {"status": "UNSUPPORTED: binary has no --verify mode"}); return
    fields = os.path.join(out, "fields"); os.makedirs(fields, exist_ok=True)
    vpath = os.path.join(out, "verify.csv")
    if os.path.exists(vpath): os.remove(vpath)
    for N, t, dt in SPATIAL + TEMPORAL:
        fp = os.path.join(fields, f"homog{N}")
        if not os.path.exists(fp + "_k.bin"):
            vlib.gen_field(a.gen, N, "homogeneous", fp, k=1.0, rhoc=1.0)
        cmd = [a.bin, str(N), str(t), str(dt), fp, out, "--verify", "--kernel", a.kernel, "--scatter", a.scatter]
        vlib.run(cmd, log=log, np_=a.np)
    rows = vlib.read_csv(vpath)
    err = {(int(r["N"]), float(r["dt"])): float(r["rel_L2_err"]) for r in rows}

    det = {"spatial": [], "temporal": [], "checks": []}
    ok = True
    # spatial
    es = [err[(N, dt)] for N, t, dt in SPATIAL]
    for (N, t, dt), e in zip(SPATIAL, es):
        pred = predict_spatial(N, t)
        det["spatial"].append({"N": N, "err": e, "pred": pred, "ratio_to_pred": e / pred})
        ok &= 0.5 <= e / pred <= 2.0
    orders = [math.log2(es[i] / es[i + 1]) for i in range(len(es) - 1)]
    det["spatial_orders"] = orders
    ok &= 1.7 <= orders[0] <= 2.3 and orders[1] >= 1.6
    # temporal
    et = [err[(N, dt)] for N, t, dt in TEMPORAL]
    for (N, t, dt), e in zip(TEMPORAL, et):
        pred = predict_temporal(dt, t)
        det["temporal"].append({"dt": dt, "err": e, "pred": pred, "ratio_to_pred": e / pred})
        ok &= 0.5 <= e / pred <= 2.0
    torders = [math.log2(et[i] / et[i + 1]) for i in range(len(et) - 1)]
    det["temporal_orders"] = torders
    ok &= all(0.85 <= o <= 1.15 for o in torders)

    # figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        hs = [1.0 / N for N, _, _ in SPATIAL]
        ax[0].loglog(hs, es, "o-", label="measured"); ax[0].loglog(hs, [predict_spatial(N, 0.02) for N, _, _ in SPATIAL], "k--", label="pi^4 h^2 t/4")
        ax[0].set_xlabel("h"); ax[0].set_ylabel("rel L2 error"); ax[0].set_title("spatial, dt=1e-5, t=0.02"); ax[0].legend()
        dts = [dt for _, _, dt in TEMPORAL]
        ax[1].loglog(dts, et, "o-", label="measured"); ax[1].loglog(dts, [predict_temporal(dt, 0.04) for dt in dts], "k--", label="lambda^2 dt t/2")
        ax[1].set_xlabel("dt"); ax[1].set_title("temporal, N=32, t=0.04"); ax[1].legend()
        fig.suptitle(f"V0 eigenmode convergence  ({a.kernel}/{a.scatter}, np={a.np})")
        fig.tight_layout(); fig.savefig(os.path.join(out, "v0_convergence.png"), dpi=140)
    except Exception as ex:
        det["plot_error"] = str(ex)
    vlib.write_report(out, "v0", ok, det)

if __name__ == "__main__":
    main()
