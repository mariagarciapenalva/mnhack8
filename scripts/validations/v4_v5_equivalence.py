#!/usr/bin/env python3
"""
V4 -- MPI decomposition consistency.   V5 -- Kernel equivalence (gauss vs fast).

REQUIREMENT SOURCE
  V4: the halo exchange (pack, host staging, Isend/Irecv, unpack-ADD) and the
      owned-node conventions are the only code that changes between 1 and P
      ranks. Every multi-node number reported from MN5, and the level-H
      injection (which needs >= 2 ranks), rests on the P-rank solution being
      the 1-rank solution. Interconnect faults cannot be studied on a halo
      exchange that is itself wrong.
  V5: the 'fast' kernel replaces the per-matvec Gauss loop by two reference
      matrices. The claim is that A_e = rhoc_e*M_hat + dt*k_e*K_hat is
      algebraically identical to the Gauss loop on a uniform grid with
      per-element-constant coefficients. That claim is testable to roundoff
      and the speedup is a reportable number only if the answers agree.

WHAT IS CHECKED
  V4: lognormal field, N=16, clean run, np in {1, 2, 4}; reassembled global
      cleanA fields compared to np=1. Colored scatter so within-rank
      summation is fixed; only the halo add order changes -> expect ~1e-14.
      Also --verify at np=2 must reproduce the np=1 error to 1e-10.
  V5: same field and run, gauss vs fast, colored. Expect ~1e-14 relative
      field difference. solve_ms ratio recorded as the speedup.

PASS
  V4: rel diff <= 1e-12 for every P; verify errors agree to 1e-10.
  V5: rel diff <= 1e-12; speedup recorded (no threshold, it is a measurement).
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import vlib

def run_clean(a, fp, d, np_, kernel, scatter, tag="eq"):
    os.makedirs(d, exist_ok=True)
    vlib.run([a.bin, str(a.N), str(a.t), str(a.dt), fp, d, "--tag", tag, "--snap", "0",
              "--kernel", kernel, "--scatter", scatter], log=os.path.join(a.out, "v4v5.log"), np_=np_)
    T = vlib.assemble_field(d, f"{tag}_clean", a.N, np_)
    sm = vlib.read_csv(os.path.join(d, f"fi_summary_{tag}.csv"))[0]
    return T, sm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--gen", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=16); ap.add_argument("--t", type=float, default=0.02)
    ap.add_argument("--dt", type=float, default=5e-4); ap.add_argument("--mpi", type=int, default=4, help="max ranks to test")
    a = ap.parse_args()
    caps = vlib.capabilities(a.bin)
    if "--kernel" not in caps:
        for name in ["v4", "v5"]:
            vlib.write_report(os.path.join(a.out, name), name, False, {"status": "UNSUPPORTED: needs --kernel/--scatter"})
        return
    fp = os.path.join(a.out, f"ln{a.N}")
    vlib.gen_field(a.gen, a.N, "lognormal", fp, sigma=1.0, L=0.125, seed=7)
    # ---- V4 ----
    out4 = os.path.join(a.out, "v4"); os.makedirs(out4, exist_ok=True)
    T1, _ = run_clean(a, fp, os.path.join(out4, "np1"), 1, "fast", "colored")
    det = {"field_rel_diff": {}, "verify": {}}; ok4 = True
    for P in [2, 4]:
        if P > a.mpi or a.N % P: continue
        TP, _ = run_clean(a, fp, os.path.join(out4, f"np{P}"), P, "fast", "colored")
        r = vlib.rel_l2(TP, T1); det["field_rel_diff"][P] = r; ok4 &= r <= 1e-12
    # verify at np=1 and np=2
    fh = os.path.join(out4, "homog"); vlib.gen_field(a.gen, a.N, "homogeneous", fh)
    for P in [1, 2]:
        if P > a.mpi: continue
        d = os.path.join(out4, f"verify_np{P}"); os.makedirs(d, exist_ok=True)
        vlib.run([a.bin, str(a.N), "0.02", "1e-4", fh, d, "--verify", "--kernel", "fast", "--scatter", "colored"],
                 log=os.path.join(a.out, "v4v5.log"), np_=P)
        det["verify"][P] = float(vlib.read_csv(os.path.join(d, "verify.csv"))[0]["rel_L2_err"])
    if 2 in det["verify"]:
        ok4 &= abs(det["verify"][2] - det["verify"][1]) / det["verify"][1] <= 1e-10
    vlib.write_report(out4, "v4", ok4, det)
    # ---- V5 ----
    out5 = os.path.join(a.out, "v5"); os.makedirs(out5, exist_ok=True)
    Tg, sg = run_clean(a, fp, os.path.join(out5, "gauss"), 1, "gauss", "colored")
    Tf, sf = run_clean(a, fp, os.path.join(out5, "fast"), 1, "fast", "colored")
    r = vlib.rel_l2(Tf, Tg)
    det5 = {"field_rel_diff": r, "solve_ms_gauss": float(sg["solve_ms_A"]), "solve_ms_fast": float(sf["solve_ms_A"]),
            "speedup": float(sg["solve_ms_A"]) / float(sf["solve_ms_A"]),
            "mean_cg_iters": {"gauss": float(sg["mean_cg_iters_A"]), "fast": float(sf["mean_cg_iters_A"])}}
    vlib.write_report(out5, "v5", r <= 1e-12, det5)

if __name__ == "__main__":
    main()
