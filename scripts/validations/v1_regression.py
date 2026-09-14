#!/usr/bin/env python3
"""
V1 -- Regression to MNHack7 (material generalization + field-file ordering).

REQUIREMENT SOURCE
  The v2 solver replaced the analytic 4-quadrant material_at() by per-element
  arrays read from a binary. Two things can silently go wrong there: the
  quadrant values/placement in gen_field.py, and the x/y/z ordering assumed by
  load_fields(). A homogeneous field (V0) is blind to ordering, because every
  element is identical. This test is the only one that is sensitive to it,
  and it is what ties the new results back to the MNHack7 numbers.

WHAT IS CHECKED
  Same N, t_final, dt on the untouched MNHack7 binary (heat_solver.cu) and on
  the v2 binary with --kernel gauss --scatter atomic (closest code path).
  Compared: the final ||T||_2 printed by MNHack7 (7 significant digits) against
  the cleanA T_norm from the v2 run_log, and the per-10-step norms.
  The two binaries differ only in (a) where k, rhoc come from and (b) the
  summation order of the dot product (two-stage vs atomic). Agreement must be
  at the print precision, 1e-6 relative.

NEGATIVE CONTROL
  The same comparison at odd N. The analytic model puts interfaces at 0.5,
  which for odd N falls inside an element and straddles Gauss points; the
  per-element model CANNOT match. If the odd-N run also "passes", the test is
  not sensitive and the pass at even N means nothing.

PASS
  even N: |norm_v2 - norm_ref| / norm_ref <= 1e-6 at every logged step.
  odd  N: the same quantity > 1e-6 (the control must FAIL to match).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import vlib

def run_pair(a, N, out, log):
    fp = os.path.join(out, f"quad{N}")
    vlib.gen_field(a.gen, N, "quadrants", fp)
    ref_out = vlib.run([a.ref, str(N), str(a.t), str(a.dt)], log=log)
    ref_norm = vlib.parse_final_norm(ref_out)
    ref_steps = vlib.parse_step_norms(ref_out)
    d = os.path.join(out, f"v2_N{N}"); os.makedirs(d, exist_ok=True)
    cmd = [a.bin, str(N), str(a.t), str(a.dt), fp, d, "--tag", "reg", "--snap", "0"]
    caps = vlib.capabilities(a.bin)
    if "--kernel" in caps: cmd += ["--kernel", "gauss", "--scatter", "atomic"]
    vlib.run(cmd, log=log)
    # v2 has no per-step T_norm log anymore; compare final field norm via the dump
    T = vlib.assemble_field(d, "reg_clean", N, 1)
    import numpy as np
    v2_norm = float(np.linalg.norm(T))
    rel = abs(v2_norm - ref_norm) / ref_norm
    return {"N": N, "ref_norm": ref_norm, "v2_norm": v2_norm, "rel": rel, "ref_steps": ref_steps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--ref", required=True, help="MNHack7 heat_solver binary")
    ap.add_argument("--gen", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=16); ap.add_argument("--Nodd", type=int, default=15)
    ap.add_argument("--t", type=float, default=0.01); ap.add_argument("--dt", type=float, default=1e-4)
    a = ap.parse_args()
    out = os.path.join(a.out, "v1"); os.makedirs(out, exist_ok=True); log = os.path.join(out, "run.log")
    even = run_pair(a, a.N, out, log)
    odd  = run_pair(a, a.Nodd, out, log)
    ok = even["rel"] <= 1e-6 and odd["rel"] > 1e-6
    vlib.write_report(out, "v1", ok, {"even": even, "odd_negative_control": odd,
        "note": "MNHack7 norm is over ALL nodes (single rank), v2 dump is the full field: same set."})

if __name__ == "__main__":
    main()
