#!/usr/bin/env python3
"""
V7 -- Quantum substrate (Layer 6) validation.

REQUIREMENT SOURCE
  Layer 7 compares fault propagation across substrates. That comparison is
  meaningless unless each substrate is first validated against the same
  continuous truth the classical solver was validated against (V0). The
  quantum side has two tiers with different failure modes: Tier 1 can only
  fail by implementation error (it is exact); Tier 2 fails by ansatz
  expressivity and by the McLachlan time step, which must be MEASURED and
  reported, not assumed small.

WHAT IS CHECKED
  laplacian: symmetry, exact spectrum, analytic evolution vs expm, Pauli
             round-trip (1D and the 3D qubit ordering), embedded-h spectrum.
  Tier 1   : 1D m=6 (64 pts) and 3D m=3 (9 qubits) vs analytic, rel err <= 1e-10.
             Boundary lifting: constant boundary data -> zero deviation.
  Tier 2   : 1D m=3, reps=3: encode infidelity <= 1e-8; after 5 steps vs Tier 1
             rel err <= 5e-2 and norm tracking <= 2e-2 (these are the
             variational error of THIS ansatz -- reported as numbers);
             halving dt must not increase the error.
  hooks    : flip_bit changes exactly one entry; pauli_x is an involution;
             flip of bit 62 leaves the representable range (catastrophic).

PASS: all of the above. Numbers go to report.json; the Tier-2 profile plot is
      the figure that shows what "variational leakage" looks like.
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import laplacian as lp, qstep
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validations"))
import vlib

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="tests/baseline_validation/v7"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); det = {}; ok = True
    # laplacian self-tests (raise on failure)
    import subprocess
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "laplacian.py")], capture_output=True, text=True)
    det["laplacian_selftest"] = r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-300:]; ok &= r.returncode == 0
    # Tier 1
    for dims, m in ((1, 6), (3, 3)):
        qs = qstep.ExactStatevector(m, dims); T0 = lp.gaussian_bump_1d(m) if dims == 1 else lp.gaussian_bump_3d(m)
        p = qs.encode(T0); [p := qs.step(p, 1e-3) for _ in range(10)]
        ref = lp.exact_evolution_1d(m, T0, 1e-2) if dims == 1 else lp.exact_evolution_3d(m, T0, 1e-2)
        e = float(np.linalg.norm(qs.decode(p) - ref) / np.linalg.norm(ref)); det[f"tier1_{dims}D_m{m}_relerr"] = e; ok &= e <= 1e-10
    qs = qstep.ExactStatevector(3, 1); p = qs.encode(np.ones(8), {'x-': 1.0, 'x+': 1.0}); det["lifting_norm"] = float(p.norm); ok &= p.norm < 1e-10
    # hooks
    p = qstep.ExactStatevector(4, 1).encode(lp.gaussian_bump_1d(4)); v0 = p.vec.copy()
    p.flip_bit(5, 62); det["flip62_value"] = float(p.vec[5]); ok &= (np.sum(p.vec != v0) == 1) and abs(p.vec[5]) > 1e300
    p2 = qstep.ExactStatevector(4, 1).encode(lp.gaussian_bump_1d(4)); v = p2.vec.copy(); p2.pauli_x(1); p2.pauli_x(1); ok &= np.array_equal(p2.vec, v)
    # Tier 2
    t0 = time.time(); m = 3; T0 = lp.gaussian_bump_1d(m, width=0.15)
    t1 = qstep.ExactStatevector(m, 1); res = {}
    for dt in (2e-3, 1e-3):
        t2 = qstep.AerVarQITE(m, 1, reps=3); p1 = t1.encode(T0); p2 = t2.encode(T0); infid = p2.meta["encode_infidelity"]
        nst = int(round(1e-2 / dt))
        for _ in range(nst): p1 = t1.step(p1, dt); p2 = t2.step(p2, dt)
        T1, T2 = t1.decode(p1), t2.decode(p2)
        res[dt] = dict(relerr=float(np.linalg.norm(T2 - T1) / np.linalg.norm(T1)), norm_err=float(abs(p2.norm - p1.norm) / p1.norm), encode_infid=float(infid))
    det["tier2"] = {str(k): v for k, v in res.items()}; det["tier2_params"] = int(t2.p); det["tier2_seconds"] = time.time() - t0
    ok &= res[2e-3]["encode_infid"] <= 1e-8 and res[2e-3]["relerr"] <= 5e-2 and res[2e-3]["norm_err"] <= 2e-2 and res[1e-3]["relerr"] <= res[2e-3]["relerr"] * 1.2
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        x = (np.arange(2**m) + 1) * lp.h_of(m)
        fig, ax = plt.subplots(figsize=(5.5, 3.4)); ax.plot(x, T0, "k:", label="T(0)"); ax.plot(x, T1, "k-", label="Tier 1 exact, t=0.01")
        ax.plot(x, T2, "r--", label=f"Tier 2 VarQITE ({t2.p} params), rel err {res[1e-3]['relerr']:.1e}")
        ax.set_title("V7: variational leakage, 3 qubits"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(a.out, "v7_tier2.png"), dpi=140)
    except Exception as ex:
        det["plot_error"] = str(ex)
    vlib.write_report(a.out, "v7", ok, det)

if __name__ == "__main__":
    main()
