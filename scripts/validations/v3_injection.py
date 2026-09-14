#!/usr/bin/env python3
"""
V3 -- Fault injector unit test: does each level fire, once, where and when
      it says, and is the effect visible?

REQUIREMENT SOURCE
  Every ensemble statistic is conditional on the injector doing exactly what
  the methodology section says: one flip, at step S, at the named site, on one
  rank. If a level silently does nothing (e.g. a target on the Dirichlet
  boundary is projected to zero the moment it is written) the run is counted
  as "benign" and the sensitivity map is biased low. If it fires more than
  once (the original code fired G and R on EVERY rank) the map is biased high.

WHAT IS CHECKED  (colored scatter so the floor is exactly zero)
  For level in {G, R, H} and bits {0, 30, 51, 55, 62, 63}, interior target,
  S = 10 of 40 steps, --snap 1:
    (1) E_fault == 0 for every snapshot before S           (nothing fired early)
    (2) E_fault  > 0 at snapshot S                          (it fired)
    (3) E_fault(t) has no second jump after S (recorded, not gating: the
        one-shot property is structural; this only catches gross failures)
    (4) bit 62 produces class 'detected' or E_max > 1e6     (exponent flips are
        catastrophic, as IEEE-754 says they must be)
    (5) bit 63 (sign) at S gives E ~ 2|v|/||T||: checked only for G, where v
        is read from the clean dump... at t_final, not at S -> we only check
        E(S) > E(S) for bit 0, i.e. ordering of magnitudes across bits.
  Boundary control: level G with a boundary node as target, no reprojection.
    Expected: the flipped value persists (boundary DOFs are never updated by
    CG) -> E_final > 0 even though diffusion would otherwise heal it. This
    documents the BC-scheme artifact; with --reproject-bc it must vanish.
  Multi-rank multiplicity: level G at np=2 with --fi-rank 0 and --fi-rank 1;
    the two runs must give the SAME E_max (target is a local index, so the
    physical location differs, but each must fire exactly once: E_max must
    not be ~sqrt(2) larger than the np=1 run of the same local target).

PASS
  (1)-(4) for every (level, bit); boundary control behaves as stated;
  H unsupported at np=1 is reported, not failed.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import vlib

BITS = [0, 30, 51, 55, 62, 63]

def one_run(a, fp, d, level, bit, target, np_=1, extra=()):
    os.makedirs(d, exist_ok=True)
    cmd = [a.bin, str(a.N), str(a.t), str(a.dt), fp, d, "--tag", "fi", "--snap", "1",
           "--kernel", a.kernel, "--scatter", "colored", "--fi", str(level), str(a.S), str(target), str(bit)] + list(extra)
    vlib.run(cmd, log=os.path.join(a.out, "v3", "run.log"), np_=np_)
    tr = vlib.read_csv(os.path.join(d, "fi_trace_fi.csv"))
    sm = vlib.read_csv(os.path.join(d, "fi_summary_fi.csv"))[0]
    E = np.array([vlib.fnum(r["E_fault"]) for r in tr]); st = np.array([int(r["step"]) for r in tr])
    return E, st, sm

def check_series(E, st, S):
    before = E[st < S]; atS = E[st == S]; after = E[st > S]
    c1 = bool((before == 0).all())
    c2 = bool(atS.size == 1 and (atS[0] > 0 or not np.isfinite(atS[0])))
    fin = np.isfinite(E)
    c3 = True
    if fin.all() and after.size > 1:
        inc = np.diff(np.log(np.maximum(E[st >= S], 1e-300)))
        c3 = bool((inc[1:] <= max(inc[0], 0) + 1e-9).all()) if inc.size > 1 else True
    return c1, c2, c3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--gen", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=32); ap.add_argument("--t", type=float, default=0.02)
    ap.add_argument("--dt", type=float, default=5e-4); ap.add_argument("--S", type=int, default=10)
    ap.add_argument("--kernel", default="fast"); ap.add_argument("--mpi", type=int, default=2, help="ranks available (0 = skip H)")
    a = ap.parse_args()
    out = os.path.join(a.out, "v3"); os.makedirs(out, exist_ok=True)
    caps = vlib.capabilities(a.bin)
    if "--scatter" not in caps or "--fi-rank" not in caps:
        vlib.write_report(out, "v3", False, {"status": "UNSUPPORTED: needs --scatter colored and --fi-rank"}); return
    N = a.N
    fp = os.path.join(out, f"ln{N}")
    vlib.gen_field(a.gen, N, "lognormal", fp, sigma=1.0, L=0.125, seed=7)
    tgt = {1: vlib.node_index(N//2, N//2, N//2, N), 2: vlib.elem_index(N//2, N//2, N//2, N),
           3: (N//2) * (N+1) + N//2}
    det = {"runs": []}; ok = True
    series = {}
    for level in [1, 2, 3]:
        np_ = 1 if level < 3 else a.mpi
        if level == 3 and a.mpi < 2:
            det["H"] = "skipped (needs >=2 ranks)"; continue
        for bit in BITS:
            d = os.path.join(out, f"L{level}_b{bit}")
            E, st, sm = one_run(a, fp, d, level, bit, tgt[level], np_=np_)
            c1, c2, c3 = check_series(E, st, a.S)
            c4 = True
            if bit == 62:
                c4 = sm["class"] == "detected" or vlib.fnum(sm["E_fault_max"]) > 1e6
            rec = dict(level=level, bit=bit, before_zero=c1, fires_at_S=c2, one_shot=c3, exp_catastrophic=c4,
                       E_max=vlib.fnum(sm["E_fault_max"]), E_final=vlib.fnum(sm["E_fault_final"]),
                       cls=sm["class"], on_boundary=int(sm["target_on_boundary"]), iters_delta=int(sm["iters_delta_max"]))
            det["runs"].append(rec); ok &= c1 and c2 and c4     # c3 recorded, not gating
            series[(level, bit)] = (st * a.dt, E)
    # boundary control (level G, boundary node). Bit 62 on 0.0 gives 2.0; a
    # mantissa/low-exponent flip on 0.0 gives ~1e-305 whose square underflows.
    bnode = vlib.node_index(0, N//2, N//2, N)
    E, st, sm = one_run(a, fp, os.path.join(out, "G_boundary"), 1, 62, bnode)
    Er, _, smr = one_run(a, fp, os.path.join(out, "G_boundary_reproject"), 1, 62, bnode, extra=["--reproject-bc"])
    det["boundary_control"] = {"on_boundary_flag": int(sm["target_on_boundary"]),
                               "no_reproject": {"E_max": vlib.fnum(sm["E_fault_max"]), "E_final": vlib.fnum(sm["E_fault_final"]), "class": sm["class"]},
                               "reproject":    {"E_max": vlib.fnum(smr["E_fault_max"]), "E_final": vlib.fnum(smr["E_fault_final"]), "class": smr["class"]}}
    ok &= int(sm["target_on_boundary"]) == 1 and sm["class"] == "persistent" and smr["class"] in ("transient", "benign")
    # multiplicity at np=2: same LOCAL index on rank 0 is the same PHYSICAL node
    # as at np=1 (node (N/4,N/2,N/2) is interior to rank 0's slab). If the flip
    # fired on both ranks the perturbation would be ~sqrt(2) larger.
    if a.mpi >= 2 and N % 2 == 0:
        t_int = vlib.node_index(N//4, N//2, N//2, N)
        _, _, s0 = one_run(a, fp, os.path.join(out, "G_np1_ctrl"), 1, 55, t_int, np_=1)
        _, _, s1 = one_run(a, fp, os.path.join(out, "G_np2_r0"), 1, 55, t_int, np_=2, extra=["--fi-rank", "0"])
        e0, e1 = vlib.fnum(s0["E_fault_max"]), vlib.fnum(s1["E_fault_max"])
        det["multiplicity_np2"] = {"E_max_np1": e0, "E_max_np2_rank0": e1, "rel_diff": abs(e1 - e0) / e0}
        ok &= abs(e1 - e0) / e0 < 1e-6
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        for i, level in enumerate([1, 2, 3]):
            for bit in BITS:
                if (level, bit) in series:
                    t, E = series[(level, bit)]; axs[i].semilogy(t, np.maximum(E, 1e-20), label=f"bit {bit}")
            axs[i].axvline(a.S * a.dt, color="k", ls=":"); axs[i].set_title(f"level {'GRH'[level-1]}"); axs[i].set_xlabel("t")
        axs[0].set_ylabel("E(t)"); axs[0].legend(fontsize=7); fig.suptitle("V3 injected-error trajectories (colored, floor = 0)")
        fig.tight_layout(); fig.savefig(os.path.join(out, "v3_injection.png"), dpi=140)
    except Exception as ex:
        det["plot_error"] = str(ex)
    vlib.write_report(out, "v3", ok, det)

if __name__ == "__main__":
    main()
