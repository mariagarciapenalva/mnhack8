#!/usr/bin/env python3
"""
V8 -- Hybrid coupling (Layer 8) validated without CUDA or MPI.

REQUIREMENT SOURCE
  Before the CUDA solver talks to qrank.py, the coupling scheme itself must be
  shown consistent: a block evolved by the quantum step with boundary data
  FROZEN at t_n, substituted into a full-domain classical step, must converge
  to the pure classical solution as dt -> 0. If it does not, every hybrid
  result is a coupling artifact, not a fault-tolerance result.

WHAT IS CHECKED (everything in NumPy: FD Laplacian on 2^M interior points,
backward Euler via the tensor eigenbasis as the "classical" side; the SAME
qstep.ExactStatevector used by qrank.py as the quantum block)
  (1) Consistency: E_block(t_final) = ||T_hybrid - T_classical|| / ||T_classical||
      on the block, for dt and dt/2 at fixed t_final -> ratio in [1.6, 2.4]
      (first-order splitting: frozen boundary + exp vs BE both O(dt) accumulated).
  (2) Zero-fault baseline: D4 checksum mismatch == 0 on every step.
  (3) Handoff injections, one at a time at step S, bit 55:
        H1 encode  : flip amplitude[j] after encoding      -> caught by PHYSICS (block err jump), NOT by checksum
        H2 transit : flip payload[j] on the wire           -> caught by CHECKSUM (d4_in > 0)
        H3 decode  : flip decoded float before reply       -> caught by PHYSICS only if the flip is large
      The asymmetry is the finding: a transit checksum sees H2 and nothing
      else; the classical-vs-quantum block comparison is the only signal for
      H1/H3. Both are needed, and both are what run_case + qrank.py compute.

PASS: (1) ratio in range, (2) exact zeros, (3) H2 caught by checksum at step S
      and H1 caught by physics at step S.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import laplacian as lp, qstep
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validations"))
import vlib

class ClassicalFD:
    """Full-domain backward Euler on 2^M interior nodes per axis, spacing h."""
    def __init__(self, M, alpha, h):
        self.M, self.n, self.h = M, 2**M, h
        self.lam, self.V = lp.eigen1d(M, alpha, h)
        self.L3 = self.lam[:, None, None] + self.lam[None, :, None] + self.lam[None, None, :]
    def step(self, T, dt):
        A = T.reshape(self.n, self.n, self.n)
        C = np.einsum('ia,jb,kc,ijk->abc', self.V, self.V, self.V, A) / (1.0 + dt * self.L3)
        return np.einsum('ia,jb,kc,abc->ijk', self.V, self.V, self.V, C).reshape(-1)

def gaussian_full(M, h, center=0.5, width=0.12):
    x = (np.arange(2**M) + 1) * h
    b = np.exp(-(x - center)**2 / (2 * width**2))
    return np.einsum('i,j,k->ijk', b, b, b).reshape(-1)

def run(M, m, off, alpha, dt, n_steps, fault=None, S=5, bit=55):
    """fault in {None,'H1','H2','H3'}, injected at the block centre. Returns per-step block err, d4_in, final fields."""
    h = 1.0 / (2**M + 1); n = 2**M; bs, side = 2**m, 2**m + 2
    j_pay = (side//2)*side*side + (side//2)*side + side//2          # centre of the payload cube (interior node)
    j_in  = (bs//2)*bs*bs + (bs//2)*bs + bs//2                      # centre of the block
    cls = ClassicalFD(M, alpha, h); qs = qstep.ExactStatevector(m, 3, alpha, h=h)
    Tc = gaussian_full(M, h); Th = Tc.copy()
    sl = slice(off - 1, off + bs + 1)                                # block + halo in interior index space
    blk_err, d4s = [], []
    for s in range(n_steps):
        # ---- quantum side receives block+halo of T^n (payload), lifts, steps
        A = Th.reshape(n, n, n)
        pay = A[sl, sl, sl].copy().reshape(-1)
        chk = (pay.sum(), np.linalg.norm(pay))
        if fault == 'H2' and s == S: pay[j_pay] = qstep._flip_bit_float(pay[j_pay], bit)
        d4 = max(abs(pay.sum() - chk[0]) / (abs(chk[0]) + 1e-300), abs(np.linalg.norm(pay) - chk[1]) / (abs(chk[1]) + 1e-300))
        B = pay.reshape(side, side, side)
        faces = {'x-': B[0, 1:-1, 1:-1], 'x+': B[-1, 1:-1, 1:-1], 'y-': B[1:-1, 0, 1:-1], 'y+': B[1:-1, -1, 1:-1],
                 'z-': B[1:-1, 1:-1, 0], 'z+': B[1:-1, 1:-1, -1]}
        p = qs.encode(B[1:-1, 1:-1, 1:-1].reshape(-1), faces)
        if fault == 'H1' and s == S: p.flip_bit(j_in, bit)
        p = qs.step(p, dt)
        Tq = qs.decode(p)
        if fault == 'H3' and s == S: Tq[j_in] = qstep._flip_bit_float(Tq[j_in], bit)
        # ---- classical full-domain step (both trajectories)
        Tc = cls.step(Tc, dt); Th = cls.step(Th, dt)
        # ---- substitute the quantum block into the hybrid trajectory; block error vs classical
        Hn = Th.reshape(n, n, n); Cn = Tc.reshape(n, n, n)
        inner = slice(off, off + bs)
        c_blk = Hn[inner, inner, inner].reshape(-1)                  # classical prediction on the block (hybrid trajectory)
        blk_err.append(np.linalg.norm(Tq - c_blk) / np.linalg.norm(c_blk))
        Hn[inner, inner, inner] = Tq.reshape(bs, bs, bs); Th = Hn.reshape(-1)
        d4s.append(d4)
    return np.array(blk_err), np.array(d4s), Tc, Th

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/baseline_validation/v8"); ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--m", type=int, default=3); ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=2e-4); ap.add_argument("--t", type=float, default=4e-3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    off = (2**a.M - 2**a.m) // 2                                   # centred block
    det = {}; ok = True
    # (1) consistency
    errs = {}
    for dt in (a.dt, a.dt / 2):
        n_steps = int(round(a.t / dt))
        be, d4, Tc, Th = run(a.M, a.m, off, a.alpha, dt, n_steps)
        n = 2**a.M; inner = slice(off, off + 2**a.m)
        cb = Tc.reshape(n, n, n)[inner, inner, inner]; hb = Th.reshape(n, n, n)[inner, inner, inner]
        errs[dt] = np.linalg.norm(hb - cb) / np.linalg.norm(cb)
        det[f"block_err_final_dt{dt:g}"] = float(errs[dt]); det[f"block_err_per_step_max_dt{dt:g}"] = float(be.max())
        if dt == a.dt: det["d4_clean_max"] = float(d4.max()); ok &= bool((d4 == 0).all())
    ratio = errs[a.dt] / errs[a.dt / 2]; det["consistency_ratio_dt_over_dt2"] = float(ratio)
    ok &= 1.6 <= ratio <= 2.4
    # (3) handoff injections
    n_steps = int(round(a.t / a.dt)); S = n_steps // 2
    be0, _, _, _ = run(a.M, a.m, off, a.alpha, a.dt, n_steps)
    cov = {}
    for f in ("H1", "H2", "H3"):
        be, d4, _, _ = run(a.M, a.m, off, a.alpha, a.dt, n_steps, fault=f, S=S)
        phys_first = int(np.argmax(be > 10 * be0.max())) if (be > 10 * be0.max()).any() else -1
        chk_first = int(np.argmax(d4 > 0)) if (d4 > 0).any() else -1
        cov[f] = dict(checksum_first=chk_first, physics_first=phys_first, block_err_at_S=float(be[S]), block_err_clean_at_S=float(be0[S]))
    det["handoff_coverage"] = cov
    ok &= cov["H2"]["checksum_first"] == S and cov["H1"]["physics_first"] == S
    ok &= cov["H1"]["checksum_first"] == -1 and cov["H3"]["checksum_first"] == -1   # the asymmetry is real
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.4)); t = np.arange(n_steps) * a.dt
        ax.semilogy(t, be0, "k-", label="clean (coupling error)")
        for f in ("H1", "H2", "H3"):
            be, _, _, _ = run(a.M, a.m, off, a.alpha, a.dt, n_steps, fault=f, S=S); ax.semilogy(t, be, label=f"{f} bit 55 @S")
        ax.axvline(S * a.dt, color="gray", ls=":"); ax.set_xlabel("t"); ax.set_ylabel("block err vs classical"); ax.legend(fontsize=8)
        ax.set_title(f"V8 hybrid block coupling, {2**a.m}^3 block in {2**a.M}^3, {3*a.m} qubits"); fig.tight_layout()
        fig.savefig(os.path.join(a.out, "v8_hybrid.png"), dpi=140)
    except Exception as ex:
        det["plot_error"] = str(ex)
    vlib.write_report(a.out, "v8", ok, det)

if __name__ == "__main__":
    main()
