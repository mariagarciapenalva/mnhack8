#!/usr/bin/env python3
"""
scale_matched.py -- Layer 7: identical problem, identical grid, identical fault,
two substrates.

Grid matching is exact: the classical solver with N = 2^m + 1 cells has
2^m interior nodes per axis at spacing h = 1/N -- the same points the quantum
block uses (h_of(m) = 1/(2^m+1)). So N=9 <-> 9 qubits, N=17 <-> 12, N=33 <-> 15.

Problem: source-free, Gaussian-bump initial condition, homogeneous alpha = 1.
Faults (all one-shot at step S, colored scatter so the classical floor is 0):
  classical  G  : IEEE flip of bit b on node (ix,iy,iz)   [--bin needed]
  quantum T1 A  : IEEE flip of bit b on amplitude j of the SAME node
  quantum T1 X  : Pauli-X on qubit q (quantum-native channel, own class)
  quantum T2 th : IEEE flip of bit b on theta[k]           [--tier2, small m]
Measurements on every side, identical definitions:
  E(t) = ||T_f - T_clean|| / ||T_clean||   on interior nodes
  spectral bands of delta T: energy fraction in the lowest / middle / highest
  third of the discrete Laplacian spectrum (classical prediction: each mode
  decays as exp(-lambda_k t); the high band must vanish first).
Without --bin the classical side is the NumPy FD backward-Euler stand-in
(same operator as the quantum block; labelled as such in the plots).
"""
import argparse, os, sys, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import laplacian as lp, qstep
from v8_hybrid_mock import ClassicalFD, gaussian_full
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validations"))
import vlib

def bands(delta, m, h):
    """Energy fraction of delta in low/mid/high thirds of the 3D spectrum."""
    lam, V = lp.eigen1d(m, 1.0, h); n = 2**m
    C = np.einsum('ia,jb,kc,ijk->abc', V, V, V, delta.reshape(n, n, n))
    L3 = lam[:, None, None] + lam[None, :, None] + lam[None, None, :]
    e = C**2; tot = e.sum() + 1e-300
    q1, q2 = np.percentile(L3, [33.3, 66.7])
    return np.array([e[L3 <= q1].sum(), e[(L3 > q1) & (L3 <= q2)].sum(), e[L3 > q2].sum()]) / tot

def classical_cuda(a, N, S, target, bit, out):
    fp = os.path.join(out, f"homog{N}"); vlib.gen_field(a.gen, N, "homogeneous", fp)
    for tag, fi in (("clean", []), ("fault", ["--fi", "1", str(S), str(target), str(bit)])):
        d = os.path.join(out, f"cuda_{tag}"); os.makedirs(d, exist_ok=True)
        vlib.run([a.bin, str(N), str(a.t), str(a.dt), fp, d, "--tag", "sm", "--snap", "1", "--ic", "gaussian", "--no-source",
                  "--kernel", "fast", "--scatter", "colored", "--dump-snaps"] + fi, log=os.path.join(out, "run.log"))
    n_steps = int(round(a.t / a.dt)); nn = N + 1; E, B = [], []
    for s in range(n_steps):
        A = np.fromfile(os.path.join(out, "cuda_clean", f"snap_sm_cleanA_{s:05d}_rank0.bin")).reshape(nn, nn, nn)[1:N, 1:N, 1:N].reshape(-1)
        F = np.fromfile(os.path.join(out, "cuda_fault", f"snap_sm_fault_{s:05d}_rank0.bin")).reshape(nn, nn, nn)[1:N, 1:N, 1:N].reshape(-1)
        E.append(np.linalg.norm(F - A) / np.linalg.norm(A)); B.append(bands(F - A, a.m, 1.0 / N))
    return np.array(E), np.array(B), "classical (CUDA FEM, G bit %d)" % bit

def classical_mock(a, N, S, j, bit):
    h = 1.0 / N; cls = ClassicalFD(a.m, 1.0, h); T = gaussian_full(a.m, h); Tf = T.copy(); E, B = [], []
    for s in range(int(round(a.t / a.dt))):
        if s == S: Tf[j] = qstep._flip_bit_float(Tf[j], bit)
        T = cls.step(T, a.dt); Tf = cls.step(Tf, a.dt)
        E.append(np.linalg.norm(Tf - T) / np.linalg.norm(T)); B.append(bands(Tf - T, a.m, h))
    return np.array(E), np.array(B), "classical (NumPy FD stand-in, bit %d)" % bit

def quantum_tier1(a, N, S, j, bit, q):
    h = 1.0 / N; qs = qstep.ExactStatevector(a.m, 3, 1.0, h=h); T0 = gaussian_full(a.m, h)
    pc, pa, px = qs.encode(T0), qs.encode(T0), qs.encode(T0); Ea, Ex, Ba, Bx = [], [], [], []
    for s in range(int(round(a.t / a.dt))):
        if s == S: pa.flip_bit(j, bit); px.pauli_x(q)
        pc, pa, px = qs.step(pc, a.dt), qs.step(pa, a.dt), qs.step(px, a.dt)
        Tc, Ta, Tx = qs.decode(pc), qs.decode(pa), qs.decode(px)
        Ea.append(np.linalg.norm(Ta - Tc) / np.linalg.norm(Tc)); Ba.append(bands(Ta - Tc, a.m, h))
        Ex.append(np.linalg.norm(Tx - Tc) / np.linalg.norm(Tc)); Bx.append(bands(Tx - Tc, a.m, h))
    return (np.array(Ea), np.array(Ba), f"quantum T1 (amplitude bit {bit})"), (np.array(Ex), np.array(Bx), f"quantum T1 (Pauli-X q{q})")

def quantum_tier2(a, N, S, k, bit):
    h = 1.0 / N; qs = qstep.AerVarQITE(a.m, 3, 1.0, reps=2, h=h); T0 = gaussian_full(a.m, h)
    pc, pf = qs.encode(T0), qs.encode(T0); E, B = [], []
    for s in range(int(round(a.t / a.dt))):
        if s == S: pf.flip_bit(k % pf.vec.size, bit)
        pc, pf = qs.step(pc, a.dt), qs.step(pf, a.dt); Tc, Tf = qs.decode(pc), qs.decode(pf)
        E.append(np.linalg.norm(Tf - Tc) / np.linalg.norm(Tc)); B.append(bands(Tf - Tc, a.m, h))
    return np.array(E), np.array(B), f"quantum T2 VarQITE (theta[{k}] bit {bit}, {qs.p} params)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin"); ap.add_argument("--gen", default="scripts/gen_field.py"); ap.add_argument("--out", default="results/scale_matched")
    ap.add_argument("--m", type=int, default=3); ap.add_argument("--t", type=float, default=4e-3); ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--S", type=int, default=10); ap.add_argument("--bit", type=int, default=55); ap.add_argument("--qubit", type=int, default=0)
    ap.add_argument("--tier2", action="store_true", help="also run VarQITE (m<=3 recommended)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    N = 2**a.m + 1; bs = 2**a.m; c = bs // 2
    j = c * bs * bs + c * bs + c                           # centre node in interior (quantum) indexing
    target = vlib.node_index(c + 1, c + 1, c + 1, N)     # the same physical node in the solver's numbering
    series = []
    series.append(classical_cuda(a, N, a.S, target, a.bit, a.out) if a.bin else classical_mock(a, N, a.S, j, a.bit))
    q1a, q1x = quantum_tier1(a, N, a.S, j, a.bit, a.qubit); series += [q1a, q1x]
    if a.tier2: series.append(quantum_tier2(a, N, a.S, 3, a.bit))
    t = (np.arange(len(series[0][0])) + 1) * a.dt
    with open(os.path.join(a.out, "scale_matched.csv"), "w") as f:
        f.write("substrate,E_max,step_at_max,E_final,healing_ratio,low_band_final,high_band_at_S\n")
        for E, B, lab in series:
            i = int(np.nanargmax(np.where(np.isfinite(E), E, -1)))
            f.write(f"{lab},{E[i]:.3e},{i},{E[-1]:.3e},{E[-1]/E[i] if E[i]>0 else 0:.3e},{B[-1][0]:.3f},{B[a.S][2]:.3f}\n")
    print(open(os.path.join(a.out, "scale_matched.csv")).read())
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for E, B, lab in series: ax[0].semilogy(t, np.maximum(E, 1e-18), label=lab)
        ax[0].axvline((a.S + 1) * a.dt, color="gray", ls=":"); ax[0].set_xlabel("t"); ax[0].set_ylabel("E(t)"); ax[0].legend(fontsize=7)
        ax[0].set_title(f"scale-matched: N={N} <-> {3*a.m} qubits, fault at step {a.S}")
        for (E, B, lab), ls in zip(series, ("-", "--", ":", "-.")):
            ax[1].plot(t, B[:, 2], "r" + ls, label=f"high band, {lab.split('(')[0]}"); ax[1].plot(t, B[:, 0], "b" + ls, label=f"low band")
        ax[1].set_xlabel("t"); ax[1].set_ylabel("energy fraction of delta T"); ax[1].set_title("spectral bands of the perturbation"); ax[1].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "scale_matched.png"), dpi=140)
    except Exception as ex:
        print("plot skipped:", ex)

if __name__ == "__main__":
    main()
