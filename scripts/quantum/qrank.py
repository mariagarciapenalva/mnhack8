#!/usr/bin/env python3
"""
qrank.py -- the quantum rank of the hybrid pipeline (Layer 8).

Launched as the LAST rank of an MPMD job, next to the CUDA solver:

    mpirun -np P build/heat_solver_het N t dt field out --qrank --qblock X0 Y0 Z0 m [...] \
         : -np 1 python3 scripts/quantum/qrank.py --backend exact|varqite [--qfi encode|decode STEP INDEX BIT] [--qfi-x STEP QUBIT]

Protocol (all on MPI_COMM_WORLD, peer = the solver rank that owns the block):
    handshake  tag 999   : 6 doubles [m, n_steps, dt, alpha, n_cases, N]
    per case, per step:
      recv     tag 1000+s: (2^m+2)^3 doubles (block + one-node halo, x slowest) + [sum, norm]
      send     tag 2000+s: (2^m)^3 doubles (block interior after the step)      + [sum, norm]

Each step: lift the boundary (harmonic extension of the halo) -> w = T - T_ss
-> encode -> [H1 injection] -> imaginary-time step -> decode -> [H3 injection]
-> unlift -> reply. D4 on this side: verify the incoming checksum, log mismatches.

Handoff injection (one-shot, on the faulted case only = the last case):
  --qfi encode STEP INDEX BIT : flip a bit of the ENCODED carrier (theta[INDEX] for
                                varqite, amplitude[INDEX] for exact) before the step
  --qfi decode STEP INDEX BIT : flip a bit of the DECODED float array before reply
  --qfi-x STEP QUBIT          : Pauli-X on the state before the step (quantum-native)
H2 (transit) is injected on the solver side: --fi 4 STEP INDEX BIT.
"""
import argparse, csv, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import qstep, laplacian as lp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["exact", "varqite"], default="exact")
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--decode", choices=["exact", "shots"], default="exact")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--qfi", nargs=4, metavar=("SITE", "STEP", "INDEX", "BIT"))
    ap.add_argument("--qfi-x", nargs=2, type=int, metavar=("STEP", "QUBIT"))
    ap.add_argument("--log", default="qrank_log.csv")
    a = ap.parse_args()
    from mpi4py import MPI
    world = MPI.COMM_WORLD; wr = world.Get_rank(); ws = world.Get_size()
    solver = world.Split(1 if wr == ws - 1 else 0, wr)   # mirror the solver's split; we never use it
    assert wr == ws - 1, "qrank.py must be the last world rank"
    st = MPI.Status()
    cfg = np.empty(6); world.Recv(cfg, source=MPI.ANY_SOURCE, tag=999, status=st)
    peer = st.Get_source()
    m, n_steps, dt, alpha, n_cases, N = int(cfg[0]), int(cfg[1]), float(cfg[2]), float(cfg[3]), int(cfg[4]), int(cfg[5])
    bs, side, h = 2**m, 2**m + 2, 1.0 / N          # the block is embedded: physical spacing is the solver's
    if a.backend == "exact":
        qs = qstep.ExactStatevector(m, 3, alpha, h=h)
    else:
        qs = qstep.AerVarQITE(m, 3, alpha, reps=a.reps, decode=a.decode, shots=a.shots, h=h)
    print(f"[qrank] peer {peer}, block {bs}^3 ({3*m} qubits), {n_cases} cases x {n_steps} steps, dt {dt}, alpha {alpha}, backend {a.backend}", flush=True)

    fi_site, fi_step, fi_idx, fi_bit = (a.qfi[0], int(a.qfi[1]), int(a.qfi[2]), int(a.qfi[3])) if a.qfi else (None, -1, 0, 0)
    x_step, x_q = (a.qfi_x[0], a.qfi_x[1]) if a.qfi_x else (-1, -1)
    log = open(a.log, "w", newline=""); w = csv.writer(log)
    w.writerow(["case", "step", "d4_in_mismatch", "encode_infidelity", "E0", "E1", "norm", "t_step_s"])
    buf_in = np.empty(side**3 + 2); buf_out = np.empty(bs**3 + 2)
    for case in range(n_cases):
        faulted = (case == n_cases - 1) and n_cases == 3
        for s in range(n_steps):
            t0 = time.time()
            world.Recv(buf_in, source=peer, tag=1000 + s)
            pay = buf_in[:-2]; chk_sum, chk_nrm = buf_in[-2], buf_in[-1]
            d4 = max(abs(pay.sum() - chk_sum) / (abs(chk_sum) + 1e-300), abs(np.linalg.norm(pay) - chk_nrm) / (abs(chk_nrm) + 1e-300))
            B = pay.reshape(side, side, side)
            faces = {'x-': B[0, 1:-1, 1:-1], 'x+': B[-1, 1:-1, 1:-1], 'y-': B[1:-1, 0, 1:-1], 'y+': B[1:-1, -1, 1:-1],
                     'z-': B[1:-1, 1:-1, 0], 'z+': B[1:-1, 1:-1, -1]}
            T = B[1:-1, 1:-1, 1:-1].reshape(-1)
            p = qs.encode(T, faces)
            if faulted and fi_site == "encode" and s == fi_step: p.flip_bit(fi_idx % p.vec.size, fi_bit)
            if faulted and s == x_step: p.pauli_x(x_q)
            p = qs.step(p, dt)
            Tn = qs.decode(p)
            if faulted and fi_site == "decode" and s == fi_step: Tn[fi_idx % Tn.size] = qstep._flip_bit_float(Tn[fi_idx % Tn.size], fi_bit)
            buf_out[:-2] = Tn; buf_out[-2] = Tn.sum(); buf_out[-1] = np.linalg.norm(Tn)
            world.Send(buf_out, dest=peer, tag=2000 + s)
            w.writerow([case, s, f"{d4:.3e}", f"{p.meta.get('encode_infidelity', 0):.3e}", f"{p.meta.get('E0', 0):.6e}",
                        f"{p.meta.get('E1', 0):.6e}", f"{p.norm:.6e}", f"{time.time()-t0:.3f}"])
        log.flush()
    log.close()
    print("[qrank] done", flush=True)

if __name__ == "__main__":
    main()
