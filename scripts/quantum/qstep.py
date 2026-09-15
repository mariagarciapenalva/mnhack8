"""
qstep.py -- Layer 6: the quantum substrate behind one interface.

    class QuantumStep:
        encode(T, boundary=None) -> Params     # floats -> (norm, theta | amplitudes)
        step(params, dtau)       -> Params     # imaginary-time evolution by dtau
        decode(params)           -> np.ndarray # -> floats (T on the block)

Fault hooks live on Params, NOT on the backend, so the same injection means the
same thing on the NumPy tier, the Aer tier, and (later) a hardware tier:
    params.flip_bit(index, bit)   IEEE-754 bit flip on one stored float
                                  (theta[index] for VarQITE, amplitude for Tier 1)
    params.pauli_x(qubit)         hardware-native bit flip on the quantum state

Physics of the coupling (Layer 8): the block is evolved for the DEVIATION
w = T - T_ss, where T_ss is the harmonic extension of the block's boundary
data (solve L T_ss = boundary contribution). Then dw/dt = -H w exactly, i.e.
pure imaginary-time evolution with no source term, and T = w + T_ss on decode.
The norm ||w|| is a classical float carried in Params; for VarQITE it is
propagated by d ln||w||/dt = -<H> (trapezoid on the step's endpoints).

Tier 1  ExactStatevector : exp(-dt H) applied through the exact eigen-expansion
                           (tensor-structured; n = 3m qubits up to ~18 in NumPy,
                           more with CuPy). Ground truth on the quantum
                           representation. No ansatz, no variational error.
Tier 2  AerVarQITE       : RealAmplitudes ansatz, McLachlan variational principle
                           (qiskit_algorithms.VarQITE), Aer Estimator. theta is
                           the classical float vector of the hybrid loop.
                           decode='exact' reads the statevector; decode='shots'
                           estimates |a_j|^2 from counts and takes signs from the
                           previous decode (sign continuity -- a stated limitation;
                           sign-resolving readout needs interference circuits).
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import laplacian as lp

# ---------------------------------------------------------------------------
def _flip_bit_float(x: float, bit: int) -> float:
    u = np.array([x], dtype=np.float64).view(np.uint64)
    u ^= np.uint64(1) << np.uint64(bit)
    return float(u.view(np.float64)[0])

@dataclass
class Params:
    """Classical carrier of the quantum state between handoffs."""
    kind: str                       # 'amplitudes' (Tier 1) or 'theta' (Tier 2)
    vec: np.ndarray                 # amplitudes (unit norm) or ansatz parameters
    norm: float                     # ||w||, carried classically
    n_qubits: int
    meta: dict = field(default_factory=dict)
    def flip_bit(self, index: int, bit: int):
        self.vec = self.vec.copy(); self.vec[index] = _flip_bit_float(self.vec[index], bit)
    def flip_norm_bit(self, bit: int):
        self.norm = _flip_bit_float(self.norm, bit)
    def pauli_x(self, qubit: int):
        if self.kind == 'amplitudes':
            n = self.vec.size; j = np.arange(n); self.vec = self.vec[j ^ (1 << qubit)]
        else:
            self.meta.setdefault('pending_x', []).append(qubit)   # applied by the backend at next step
    def checksum(self):
        return float(np.sum(self.vec)), float(np.linalg.norm(self.vec)), self.norm

# ---------------------------------------------------------------------------
def harmonic_extension(m, dims, faces, alpha=1.0, h=None):
    """T_ss on the 2^m^dims interior block with Dirichlet data `faces` on the
    surrounding layer. faces: dict {'x-':A,'x+':A,'y-':..} of (2^m)^(dims-1)
    arrays (dims=3) or scalars (dims=1). Solves L T_ss = rhs with rhs from the
    boundary neighbours (FD: alpha/h^2 * neighbour value)."""
    n = 2**m; hp = lp.h_of(m, h); c = alpha / hp**2
    if dims == 1:
        rhs = np.zeros(n); rhs[0] += c * faces['x-']; rhs[-1] += c * faces['x+']
        return np.linalg.solve(lp.lap1d(m, alpha, h), rhs)
    rhs = np.zeros((n, n, n))
    rhs[0, :, :]  += c * faces['x-']; rhs[-1, :, :] += c * faces['x+']
    rhs[:, 0, :]  += c * faces['y-']; rhs[:, -1, :] += c * faces['y+']
    rhs[:, :, 0]  += c * faces['z-']; rhs[:, :, -1] += c * faces['z+']
    # solve with the tensor eigenbasis: L = sum of 1D operators -> diagonal in V(x)V(x)V
    lam, V = lp.eigen1d(m, alpha, h)
    C = np.einsum('ia,jb,kc,ijk->abc', V, V, V, rhs)
    L3 = lam[:, None, None] + lam[None, :, None] + lam[None, None, :]
    return np.einsum('ia,jb,kc,abc->ijk', V, V, V, C / L3).reshape(-1)

# ---------------------------------------------------------------------------
class QuantumStep:
    def __init__(self, m, dims=1, alpha=1.0, h=None):
        self.m, self.dims, self.alpha, self.h = m, dims, alpha, h
        self.n_qubits = m * dims
        self.N = 2**self.n_qubits
        self.T_ss = np.zeros(self.N)
    # -- boundary lifting shared by both tiers
    def _lift(self, T, faces):
        self.T_ss = harmonic_extension(self.m, self.dims, faces, self.alpha, self.h) if faces is not None else np.zeros(self.N)
        return T - self.T_ss
    def _unlift(self, w):
        return w + self.T_ss
    def encode(self, T, faces=None) -> Params: raise NotImplementedError
    def step(self, p: Params, dtau: float) -> Params: raise NotImplementedError
    def decode(self, p: Params) -> np.ndarray: raise NotImplementedError

# ---------------------------------------------------------------------------
class ExactStatevector(QuantumStep):
    """Tier 1."""
    def encode(self, T, faces=None):
        w = self._lift(np.asarray(T, float), faces); nrm = np.linalg.norm(w)
        return Params('amplitudes', w / nrm if nrm > 0 else w, nrm, self.n_qubits)
    def step(self, p, dtau):
        w = p.vec * p.norm
        w = lp.exact_evolution_1d(self.m, w, dtau, self.alpha, self.h) if self.dims == 1 else lp.exact_evolution_3d(self.m, w, dtau, self.alpha, self.h)
        nrm = np.linalg.norm(w)
        return Params('amplitudes', w / nrm if nrm > 0 else w, nrm, self.n_qubits, dict(p.meta))
    def decode(self, p):
        return self._unlift(p.vec * p.norm)

# ---------------------------------------------------------------------------
class AerVarQITE(QuantumStep):
    """Tier 2. reps = ansatz depth. decode: 'exact' | 'shots'."""
    def __init__(self, m, dims=1, alpha=1.0, reps=3, decode='exact', shots=4096, seed=7, fit_restarts=3, h=None):
        super().__init__(m, dims, alpha, h)
        self.H = lp.pauli_1d(m, alpha, h) if dims == 1 else lp.pauli_3d(m, alpha, h)
        # A flat circuit of ry/cx, not a composite gate: Aer's assembler
        # rejects the RealAmplitudes block, and VarQITE needs plain parameters.
        try:
            from qiskit.circuit.library import real_amplitudes
            self.ansatz = real_amplitudes(self.n_qubits, reps=reps, entanglement='reverse_linear')
        except ImportError:
            from qiskit.circuit.library import RealAmplitudes
            self.ansatz = RealAmplitudes(self.n_qubits, reps=reps, entanglement='reverse_linear').decompose()
        self.p = self.ansatz.num_parameters
        self.decode_mode, self.shots, self.seed, self.fit_restarts = decode, shots, seed, fit_restarts
        self._last_sign = None
    # -- exact statevector of the ansatz (simulation side; on hardware this is state prep)
    def _state(self, theta):
        from qiskit.quantum_info import Statevector
        return np.real(Statevector(self.ansatz.assign_parameters(theta)).data)
    def _energy(self, theta):
        from qiskit.quantum_info import Statevector
        return float(np.real(Statevector(self.ansatz.assign_parameters(theta)).expectation_value(self.H)))
    def fit(self, target):
        """theta = argmax |<target|U(theta)|0>|^2 (classical state-preparation fit)."""
        from scipy.optimize import minimize
        rng = np.random.default_rng(self.seed)
        best = None
        for r in range(self.fit_restarts):
            th0 = rng.uniform(-np.pi, np.pi, self.p) if r else np.zeros(self.p)
            res = minimize(lambda th: 1.0 - (self._state(th) @ target)**2, th0, method='L-BFGS-B',
                           options=dict(maxiter=400))
            if best is None or res.fun < best.fun: best = res
        return best.x, float(best.fun)
    def encode(self, T, faces=None):
        w = self._lift(np.asarray(T, float), faces); nrm = np.linalg.norm(w)
        theta, infid = self.fit(w / nrm)
        return Params('theta', theta, nrm, self.n_qubits, {'encode_infidelity': infid})
    def step(self, p, dtau):
        from qiskit_algorithms import VarQITE, TimeEvolutionProblem
        from qiskit_algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
        from qiskit_aer.primitives import Estimator
        theta = p.vec.copy(); meta = dict(p.meta)
        for q in meta.pop('pending_x', []):            # hardware X: re-fit X|psi>
            s = self._state(theta); j = np.arange(s.size); theta, _ = self.fit(s[j ^ (1 << q)])
        E0 = self._energy(theta)
        est = Estimator(run_options={'shots': None}, approximation=True)
        vq = VarQITE(self.ansatz, theta, ImaginaryMcLachlanPrinciple(), est, num_timesteps=1)
        res = vq.evolve(TimeEvolutionProblem(self.H, dtau))
        theta1 = np.array(res.parameter_values[-1], float)
        E1 = self._energy(theta1)
        nrm = p.norm * np.exp(-0.5 * (E0 + E1) * dtau)     # d ln||w||/dt = -<H>
        meta.update(E0=E0, E1=E1)
        return Params('theta', theta1, nrm, self.n_qubits, meta)
    def decode(self, p):
        if self.decode_mode == 'exact':
            a = self._state(p.vec)
        else:
            from qiskit_aer import AerSimulator
            qc = self.ansatz.assign_parameters(p.vec); qc.measure_all()
            counts = AerSimulator(seed_simulator=self.seed).run(qc, shots=self.shots).result().get_counts()
            prob = np.zeros(self.N)
            for k, v in counts.items(): prob[int(k, 2)] = v / self.shots
            ref = self._last_sign if self._last_sign is not None else np.ones(self.N)
            a = np.sign(ref) * np.sqrt(prob)
        self._last_sign = np.where(a == 0, 1.0, np.sign(a))
        return self._unlift(a * p.norm)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    # Tier 1 vs analytic, 1D and 3D, with lifting
    for dims, m in ((1, 4), (3, 2)):
        qs = ExactStatevector(m, dims)
        T0 = lp.gaussian_bump_1d(m) if dims == 1 else lp.gaussian_bump_3d(m)
        faces = {'x-': 0.0, 'x+': 0.0} if dims == 1 else {k: np.zeros((2**m, 2**m)) for k in ('x-','x+','y-','y+','z-','z+')}
        p = qs.encode(T0, faces)
        for _ in range(10): p = qs.step(p, 1e-3)
        Tq = qs.decode(p)
        Tref = lp.exact_evolution_1d(m, T0, 1e-2) if dims == 1 else lp.exact_evolution_3d(m, T0, 1e-2)
        err = np.linalg.norm(Tq - Tref) / np.linalg.norm(Tref)
        print(f"Tier1 {dims}D m={m} ({m*dims} qubits): rel err vs analytic {err:.2e}"); assert err < 1e-10
    # lifting check: constant boundary data 1.0, T0 = 1 everywhere -> steady, w = 0
    m = 3; qs = ExactStatevector(m, 1); p = qs.encode(np.ones(2**m), {'x-': 1.0, 'x+': 1.0})
    assert abs(p.norm) < 1e-10, "harmonic extension of constant boundary data is the constant"
    print("lifting OK")
    # fault hooks
    p = ExactStatevector(4, 1).encode(lp.gaussian_bump_1d(4))
    p2 = Params(p.kind, p.vec.copy(), p.norm, p.n_qubits); p2.flip_bit(5, 62)
    print(f"flip_bit(5,62): {p.vec[5]:.4e} -> {p2.vec[5]:.4e}")
    p3 = Params(p.kind, p.vec.copy(), p.norm, p.n_qubits); p3.pauli_x(0)
    assert np.allclose(p3.vec[::2], p.vec[1::2]), "pauli_x(0) swaps neighbours"
    # Tier 2 vs Tier 1, 1D, 3 qubits
    m = 3; t0 = time.time()
    t1 = ExactStatevector(m, 1); t2 = AerVarQITE(m, 1, reps=3)
    T0 = lp.gaussian_bump_1d(m, width=0.15)
    p1 = t1.encode(T0); p2 = t2.encode(T0)
    print(f"Tier2 encode infidelity: {p2.meta['encode_infidelity']:.2e}  (p={t2.p} params)")
    for _ in range(5): p1 = t1.step(p1, 2e-3); p2 = t2.step(p2, 2e-3)
    T1, T2 = t1.decode(p1), t2.decode(p2)
    print(f"Tier2 vs Tier1 after 5 steps: rel err {np.linalg.norm(T2-T1)/np.linalg.norm(T1):.2e}, "
          f"norm T1 {p1.norm:.6f} T2 {p2.norm:.6f}   ({time.time()-t0:.1f}s)")
    T2s = AerVarQITE(m, 1, reps=3, decode='shots', shots=20000); T2s.T_ss = t2.T_ss
    Ts = T2s.decode(p2)
    print(f"Tier2 shots decode (20k) vs exact decode: rel err {np.linalg.norm(Ts-T2)/np.linalg.norm(T2):.2e}")
    print("qstep.py self-tests OK")
