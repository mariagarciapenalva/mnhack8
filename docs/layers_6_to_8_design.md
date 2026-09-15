# Layers 6–8: quantum substrate, scale-matched comparison, hybrid pipeline

Status: design, decisions made, no code yet. Built on Layers 0–5 being validated
(V0–V6 green, pilot ensemble run). Every section ends with what it produces and
what MN5 is specifically needed for.

---

## Layer 6 — Quantum substrate

### 6.1 The problem the quantum side solves (make it the SAME problem)

Source-free heat equation, the case V0 already validates exactly:

    dT/dt = alpha * Laplacian(T),  T = 0 on the boundary,  T(x,0) = T0(x)

Discretized on 2^m points per axis (m qubits per axis, n = 3m total in 3D,
n = m in 1D), with the uniform-grid FEM operator L = M^-1 K (symmetric positive
semidefinite after the Dirichlet rows are removed). Then

    T(t) = exp(-t L) T0

is *literally* imaginary-time evolution under the Hamiltonian H = L with
tau = t. Amplitude-encode the field: |psi(t)> proportional to sum_i T_i(t) |i>.
Nothing is approximated in the mapping; the only thing that differs between
the substrates is how exp(-t L) is applied and where the floats live.

Initial condition: a Gaussian bump, NOT the sine eigenmode. The eigenmode is an
eigenstate of H and just decays — trivial on both sides. A bump is spectrally
broadband and makes propagation visible. Add `--ic gaussian` to the classical
solver (a 10-line kernel next to `exact_mode_kernel`); the analytic reference
is the sine-series expansion of the bump, computed in NumPy.

3D Hamiltonian by tensor structure:  H = L1 (x) I (x) I + I (x) L1 (x) I + I (x) I (x) L1.
Only the 1D operator L1 (m qubits) is ever decomposed into Paulis. For m <= 8
use `SparsePauliOp.from_operator` on the dense 2^m x 2^m matrix (cost 4^m,
fine). For larger m there are O(m)-term decompositions of the discrete
Laplacian in the literature (Sato et al. 2021 for the Poisson equation is the
usual starting point — verify the reference; I cannot check it here).

### 6.2 Two tiers, both required, for different reasons

**Tier 1 — exact imaginary-time statevector.** psi <- normalize(exp(-dt H) psi)
per step, dense exp via eigendecomposition for n <= 14, cuStateVec / CuPy for
bigger. No ansatz, no variational error. This is the *ground truth on the
quantum representation* and the fair partner for the classical solver in Layer
7. Fault classes on it:
  (a) IEEE-754 bit flip on a stored amplitude — the identical fault model to the
      classical side (same `bitflip` semantics, same bit taxonomy);
  (b) Pauli-X on qubit k applied to the state — the hardware-native quantum
      fault, reported separately, never called "equivalent".

**Tier 2 — VarQITE (McLachlan variational principle).** Hardware-efficient
ansatz U(theta), theta in R^p, p ~ 2 n * layers. Each step solves the p x p
linear system  A(theta) theta_dot = C(theta)  where A is the quantum geometric
tensor and C the energy gradient, both estimated from circuit expectation
values, then theta <- theta + dt theta_dot. This tier is what makes Layer 8
possible at all: **theta is a classical float vector that lives in a hybrid
loop**, so encode / transit / decode handoffs exist as real buffers. Tier 1 has
no such buffers. This is also the resolution of the proposal's open
"equivalent injection" question: the IEEE fault on theta *is* the classical
fault, bit for bit; Pauli-X remains a distinct channel.

Cost per VarQITE step: O(p^2) circuit evaluations for A, O(p) for C, each a
statevector evaluation of size 2^n. At n = 10, layers = 3: p ~ 60, ~ 4000
circuits/step — seconds on a GPU. n = 16 is MN5 territory. That cost is the
honest MN5 justification for this layer.

### 6.3 Software decisions

| Concern | Decision | Why |
|---|---|---|
| Framework for the hackathon | **Qiskit 1.x + Aer + qiskit-algorithms** (`VarQITE`, `ImaginaryMcLachlanPrinciple`, `EfficientSU2`, `SparsePauliOp`) | Only stack with a maintained, tested VarQITE; already in your abstract to Sergio; Aer has a cuStateVec-backed GPU build. |
| GPU on MN5 | `qiskit-aer-gpu` with `cuStateVec_enable=True` if the module exists; fallback: Tier 1 in CuPy (own code, 60 lines) and VarQITE on CPU Aer for n <= 12 | Do not make the hackathon depend on an unverified module; ask Sergio which of `qiskit-aer-gpu` / `cuquantum` are installed on ACC. |
| Real-QPU path at BSC | **Qibo / Qibolab** — the native stack of the Qilimanjaro / Ona hardware | Verify with Sergio. Qiskit is not the interface to that machine. Consequence: code against an internal interface, not against Qiskit directly. |
| Version pinning | `requirements-quantum.txt` with exact versions, committed | qiskit-algorithms has broken VarQITE across minor versions before; reproducibility on MN5 needs a lockfile. |

The internal interface (one file, `scripts/quantum/qstep.py`):

    class QuantumStep:
        def encode(self, T_sub: np.ndarray) -> Params      # floats -> (norm, theta)
        def step(self, params: Params, dtau: float) -> Params
        def decode(self, params: Params) -> np.ndarray     # -> floats (with shot noise if shots)
    class ExactStatevector(QuantumStep)   # Tier 1, NumPy/CuPy
    class AerVarQITE(QuantumStep)         # Tier 2, Aer statevector or shots(+FakeTorino noise)
    class QiboVarQITE(QuantumStep)        # Tier 2 on BSC hardware, later

Fault hooks are methods on `Params`, not on the backend: `flip_bit(index, bit)`
and `pauli_x(qubit)`. That is what keeps the injection model identical across
simulator, GPU simulator and hardware.

### 6.4 Layer 6 deliverables
`scripts/quantum/qstep.py`, `scripts/quantum/laplacian.py` (1D/3D H
construction + tests: H symmetric, row sums 0 on interior, eigenvalues match
the FEM discrete Laplacian), `scripts/quantum/v7_quantum.py` (validation:
Tier 1 vs analytic decay of the bump to 1e-10; Tier 2 vs Tier 1 to the
variational error, reported, not hidden).

---

## Layer 7 — Scale-matched comparison

DOF matching: classical grid N with (N+1)^3 nodes = 2^n quantum amplitudes.
N = 15 -> 16^3 = 2^12 (12 qubits, 4/axis), N = 31 -> 2^15, N = 63 -> 2^18.
Classical solver runs at N = 15 / 31 in verify-like mode (no source, Gaussian
IC). 12 qubits is the hackathon working size for Tier 2; 15–18 for Tier 1 on
MN5.

Same measurements as Layer 3 on both substrates, same fault taxonomy:
  - E(t) after an identical IEEE bit flip (state on the classical side,
    amplitude for Tier 1, theta for Tier 2) at the same physical location
    (same node index i in the amplitude encoding) and same step S.
  - E(t) after Pauli-X on qubit k (quantum only), reported as its own class.
  - Spectral view: project delta T onto the sine eigenbasis (both sides).
    Classical: each mode decays as exp(-lambda_k t) — this is exact, and V0
    already proved the solver reproduces it. Tier 1: identical by
    construction (same H), so any difference is the fault model. Tier 2: the
    ansatz cannot represent all modes; the *variational leakage* shows up as
    error that never decays — that is a real result about VarQITE as a
    fault-tolerance substrate, not a bug.
  - Healing time: first t at which E drops below a fixed fraction of E_max.

Deliverable: `scripts/quantum/scale_matched.py` -> side-by-side E(t) and
spectral plots + a table (substrate x fault class -> E_max, healing time,
persistent-error floor). MN5: 15–18 qubit Tier 1 sweeps and the ensembles.

---

## Layer 8 — Hybrid pipeline with handoff injection

### 8.1 Architecture: the quantum step is a rank

Do not build a new pipeline; extend the one you have. The classical solver
already does per-step halo exchanges between MPI ranks through host staging
buffers, and that is exactly a handoff. Treat the quantum step as one more
rank that owns a subdomain Omega_q (a line of 2^m nodes, or a small block):

    every time step n -> n+1:
      classical ranks:  T^{n+1} = BE step on the full domain (as today)
      quantum rank:     receives T^n|Omega_q  ->  encode  ->  VarQITE step
                        ->  decode  ->  sends T^{n+1}|Omega_q back
      coupling:         the quantum result overwrites (or is blended into)
                        Omega_q in the classical field; boundary values of
                        Omega_q come from the classical field (Dirichlet
                        coupling, one Schwarz iteration per step)

Handoff points, mapped to what already exists:

| Handoff | What is in the buffer | Injection | Analogue in the classical solver |
|---|---|---|---|
| **H1 encode** | T^n restricted to Omega_q, normalized, then theta after the state-preparation fit | flip bit in theta / in the normalized amplitude buffer *after* encoding, before it is sent | level R (an intermediate that no ECC sees) |
| **H2 transit** | the serialized theta (or amplitudes) crossing from the CUDA process to the Python process | flip bit in the message buffer between pack and send | level H, identical mechanism, real MPI message on MN5 |
| **H3 decode** | measured / estimated amplitudes parsed to floats, un-normalized, about to be written into d_T | flip bit in the parsed float array before write-back | level G, but at the moment of re-entry |

The scientific question of the proposal ("which handoff is silent") becomes:
for each Hk, what fraction of injections is caught by D1–D3 (already built)
plus **D4**, a handoff checksum: sum and norm of the buffer computed on both
sides of every handoff and compared, plus the physical check that the energy
of the returned subdomain is within tolerance of what the classical operator
would have produced (the classical BE step on Omega_q is cheap and serves as a
reference — that is redundancy, the one thing that can see G-type faults).

### 8.2 Process model

Hackathon / local: one MPI job, P CUDA ranks (the existing binary, extended
with a `--quantum-rank R --qsub x0,y0,z0,m` option that designates Omega_q and
opens an intercommunicator) + 1 Python rank (`mpi4py`, `qstep.py`). Launch as
MPMD: `mpirun -np P ./heat_solver_het ... : -np 1 python3 qrank.py ...`.
All three handoffs are then real buffers in real processes; H2 is a real
MPI message. Locally both sides share the GPU; on MN5 the quantum rank gets
its own H100 for Aer-GPU.

MN5: the same MPMD launch across nodes (H2 crosses a real NIC), and a job
array of independent pipelines for the ensemble (each array task = one
injection configuration; `run_ensemble.py --shard i/n` already supports this
pattern). Scalability claim, stated honestly: the classical part scales as
Layer 9 measures; the quantum rank scales in qubits (Tier 1 to ~20 on one
H100) and in *count* (many pipelines, not one big one). That is what the
statistics need, and it is what a QPU-era deployment looks like too: many
small quantum sub-tasks coupled to a large classical solver, not one giant
quantum state.

### 8.3 Real-QPU path (what makes the post-hackathon proposal credible)

Swap `AerVarQITE` for `QiboVarQITE` behind the same interface. H1 and H3
become *physical* handoffs (pulse compilation in, readout parsing out), and
the same injection hooks still apply — corrupt theta before compilation,
corrupt the readout floats before write-back — plus a fourth, uncontrolled
channel: real hardware noise, characterized against the FakeTorino-noise
simulator runs as the control. Sized to Ona: m qubits per axis such that
n <= the machine's usable qubit count, shots budget from the D4 tolerance.

What Sergio's group would need to see to grant QPU time: (1) the pipeline
running end-to-end with simulator backends, (2) the handoff sensitivity map
H1/H2/H3 x bit class with CIs, (3) D4 coverage, (4) a shot/qubit-budgeted run
plan. (1)–(3) are hackathon deliverables if Layers 6–7 are prototyped locally
before the event.

---

## Order and dependencies

    Layer 4 (done, validate with V6)  ->  Layer 5 pilot locally (this week)
    ->  Layer 6 Tier 1 + laplacian tests (local, NumPy)  ->  Layer 6 Tier 2 (Aer, n <= 10)
    ->  Layer 7 at N = 15 / 12 qubits (local)  ->  Layer 8 with the Python rank (local, both on the 3060)
    ->  MN5: Layer 9 scaling, Layer 5 at scale, Layer 8 across nodes, Tier 1 at 18 qubits

Nothing in 6–8 needs MN5 to be *built*; MN5 is where it is *scaled*. Building
it locally first is what makes the MN5 time count.
