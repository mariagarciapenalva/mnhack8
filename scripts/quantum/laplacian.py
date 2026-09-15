"""
laplacian.py -- the Hamiltonian of the quantum substrate.

H = -alpha * Laplacian_h  on 2^m interior points per axis, homogeneous
Dirichlet, 7-point (3D) / 3-point (1D) finite-difference stencil. The
source-free heat equation is then exactly imaginary-time evolution:

    T(t) = exp(-t H) T(0)

DISCRETIZATION NOTE (state this in the write-up): the classical solver uses
Q1 FEM with consistent mass; the quantum side uses the FD Laplacian, which
equals lumped-mass Q1 in 1D and the 7-point stencil in 3D. Both converge to
the same PDE at O(h^2). Each substrate is validated against the ANALYTIC
solution (V0 for the classical side, V7 for the quantum side), so the
comparison in Layer 7 is between two validated discretizations of one
continuous problem, at matched DOF count -- not "the same matrix".

Grid convention: 2^m interior points x_j = (j+1) h, j = 0..2^m-1, h = 1/(2^m+1),
so the domain is [0,1] with boundary nodes at 0 and 1 excluded (they are 0).
Amplitude encoding: node index j <-> computational basis state |j>, with the
3D index j = jx*4^m ... i.e. j = (jx * 2^m + jy) * 2^m + jz, jz fastest --
the same x-slowest ordering as the classical solver.
"""
import numpy as np

def h_of(m, h=None):
    """Grid spacing. Default: the block spans [0,1] (2^m interior points).
    Pass h explicitly when the block is embedded in a larger grid (h = 1/N)."""
    return (1.0 / (2**m + 1)) if h is None else h

def lap1d(m, alpha=1.0, h=None):
    """Dense -alpha * Laplacian_h on 2^m interior points (SPD)."""
    n = 2**m; h = h_of(m, h)
    L = np.zeros((n, n))
    L[np.arange(n), np.arange(n)] = 2.0
    L[np.arange(n-1), np.arange(1, n)] = -1.0
    L[np.arange(1, n), np.arange(n-1)] = -1.0
    return alpha * L / h**2

def lap3d(m, alpha=1.0, h=None):
    """Dense 3D operator via tensor structure. Only for small m (n = 3m <= 12)."""
    L1 = lap1d(m, alpha, h); I = np.eye(2**m)
    return np.kron(np.kron(L1, I), I) + np.kron(np.kron(I, L1), I) + np.kron(np.kron(I, I), L1)

def eigen1d(m, alpha=1.0, h=None):
    """Exact eigenpairs of lap1d: lambda_k = (4 alpha/h^2) sin^2(k pi h0/2), v_k(j) = sin(k pi j h0),
    h0 = 1/(2^m+1) the reference spacing; the physical h only scales lambda."""
    n = 2**m; hp = h_of(m, h); h = h_of(m)
    k = np.arange(1, n+1); x = (np.arange(n) + 1) * h
    lam = 4 * alpha / hp**2 * np.sin(k * np.pi * h / 2)**2
    V = np.sqrt(2 * h) * np.sin(np.outer(x, k) * np.pi)     # orthonormal columns
    return lam, V

def gaussian_bump_1d(m, center=0.5, width=0.1):
    x = (np.arange(2**m) + 1) * h_of(m)
    return np.exp(-(x - center)**2 / (2 * width**2))

def gaussian_bump_3d(m, center=0.5, width=0.1):
    b = gaussian_bump_1d(m, center, width)
    return np.einsum('i,j,k->ijk', b, b, b).reshape(-1)

def exact_evolution_1d(m, T0, t, alpha=1.0, h=None):
    """T(t) = V exp(-lam t) V^T T0 -- the analytic solution of the discrete problem."""
    lam, V = eigen1d(m, alpha, h)
    return V @ (np.exp(-lam * t) * (V.T @ T0))

def exact_evolution_3d(m, T0, t, alpha=1.0, h=None):
    """Tensor-product eigen-expansion; O(n 2^{3m}) via three mode-wise contractions."""
    lam, V = eigen1d(m, alpha, h); n = 2**m
    A = T0.reshape(n, n, n)
    C = np.einsum('ia,jb,kc,ijk->abc', V, V, V, A)             # spectral coefficients
    L3 = lam[:, None, None] + lam[None, :, None] + lam[None, None, :]
    C = C * np.exp(-L3 * t)
    return np.einsum('ia,jb,kc,abc->ijk', V, V, V, C).reshape(-1)

# ---------------------------------------------------------------------------
# Pauli decomposition (tensor structured: only the 1D operator is decomposed)
# ---------------------------------------------------------------------------
def pauli_1d(m, alpha=1.0, h=None):
    """SparsePauliOp of lap1d on m qubits. Numerical decomposition: cost 4^m,
    fine for m <= 8. For larger m use an analytic decomposition (see docs)."""
    from qiskit.quantum_info import SparsePauliOp
    return SparsePauliOp.from_operator(lap1d(m, alpha, h)).simplify()

def pauli_3d(m, alpha=1.0, h=None):
    """H = L1 (x) I (x) I + I (x) L1 (x) I + I (x) I (x) L1 on 3m qubits.
    Qiskit's tensor order: qubit 0 is the rightmost factor. With basis state
    index j = (jx*2^m + jy)*2^m + jz, jz occupies the low qubits (0..m-1),
    jy the middle, jx the high (2m..3m-1)."""
    from qiskit.quantum_info import SparsePauliOp
    L1 = pauli_1d(m, alpha, h); I1 = SparsePauliOp.from_list([("I"*m, 1.0)])
    return (L1.tensor(I1).tensor(I1) + I1.tensor(L1).tensor(I1) + I1.tensor(I1).tensor(L1)).simplify()

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # self-tests: symmetry, spectrum, analytic solution, Pauli round-trip
    for m in (3, 4):
        L = lap1d(m); assert np.allclose(L, L.T)
        lam, V = eigen1d(m)
        assert np.allclose(np.sort(np.linalg.eigvalsh(L)), np.sort(lam)), "1D spectrum"
        assert np.allclose(V.T @ V, np.eye(2**m)), "eigenbasis orthonormal"
        T0 = gaussian_bump_1d(m); t = 0.01
        Tt = exact_evolution_1d(m, T0, t)
        from scipy.linalg import expm
        assert np.allclose(Tt, expm(-t * L) @ T0, atol=1e-12), "1D exact evolution"
    L = lap1d(3, h=1/17); lam, V = eigen1d(3, h=1/17)
    assert np.allclose(np.sort(np.linalg.eigvalsh(L)), np.sort(lam)), "embedded-h spectrum"
    m = 2
    L3 = lap3d(m); assert np.allclose(L3, L3.T)
    T0 = gaussian_bump_3d(m); t = 0.005
    assert np.allclose(exact_evolution_3d(m, T0, t), expm(-t * L3) @ T0, atol=1e-12), "3D exact evolution"
    try:
        from qiskit.quantum_info import Operator
        assert np.allclose(pauli_1d(3).to_matrix(), lap1d(3)), "1D Pauli round-trip"
        assert np.allclose(pauli_3d(2).to_matrix(), lap3d(2)), "3D Pauli ordering"
        print("pauli terms: 1D m=3:", len(pauli_1d(3)), " 3D m=2:", len(pauli_3d(2)))
    except ImportError:
        print("qiskit not installed: Pauli tests skipped")
    print("laplacian.py self-tests OK")
