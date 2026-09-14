// =============================================================================
// heat_solver_het.cu  --  v2, validated-baseline candidate (September 2026)
//
// Copyright 2026 Maria Garcia Penalva
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Assembly-free FEM solver for the transient heat equation in 3D heterogeneous
// materials, with single-event-upset (SEU) fault injection and a twin-run
// measurement protocol. Built on the MNHack7 prototype (heat_solver.cu).
//
//   PDE:   rho(x) c(x) dT/dt = div( k(x) grad T ) + Q(x)      in [0,1]^3
//          T = 0 on the boundary (homogeneous Dirichlet)
//   Space: trilinear hexahedral (Q1) FEM on a uniform axis-aligned grid.
//          2x2x2 Gauss quadrature, which is exact for the Q1 mass and
//          stiffness integrands on this mesh.
//   Time:  backward Euler, (M + dt K) T^{n+1} = M T^n + dt F.
//          Chosen for L-stability: an injected bit flip is spectrally
//          broadband and BE damps every mode monotonically, amplification
//          1/(1 + lambda dt). Crank-Nicolson would ring at -1.
//   Solve: matrix-free Jacobi-PCG. 1D MPI decomposition along x. Halo data
//          is staged through host buffers (not CUDA-aware MPI).
//
// MATERIAL: piecewise constant per element, read from gen_field.py binaries
//   with ordering e = ix*ny*nz + iy*nz + iz (x slowest). Any sub-element
//   structure is upscaled by gen_field.py (harmonic mean for k, volume-
//   weighted arithmetic mean for rho*c). The solver itself never averages.
//
// OPERATOR KERNELS (one binary, runtime --kernel):
//   gauss : 2x2x2 Gauss loop evaluated per element per matvec (MNHack7 form).
//           Largest arithmetic surface; kept as the fault-injection reference.
//   fast  : A_e = rhoc_e*M_hat + dt*k_e*K_hat using two 8x8 reference
//           matrices computed once at startup. Exact on a uniform grid with
//           per-element-constant coefficients (element congruency; cf. Yadav
//           & Suresh). ~20-30x fewer flops in the hot kernel.
//
// SCATTER (runtime --scatter):
//   atomic  : thread per element, atomicAdd on shared nodes. FP atomics are
//             order-nondeterministic, so clean-vs-clean is not bitwise zero.
//             This is the production pattern; its noise floor is measured.
//   colored : 8-color element ordering (parity of ix,iy,iz). Elements of one
//             color share no node, so plain += is race-free and the operator
//             is bitwise reproducible. Dot products use a two-stage fixed-
//             order reduction in BOTH modes, so the only nondeterminism in
//             atomic mode is the element scatter itself.
//
// FAULT INJECTION (one SEU per run: --fi LEVEL STEP TARGET BIT [--fi-rank r]):
//   G (1): XOR one bit of d_T[TARGET % n_local] in global memory, before the
//          RHS of step STEP is built.                           (ECC-covered)
//   R (2): XOR one bit of element accumulator ye[0] of local element TARGET in
//          the FIRST matvec of the CG solve at step STEP, before scatter.
//          Stands in for an arbitrary register-resident intermediate; it is
//          not a literal hardware register.                     (unprotected)
//   H (3): XOR one bit of the host staging buffer of the right-going halo face
//          of rank --fi-rank during the RHS halo exchange at step STEP. Models
//          the gap between GPU ECC and NIC CRC. Needs >= 2 ranks; does NOT
//          need separate nodes.                                 (in-transit)
//   Fires exactly once, on exactly one rank (--fi-rank, default 0).
//
// TWIN-RUN PROTOCOL: cleanA (reference), cleanB (noise floor), fault.
//   E(t) = ||T_x(t)-T_A(t)|| / ||T_A(t)|| over owned nodes at snapshot steps.
//   The heat equation is dissipative: any finite perturbation decays. So the
//   class is decided on max_t E(t), not on E(t_final):
//     benign     : E_max <= thr
//     transient  : E_max  > thr and E_final <= thr   (healed by diffusion)
//     persistent : E_max  > thr and E_final  > thr
//     detected   : a non-finite value was observed
//   thr = max(--thr, 10 * E_floor_max). With --scatter colored, E_floor == 0.
//
// STARTUP SELF-CHECK: M_hat and K_hat are compared with their closed forms
//   (M: V/216*{8,4,2,1} by node distance, sum = V; K: zero row sums, diagonal
//   (1/9) sum_i h_j h_k / h_i). Any mismatch > 1e-12 aborts the run.
//
// --verify: requires a homogeneous field; source off; T(x,0) = sin(pi x)
//   sin(pi y) sin(pi z); exact T(x,t) = exp(-3 pi^2 alpha t) T(x,0),
//   alpha = k/(rho c). Reports the relative L2 error at t_end = n_steps*dt.
// =============================================================================

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

typedef double real_t;
#define MPI_REAL_T MPI_DOUBLE

#define CUDA_CHECK(call) do {                                                  \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
        fprintf(stderr, "[CUDA] %s:%d: %s\n",                                  \
                __FILE__, __LINE__, cudaGetErrorString(_e));                   \
        MPI_Abort(MPI_COMM_WORLD, 1);                                          \
    }                                                                          \
} while (0)

#define CUDA_CHECK_LAUNCH() do {                                               \
    CUDA_CHECK(cudaPeekAtLastError());                                         \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
} while (0)

// -----------------------------------------------------------------------------
// Grid descriptor (passed by value to kernels)
// -----------------------------------------------------------------------------
struct Grid {
    int  nx, ny, nz;
    real_t hx, hy, hz;
    int  rank, nprocs;
    int  nx_local, ix_start;
    int  nnx_local, nny, nnz;
    long nnodes_local, ne_local;
    const real_t* k_cell;      // conductivity per local element
    const real_t* rhoc_cell;   // rho*c per local element
    int  verify;               // 1: source term off (Layer-0 check)
};

// -----------------------------------------------------------------------------
// Fault injection configuration
// -----------------------------------------------------------------------------
struct FIConfig {
    int  level  = 0;    // 0 none, 1 G, 2 R, 3 H
    int  step   = -1;
    long target = 0;
    int  bit    = 0;    // 0-51 mantissa, 52-62 exponent, 63 sign
    int  rank   = 0;    // the ONE rank that fires
    int  armed  = 0;
};

__global__ void flip_bit_global_kernel(real_t* v, long idx, int bit) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long* w = reinterpret_cast<unsigned long long*>(&v[idx]);
        *w ^= (1ULL << bit);
    }
}

// -----------------------------------------------------------------------------
// Reference element matrices (element congruency). Filled once at setup.
// -----------------------------------------------------------------------------
__constant__ real_t c_Mhat[64];
__constant__ real_t c_Khat[64];

// -----------------------------------------------------------------------------
// Indexing helpers
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
long lnid(int ix_local, int iy, int iz, const Grid& g) {
    return (long)ix_local * (g.nny * g.nnz) + (long)iy * g.nnz + iz;
}

__host__ __device__ __forceinline__
void node_coords(int ix_local, int iy, int iz, const Grid& g,
                 real_t& x, real_t& y, real_t& z) {
    x = (real_t)(g.ix_start + ix_local) * g.hx;
    y = (real_t)iy * g.hy;
    z = (real_t)iz * g.hz;
}

__host__ __device__ __forceinline__
bool is_boundary(int ix_local, int iy, int iz, const Grid& g) {
    int ix_global = g.ix_start + ix_local;
    return (ix_global == 0 || ix_global == g.nx ||
            iy == 0 || iy == g.ny || iz == 0 || iz == g.nz);
}

// Rank r owns ix_local in [0, nx_local); the last rank also owns nx_local.
__host__ __device__ __forceinline__
bool is_owned(int ix_local, const Grid& g) {
    if (g.rank == g.nprocs - 1) return ix_local <= g.nx_local;
    return ix_local < g.nx_local;
}

__device__ __forceinline__
real_t source_at(real_t x, real_t y, real_t z, const Grid& g) {
    if (g.verify) return 0.0;
    const real_t cx = 0.5, cy = 0.5, cz = 0.5;
    const real_t sigma = 0.1;
    const real_t coeff = 1000.0;
    real_t r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz);
    return coeff * exp(-r2 / (2.0 * sigma * sigma));
}

// Trilinear hex shape functions on [-1,1]^3.
//   N0:(-1,-1,-1) N1:(+1,-1,-1) N2:(+1,+1,-1) N3:(-1,+1,-1)
//   N4:(-1,-1,+1) N5:(+1,-1,+1) N6:(+1,+1,+1) N7:(-1,+1,+1)
__host__ __device__ __forceinline__
void shape_func_grad(real_t xi, real_t eta, real_t zeta,
                     real_t N[8], real_t dN[8][3]) {
    const int sgn[8][3] = {
        {-1,-1,-1}, { 1,-1,-1}, { 1, 1,-1}, {-1, 1,-1},
        {-1,-1, 1}, { 1,-1, 1}, { 1, 1, 1}, {-1, 1, 1}
    };
    for (int i = 0; i < 8; ++i) {
        real_t sx = (real_t)sgn[i][0], sy = (real_t)sgn[i][1], sz = (real_t)sgn[i][2];
        real_t a = 1 + sx*xi, b = 1 + sy*eta, c = 1 + sz*zeta;
        N[i]     = 0.125 * a * b * c;
        dN[i][0] = 0.125 * sx * b * c;
        dN[i][1] = 0.125 * a * sy * c;
        dN[i][2] = 0.125 * a * b  * sz;
    }
}

__device__ __forceinline__
void element_nodes(int ix_e, int iy_e, int iz_e, const Grid& g, long gid[8]) {
    const int off[8][3] = {
        {0,0,0},{1,0,0},{1,1,0},{0,1,0},
        {0,0,1},{1,0,1},{1,1,1},{0,1,1}
    };
    for (int i = 0; i < 8; ++i)
        gid[i] = lnid(ix_e + off[i][0], iy_e + off[i][1], iz_e + off[i][2], g);
}

// -----------------------------------------------------------------------------
// Element index resolution. color < 0: thread tid -> element tid (all
// elements). color in 0..7: thread tid -> the tid-th element whose
// (ix%2, iy%2, iz%2) == (cx, cy, cz). Elements of one color share no node.
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__
long color_count(int color, const Grid& g) {
    int cx = (color >> 2) & 1, cy = (color >> 1) & 1, cz = color & 1;
    long nxa = (g.nx_local - cx + 1) / 2;
    long nyb = (g.ny - cy + 1) / 2;
    long nzc = (g.nz - cz + 1) / 2;
    return nxa * nyb * nzc;
}

__device__ __forceinline__
bool resolve_element(long tid, int color, const Grid& g,
                     long& e, int& ix_e, int& iy_e, int& iz_e) {
    if (color < 0) {
        if (tid >= g.ne_local) return false;
        e = tid;
        ix_e = (int)(e / ((long)g.ny * g.nz));
        iy_e = (int)((e / g.nz) % g.ny);
        iz_e = (int)(e % g.nz);
        return true;
    }
    int cx = (color >> 2) & 1, cy = (color >> 1) & 1, cz = color & 1;
    long nyb = (g.ny - cy + 1) / 2;
    long nzc = (g.nz - cz + 1) / 2;
    if (tid >= color_count(color, g)) return false;
    int a = (int)(tid / (nyb * nzc));
    int b = (int)((tid / nzc) % nyb);
    int c = (int)(tid % nzc);
    ix_e = 2*a + cx; iy_e = 2*b + cy; iz_e = 2*c + cz;
    e = (long)ix_e * g.ny * g.nz + (long)iy_e * g.nz + iz_e;
    return true;
}

// -----------------------------------------------------------------------------
// Element operator: ye = (rhoc_e M_e + dt k_e K_e) xe.
// gauss: integrate on the fly (MNHack7 form). fast: reference matrices.
// -----------------------------------------------------------------------------
__device__ __forceinline__
void elem_apply_gauss(const real_t xe[8], real_t ye[8],
                      real_t k_e, real_t rhoc_e, real_t dt, const Grid& g) {
    const real_t det_J  = g.hx * g.hy * g.hz / 8.0;
    const real_t inv_Jx = 2.0 / g.hx, inv_Jy = 2.0 / g.hy, inv_Jz = 2.0 / g.hz;
    const real_t gp = 1.0 / 1.7320508075688772;
    const real_t pts[2] = { -gp, gp };
    for (int i = 0; i < 8; ++i) ye[i] = 0.0;
    for (int gi = 0; gi < 2; ++gi)
    for (int gj = 0; gj < 2; ++gj)
    for (int gk = 0; gk < 2; ++gk) {
        real_t N[8], dNref[8][3];
        shape_func_grad(pts[gi], pts[gj], pts[gk], N, dNref);
        real_t dNx[8], dNy[8], dNz[8];
        for (int i = 0; i < 8; ++i) {
            dNx[i] = inv_Jx * dNref[i][0];
            dNy[i] = inv_Jy * dNref[i][1];
            dNz[i] = inv_Jz * dNref[i][2];
        }
        for (int i = 0; i < 8; ++i) {
            real_t Mx = 0.0, Kx = 0.0;
            for (int j = 0; j < 8; ++j) {
                Mx += rhoc_e * N[i] * N[j] * xe[j];
                Kx += k_e * (dNx[i]*dNx[j] + dNy[i]*dNy[j] + dNz[i]*dNz[j]) * xe[j];
            }
            ye[i] += det_J * (Mx + dt * Kx);
        }
    }
}

__device__ __forceinline__
void elem_apply_fast(const real_t xe[8], real_t ye[8],
                     real_t k_e, real_t rhoc_e, real_t dt) {
    for (int i = 0; i < 8; ++i) {
        real_t acc = 0.0;
        for (int j = 0; j < 8; ++j)
            acc += (rhoc_e * c_Mhat[i*8+j] + dt * k_e * c_Khat[i*8+j]) * xe[j];
        ye[i] = acc;
    }
}

// y += (M + dt_eff K) x, element-by-element. dt_eff = 0 gives y += M x.
template<int FAST>
__global__
void operator_kernel(const real_t* __restrict__ x, real_t* __restrict__ y,
                     Grid g, real_t dt_eff, FIConfig fi, int color) {
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long e; int ix_e, iy_e, iz_e;
    if (!resolve_element(tid, color, g, e, ix_e, iy_e, iz_e)) return;

    long gid[8];
    element_nodes(ix_e, iy_e, iz_e, g, gid);
    real_t xe[8], ye[8];
    for (int i = 0; i < 8; ++i) xe[i] = x[gid[i]];

    const real_t k_e = g.k_cell[e], rhoc_e = g.rhoc_cell[e];
    if (FAST) elem_apply_fast(xe, ye, k_e, rhoc_e, dt_eff);
    else      elem_apply_gauss(xe, ye, k_e, rhoc_e, dt_eff, g);

    // Level R: corrupt a register-resident accumulator before it is scattered.
    if (fi.armed && fi.level == 2 && g.rank == fi.rank && e == fi.target) {
        unsigned long long w64 = __double_as_longlong(ye[0]);
        w64 ^= (1ULL << fi.bit);
        ye[0] = __longlong_as_double(w64);
    }

    if (color < 0) { for (int i = 0; i < 8; ++i) atomicAdd(&y[gid[i]], ye[i]); }
    else           { for (int i = 0; i < 8; ++i) y[gid[i]] += ye[i]; }
}

// F_i = int Q N_i dOmega (2x2x2 Gauss; Q varies inside the element).
__global__
void source_kernel(real_t* __restrict__ F, Grid g, int color) {
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long e; int ix_e, iy_e, iz_e;
    if (!resolve_element(tid, color, g, e, ix_e, iy_e, iz_e)) return;
    long gid[8];
    element_nodes(ix_e, iy_e, iz_e, g, gid);
    real_t x0, y0, z0;
    node_coords(ix_e, iy_e, iz_e, g, x0, y0, z0);
    const real_t det_J = g.hx * g.hy * g.hz / 8.0;
    const real_t gp = 1.0 / 1.7320508075688772;
    const real_t pts[2] = { -gp, gp };
    real_t Fe[8] = {0};
    for (int gi = 0; gi < 2; ++gi)
    for (int gj = 0; gj < 2; ++gj)
    for (int gk = 0; gk < 2; ++gk) {
        real_t xg = x0 + (pts[gi] + 1.0) * 0.5 * g.hx;
        real_t yg = y0 + (pts[gj] + 1.0) * 0.5 * g.hy;
        real_t zg = z0 + (pts[gk] + 1.0) * 0.5 * g.hz;
        real_t Q = source_at(xg, yg, zg, g);
        real_t N[8], dN[8][3];
        shape_func_grad(pts[gi], pts[gj], pts[gk], N, dN);
        for (int i = 0; i < 8; ++i) Fe[i] += det_J * Q * N[i];
    }
    if (color < 0) { for (int i = 0; i < 8; ++i) atomicAdd(&F[gid[i]], Fe[i]); }
    else           { for (int i = 0; i < 8; ++i) F[gid[i]] += Fe[i]; }
}

// diag(A)_ii = rhoc_e M_hat_ii + dt k_e K_hat_ii, summed over elements.
__global__
void diag_kernel(real_t* __restrict__ d, Grid g, real_t dt, int color) {
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long e; int ix_e, iy_e, iz_e;
    if (!resolve_element(tid, color, g, e, ix_e, iy_e, iz_e)) return;
    long gid[8];
    element_nodes(ix_e, iy_e, iz_e, g, gid);
    const real_t k_e = g.k_cell[e], rhoc_e = g.rhoc_cell[e];
    for (int i = 0; i < 8; ++i) {
        real_t de = rhoc_e * c_Mhat[i*9] + dt * k_e * c_Khat[i*9];
        if (color < 0) atomicAdd(&d[gid[i]], de); else d[gid[i]] += de;
    }
}

// -----------------------------------------------------------------------------
// Boundary conditions, vector kernels, reductions
// -----------------------------------------------------------------------------
__global__ void apply_dirichlet_value_kernel(real_t* v, Grid g, real_t val) {
    long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= g.nnodes_local) return;
    int ix_local = (int)(n / (g.nny * g.nnz));
    int iy = (int)((n / g.nnz) % g.nny);
    int iz = (int)(n % g.nnz);
    if (is_boundary(ix_local, iy, iz, g)) v[n] = val;
}

__global__ void axpy_kernel(real_t a, const real_t* x, real_t* y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
__global__ void aypx_kernel(real_t a, const real_t* x, real_t* y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + a * y[i];
}
__global__ void copy_kernel(const real_t* x, real_t* y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
__global__ void zero_kernel(real_t* x, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 0.0;
}
__global__ void jacobi_apply_kernel(const real_t* r, const real_t* d,
                                    real_t* z, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = r[i] / d[i];
}

// Two-stage deterministic dot product over owned nodes.
__global__
void owned_dot_partial_kernel(const real_t* x, const real_t* y,
                              real_t* partial, Grid g) {
    extern __shared__ real_t sdata[];
    int tid = threadIdx.x;
    long i = (long)blockIdx.x * blockDim.x + tid;
    real_t v = 0.0;
    if (i < g.nnodes_local) {
        int ix_local = (int)(i / (g.nny * g.nnz));
        if (is_owned(ix_local, g)) v = x[i] * y[i];
    }
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

__global__
void reduce_partials_kernel(const real_t* partial, long n, real_t* result) {
    extern __shared__ real_t sdata[];
    int tid = threadIdx.x;
    real_t s = 0.0;
    for (long i = tid; i < n; i += blockDim.x) s += partial[i];   // fixed order
    sdata[tid] = s;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) sdata[tid] += sdata[tid + st];
        __syncthreads();
    }
    if (tid == 0) *result = sdata[0];
}

// Halo exchange pack/unpack
__global__
void pack_face_kernel(const real_t* v, real_t* buf, Grid g, int x_local) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long face = (long)g.nny * g.nnz;
    if (idx >= face) return;
    int iy = (int)(idx / g.nnz), iz = (int)(idx % g.nnz);
    buf[idx] = v[lnid(x_local, iy, iz, g)];
}
__global__
void unpack_face_add_kernel(real_t* v, const real_t* buf, Grid g, int x_local) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long face = (long)g.nny * g.nnz;
    if (idx >= face) return;
    int iy = (int)(idx / g.nnz), iz = (int)(idx % g.nnz);
    v[lnid(x_local, iy, iz, g)] += buf[idx];
}

// Layer-0 check: exact eigenmode of the homogeneous source-free problem.
__global__
void exact_mode_kernel(real_t* v, Grid g, real_t t, real_t alpha) {
    long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= g.nnodes_local) return;
    int ix_local = (int)(n / (g.nny * g.nnz));
    int iy = (int)((n / g.nnz) % g.nny);
    int iz = (int)(n % g.nnz);
    real_t x, y, z;
    node_coords(ix_local, iy, iz, g, x, y, z);
    const real_t pi = 3.14159265358979323846;
    real_t amp = exp(-3.0 * pi * pi * alpha * t);
    v[n] = amp * sin(pi*x) * sin(pi*y) * sin(pi*z);
    if (is_boundary(ix_local, iy, iz, g)) v[n] = 0.0;
}

// =============================================================================
// HeatOperator
// =============================================================================
class HeatOperator {
public:
    Grid g;
    real_t dt;
    FIConfig fi;
    int kernel_fast = 1;   // --kernel fast|gauss
    int colored     = 0;   // --scatter colored|atomic

    real_t *d_kcell = nullptr, *d_rhoccell = nullptr;
    real_t *d_diag = nullptr, *d_F = nullptr;
    real_t *d_send_l = nullptr, *d_send_r = nullptr;
    real_t *d_recv_l = nullptr, *d_recv_r = nullptr;
    real_t *h_send_l = nullptr, *h_send_r = nullptr;
    real_t *h_recv_l = nullptr, *h_recv_r = nullptr;
    real_t *d_partial = nullptr, *d_dot_result = nullptr;

    int  threads_node = 256; long blocks_node = 0;
    int  threads_elem = 128; long blocks_elem = 0;
    long color_blocks[8] = {0};

    // material statistics (host, global over ranks)
    real_t kmin = 0, kmax = 0, rhocmin = 0, rhocmax = 0;

    // ---- reference matrices -------------------------------------------------
    void build_reference_matrices() {
        real_t Mh[64] = {0}, Kh[64] = {0};
        const real_t detJ = g.hx * g.hy * g.hz / 8.0;
        const real_t iJ[3] = { 2.0/g.hx, 2.0/g.hy, 2.0/g.hz };
        const real_t gp = 1.0 / 1.7320508075688772;
        const real_t pts[2] = { -gp, gp };
        for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int c = 0; c < 2; ++c) {
            real_t N[8], dN[8][3];
            shape_func_grad(pts[a], pts[b], pts[c], N, dN);
            for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) {
                Mh[i*8+j] += detJ * N[i] * N[j];
                Kh[i*8+j] += detJ * ( iJ[0]*dN[i][0] * iJ[0]*dN[j][0]
                                    + iJ[1]*dN[i][1] * iJ[1]*dN[j][1]
                                    + iJ[2]*dN[i][2] * iJ[2]*dN[j][2] );
            }
        }
        // ---- closed-form checks (abort on failure) ----
        const int sgn[8][3] = {
            {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
            {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1} };
        const real_t V = g.hx * g.hy * g.hz;
        real_t errM = 0, sumM = 0, errKrow = 0, errKdiag = 0, errSym = 0;
        const real_t Kdiag = (g.hy*g.hz/g.hx + g.hx*g.hz/g.hy + g.hx*g.hy/g.hz) / 9.0;
        for (int i = 0; i < 8; ++i) {
            real_t rs = 0;
            for (int j = 0; j < 8; ++j) {
                int d = (sgn[i][0]!=sgn[j][0]) + (sgn[i][1]!=sgn[j][1]) + (sgn[i][2]!=sgn[j][2]);
                real_t Mref = V / 216.0 * (real_t)(8 >> d);
                errM = fmax(errM, fabs(Mh[i*8+j] - Mref) / (V/216.0*8));
                sumM += Mh[i*8+j];
                rs += Kh[i*8+j];
                errSym = fmax(errSym, fabs(Kh[i*8+j] - Kh[j*8+i]) + fabs(Mh[i*8+j] - Mh[j*8+i]));
            }
            errKrow  = fmax(errKrow,  fabs(rs) / Kdiag);
            errKdiag = fmax(errKdiag, fabs(Kh[i*9] - Kdiag) / Kdiag);
        }
        real_t errSum = fabs(sumM - V) / V;
        int ok = (errM < 1e-12 && errSum < 1e-12 && errKrow < 1e-12 &&
                  errKdiag < 1e-12 && errSym < 1e-12 * Kdiag);
        if (g.rank == 0)
            printf("Reference matrices: M pattern %.1e, sum(M)-V %.1e, K rowsum %.1e, "
                   "K diag %.1e, symmetry %.1e  -> %s\n",
                   (double)errM, (double)errSum, (double)errKrow, (double)errKdiag,
                   (double)errSym, ok ? "OK" : "FAIL");
        if (!ok) MPI_Abort(MPI_COMM_WORLD, 2);
        CUDA_CHECK(cudaMemcpyToSymbol(c_Mhat, Mh, 64 * sizeof(real_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_Khat, Kh, 64 * sizeof(real_t)));
    }

    // ---- fields -------------------------------------------------------------
    void load_fields(const char* prefix) {
        long ne_slab = (long)g.nx_local * g.ny * g.nz;
        long offset  = (long)g.ix_start * g.ny * g.nz;
        std::vector<real_t> h_k(ne_slab), h_rhoc(ne_slab);
        char fk[512], fr[512];
        snprintf(fk, sizeof(fk), "%s_k.bin", prefix);
        snprintf(fr, sizeof(fr), "%s_rhoc.bin", prefix);
        FILE* f1 = fopen(fk, "rb");
        FILE* f2 = fopen(fr, "rb");
        if (!f1 || !f2) {
            if (g.rank == 0) fprintf(stderr, "Cannot open %s / %s (run gen_field.py)\n", fk, fr);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fseek(f1, offset * sizeof(real_t), SEEK_SET);
        fseek(f2, offset * sizeof(real_t), SEEK_SET);
        if (fread(h_k.data(), sizeof(real_t), ne_slab, f1) != (size_t)ne_slab ||
            fread(h_rhoc.data(), sizeof(real_t), ne_slab, f2) != (size_t)ne_slab) {
            if (g.rank == 0)
                fprintf(stderr, "Field file too small: expected %ld doubles for this rank "
                        "(grid mismatch? regenerate with gen_field.py --n %d)\n", ne_slab, g.nx);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fclose(f1); fclose(f2);
        real_t lmin[2] = {h_k[0], h_rhoc[0]}, lmax[2] = {h_k[0], h_rhoc[0]};
        for (long i = 0; i < ne_slab; ++i) {
            if (!(h_k[i] > 0) || !(h_rhoc[i] > 0)) {
                fprintf(stderr, "rank %d: non-positive or NaN coefficient at element %ld\n", g.rank, i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            lmin[0] = fmin(lmin[0], h_k[i]);    lmax[0] = fmax(lmax[0], h_k[i]);
            lmin[1] = fmin(lmin[1], h_rhoc[i]); lmax[1] = fmax(lmax[1], h_rhoc[i]);
        }
        real_t gmin[2], gmax[2];
        MPI_Allreduce(lmin, gmin, 2, MPI_REAL_T, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(lmax, gmax, 2, MPI_REAL_T, MPI_MAX, MPI_COMM_WORLD);
        kmin = gmin[0]; kmax = gmax[0]; rhocmin = gmin[1]; rhocmax = gmax[1];
        CUDA_CHECK(cudaMalloc(&d_kcell,    ne_slab * sizeof(real_t)));
        CUDA_CHECK(cudaMalloc(&d_rhoccell, ne_slab * sizeof(real_t)));
        CUDA_CHECK(cudaMemcpy(d_kcell, h_k.data(), ne_slab * sizeof(real_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rhoccell, h_rhoc.data(), ne_slab * sizeof(real_t), cudaMemcpyHostToDevice));
        g.k_cell = d_kcell; g.rhoc_cell = d_rhoccell;
    }

    // ---- setup --------------------------------------------------------------
    void setup(int nx, int ny, int nz, real_t Lx, real_t Ly, real_t Lz,
               real_t dt_in, MPI_Comm comm, const char* field_prefix,
               int verify, int kfast, int use_colored) {
        MPI_Comm_rank(comm, &g.rank);
        MPI_Comm_size(comm, &g.nprocs);
        g.nx = nx; g.ny = ny; g.nz = nz;
        g.hx = Lx / nx; g.hy = Ly / ny; g.hz = Lz / nz;
        g.verify = verify;
        kernel_fast = kfast; colored = use_colored;
        if (nx % g.nprocs != 0) {
            if (g.rank == 0) fprintf(stderr, "nx (%d) must be divisible by nprocs (%d)\n", nx, g.nprocs);
            MPI_Abort(comm, 1);
        }
        g.nx_local = nx / g.nprocs;
        g.ix_start = g.rank * g.nx_local;
        g.nnx_local = g.nx_local + 1;
        g.nny = ny + 1; g.nnz = nz + 1;
        g.nnodes_local = (long)g.nnx_local * g.nny * g.nnz;
        g.ne_local = (long)g.nx_local * g.ny * g.nz;
        dt = dt_in;

        long face = (long)g.nny * g.nnz;
        CUDA_CHECK(cudaMalloc(&d_diag, g.nnodes_local * sizeof(real_t)));
        CUDA_CHECK(cudaMalloc(&d_F,    g.nnodes_local * sizeof(real_t)));
        CUDA_CHECK(cudaMalloc(&d_send_l, face * sizeof(real_t)));
        CUDA_CHECK(cudaMalloc(&d_send_r, face * sizeof(real_t)));
        CUDA_CHECK(cudaMalloc(&d_recv_l, face * sizeof(real_t)));
        CUDA_CHECK(cudaMalloc(&d_recv_r, face * sizeof(real_t)));
        h_send_l = (real_t*)malloc(face * sizeof(real_t));
        h_send_r = (real_t*)malloc(face * sizeof(real_t));
        h_recv_l = (real_t*)malloc(face * sizeof(real_t));
        h_recv_r = (real_t*)malloc(face * sizeof(real_t));
        blocks_node = (g.nnodes_local + threads_node - 1) / threads_node;
        blocks_elem = (g.ne_local + threads_elem - 1) / threads_elem;
        for (int c = 0; c < 8; ++c)
            color_blocks[c] = (color_count(c, g) + threads_elem - 1) / threads_elem;
        CUDA_CHECK(cudaMalloc(&d_partial, blocks_node * sizeof(real_t)));
        CUDA_CHECK(cudaMalloc(&d_dot_result, sizeof(real_t)));

        load_fields(field_prefix);
        build_reference_matrices();

        // Jacobi diagonal
        CUDA_CHECK(cudaMemset(d_diag, 0, g.nnodes_local * sizeof(real_t)));
        if (!colored) { diag_kernel<<<blocks_elem, threads_elem>>>(d_diag, g, dt, -1); CUDA_CHECK_LAUNCH(); }
        else for (int c = 0; c < 8; ++c) { diag_kernel<<<color_blocks[c], threads_elem>>>(d_diag, g, dt, c); CUDA_CHECK_LAUNCH(); }
        halo_exchange_sum(d_diag);
        apply_dirichlet_value_kernel<<<blocks_node, threads_node>>>(d_diag, g, 1.0);
        CUDA_CHECK_LAUNCH();

        // Source vector (time-independent): computed once.
        CUDA_CHECK(cudaMemset(d_F, 0, g.nnodes_local * sizeof(real_t)));
        if (!g.verify) {
            if (!colored) { source_kernel<<<blocks_elem, threads_elem>>>(d_F, g, -1); CUDA_CHECK_LAUNCH(); }
            else for (int c = 0; c < 8; ++c) { source_kernel<<<color_blocks[c], threads_elem>>>(d_F, g, c); CUDA_CHECK_LAUNCH(); }
            halo_exchange_sum(d_F);
            apply_dirichlet_value_kernel<<<blocks_node, threads_node>>>(d_F, g, 0.0);
            CUDA_CHECK_LAUNCH();
        }
    }

    void cleanup() {
        cudaFree(d_kcell); cudaFree(d_rhoccell); cudaFree(d_diag); cudaFree(d_F);
        cudaFree(d_send_l); cudaFree(d_send_r); cudaFree(d_recv_l); cudaFree(d_recv_r);
        cudaFree(d_partial); cudaFree(d_dot_result);
        free(h_send_l); free(h_send_r); free(h_recv_l); free(h_recv_r);
    }

    // ---- halo ---------------------------------------------------------------
    void halo_exchange_sum(real_t* d_v) {
        if (g.nprocs == 1) return;
        long face = (long)g.nny * g.nnz;
        int t = 256;
        long b = (face + t - 1) / t;
        if (g.rank > 0)             pack_face_kernel<<<b, t>>>(d_v, d_send_l, g, 0);
        if (g.rank < g.nprocs - 1)  pack_face_kernel<<<b, t>>>(d_v, d_send_r, g, g.nx_local);
        CUDA_CHECK_LAUNCH();
        if (g.rank > 0)
            CUDA_CHECK(cudaMemcpy(h_send_l, d_send_l, face * sizeof(real_t), cudaMemcpyDeviceToHost));
        if (g.rank < g.nprocs - 1)
            CUDA_CHECK(cudaMemcpy(h_send_r, d_send_r, face * sizeof(real_t), cudaMemcpyDeviceToHost));

        // Level H: corrupt the staging buffer of the right-going face, once.
        if (fi.armed && fi.level == 3 && g.rank == fi.rank && g.rank < g.nprocs - 1) {
            long idx = fi.target % face;
            unsigned long long* w = reinterpret_cast<unsigned long long*>(&h_send_r[idx]);
            *w ^= (1ULL << fi.bit);
            fi.armed = 0;
        }

        MPI_Request reqs[4]; int nr = 0;
        if (g.rank > 0) {
            MPI_Irecv(h_recv_l, face, MPI_REAL_T, g.rank-1, 0, MPI_COMM_WORLD, &reqs[nr++]);
            MPI_Isend(h_send_l, face, MPI_REAL_T, g.rank-1, 1, MPI_COMM_WORLD, &reqs[nr++]);
        }
        if (g.rank < g.nprocs - 1) {
            MPI_Irecv(h_recv_r, face, MPI_REAL_T, g.rank+1, 1, MPI_COMM_WORLD, &reqs[nr++]);
            MPI_Isend(h_send_r, face, MPI_REAL_T, g.rank+1, 0, MPI_COMM_WORLD, &reqs[nr++]);
        }
        MPI_Waitall(nr, reqs, MPI_STATUSES_IGNORE);
        if (g.rank > 0) {
            CUDA_CHECK(cudaMemcpy(d_recv_l, h_recv_l, face * sizeof(real_t), cudaMemcpyHostToDevice));
            unpack_face_add_kernel<<<b, t>>>(d_v, d_recv_l, g, 0);
        }
        if (g.rank < g.nprocs - 1) {
            CUDA_CHECK(cudaMemcpy(d_recv_r, h_recv_r, face * sizeof(real_t), cudaMemcpyHostToDevice));
            unpack_face_add_kernel<<<b, t>>>(d_v, d_recv_r, g, g.nx_local);
        }
        CUDA_CHECK_LAUNCH();
    }

    // ---- operator -----------------------------------------------------------
    // y = (M + dt_eff K) x, with halo sum and Dirichlet projection.
    // allow_fi: whether a pending level-R injection may fire in this call.
    void apply_operator(const real_t* d_x, real_t* d_y, real_t dt_eff, bool allow_fi) {
        zero_kernel<<<blocks_node, threads_node>>>(d_y, g.nnodes_local);
        CUDA_CHECK_LAUNCH();
        FIConfig f = fi;
        if (!allow_fi) f.armed = 0;
        if (!colored) {
            if (kernel_fast) operator_kernel<1><<<blocks_elem, threads_elem>>>(d_x, d_y, g, dt_eff, f, -1);
            else             operator_kernel<0><<<blocks_elem, threads_elem>>>(d_x, d_y, g, dt_eff, f, -1);
            CUDA_CHECK_LAUNCH();
        } else {
            for (int c = 0; c < 8; ++c) {
                if (kernel_fast) operator_kernel<1><<<color_blocks[c], threads_elem>>>(d_x, d_y, g, dt_eff, f, c);
                else             operator_kernel<0><<<color_blocks[c], threads_elem>>>(d_x, d_y, g, dt_eff, f, c);
                CUDA_CHECK_LAUNCH();
            }
        }
        if (allow_fi && fi.armed && fi.level == 2) fi.armed = 0;   // one-shot
        halo_exchange_sum(d_y);
        apply_dirichlet_value_kernel<<<blocks_node, threads_node>>>(d_y, g, 0.0);
        CUDA_CHECK_LAUNCH();
    }

    void apply_matvec(const real_t* d_x, real_t* d_y) { apply_operator(d_x, d_y, dt, true); }

    void apply_preconditioner(const real_t* d_r, real_t* d_z) {
        jacobi_apply_kernel<<<blocks_node, threads_node>>>(d_r, d_diag, d_z, g.nnodes_local);
        CUDA_CHECK_LAUNCH();
    }

    // b = M T^n + dt F. Level H (if armed) fires in this call's halo exchange.
    void build_rhs(const real_t* d_Tn, real_t* d_b) {
        apply_operator(d_Tn, d_b, 0.0, false);
        if (!g.verify) {
            axpy_kernel<<<blocks_node, threads_node>>>(dt, d_F, d_b, g.nnodes_local);
            CUDA_CHECK_LAUNCH();
        }
    }

    real_t dot(const real_t* d_x, const real_t* d_y) {
        owned_dot_partial_kernel<<<blocks_node, threads_node, threads_node*sizeof(real_t)>>>(
            d_x, d_y, d_partial, g);
        CUDA_CHECK_LAUNCH();
        reduce_partials_kernel<<<1, 1024, 1024*sizeof(real_t)>>>(d_partial, blocks_node, d_dot_result);
        CUDA_CHECK_LAUNCH();
        real_t local, global;
        CUDA_CHECK(cudaMemcpy(&local, d_dot_result, sizeof(real_t), cudaMemcpyDeviceToHost));
        MPI_Allreduce(&local, &global, 1, MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);
        return global;
    }
    real_t norm2(const real_t* d_x) { return sqrt(dot(d_x, d_x)); }
};

// =============================================================================
// Matrix-free PCG
// =============================================================================
int pcg_solve(HeatOperator& op, const real_t* d_b, real_t* d_x, int max_iter, real_t tol,
              real_t* d_r, real_t* d_z, real_t* d_p, real_t* d_Ap) {
    long n = op.g.nnodes_local; int tn = op.threads_node; long bn = op.blocks_node;
    (void)tn; (void)bn;
    op.apply_matvec(d_x, d_Ap);                       // level R fires here if armed
    copy_kernel<<<bn, tn>>>(d_b, d_r, n); CUDA_CHECK_LAUNCH();
    axpy_kernel<<<bn, tn>>>(-1.0, d_Ap, d_r, n); CUDA_CHECK_LAUNCH();
    real_t bnorm = op.norm2(d_b);
    if (bnorm < 1e-30) bnorm = 1.0;
    op.apply_preconditioner(d_r, d_z);
    copy_kernel<<<bn, tn>>>(d_z, d_p, n); CUDA_CHECK_LAUNCH();
    real_t rho = op.dot(d_r, d_z);
    int iter = 0;
    for (iter = 0; iter < max_iter; ++iter) {
        op.apply_matvec(d_p, d_Ap);
        real_t pAp = op.dot(d_p, d_Ap);
        if (fabs(pAp) < 1e-30) break;
        real_t alpha = rho / pAp;
        axpy_kernel<<<bn, tn>>>( alpha, d_p,  d_x, n); CUDA_CHECK_LAUNCH();
        axpy_kernel<<<bn, tn>>>(-alpha, d_Ap, d_r, n); CUDA_CHECK_LAUNCH();
        real_t rnorm = op.norm2(d_r);
        if (rnorm / bnorm < tol) { iter++; break; }
        op.apply_preconditioner(d_r, d_z);
        real_t rho_new = op.dot(d_r, d_z);
        real_t beta = rho_new / rho;
        aypx_kernel<<<bn, tn>>>(beta, d_z, d_p, n); CUDA_CHECK_LAUNCH();
        rho = rho_new;
    }
    return iter;
}

// =============================================================================
// Host-side helpers
// =============================================================================
static void host_snapshot(HeatOperator& op, const real_t* d_T, std::vector<real_t>& out) {
    out.resize(op.g.nnodes_local);
    CUDA_CHECK(cudaMemcpy(out.data(), d_T, op.g.nnodes_local * sizeof(real_t), cudaMemcpyDeviceToHost));
}

// ||a - b|| / ||b|| over owned nodes, global.
static real_t global_rel_err(HeatOperator& op, const std::vector<real_t>& a, const std::vector<real_t>& b) {
    const Grid& g = op.g;
    double num = 0.0, den = 0.0;
    for (long n = 0; n < g.nnodes_local; ++n) {
        if (!is_owned((int)(n / (g.nny * g.nnz)), g)) continue;
        double d = (double)a[n] - (double)b[n];
        num += d * d; den += (double)b[n] * (double)b[n];
    }
    double s[2] = {num, den}, gl[2];
    MPI_Allreduce(s, gl, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (gl[1] < 1e-300) gl[1] = 1.0;
    return sqrt(gl[0] / gl[1]);
}

struct RunResult {
    std::vector<int>    snap_steps;
    std::vector<real_t> E;          // vs reference (empty for the reference run)
    std::vector<int>    cg_iters;   // per step
    double solve_ms = 0.0;          // build_rhs + pcg only (no snapshots, no logging)
    int nonfinite = 0;
};

// One full simulation from T=0. If ref != nullptr, E(t) vs *ref is computed at
// snapshot steps (streaming: only the reference run keeps its snapshots).
static void run_case(HeatOperator& op, int n_steps, int snap_every, FIConfig fi_cfg,
                     int reproject_bc,
                     real_t* d_T, real_t* d_b, real_t* d_r, real_t* d_z, real_t* d_p, real_t* d_Ap,
                     std::vector<std::vector<real_t>>* keep_snaps,
                     const std::vector<std::vector<real_t>>* ref,
                     RunResult& res) {
    long n = op.g.nnodes_local;
    CUDA_CHECK(cudaMemset(d_T, 0, n * sizeof(real_t)));
    op.fi = FIConfig();
    res = RunResult();
    if (keep_snaps) keep_snaps->clear();
    std::vector<real_t> tmp;
    size_t si = 0;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    for (int step = 0; step < n_steps; ++step) {
        if (fi_cfg.level > 0 && step == fi_cfg.step) {
            if (fi_cfg.level == 1) {
                if (op.g.rank == fi_cfg.rank) {
                    flip_bit_global_kernel<<<1, 1>>>(d_T, fi_cfg.target % n, fi_cfg.bit);
                    CUDA_CHECK_LAUNCH();
                }
            } else {
                op.fi = fi_cfg; op.fi.armed = 1;
            }
        }
        cudaEventRecord(t0);
        op.build_rhs(d_T, d_b);
        int it = pcg_solve(op, d_b, d_T, 500, 1e-8, d_r, d_z, d_p, d_Ap);
        if (reproject_bc) {
            apply_dirichlet_value_kernel<<<op.blocks_node, op.threads_node>>>(d_T, op.g, 0.0);
            CUDA_CHECK_LAUNCH();
        }
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1); res.solve_ms += ms;
        op.fi.armed = 0;
        res.cg_iters.push_back(it);

        bool snap = (snap_every > 0 && step % snap_every == 0) || step == n_steps - 1;
        if (snap) {
            if (keep_snaps) { keep_snaps->emplace_back(); host_snapshot(op, d_T, keep_snaps->back()); }
            if (ref) {
                host_snapshot(op, d_T, tmp);
                real_t E = global_rel_err(op, tmp, (*ref)[si]);
                if (!std::isfinite((double)E)) res.nonfinite = 1;
                res.E.push_back(E);
            }
            res.snap_steps.push_back(step);
            ++si;
        }
    }
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

static void write_field(HeatOperator& op, const std::vector<real_t>& T,
                        const std::string& outdir, const std::string& tag) {
    char path[600];
    snprintf(path, sizeof(path), "%s/T_final_%s_rank%d.bin", outdir.c_str(), tag.c_str(), op.g.rank);
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); MPI_Abort(MPI_COMM_WORLD, 1); }
    fwrite(T.data(), sizeof(real_t), T.size(), f);
    fclose(f);
}

// Is the injection target on the Dirichlet boundary? (decided on fi.rank)
static int target_on_boundary(HeatOperator& op, const FIConfig& fi) {
    const Grid& g = op.g; int flag = 0;
    if (g.rank == fi.rank) {
        if (fi.level == 1) {
            long nd = fi.target % g.nnodes_local;
            flag = is_boundary((int)(nd / (g.nny*g.nnz)), (int)((nd / g.nnz) % g.nny), (int)(nd % g.nnz), g);
        } else if (fi.level == 2) {
            long e = fi.target % g.ne_local;
            int ix_e = (int)(e / ((long)g.ny*g.nz)), iy_e = (int)((e / g.nz) % g.ny), iz_e = (int)(e % g.nz);
            flag = is_boundary(ix_e, iy_e, iz_e, g);           // node 0 of the element
        } else if (fi.level == 3) {
            long face = (long)g.nny * g.nnz, idx = fi.target % face;
            int iy = (int)(idx / g.nnz), iz = (int)(idx % g.nnz);
            flag = (iy == 0 || iy == g.ny || iz == 0 || iz == g.nz);
        }
    }
    MPI_Bcast(&flag, 1, MPI_INT, fi.rank, MPI_COMM_WORLD);
    return flag;
}

// Layer-0 check. Requires a homogeneous field.
static void run_verify(HeatOperator& op, int n_steps, real_t dt, const std::string& outdir,
                       real_t* d_T, real_t* d_b, real_t* d_r, real_t* d_z, real_t* d_p, real_t* d_Ap) {
    if (op.kmin != op.kmax || op.rhocmin != op.rhocmax) {
        if (op.g.rank == 0)
            fprintf(stderr, "--verify needs a homogeneous field (k %g..%g, rhoc %g..%g)\n",
                    (double)op.kmin, (double)op.kmax, (double)op.rhocmin, (double)op.rhocmax);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const real_t alpha = op.kmin / op.rhocmin;
    long n = op.g.nnodes_local;
    exact_mode_kernel<<<op.blocks_node, op.threads_node>>>(d_T, op.g, 0.0, alpha);
    CUDA_CHECK_LAUNCH();
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    long total_it = 0;
    for (int step = 0; step < n_steps; ++step) {
        op.build_rhs(d_T, d_b);
        total_it += pcg_solve(op, d_b, d_T, 500, 1e-8, d_r, d_z, d_p, d_Ap);
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    real_t t_end = n_steps * dt;
    exact_mode_kernel<<<op.blocks_node, op.threads_node>>>(d_r, op.g, t_end, alpha);
    CUDA_CHECK_LAUNCH();
    real_t exact_norm = op.norm2(d_r);
    axpy_kernel<<<op.blocks_node, op.threads_node>>>(-1.0, d_T, d_r, n);
    CUDA_CHECK_LAUNCH();
    real_t rel_err = op.norm2(d_r) / exact_norm;
    if (op.g.rank == 0) {
        printf("VERIFY: N=%d dt=%.4e t_end=%.6e alpha=%g rel_L2_err=%.6e  (%s/%s, %.1f ms, %.1f CG it/step)\n",
               op.g.nx, (double)dt, (double)t_end, (double)alpha, (double)rel_err,
               op.kernel_fast ? "fast" : "gauss", op.colored ? "colored" : "atomic",
               ms, (double)total_it / n_steps);
        std::string p = outdir + "/verify.csv";
        FILE* vf = fopen(p.c_str(), "r"); int fresh = (vf == nullptr); if (vf) fclose(vf);
        vf = fopen(p.c_str(), "a");
        if (!vf) { fprintf(stderr, "cannot write %s\n", p.c_str()); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fresh) fprintf(vf, "N,dt,t_end,nprocs,kernel,scatter,rel_L2_err,solve_ms,avg_cg_iters\n");
        fprintf(vf, "%d,%.6e,%.6e,%d,%s,%s,%.10e,%.3f,%.3f\n",
                op.g.nx, (double)dt, (double)t_end, op.g.nprocs,
                op.kernel_fast ? "fast" : "gauss", op.colored ? "colored" : "atomic",
                (double)rel_err, ms, (double)total_it / n_steps);
        fclose(vf);
    }
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

static void usage(const char* prog) {
    fprintf(stderr,
      "usage: %s N t_final dt field_prefix [outdir]\n"
      "   [--fi LEVEL STEP TARGET BIT] [--fi-rank R] [--snap S] [--tag NAME]\n"
      "   [--kernel gauss|fast] [--scatter atomic|colored] [--thr X]\n"
      "   [--reproject-bc] [--verify]\n"
      "  LEVEL: 1=G global memory, 2=R accumulator, 3=H halo staging buffer\n"
      "  BIT:   0-51 mantissa, 52-62 exponent, 63 sign\n", prog);
}

// =============================================================================
// Driver
// =============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n_gpus = 0; cudaGetDeviceCount(&n_gpus);
    if (n_gpus == 0) { if (rank == 0) fprintf(stderr, "No CUDA devices\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaSetDevice(rank % n_gpus);

    if (argc < 5) { if (rank == 0) usage(argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }
    int    N       = atoi(argv[1]);
    real_t t_final = atof(argv[2]);
    real_t dt      = atof(argv[3]);
    const char* field_prefix = argv[4];
    std::string outdir = "out", tag = "run";
    FIConfig fi_cfg;
    int snap_every = 5, verify_mode = 0, kernel_fast = 1, colored = 0, reproject = 0;
    real_t thr_user = 1e-13;
    for (int i = 5; i < argc; ++i) {
        if (!strcmp(argv[i], "--fi") && i + 4 < argc) {
            fi_cfg.level = atoi(argv[++i]); fi_cfg.step = atoi(argv[++i]);
            fi_cfg.target = atol(argv[++i]); fi_cfg.bit = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--fi-rank") && i + 1 < argc) { fi_cfg.rank = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--snap") && i + 1 < argc)    { snap_every = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--tag") && i + 1 < argc)     { tag = argv[++i];
        } else if (!strcmp(argv[i], "--thr") && i + 1 < argc)     { thr_user = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--kernel") && i + 1 < argc)  { kernel_fast = strcmp(argv[++i], "gauss") != 0;
        } else if (!strcmp(argv[i], "--scatter") && i + 1 < argc) { colored = !strcmp(argv[++i], "colored");
        } else if (!strcmp(argv[i], "--reproject-bc")) { reproject = 1;
        } else if (!strcmp(argv[i], "--verify"))       { verify_mode = 1;
        } else if (argv[i][0] != '-') { outdir = argv[i];
        } else { if (rank == 0) { fprintf(stderr, "unknown option %s\n", argv[i]); usage(argv[0]); } MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    if (fi_cfg.level > 0 && (fi_cfg.rank < 0 || fi_cfg.rank >= nprocs)) {
        if (rank == 0) fprintf(stderr, "--fi-rank must be in [0,%d)\n", nprocs);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (fi_cfg.level == 3 && (nprocs < 2 || fi_cfg.rank >= nprocs - 1)) {
        if (rank == 0) fprintf(stderr, "level H needs >= 2 ranks and --fi-rank < nprocs-1\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (fi_cfg.level > 0 && (fi_cfg.bit < 0 || fi_cfg.bit > 63)) {
        if (rank == 0) fprintf(stderr, "BIT must be in 0..63\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int nx = N, ny = N, nz = N;
    int n_steps = (int)(t_final / dt + 0.5);
    if (rank == 0) {
        printf("=== Assembly-Free FEM Heat Solver v2 (Backward Euler + Jacobi-PCG) ===\n");
        printf("Grid %d^3 (DOFs %ld)  ranks %d (1D x-decomp, %d cells/rank)\n",
               N, (long)(N+1)*(N+1)*(N+1), nprocs, N/nprocs);
        printf("dt %.4e  t_final %.4e  steps %d  kernel %s  scatter %s  mode %s\n",
               (double)dt, (double)t_final, n_steps, kernel_fast ? "fast" : "gauss",
               colored ? "colored" : "atomic", verify_mode ? "VERIFY" : "production");
    }

    HeatOperator op;
    op.setup(nx, ny, nz, 1.0, 1.0, 1.0, dt, MPI_COMM_WORLD, field_prefix, verify_mode, kernel_fast, colored);
    if (rank == 0)
        printf("Material: k in [%g, %g] (contrast %.3g), rho*c in [%g, %g]\n",
               (double)op.kmin, (double)op.kmax, (double)(op.kmax/op.kmin), (double)op.rhocmin, (double)op.rhocmax);

    long n = op.g.nnodes_local;
    real_t *d_T, *d_b, *d_r, *d_z, *d_p, *d_Ap;
    CUDA_CHECK(cudaMalloc(&d_T,  n * sizeof(real_t))); CUDA_CHECK(cudaMalloc(&d_b,  n * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_r,  n * sizeof(real_t))); CUDA_CHECK(cudaMalloc(&d_z,  n * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_p,  n * sizeof(real_t))); CUDA_CHECK(cudaMalloc(&d_Ap, n * sizeof(real_t)));

    if (verify_mode) {
        run_verify(op, n_steps, dt, outdir, d_T, d_b, d_r, d_z, d_p, d_Ap);
    } else {
        // ---- twin-run protocol ---------------------------------------------
        std::vector<std::vector<real_t>> snapA;
        RunResult rA, rB, rF;
        run_case(op, n_steps, snap_every, FIConfig(), reproject, d_T, d_b, d_r, d_z, d_p, d_Ap, &snapA, nullptr, rA);
        write_field(op, snapA.back(), outdir, tag + "_clean");
        run_case(op, n_steps, snap_every, FIConfig(), reproject, d_T, d_b, d_r, d_z, d_p, d_Ap, nullptr, &snapA, rB);
        int on_bnd = -1;
        if (fi_cfg.level > 0) {
            on_bnd = target_on_boundary(op, fi_cfg);
            run_case(op, n_steps, snap_every, fi_cfg, reproject, d_T, d_b, d_r, d_z, d_p, d_Ap, nullptr, &snapA, rF);
            std::vector<real_t> Tf; host_snapshot(op, d_T, Tf);
            write_field(op, Tf, outdir, tag + "_fault");
        }

        // ---- trace + summary (rank 0) -----------------------------------------
        real_t Efl_max = 0, Efa_max = 0, Efa_int = 0, Efa_final = 0, Efl_final = 0;
        int step_max = -1, iters_delta_max = 0;
        for (size_t s = 0; s < rB.E.size(); ++s) {
            Efl_max = fmax(Efl_max, rB.E[s]); Efl_final = rB.E[s];
            if (fi_cfg.level > 0) {
                real_t E = rF.E[s];
                if (std::isfinite((double)E) && E > Efa_max) { Efa_max = E; step_max = rB.snap_steps[s]; }
                Efa_final = E;
                real_t w = (s + 1 < rB.E.size()) ? (rB.snap_steps[s+1] - rB.snap_steps[s]) * dt : 0.0;
                if (std::isfinite((double)E)) Efa_int += E * w;
            }
        }
        for (size_t s = 0; fi_cfg.level > 0 && s < rF.cg_iters.size(); ++s)
            iters_delta_max = std::max(iters_delta_max, rF.cg_iters[s] - rA.cg_iters[s]);
        real_t thr = fmax(thr_user, 10.0 * Efl_max);
        const char* cls = "clean";
        if (fi_cfg.level > 0) {
            if (rF.nonfinite)          cls = "detected";
            else if (Efa_max <= thr)   cls = "benign";
            else if (Efa_final <= thr) cls = "transient";
            else                       cls = "persistent";
        }
        if (rank == 0) {
            char tp[600]; snprintf(tp, sizeof(tp), "%s/fi_trace_%s.csv", outdir.c_str(), tag.c_str());
            FILE* tf = fopen(tp, "w");
            if (!tf) { fprintf(stderr, "cannot write %s\n", tp); MPI_Abort(MPI_COMM_WORLD, 1); }
            fprintf(tf, "step,t,E_noise_floor,E_fault,cg_A,cg_B,cg_F\n");
            for (size_t s = 0; s < rB.E.size(); ++s) {
                int st = rB.snap_steps[s];
                fprintf(tf, "%d,%.6e,%.10e,%.10e,%d,%d,%d\n", st, st*dt, (double)rB.E[s],
                        fi_cfg.level > 0 ? (double)rF.E[s] : 0.0,
                        rA.cg_iters[st], rB.cg_iters[st], fi_cfg.level > 0 ? rF.cg_iters[st] : -1);
            }
            fclose(tf);
            double meanA = 0; for (int it : rA.cg_iters) meanA += it; meanA /= std::max<size_t>(1, rA.cg_iters.size());
            char sp[600]; snprintf(sp, sizeof(sp), "%s/fi_summary_%s.csv", outdir.c_str(), tag.c_str());
            FILE* sf = fopen(sp, "w");
            fprintf(sf, "tag,N,dt,n_steps,nprocs,kernel,scatter,reproject_bc,kmin,kmax,rhocmin,rhocmax,"
                        "fi_level,fi_rank,fi_step,fi_target,fi_bit,target_on_boundary,"
                        "E_floor_max,E_floor_final,E_fault_max,step_at_max,E_fault_int,E_fault_final,"
                        "thr,iters_delta_max,mean_cg_iters_A,solve_ms_A,solve_ms_B,solve_ms_F,class\n");
            fprintf(sf, "%s,%d,%.3e,%d,%d,%s,%s,%d,%g,%g,%g,%g,%d,%d,%d,%ld,%d,%d,"
                        "%.6e,%.6e,%.6e,%d,%.6e,%.6e,%.3e,%d,%.2f,%.3f,%.3f,%.3f,%s\n",
                    tag.c_str(), N, (double)dt, n_steps, nprocs, kernel_fast ? "fast" : "gauss",
                    colored ? "colored" : "atomic", reproject,
                    (double)op.kmin, (double)op.kmax, (double)op.rhocmin, (double)op.rhocmax,
                    fi_cfg.level, fi_cfg.rank, fi_cfg.step, fi_cfg.target, fi_cfg.bit, on_bnd,
                    (double)Efl_max, (double)Efl_final, (double)Efa_max, step_max, (double)Efa_int,
                    (double)Efa_final, (double)thr, iters_delta_max, meanA,
                    rA.solve_ms, rB.solve_ms, rF.solve_ms, cls);
            fclose(sf);
            printf("\n=== Done (%s) ===\n", tag.c_str());
            printf("solve wall A/B/F (ms): %.1f / %.1f / %.1f   mean CG iters A: %.2f\n",
                   rA.solve_ms, rB.solve_ms, rF.solve_ms, meanA);
            printf("noise floor  max %.3e  final %.3e\n", (double)Efl_max, (double)Efl_final);
            if (fi_cfg.level > 0)
                printf("fault        max %.3e @step %d  int %.3e  final %.3e  iters+%d  bnd %d  -> %s\n",
                       (double)Efa_max, step_max, (double)Efa_int, (double)Efa_final, iters_delta_max, on_bnd, cls);
            char mp[600]; snprintf(mp, sizeof(mp), "%s/meta_%s.txt", outdir.c_str(), tag.c_str());
            FILE* mf = fopen(mp, "w");
            fprintf(mf, "N %d\nnprocs %d\ndt %g\nn_steps %d\nfield %s\nkernel %s\nscatter %s\n",
                    N, nprocs, (double)dt, n_steps, field_prefix, kernel_fast ? "fast" : "gauss",
                    colored ? "colored" : "atomic");
            fclose(mf);
        }
    }

    cudaFree(d_T); cudaFree(d_b); cudaFree(d_r); cudaFree(d_z); cudaFree(d_p); cudaFree(d_Ap);
    op.cleanup();
    MPI_Finalize();
    return 0;
}
