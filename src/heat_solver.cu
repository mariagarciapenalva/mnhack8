// =============================================================================
// heat_solver.cu
//
// Assembly-free FEM solver for the transient heat equation in 3D heterogeneous
// materials. Backward Euler time stepping. Matrix-free Jacobi-preconditioned
// conjugate gradient. 1D MPI domain decomposition along the x-axis. Designed
// for integration with the GlaSs library (the matvec and preconditioner are
// exposed as clean callable methods on HeatOperator).
//
// Built on the MNHack7 prototype by María García Penalva, with corrections
// and extensions per code review (May 2026).
//
// PDE:
//   ρ(x) c(x) ∂T/∂t = ∇·(k(x) ∇T) + Q(x,t)    in Ω = [0,1]^3
//   T = 0                                       on ∂Ω (Dirichlet)
//
// Discretisation: trilinear hexahedral FEM on uniform grid.
// Time integration: backward Euler.
// Solver: matrix-free preconditioned CG with Jacobi preconditioner.
// =============================================================================

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// -----------------------------------------------------------------------------
// Precision
// -----------------------------------------------------------------------------
typedef double real_t;
#define MPI_REAL_T MPI_DOUBLE

// -----------------------------------------------------------------------------
// CUDA error checking
// -----------------------------------------------------------------------------
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
    int  nx, ny, nz;          // global cells in each direction
    real_t hx, hy, hz;        // cell sizes (uniform)
    int  rank, nprocs;        // MPI info
    int  nx_local;            // local cells in x for this rank
    int  ix_start;            // global x cell index where this rank starts
    int  nnx_local;           // local nodes in x = nx_local + 1
    int  nny;                 // = ny + 1
    int  nnz;                 // = nz + 1
    long nnodes_local;        // total nodes stored locally
    long ne_local;            // total elements owned locally
};

// Linear index of a local node (ix_local, iy, iz) in the local node array.
__host__ __device__ __forceinline__
long lnid(int ix_local, int iy, int iz, const Grid& g) {
    return (long)ix_local * (g.nny * g.nnz) + (long)iy * g.nnz + iz;
}

// Compute physical coordinates of a node from local indices.
// Coordinates are NOT stored in memory anymore (was an O(N) memory bug source);
// they are recomputed cheaply from indices.
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
            iy == 0 || iy == g.ny ||
            iz == 0 || iz == g.nz);
}

// Owner convention for shared nodes:
//   rank r owns ix_local in [0, nx_local), and (only on the last rank)
//   also owns ix_local == nx_local. Used for non-double-counting in dot products.
__host__ __device__ __forceinline__
bool is_owned(int ix_local, const Grid& g) {
    if (g.rank == g.nprocs - 1) return ix_local <= g.nx_local;
    return ix_local < g.nx_local;
}

// -----------------------------------------------------------------------------
// Material model: piecewise constant, 4 regions (preserved from MNHack7 model)
// -----------------------------------------------------------------------------
__device__ __forceinline__
void material_at(real_t x, real_t y, real_t z, real_t& k, real_t& rho_c) {
    real_t kk, rho, c;
    if      (x <  0.5 && y <  0.5 && z < 0.5) { kk = 100.0; rho = 1.0; c = 1.0; }
    else if (x >= 0.5 && y <  0.5 && z < 0.5) { kk = 200.0; rho = 0.8; c = 1.5; }
    else if (x <  0.5 && y >= 0.5 && z < 0.5) { kk = 300.0; rho = 0.6; c = 2.0; }
    else                                      { kk = 400.0; rho = 0.5; c = 2.5; }
    k = kk;
    rho_c = rho * c;
}

__device__ __forceinline__
real_t source_at(real_t x, real_t y, real_t z) {
    const real_t cx = 0.5, cy = 0.5, cz = 0.5;
    const real_t sigma = 0.1;
    const real_t coeff = 1000.0;
    real_t r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz);
    return coeff * exp(-r2 / (2.0 * sigma * sigma));
}

// -----------------------------------------------------------------------------
// Trilinear hex shape functions on reference cube [-1, 1]^3
// Standard node ordering:
//   N0:(-1,-1,-1) N1:(+1,-1,-1) N2:(+1,+1,-1) N3:(-1,+1,-1)
//   N4:(-1,-1,+1) N5:(+1,-1,+1) N6:(+1,+1,+1) N7:(-1,+1,+1)
// -----------------------------------------------------------------------------
__device__ __forceinline__
void shape_func_grad(real_t xi, real_t eta, real_t zeta,
                     real_t N[8], real_t dN[8][3]) {
    const int sgn[8][3] = {
        {-1,-1,-1}, { 1,-1,-1}, { 1, 1,-1}, {-1, 1,-1},
        {-1,-1, 1}, { 1,-1, 1}, { 1, 1, 1}, {-1, 1, 1}
    };
    for (int i = 0; i < 8; ++i) {
        real_t sx = (real_t)sgn[i][0];
        real_t sy = (real_t)sgn[i][1];
        real_t sz = (real_t)sgn[i][2];
        real_t a = 1 + sx*xi;
        real_t b = 1 + sy*eta;
        real_t c = 1 + sz*zeta;
        N[i]     = 0.125 * a * b * c;
        dN[i][0] = 0.125 * sx * b * c;
        dN[i][1] = 0.125 * a * sy * c;
        dN[i][2] = 0.125 * a * b  * sz;
    }
}

// Gather 8 global node indices for hex element (ix_e, iy_e, iz_e).
__device__ __forceinline__
void element_nodes(int ix_e, int iy_e, int iz_e, const Grid& g, long gid[8]) {
    const int off[8][3] = {
        {0,0,0},{1,0,0},{1,1,0},{0,1,0},
        {0,0,1},{1,0,1},{1,1,1},{0,1,1}
    };
    for (int i = 0; i < 8; ++i) {
        gid[i] = lnid(ix_e + off[i][0], iy_e + off[i][1], iz_e + off[i][2], g);
    }
}

// =============================================================================
// Assembly-free matvec: computes y += A*x where A = M + dt*K
// One thread per element. Race conditions on shared nodes resolved via atomicAdd.
// =============================================================================
__global__
void matvec_kernel(const real_t* __restrict__ x,
                   real_t* __restrict__ y,
                   Grid g, real_t dt) {
    long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= g.ne_local) return;

    int ix_e = e / (g.ny * g.nz);
    int iy_e = (e / g.nz) % g.ny;
    int iz_e = e % g.nz;

    long gid[8];
    element_nodes(ix_e, iy_e, iz_e, g, gid);

    real_t xe[8], ye[8] = {0};
    for (int i = 0; i < 8; ++i) xe[i] = x[gid[i]];

    real_t x0, y0, z0;
    node_coords(ix_e, iy_e, iz_e, g, x0, y0, z0);

    // Jacobian of mapping [-1,1]^3 -> physical hex (uniform mesh)
    const real_t det_J  = g.hx * g.hy * g.hz / 8.0;
    const real_t inv_Jx = 2.0 / g.hx;
    const real_t inv_Jy = 2.0 / g.hy;
    const real_t inv_Jz = 2.0 / g.hz;

    // 2-pt Gauss quadrature
    const real_t gp_pt = 1.0 / 1.7320508075688772;  // 1/sqrt(3)
    const real_t pts[2] = { -gp_pt, gp_pt };

    for (int gi = 0; gi < 2; ++gi)
    for (int gj = 0; gj < 2; ++gj)
    for (int gk = 0; gk < 2; ++gk) {
        real_t xi = pts[gi], eta = pts[gj], zeta = pts[gk];

        real_t xg = x0 + (xi   + 1.0) * 0.5 * g.hx;
        real_t yg = y0 + (eta  + 1.0) * 0.5 * g.hy;
        real_t zg = z0 + (zeta + 1.0) * 0.5 * g.hz;

        real_t k_g, rho_c_g;
        material_at(xg, yg, zg, k_g, rho_c_g);

        real_t N[8], dNref[8][3];
        shape_func_grad(xi, eta, zeta, N, dNref);

        real_t dNx[8], dNy[8], dNz[8];
        for (int i = 0; i < 8; ++i) {
            dNx[i] = inv_Jx * dNref[i][0];
            dNy[i] = inv_Jy * dNref[i][1];
            dNz[i] = inv_Jz * dNref[i][2];
        }

        // Gauss weight is 1*1*1 = 1
        real_t w = det_J;

        // ye[i] += w * sum_j (rho_c*N_i*N_j + dt*k*grad N_i . grad N_j) * xe[j]
        for (int i = 0; i < 8; ++i) {
            real_t Mx = 0.0, Kx = 0.0;
            for (int j = 0; j < 8; ++j) {
                Mx += rho_c_g * N[i] * N[j] * xe[j];
                Kx += k_g * (dNx[i]*dNx[j] + dNy[i]*dNy[j] + dNz[i]*dNz[j]) * xe[j];
            }
            ye[i] += w * (Mx + dt * Kx);
        }
    }

    for (int i = 0; i < 8; ++i) {
        atomicAdd(&y[gid[i]], ye[i]);
    }
}

// =============================================================================
// RHS construction: b = M*T_n + dt*F  where F_i = ∫ Q*N_i dΩ
// =============================================================================
__global__
void build_rhs_kernel(const real_t* __restrict__ Tn,
                      real_t* __restrict__ b,
                      Grid g, real_t dt) {
    long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= g.ne_local) return;

    int ix_e = e / (g.ny * g.nz);
    int iy_e = (e / g.nz) % g.ny;
    int iz_e = e % g.nz;

    long gid[8];
    element_nodes(ix_e, iy_e, iz_e, g, gid);

    real_t Tne[8], be[8] = {0};
    for (int i = 0; i < 8; ++i) Tne[i] = Tn[gid[i]];

    real_t x0, y0, z0;
    node_coords(ix_e, iy_e, iz_e, g, x0, y0, z0);

    const real_t det_J = g.hx * g.hy * g.hz / 8.0;
    const real_t gp_pt = 1.0 / 1.7320508075688772;
    const real_t pts[2] = { -gp_pt, gp_pt };

    for (int gi = 0; gi < 2; ++gi)
    for (int gj = 0; gj < 2; ++gj)
    for (int gk = 0; gk < 2; ++gk) {
        real_t xi = pts[gi], eta = pts[gj], zeta = pts[gk];

        real_t xg = x0 + (xi   + 1.0) * 0.5 * g.hx;
        real_t yg = y0 + (eta  + 1.0) * 0.5 * g.hy;
        real_t zg = z0 + (zeta + 1.0) * 0.5 * g.hz;

        real_t k_g, rho_c_g;
        material_at(xg, yg, zg, k_g, rho_c_g);
        real_t Q_g = source_at(xg, yg, zg);

        real_t N[8], dNref[8][3];
        shape_func_grad(xi, eta, zeta, N, dNref);

        real_t w = det_J;
        for (int i = 0; i < 8; ++i) {
            real_t MTn = 0.0;
            for (int j = 0; j < 8; ++j) {
                MTn += rho_c_g * N[i] * N[j] * Tne[j];
            }
            be[i] += w * (MTn + dt * Q_g * N[i]);
        }
    }

    for (int i = 0; i < 8; ++i) {
        atomicAdd(&b[gid[i]], be[i]);
    }
}

// =============================================================================
// Diagonal extraction for Jacobi preconditioner: diag(A)_ii = M_ii + dt*K_ii
// =============================================================================
__global__
void diag_kernel(real_t* __restrict__ d, Grid g, real_t dt) {
    long e = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= g.ne_local) return;

    int ix_e = e / (g.ny * g.nz);
    int iy_e = (e / g.nz) % g.ny;
    int iz_e = e % g.nz;

    long gid[8];
    element_nodes(ix_e, iy_e, iz_e, g, gid);

    real_t de[8] = {0};

    real_t x0, y0, z0;
    node_coords(ix_e, iy_e, iz_e, g, x0, y0, z0);

    const real_t det_J = g.hx * g.hy * g.hz / 8.0;
    const real_t inv_Jx = 2.0 / g.hx;
    const real_t inv_Jy = 2.0 / g.hy;
    const real_t inv_Jz = 2.0 / g.hz;
    const real_t gp_pt = 1.0 / 1.7320508075688772;
    const real_t pts[2] = { -gp_pt, gp_pt };

    for (int gi = 0; gi < 2; ++gi)
    for (int gj = 0; gj < 2; ++gj)
    for (int gk = 0; gk < 2; ++gk) {
        real_t xi = pts[gi], eta = pts[gj], zeta = pts[gk];

        real_t xg = x0 + (xi   + 1.0) * 0.5 * g.hx;
        real_t yg = y0 + (eta  + 1.0) * 0.5 * g.hy;
        real_t zg = z0 + (zeta + 1.0) * 0.5 * g.hz;

        real_t k_g, rho_c_g;
        material_at(xg, yg, zg, k_g, rho_c_g);

        real_t N[8], dNref[8][3];
        shape_func_grad(xi, eta, zeta, N, dNref);

        real_t w = det_J;
        for (int i = 0; i < 8; ++i) {
            real_t dNx = inv_Jx * dNref[i][0];
            real_t dNy = inv_Jy * dNref[i][1];
            real_t dNz = inv_Jz * dNref[i][2];
            de[i] += w * (rho_c_g * N[i] * N[i]
                        + dt * k_g * (dNx*dNx + dNy*dNy + dNz*dNz));
        }
    }

    for (int i = 0; i < 8; ++i) {
        atomicAdd(&d[gid[i]], de[i]);
    }
}

// =============================================================================
// Boundary condition kernels (homogeneous Dirichlet)
// =============================================================================
__global__
void apply_dirichlet_zero_kernel(real_t* v, Grid g) {
    long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= g.nnodes_local) return;
    int ix_local = n / (g.nny * g.nnz);
    int iy = (n / g.nnz) % g.nny;
    int iz = n % g.nnz;
    if (is_boundary(ix_local, iy, iz, g)) v[n] = 0.0;
}

__global__
void apply_dirichlet_one_kernel(real_t* v, Grid g) {
    long n = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= g.nnodes_local) return;
    int ix_local = n / (g.nny * g.nnz);
    int iy = (n / g.nnz) % g.nny;
    int iz = n % g.nnz;
    if (is_boundary(ix_local, iy, iz, g)) v[n] = 1.0;
}

// =============================================================================
// Vector op kernels for CG
// =============================================================================
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

// Block-reduction dot product over OWNED nodes only (no double counting at
// shared boundaries between MPI ranks).
__global__
void owned_dot_kernel(const real_t* x, const real_t* y, real_t* result, Grid g) {
    extern __shared__ real_t sdata[];
    int tid = threadIdx.x;
    long i = (long)blockIdx.x * blockDim.x + tid;

    real_t v = 0.0;
    if (i < g.nnodes_local) {
        int ix_local = i / (g.nny * g.nnz);
        if (is_owned(ix_local, g)) v = x[i] * y[i];
    }
    sdata[tid] = v;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}

// =============================================================================
// Halo exchange: face-pack/unpack kernels
// =============================================================================
__global__
void pack_face_kernel(const real_t* v, real_t* buf, Grid g, int x_local) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int face = g.nny * g.nnz;
    if (idx >= face) return;
    int iy = idx / g.nnz;
    int iz = idx % g.nnz;
    buf[idx] = v[lnid(x_local, iy, iz, g)];
}

__global__
void unpack_face_add_kernel(real_t* v, const real_t* buf, Grid g, int x_local) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int face = g.nny * g.nnz;
    if (idx >= face) return;
    int iy = idx / g.nnz;
    int iz = idx % g.nnz;
    v[lnid(x_local, iy, iz, g)] += buf[idx];
}

// =============================================================================
// HeatOperator: encapsulates the (M + dt*K) operator and provides the matvec
// and preconditioner callbacks for an external CG solver (e.g. GlaSs).
// =============================================================================
class HeatOperator {
public:
    Grid g;
    real_t dt;
    real_t *d_diag = nullptr;
    real_t *d_send_l = nullptr, *d_send_r = nullptr;
    real_t *d_recv_l = nullptr, *d_recv_r = nullptr;
    real_t *h_send_l = nullptr, *h_send_r = nullptr;
    real_t *h_recv_l = nullptr, *h_recv_r = nullptr;
    real_t *d_dot_result = nullptr;

    int  threads_node = 256;
    long blocks_node = 0;
    int  threads_elem = 128;   // lower thread count to ease register pressure
    long blocks_elem = 0;

    void setup(int nx, int ny, int nz,
               real_t Lx, real_t Ly, real_t Lz,
               real_t dt_in, MPI_Comm comm) {
        MPI_Comm_rank(comm, &g.rank);
        MPI_Comm_size(comm, &g.nprocs);
        g.nx = nx; g.ny = ny; g.nz = nz;
        g.hx = Lx / nx; g.hy = Ly / ny; g.hz = Lz / nz;

        if (nx % g.nprocs != 0) {
            if (g.rank == 0)
                fprintf(stderr,
                    "Error: nx (%d) must be divisible by nprocs (%d)\n",
                    nx, g.nprocs);
            MPI_Abort(comm, 1);
        }
        g.nx_local = nx / g.nprocs;
        g.ix_start = g.rank * g.nx_local;
        g.nnx_local = g.nx_local + 1;
        g.nny = ny + 1;
        g.nnz = nz + 1;
        g.nnodes_local = (long)g.nnx_local * g.nny * g.nnz;
        g.ne_local = (long)g.nx_local * g.ny * g.nz;

        dt = dt_in;

        CUDA_CHECK(cudaMalloc(&d_diag, g.nnodes_local * sizeof(real_t)));

        long face = (long)g.nny * g.nnz;
        CUDA_CHECK(cudaMalloc(&d_send_l, face * sizeof(real_t)));
        CUDA_CHECK(cudaMalloc(&d_send_r, face * sizeof(real_t)));
        CUDA_CHECK(cudaMalloc(&d_recv_l, face * sizeof(real_t)));
        CUDA_CHECK(cudaMalloc(&d_recv_r, face * sizeof(real_t)));
        h_send_l = (real_t*)malloc(face * sizeof(real_t));
        h_send_r = (real_t*)malloc(face * sizeof(real_t));
        h_recv_l = (real_t*)malloc(face * sizeof(real_t));
        h_recv_r = (real_t*)malloc(face * sizeof(real_t));
        CUDA_CHECK(cudaMalloc(&d_dot_result, sizeof(real_t)));

        blocks_node = (g.nnodes_local + threads_node - 1) / threads_node;
        blocks_elem = (g.ne_local + threads_elem - 1) / threads_elem;

        // Precompute diagonal once (matrix-free Jacobi preconditioner)
        CUDA_CHECK(cudaMemset(d_diag, 0, g.nnodes_local * sizeof(real_t)));
        diag_kernel<<<blocks_elem, threads_elem>>>(d_diag, g, dt);
        CUDA_CHECK_LAUNCH();
        halo_exchange_sum(d_diag);
        apply_dirichlet_one_kernel<<<blocks_node, threads_node>>>(d_diag, g);
        CUDA_CHECK_LAUNCH();
    }

    void cleanup() {
        if (d_diag)   cudaFree(d_diag);
        if (d_send_l) cudaFree(d_send_l);
        if (d_send_r) cudaFree(d_send_r);
        if (d_recv_l) cudaFree(d_recv_l);
        if (d_recv_r) cudaFree(d_recv_r);
        if (d_dot_result) cudaFree(d_dot_result);
        free(h_send_l); free(h_send_r);
        free(h_recv_l); free(h_recv_r);
    }

    // Sum contributions across MPI rank boundaries on shared faces.
    void halo_exchange_sum(real_t* d_v) {
        if (g.nprocs == 1) return;

        long face = (long)g.nny * g.nnz;
        int t = 256;
        long b = (face + t - 1) / t;

        if (g.rank > 0)
            pack_face_kernel<<<b, t>>>(d_v, d_send_l, g, 0);
        if (g.rank < g.nprocs - 1)
            pack_face_kernel<<<b, t>>>(d_v, d_send_r, g, g.nx_local);
        CUDA_CHECK_LAUNCH();

        if (g.rank > 0) {
            CUDA_CHECK(cudaMemcpy(h_send_l, d_send_l, face * sizeof(real_t),
                                   cudaMemcpyDeviceToHost));
        }
        if (g.rank < g.nprocs - 1) {
            CUDA_CHECK(cudaMemcpy(h_send_r, d_send_r, face * sizeof(real_t),
                                   cudaMemcpyDeviceToHost));
        }

        MPI_Request reqs[4];
        int nr = 0;
        if (g.rank > 0) {
            MPI_Irecv(h_recv_l, face, MPI_REAL_T, g.rank-1, 0,
                      MPI_COMM_WORLD, &reqs[nr++]);
            MPI_Isend(h_send_l, face, MPI_REAL_T, g.rank-1, 1,
                      MPI_COMM_WORLD, &reqs[nr++]);
        }
        if (g.rank < g.nprocs - 1) {
            MPI_Irecv(h_recv_r, face, MPI_REAL_T, g.rank+1, 1,
                      MPI_COMM_WORLD, &reqs[nr++]);
            MPI_Isend(h_send_r, face, MPI_REAL_T, g.rank+1, 0,
                      MPI_COMM_WORLD, &reqs[nr++]);
        }
        MPI_Waitall(nr, reqs, MPI_STATUSES_IGNORE);

        if (g.rank > 0) {
            CUDA_CHECK(cudaMemcpy(d_recv_l, h_recv_l, face * sizeof(real_t),
                                   cudaMemcpyHostToDevice));
            unpack_face_add_kernel<<<b, t>>>(d_v, d_recv_l, g, 0);
        }
        if (g.rank < g.nprocs - 1) {
            CUDA_CHECK(cudaMemcpy(d_recv_r, h_recv_r, face * sizeof(real_t),
                                   cudaMemcpyHostToDevice));
            unpack_face_add_kernel<<<b, t>>>(d_v, d_recv_r, g, g.nx_local);
        }
        CUDA_CHECK_LAUNCH();
    }

    // ---- Public API for an external CG (GlaSs) -----------------------------

    // y = A*x  with halo sum + Dirichlet projection
    void apply_matvec(const real_t* d_x, real_t* d_y) {
        zero_kernel<<<blocks_node, threads_node>>>(d_y, g.nnodes_local);
        CUDA_CHECK_LAUNCH();
        matvec_kernel<<<blocks_elem, threads_elem>>>(d_x, d_y, g, dt);
        CUDA_CHECK_LAUNCH();
        halo_exchange_sum(d_y);
        apply_dirichlet_zero_kernel<<<blocks_node, threads_node>>>(d_y, g);
        CUDA_CHECK_LAUNCH();
    }

    // z = D^{-1} r   (Jacobi preconditioner)
    void apply_preconditioner(const real_t* d_r, real_t* d_z) {
        jacobi_apply_kernel<<<blocks_node, threads_node>>>(
            d_r, d_diag, d_z, g.nnodes_local);
        CUDA_CHECK_LAUNCH();
    }

    // b = M*T_n + dt*F  with halo sum + Dirichlet projection
    void build_rhs(const real_t* d_Tn, real_t* d_b) {
        zero_kernel<<<blocks_node, threads_node>>>(d_b, g.nnodes_local);
        CUDA_CHECK_LAUNCH();
        build_rhs_kernel<<<blocks_elem, threads_elem>>>(d_Tn, d_b, g, dt);
        CUDA_CHECK_LAUNCH();
        halo_exchange_sum(d_b);
        apply_dirichlet_zero_kernel<<<blocks_node, threads_node>>>(d_b, g);
        CUDA_CHECK_LAUNCH();
    }

    // Global owned dot product (CG inner product)
    real_t dot(const real_t* d_x, const real_t* d_y) {
        zero_kernel<<<1, 1>>>(d_dot_result, 1);
        CUDA_CHECK_LAUNCH();
        owned_dot_kernel<<<blocks_node, threads_node,
                           threads_node*sizeof(real_t)>>>(
            d_x, d_y, d_dot_result, g);
        CUDA_CHECK_LAUNCH();
        real_t local;
        CUDA_CHECK(cudaMemcpy(&local, d_dot_result, sizeof(real_t),
                               cudaMemcpyDeviceToHost));
        real_t global;
        MPI_Allreduce(&local, &global, 1, MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);
        return global;
    }

    real_t norm2(const real_t* d_x) {
        return sqrt(dot(d_x, d_x));
    }
};

// =============================================================================
// Matrix-free Preconditioned CG
// (Replace this with a call into GlaSs::pcg() once the library is wired in.
//  GlaSs needs callbacks; HeatOperator::apply_matvec and apply_preconditioner
//  are the natural entry points.)
// =============================================================================
int pcg_solve(HeatOperator& op,
              const real_t* d_b, real_t* d_x,
              int max_iter, real_t tol,
              real_t* d_r, real_t* d_z, real_t* d_p, real_t* d_Ap) {
    long n = op.g.nnodes_local;
    int  tn = op.threads_node;
    long bn = op.blocks_node;

    op.apply_matvec(d_x, d_Ap);
    copy_kernel<<<bn, tn>>>(d_b, d_r, n); CUDA_CHECK_LAUNCH();
    axpy_kernel<<<bn, tn>>>(-1.0, d_Ap, d_r, n); CUDA_CHECK_LAUNCH();

    real_t bnorm = op.norm2(d_b);
    if (bnorm < 1e-30) bnorm = 1.0;
    real_t r0_norm = op.norm2(d_r);

    op.apply_preconditioner(d_r, d_z);
    copy_kernel<<<bn, tn>>>(d_z, d_p, n); CUDA_CHECK_LAUNCH();
    real_t rho = op.dot(d_r, d_z);

    int iter = 0;
    for (iter = 0; iter < max_iter; ++iter) {
        op.apply_matvec(d_p, d_Ap);
        real_t pAp = op.dot(d_p, d_Ap);
        if (fabs(pAp) < 1e-30) break;
        real_t alpha = rho / pAp;
        axpy_kernel<<<bn, tn>>>(alpha, d_p, d_x, n); CUDA_CHECK_LAUNCH();
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
// Driver
// =============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n_gpus = 0;
    cudaGetDeviceCount(&n_gpus);
    if (n_gpus == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaSetDevice(rank % n_gpus);

    int    N       = (argc > 1) ? atoi(argv[1]) : 32;
    real_t t_final = (argc > 2) ? atof(argv[2]) : 0.1;
    real_t dt      = (argc > 3) ? atof(argv[3]) : 0.001;

    int nx = N, ny = N, nz = N;
    real_t Lx = 1.0, Ly = 1.0, Lz = 1.0;
    int n_steps = (int)(t_final / dt + 0.5);

    if (rank == 0) {
        printf("=== Assembly-Free FEM Heat Solver (Backward Euler + PCG) ===\n");
        printf("Grid:           %d x %d x %d  (DOFs: %ld)\n",
               nx, ny, nz, (long)(nx+1)*(ny+1)*(nz+1));
        printf("MPI ranks:      %d  (1D decomp along x, %d cells per rank)\n",
               nprocs, nx/nprocs);
        printf("dt:             %.4e s\n", dt);
        printf("t_final:        %.4e s\n", t_final);
        printf("Time steps:     %d\n", n_steps);
        printf("Precision:      %s\n", sizeof(real_t)==8 ? "double" : "single");
        printf("============================================================\n");
    }

    HeatOperator op;
    op.setup(nx, ny, nz, Lx, Ly, Lz, dt, MPI_COMM_WORLD);

    long n = op.g.nnodes_local;
    real_t *d_T, *d_b, *d_r, *d_z, *d_p, *d_Ap;
    CUDA_CHECK(cudaMalloc(&d_T,  n * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_b,  n * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_r,  n * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_z,  n * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_p,  n * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_Ap, n * sizeof(real_t)));

    CUDA_CHECK(cudaMemset(d_T, 0, n * sizeof(real_t)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    int total_iters = 0;
    for (int step = 0; step < n_steps; ++step) {
        op.build_rhs(d_T, d_b);
        // Initial guess: previous T (already in d_T), CG updates in-place
        int it = pcg_solve(op, d_b, d_T, 500, 1e-8, d_r, d_z, d_p, d_Ap);
        total_iters += it;
        if (rank == 0 && (step % 10 == 0 || step == n_steps-1)) {
            real_t Tn = op.norm2(d_T);
            printf("  step %4d / %4d   CG iters %4d   ||T||=%.6e\n",
                   step+1, n_steps, it, Tn);
        }
    }

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);

    real_t Tnorm = op.norm2(d_T);
    if (rank == 0) {
        printf("\n=== Done ===\n");
        printf("Total time:           %.4f s\n", ms / 1000.0);
        printf("Avg time / step:      %.4f ms\n", ms / n_steps);
        printf("Avg CG iters / step:  %.2f\n",   (double)total_iters / n_steps);
        printf("Final ||T||_2:        %.6e\n",   Tnorm);
    }

    cudaFree(d_T); cudaFree(d_b);
    cudaFree(d_r); cudaFree(d_z);
    cudaFree(d_p); cudaFree(d_Ap);
    op.cleanup();
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    MPI_Finalize();
    return 0;
}
