#!/usr/bin/env python3
"""
gen_field.py -- per-element material fields for heat_solver_het.cu.

Element ordering (MUST match the solver):  e = ix*ny*nz + iy*nz + iz,
ix slowest, iz fastest, for a cubic grid nx = ny = nz = n.

Models
  homogeneous  k = --k, rhoc = --rhoc everywhere.               (V0)
  quadrants    exact reproduction of heat_solver.cu material_at() evaluated
               at element centres. Regression bridge to MNHack7.  (V1)
               NOTE: the analytic model splits the domain at x,y,z = 0.5.
               For even n the split lies on element faces and the per-element
               model is exact; for odd n it is not, by construction.
  lognormal    k = exp(mu + sigma * G), G a Gaussian random field with
               Gaussian covariance and correlation length L (FFT synthesis).
               rhoc is either constant (--rhoc) or correlated with G via
               --rhoc-sigma. Optional --upscale r generates at (r*n)^3 and
               upscales each r^3 block: HARMONIC mean for k, ARITHMETIC mean
               for rhoc (Reuss bound for transverse layering; heat capacity is
               volume-additive).                                  (production)
  layered      k alternates between --k and --k2 in x-slabs of --period
               elements. Anisotropic reference case for the harmonic rule.

Every run writes <out>_k.bin, <out>_rhoc.bin (float64, C order) and
<out>_meta.json with the parameters, seed, kmin/kmax, and L/h.
"""
import argparse, json, sys
import numpy as np

def quadrants(n):
    h = 1.0 / n
    c = (np.arange(n) + 0.5) * h
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")     # ix slowest
    k = np.full(X.shape, 400.0); rhoc = np.full(X.shape, 0.5 * 2.5)
    m = (X < 0.5) & (Y < 0.5) & (Z < 0.5);   k[m] = 100.0; rhoc[m] = 1.0 * 1.0
    m = (X >= 0.5) & (Y < 0.5) & (Z < 0.5);  k[m] = 200.0; rhoc[m] = 0.8 * 1.5
    m = (X < 0.5) & (Y >= 0.5) & (Z < 0.5);  k[m] = 300.0; rhoc[m] = 0.6 * 2.0
    return k, rhoc

def gaussian_field(n, L, seed):
    """Unit-variance Gaussian random field on an n^3 periodic grid, Gaussian
    covariance C(r) = exp(-r^2 / L^2), domain [0,1]^3."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n, n))
    kx = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2
    S = np.exp(-k2 * L**2 / 4.0)                      # spectrum of exp(-r^2/L^2)
    f = np.fft.ifftn(np.fft.fftn(white) * np.sqrt(S)).real
    f -= f.mean(); f /= f.std()
    return f

def upscale(fine, r, mode):
    n = fine.shape[0] // r
    blocks = fine.reshape(n, r, n, r, n, r).transpose(0, 2, 4, 1, 3, 5).reshape(n, n, n, -1)
    if mode == "arithmetic":
        return blocks.mean(-1)
    if mode == "harmonic":
        return 1.0 / (1.0 / blocks).mean(-1)
    raise ValueError(mode)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--model", choices=["homogeneous", "quadrants", "lognormal", "layered"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=float, default=1.0)
    p.add_argument("--k2", type=float, default=100.0, help="layered: second conductivity")
    p.add_argument("--period", type=int, default=4, help="layered: slab thickness in elements")
    p.add_argument("--rhoc", type=float, default=1.0)
    p.add_argument("--mu", type=float, default=0.0, help="lognormal: mean of ln k")
    p.add_argument("--sigma", type=float, default=1.0, help="lognormal: std of ln k")
    p.add_argument("--rhoc-sigma", type=float, default=0.0, help="lognormal: std of ln rhoc (same field)")
    p.add_argument("--L", type=float, default=0.125, help="lognormal: correlation length (domain = 1)")
    p.add_argument("--upscale", type=int, default=1, help="lognormal: generate at (r*n)^3, upscale r^3 blocks")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    n = a.n
    meta = vars(a).copy()

    if a.model == "homogeneous":
        k = np.full((n, n, n), a.k); rhoc = np.full((n, n, n), a.rhoc)
    elif a.model == "quadrants":
        if n % 2:
            print(f"WARNING: n={n} is odd; the quadrants model cannot match heat_solver.cu "
                  f"(interfaces at 0.5 fall inside elements). Fine as a negative control.", file=sys.stderr)
        k, rhoc = quadrants(n)
    elif a.model == "layered":
        ix = np.arange(n)[:, None, None]
        k = np.where((ix // a.period) % 2 == 0, a.k, a.k2) * np.ones((n, n, n))
        rhoc = np.full((n, n, n), a.rhoc)
    else:  # lognormal
        nf = n * a.upscale
        G = gaussian_field(nf, a.L, a.seed)
        kf = np.exp(a.mu + a.sigma * G)
        rf = np.exp(np.log(a.rhoc) + a.rhoc_sigma * G)
        if a.upscale > 1:
            k = upscale(kf, a.upscale, "harmonic")
            rhoc = upscale(rf, a.upscale, "arithmetic")
            meta["upscale_rule"] = {"k": "harmonic", "rhoc": "arithmetic"}
        else:
            k, rhoc = kf, rf
        meta["L_over_h"] = a.L * n

    k = np.ascontiguousarray(k, dtype=np.float64); rhoc = np.ascontiguousarray(rhoc, dtype=np.float64)
    assert k.shape == (n, n, n)
    k.ravel(order="C").tofile(f"{a.out}_k.bin")
    rhoc.ravel(order="C").tofile(f"{a.out}_rhoc.bin")
    meta.update(dict(kmin=float(k.min()), kmax=float(k.max()), contrast=float(k.max() / k.min()),
                     rhocmin=float(rhoc.min()), rhocmax=float(rhoc.max()), n_elements=int(n**3)))
    with open(f"{a.out}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{a.model}: n={n}  k in [{k.min():.4g}, {k.max():.4g}] (contrast {k.max()/k.min():.3g})  "
          f"rhoc in [{rhoc.min():.4g}, {rhoc.max():.4g}]  -> {a.out}_{{k,rhoc}}.bin")

if __name__ == "__main__":
    main()
