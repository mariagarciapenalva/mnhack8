#!/usr/bin/env python3
"""
aggregate.py -- Layer 5 statistics from ensemble.csv. Never re-runs anything.

Produces, in <out>/:
  sensitivity.csv     P(class) per (level, bit_class) with Wilson 95% CIs
  sensitivity.png     heatmap of P(not benign) with n per cell
  detection.csv       per (level, bit_class): fraction caught by D1/D2/D3/any
  contrast.csv        E_max and P(not benign) vs sigma (material contrast)
  healing.png         E_max vs E_final scatter, coloured by class

Wilson interval: the standard choice for binomial proportions with small n;
Wald intervals go negative / exceed 1 exactly where your data is sparsest.
"""
import argparse, csv, math, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validations"))
import vlib

LEVELS = {"1": "G", "2": "R", "3": "H"}
CLASSES = ["mant_lo", "mant_hi", "exp", "sign"]

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [r for r in vlib.read_csv(a.csv) if r.get("status") == "ok"]
    print(f"{len(rows)} successful runs")

    # ---- sensitivity map ----
    cells = defaultdict(list)
    for r in rows: cells[(LEVELS[r["fi_level"]], r["bit_class"])].append(r)
    with open(os.path.join(a.out, "sensitivity.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["level", "bit_class", "n", "p_benign", "p_transient", "p_persistent", "p_detected",
                                       "p_not_benign", "ci_lo", "ci_hi", "E_max_median", "E_max_p90"])
        P = {}
        for lv in ["G", "R", "H"]:
            for bc in CLASSES:
                rs = cells.get((lv, bc), []); n = len(rs)
                if n == 0: continue
                cnt = {c: sum(r["class"] == c for r in rs) for c in ["benign", "transient", "persistent", "detected"]}
                nb = n - cnt["benign"]; p, lo, hi = wilson(nb, n)
                em = np.array([vlib.fnum(r["E_fault_max"]) for r in rs]); em = em[np.isfinite(em)]
                w.writerow([lv, bc, n] + [f"{cnt[c]/n:.3f}" for c in ["benign","transient","persistent","detected"]] +
                           [f"{p:.3f}", f"{lo:.3f}", f"{hi:.3f}",
                            f"{np.median(em):.3e}" if em.size else "", f"{np.percentile(em,90):.3e}" if em.size else ""])
                P[(lv, bc)] = (p, lo, hi, n)

    # ---- detection coverage ----
    with open(os.path.join(a.out, "detection.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["level", "bit_class", "n_not_benign", "frac_D1", "frac_D2", "frac_D3", "frac_any", "frac_cg_iters"])
        for lv in ["G", "R", "H"]:
            for bc in CLASSES:
                rs = [r for r in cells.get((lv, bc), []) if r["class"] != "benign" and r.get("detect") == "1"]
                n = len(rs)
                if n == 0: continue
                d1 = sum(int(r["d1_first"]) >= 0 for r in rs); d2 = sum(int(r["d2_first"]) >= 0 for r in rs)
                d3 = sum(int(r["d3_first"]) >= 0 for r in rs)
                anyd = sum((int(r["d1_first"]) >= 0) or (int(r["d2_first"]) >= 0) or (int(r["d3_first"]) >= 0) or r["class"] == "detected" for r in rs)
                cg = sum(int(r["iters_delta_max"]) > 0 for r in rs)
                w.writerow([lv, bc, n, f"{d1/n:.2f}", f"{d2/n:.2f}", f"{d3/n:.2f}", f"{anyd/n:.2f}", f"{cg/n:.2f}"])

    # ---- contrast dependence ----
    bys = defaultdict(list)
    for r in rows: bys[float(r["sigma"])].append(r)
    with open(os.path.join(a.out, "contrast.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sigma", "n", "contrast_kmax_kmin", "mean_cg_iters", "p_not_benign", "ci_lo", "ci_hi", "E_max_median"])
        for s in sorted(bys):
            rs = bys[s]; n = len(rs); nb = sum(r["class"] != "benign" for r in rs); p, lo, hi = wilson(nb, n)
            em = np.array([vlib.fnum(r["E_fault_max"]) for r in rs]); em = em[np.isfinite(em)]
            w.writerow([s, n, f"{np.mean([vlib.fnum(r['kmax'])/vlib.fnum(r['kmin']) for r in rs]):.3g}",
                        f"{np.mean([vlib.fnum(r['mean_cg_iters_A']) for r in rs]):.1f}",
                        f"{p:.3f}", f"{lo:.3f}", f"{hi:.3f}", f"{np.median(em):.3e}" if em.size else ""])

    # ---- figures ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.6))
        M = np.full((3, 4), np.nan)
        for i, lv in enumerate(["G", "R", "H"]):
            for j, bc in enumerate(CLASSES):
                if (lv, bc) in P: M[i, j] = P[(lv, bc)][0]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(3)); ax.set_yticklabels(["G state", "R accum", "H halo"])
        for i, lv in enumerate(["G", "R", "H"]):
            for j, bc in enumerate(CLASSES):
                if (lv, bc) in P:
                    p, lo, hi, n = P[(lv, bc)]
                    ax.text(j, i, f"{p:.2f}\n[{lo:.2f},{hi:.2f}]\nn={n}", ha="center", va="center", fontsize=7,
                            color="white" if p < 0.6 else "black")
        fig.colorbar(im, label="P(not benign)"); ax.set_title("SEU sensitivity map (Wilson 95% CI)")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "sensitivity.png"), dpi=150)
        fig, ax = plt.subplots(figsize=(5, 4))
        col = {"benign": "tab:green", "transient": "tab:blue", "persistent": "tab:red", "detected": "k"}
        for c in col:
            rs = [r for r in rows if r["class"] == c]
            if rs: ax.loglog([max(vlib.fnum(r["E_fault_max"]), 1e-18) for r in rs],
                             [max(vlib.fnum(r["E_fault_final"]), 1e-18) for r in rs], ".", label=c, color=col[c], alpha=0.6)
        ax.plot([1e-18, 1e12], [1e-18, 1e12], "k:", lw=0.5); ax.set_xlabel("E_max"); ax.set_ylabel("E_final"); ax.legend()
        ax.set_title("healing: below the diagonal = diffusion removed error"); fig.tight_layout()
        fig.savefig(os.path.join(a.out, "healing.png"), dpi=150)
        fig, ax1 = plt.subplots(figsize=(5, 3.4)); ax2 = ax1.twinx()
        sg = sorted(bys); pnb = []; cg = []; lo = []; hi = []
        for s_ in sg:
            rs = bys[s_]; n = len(rs); nb = sum(r["class"] != "benign" for r in rs); p, l, h = wilson(nb, n)
            pnb.append(p); lo.append(p - l); hi.append(h - p); cg.append(np.mean([vlib.fnum(r["mean_cg_iters_A"]) for r in rs]))
        ax1.errorbar(sg, pnb, yerr=[lo, hi], fmt="o-", color="tab:red", label="P(not benign)"); ax1.set_ylabel("P(not benign)", color="tab:red")
        ax2.plot(sg, cg, "s--", color="tab:blue", label="mean CG iters"); ax2.set_ylabel("mean CG iterations", color="tab:blue")
        ax1.set_xlabel("sigma of ln k (material contrast)"); ax1.set_title("contrast dependence"); fig.tight_layout()
        fig.savefig(os.path.join(a.out, "contrast.png"), dpi=150)
    except Exception as ex:
        print("plot skipped:", ex)
    print("wrote", os.listdir(a.out))

if __name__ == "__main__":
    main()
