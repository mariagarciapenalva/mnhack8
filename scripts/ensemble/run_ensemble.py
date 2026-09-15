#!/usr/bin/env python3
"""
run_ensemble.py -- Layer 5 driver. Runs the fault-injection grid and appends
one row per run to <out>/ensemble.csv. Resumable: rows already present are
skipped, so a killed job (or an MN5 walltime limit) just gets re-launched.

GRID (defaults chosen for a local RTX 3060 pilot; scale N/steps/reps on MN5)
  level      G, R, H            (H only if --mpi >= 2)
  bit class  mantissa-lo (bits 0-29), mantissa-hi (30-51), exponent (52-62), sign (63)
             sampled uniformly INSIDE each class, --reps draws per class
  time       injection step = 10%, 50%, 90% of the run
  target     --targets random interior nodes/elements per field (seeded)
  field      --sigmas x --seeds lognormal realizations (L/h fixed by --L)
Every row records the full summary line the solver already writes (E_max,
E_int, class, detectors, CG iterations, contrast, timings), so aggregate.py
never re-runs anything.

Run locally:
  python3 scripts/ensemble/run_ensemble.py --bin build/heat_solver_het --out results/ensemble_N32
On MN5 (see docs/running_elsewhere.md) the same script is the payload of the
sbatch job; use --shard i/n to split the grid across job-array tasks.
"""
import argparse, csv, itertools, os, random, subprocess, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validations"))
import vlib

CLASSES = {"mant_lo": range(0, 30), "mant_hi": range(30, 52), "exp": range(52, 63), "sign": [63]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--gen", default="scripts/gen_field.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=32); ap.add_argument("--t", type=float, default=0.02)
    ap.add_argument("--dt", type=float, default=5e-4)
    ap.add_argument("--sigmas", default="0.5,1.0,2.0"); ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--L", type=float, default=0.125); ap.add_argument("--upscale", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3, help="bit draws per class")
    ap.add_argument("--targets", type=int, default=3, help="interior targets per field")
    ap.add_argument("--levels", default="1,2,3"); ap.add_argument("--mpi", type=int, default=2)
    ap.add_argument("--kernel", default="fast"); ap.add_argument("--scatter", default="colored")
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: run only rows with index %% n == i")
    ap.add_argument("--rng", type=int, default=12345)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fields = os.path.join(a.out, "fields"); os.makedirs(fields, exist_ok=True)
    n_steps = int(round(a.t / a.dt)); steps = sorted({max(1, int(f * n_steps)) for f in (0.1, 0.5, 0.9)})
    levels = [int(x) for x in a.levels.split(",") if not (int(x) == 3 and a.mpi < 2)]
    sigmas = [float(x) for x in a.sigmas.split(",")]; seeds = [int(x) for x in a.seeds.split(",")]
    rng = random.Random(a.rng)
    shard_i, shard_n = (int(x) for x in a.shard.split("/"))

    # ---- build the grid deterministically ----
    grid = []
    for sigma, seed in itertools.product(sigmas, seeds):
        tag_f = f"ln{a.N}_s{sigma}_seed{seed}"
        fp = os.path.join(fields, tag_f)
        # interior targets: nodes for G, elements for R, face entries for H
        nodes = [vlib.node_index(rng.randint(2, a.N-2), rng.randint(2, a.N-2), rng.randint(2, a.N-2), a.N) for _ in range(a.targets)]
        elems = [vlib.elem_index(rng.randint(2, a.N-3), rng.randint(2, a.N-3), rng.randint(2, a.N-3), a.N) for _ in range(a.targets)]
        faces = [rng.randint(2, a.N-2) * (a.N+1) + rng.randint(2, a.N-2) for _ in range(a.targets)]
        for level in levels:
            tg = {1: nodes, 2: elems, 3: faces}[level]
            for cls, bits in CLASSES.items():
                for _ in range(a.reps):
                    bit = rng.choice(list(bits))
                    for S in steps:
                        for ti, target in enumerate(tg):
                            grid.append(dict(field=tag_f, fp=fp, sigma=sigma, seed=seed, level=level,
                                             bit_class=cls, bit=bit, step=S, target=target, tidx=ti))
    print(f"grid: {len(grid)} runs ({len(sigmas)} sigmas x {len(seeds)} seeds x {len(levels)} levels x "
          f"{len(CLASSES)} classes x {a.reps} reps x {len(steps)} steps x {a.targets} targets)")

    # ---- resume ----
    csv_path = os.path.join(a.out, "ensemble.csv")
    done = set()
    if os.path.exists(csv_path):
        for r in vlib.read_csv(csv_path):
            done.add(r["run_id"])
    header_written = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    t0 = time.time(); n_run = 0
    for idx, g in enumerate(grid):
        if idx % shard_n != shard_i: continue
        run_id = f"{g['field']}_L{g['level']}_{g['bit_class']}_b{g['bit']}_S{g['step']}_t{g['tidx']}"
        if run_id in done: continue
        if not os.path.exists(g["fp"] + "_k.bin"):
            vlib.gen_field(a.gen, a.N, "lognormal", g["fp"], sigma=g["sigma"], L=a.L, upscale=a.upscale, seed=g["seed"])
        d = os.path.join(a.out, "runs", run_id); os.makedirs(d, exist_ok=True)
        cmd = [a.bin, str(a.N), str(a.t), str(a.dt), g["fp"], d, "--tag", "e", "--snap", "5",
               "--kernel", a.kernel, "--scatter", a.scatter,
               "--fi", str(g["level"]), str(g["step"]), str(g["target"]), str(g["bit"])]
        if a.detect: cmd.append("--detect")
        try:
            vlib.run(cmd, log=os.path.join(a.out, "run.log"), np_=(a.mpi if g["level"] == 3 else 1))
            row = vlib.read_csv(os.path.join(d, "fi_summary_e.csv"))[0]
            row.update(run_id=run_id, field=g["field"], sigma=g["sigma"], seed=g["seed"], bit_class=g["bit_class"], status="ok")
        except Exception as ex:
            row = dict(run_id=run_id, field=g["field"], sigma=g["sigma"], seed=g["seed"], bit_class=g["bit_class"],
                       fi_level=g["level"], fi_bit=g["bit"], fi_step=g["step"], status=f"error: {str(ex)[:120]}")
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(set(row.keys()) | {"run_id","field","sigma","seed","bit_class","status"}), extrasaction="ignore")
            if not header_written: w.writeheader(); header_written = True
            w.writerow(row)
        # keep only the summary/trace; the field dumps are large
        for fn in os.listdir(d):
            if fn.startswith("T_final_"): os.remove(os.path.join(d, fn))
        n_run += 1
        if n_run % 10 == 0:
            el = time.time() - t0
            print(f"[{n_run} runs, {el/60:.1f} min, {el/n_run:.1f} s/run]", flush=True)
    print(f"done: {n_run} new runs -> {csv_path}")

if __name__ == "__main__":
    main()
