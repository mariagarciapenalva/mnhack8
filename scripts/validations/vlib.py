"""
vlib.py -- shared helpers for scripts/validations/v*.py

Conventions
  - Every validation writes into  <outroot>/<test-name>/  : raw solver outputs,
    a report.json with PASS/FAIL and the numbers behind it, and figures.
  - Binaries are never modified by the scripts; capabilities are probed from
    the usage text so the legacy (pre-v2) binary degrades to "UNSUPPORTED"
    instead of crashing.
"""
import csv, json, os, re, subprocess, sys, time
import numpy as np

# ----------------------------------------------------------------------------
def run(cmd, cwd=None, log=None, np_=1, env=None):
    """Run a command (list). np_>1 prefixes mpirun. Returns stdout; raises on error."""
    if np_ > 1:
        cmd = ["mpirun", "--oversubscribe", "-np", str(np_)] + cmd
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if log:
        with open(log, "a") as f:
            f.write(f"$ {' '.join(cmd)}\n[{dt:.1f}s, rc={r.returncode}]\n{r.stdout}\n{r.stderr}\n")
    if r.returncode != 0:
        raise RuntimeError(f"command failed (rc={r.returncode}): {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout

def capabilities(binary):
    """Probe the usage text. Returns set of supported flags."""
    r = subprocess.run([binary], capture_output=True, text=True)
    txt = r.stdout + r.stderr
    return {f for f in ["--verify", "--kernel", "--scatter", "--fi-rank", "--thr", "--reproject-bc"] if f in txt}

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def fnum(x):
    try: return float(x)
    except Exception: return float("nan")

def write_report(outdir, name, passed, details):
    os.makedirs(outdir, exist_ok=True)
    rep = {"test": name, "pass": bool(passed), "details": details}
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print(f"[{name}] {'PASS' if passed else 'FAIL'}")
    return rep

def rel_l2(a, b):
    b = np.asarray(b); a = np.asarray(a)
    den = np.linalg.norm(b)
    return np.linalg.norm(a - b) / (den if den > 0 else 1.0)

# ----------------------------------------------------------------------------
def assemble_field(outdir, tag, N, nprocs):
    """Reassemble the global (N+1)^3 node field from T_final_<tag>_rank*.bin.
    Rank r stores nodes ix_local in [0, nx_local]; the shared face is taken from
    the owner (rank r owns [0,nx_local), the last rank also owns nx_local)."""
    nx_local = N // nprocs
    nn = N + 1
    T = np.zeros((nn, nn, nn))
    for r in range(nprocs):
        a = np.fromfile(os.path.join(outdir, f"T_final_{tag}_rank{r}.bin")).reshape(nx_local + 1, nn, nn)
        n_own = nx_local + (1 if r == nprocs - 1 else 0)
        T[r * nx_local : r * nx_local + n_own] = a[:n_own]
    return T

def node_index(ix, iy, iz, N):
    nn = N + 1
    return ix * nn * nn + iy * nn + iz

def elem_index(ix, iy, iz, N):
    return ix * N * N + iy * N + iz

def gen_field(gen, n, model, out, **kw):
    cmd = [sys.executable, gen, "--n", str(n), "--model", model, "--out", out]
    for k, v in kw.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    return run(cmd)

def parse_final_norm(stdout):
    """From heat_solver.cu (MNHack7): 'Final ||T||_2:        1.234567e+00'"""
    m = re.search(r"Final \|\|T\|\|_2:\s*([0-9.eE+-]+)", stdout)
    return float(m.group(1)) if m else float("nan")

def parse_step_norms(stdout):
    """From heat_solver.cu: 'step   10 /  100   CG iters   12   ||T||=1.234567e+00'"""
    out = {}
    for m in re.finditer(r"step\s+(\d+)\s*/\s*\d+\s+CG iters\s+(\d+)\s+\|\|T\|\|=([0-9.eE+-]+)", stdout):
        out[int(m.group(1))] = (int(m.group(2)), float(m.group(3)))
    return out
