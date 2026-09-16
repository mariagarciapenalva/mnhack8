# Assembly-Free FEM Heat Solver: SEU Fault-Tolerance Characterization.
## Build + validation guide
## 1. Overview

The project contains two solver implementations that are validated against
each other:

- `src/heat_solver.cu` — host/CPU reference implementation
- `src/heat_solver_het.cu` — CUDA (GPU) implementation

The validation suite (`scripts/validations/run_validations.sh`) runs stages
V0–V8, comparing the two implementations under normal and fault-injected
conditions, and reports PASS/FAIL per case with accompanying plots.

## 2. Requirements

### Local machine
- C++ compiler (g++ or equivalent) for the host build
- NVIDIA CUDA Toolkit (`nvcc`) and a CUDA-capable GPU for the GPU build
  - Target architecture used in this project: `sm_86`. check GPU with `nvidia-smi` or the CUDA `deviceQuery`
    sample)
- Python 3, with the packages in `requirements-quantum.txt` (Qiskit/Aer for
  the V7 quantum-substrate stage, plus NumPy for V8 and the plotting stages)
- Git
### MN5


## 3. Instructions
\`\`\`bash
git clone <REPO_URL> mnhack8
cd mnhack8
pip install -r requirements-quantum.txt
scripts/validations/run_validations.sh v7 src/heat_solver_het.cu src/heat_solver.cu sm_86 2
\`\`\`
Arguments, in order:
1. Output directory / run label under `tests/baseline_validation/`
2. Path to the CUDA solver source
3. Path to the host solver source
4. Target CUDA architecture
5. Number of processes/ranks

This single command builds both binaries and runs every validation stage.

## 6. Review results

\`\`\`bash
cat tests/baseline_validation/<run_label>/summary.txt
\`\`\`

Prints PASS/FAIL per test case. Each stage also produces a plot for visual
inspection, e.g.:

| Stage | Plot |
|---|---|
| V0 | `v0_convergence.png` |
| V2 | `v2_noise_floor.png` |
| V3 | `v3_injection.png` |
| V6 | `v6_coverage.png` (+ `coverage.csv`) |
| V7 | `v7_tier2.png` |
| V8 | `v8_hybrid.png` |

For numeric detail behind any case, inspect the corresponding `report.json`.Arguments, in order:
1. Output directory / run label under `tests/baseline_validation/`
2. Path to the CUDA solver source
3. Path to the host solver source
4. Target CUDA architecture
5. Number of processes/ranks

This single command builds both binaries and runs every validation stage.

## 6. Review results

\`\`\`bash
cat tests/baseline_validation/<run_label>/summary.txt
\`\`\`

Prints PASS/FAIL per test case. Each stage also produces a plot for visual
inspection, e.g.:

| Stage | Plot |
|---|---|
| V0 | `v0_convergence.png` |
| V2 | `v2_noise_floor.png` |
| V3 | `v3_injection.png` |
| V6 | `v6_coverage.png` (+ `coverage.csv`) |
| V7 | `v7_tier2.png` |
| V8 | `v8_hybrid.png` |

For numeric detail behind any case, inspect the corresponding `report.json`.

