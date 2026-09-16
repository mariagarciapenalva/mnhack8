ARCH ?= sm_86        # RTX 3060: sm_86  |  T4 (Google Colab attempt): sm_75  |  H100 (MN5): sm_90
NVCC  = nvcc -O3 -arch=$(ARCH) -ccbin=mpicxx -lineinfo

all: build/heat_solver_het build/heat_solver_ref

build/heat_solver_het: src/heat_solver_het.cu
	mkdir -p build && $(NVCC) $< -o $@

build/heat_solver_ref: src/heat_solver.cu
	mkdir -p build && $(NVCC) $< -o $@

validate:
	scripts/validations/run_validations.sh v2 src/heat_solver_het.cu src/heat_solver.cu $(ARCH)

clean:
	rm -rf build
