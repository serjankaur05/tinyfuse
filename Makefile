CC      ?= gcc
MARCH   ?= native
CFLAGS  ?= -O3 -march=$(MARCH) -fopenmp -Wall -Wextra
BATCH   ?= 256
REPEATS ?= 50

.PHONY: all build test bench profile clean

all: build test

build: build/bench build/libtfkernels.so build/kprof

build/bench: bench/bench.c kernels/kernels.c kernels/kernels.h | build-dir
	$(CC) $(CFLAGS) -o $@ bench/bench.c kernels/kernels.c -lm

build/libtfkernels.so: kernels/kernels.c kernels/kernels.h | build-dir
	$(CC) $(CFLAGS) -fPIC -shared -o $@ kernels/kernels.c -lm

build/kprof: $(wildcard profiler/*.go) profiler/go.mod | build-dir
	cd profiler && go build -o ../build/kprof .

build-dir:
	@mkdir -p build results

# C kernel correctness is checked inside bench (exits non-zero on mismatch).
test: build
	./build/bench 37 3 build/smoke_trace.json > /dev/null
	TINYFUSE_MARCH=$(MARCH) python3 -m unittest discover -s tests -v
	cd profiler && go vet ./... && go test ./...

bench: build
	./build/bench $(BATCH) $(REPEATS) results/kernel_trace.json
	TINYFUSE_MARCH=$(MARCH) python3 bench/bench_model.py --batch $(BATCH) --repeats $(REPEATS)

profile: bench
	./build/kprof -trace results/kernel_trace.json -compare naive,simd_fused_omp -html results/kernel_report.html
	./build/kprof -trace results/model_trace.json -html results/model_report.html

clean:
	rm -rf build
