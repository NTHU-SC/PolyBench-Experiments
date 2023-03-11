# QCT oneAPI Experiments - Polybench

- Variants
    - CUDA @ GPU `CUDA`
    - CUDA -> SYCL using DPCT @ CPU `DPCT_CPU`
    - CUDA -> SYCL using DPCT @ GPU `DPCT_GPU`
    - sycl-bench SYCL @ CPU `SYCL_CPU`
    - sycl-bench SYCL @ GPU `SYCL_GPU`

## Running experiments

### CUDA @ GPU `CUDA`
> Requires CUDA toolkit

1. `make_all_cuda.sh`
    - Build all CUDA benchmarks using `nvcc`
    > Makefile is at `../CUDA/utilities/common.mk`, target `cuda`
2. `run_all_cuda.sh`
    - Run all CUDA benchmarks
    > Outputs are written to `output.cuda.txt` in the testcase folders

### CUDA -> SYCL using DPCT @ CPU `DPCT_CPU`
> Requires oneAPI DPC++ compiler & DPCT

1. `dpct_convert_all.sh`
    - Generates SYCL code from CUDA code using DPCT
    > Converted code are already in the repo, so this step is not required.
2. `make_all_dpct_cpu.sh`
    - Build all DPCT converted SYCL code targeting Intel CPU with `icpx`

    > Makefile is at `../CUDA/utilities/common.mk`, target `dpct_cpu`
3. `run_all_dpct_cpu.sh`
    - Run all DPCT benchmarks built with CPU target
    > Run 5 times by default. Override using `REPEAT=x ./run_all_dpct_cpu.sh`

    > Outputs are written to `output.dpct_cpu.txt` in the testcase folders

### CUDA -> SYCL using DPCT @ GPU `DPCT_GPU`
> Requires intel/llvm built with CUDA & DPCT

1. `dpct_convert_all.sh`
    - Generates SYCL code from CUDA code using DPCT
    > Converted code are already in the repo, so this step is not required.
2. `make_all_dpct_gpu.sh`
    - Build all DPCT converted SYCL code targeting NVIDIA GPU with intel/llvm's `clang++`
    > Makefile is at `../CUDA/utilities/common.mk`, target `dpct_cpu`

    > Note: change `--cuda-gpu-arch=` if you are not using Volta architecture GPU
3. `run_all_dpct_gpu.sh`
    - Run all DPCT benchmarks built with GPU target
    > Run 5 times by default. Override using `REPEAT=x ./run_all_dpct_cpu.sh`

    > Outputs are written to `output.dpct_gpu.txt` in the testcase folders

### sycl-bench SYCL @ CPU `SYCL_CPU`
> Requires oneAPI DPC++ compiler

> Make sure the `sycl-bench` submodule is cloned.
> If not, run `git submodule update --init`

1. `make_all_sycl_cpu.sh`
    - Build all SYCL-Bench polybench codes targeting Intel CPU with `icpx`
2. `run_all_sycl_cpu.sh`
    - Run all SYCL polybench benchmarks built with CPU target
    > Run 5 times by default. Override using `REPEAT=x ./run_all_dpct_cpu.sh`
    > Outputs are written to `output.sycl_cpu.txt` in `sycl-bench/build`

### sycl-bench SYCL @ CPU `SYCL_GPU`
> Requires oneAPI DPC++ compiler

> Make sure the `sycl-bench` submodule is cloned.
> If not, run `git submodule update --init`

1. `make_all_sycl_gpu.sh`
    - Build all SYCL-Bench polybench codes targeting NVIDIA GPU with intel/llvm's `clang++`
    > Note: change `--cuda-gpu-arch=` if you are not using Volta architecture GPU
2. `run_all_sycl_gpu.sh`
    - Run all SYCL polybench benchmarks built with GPU target
    > Run 5 times by default. Override using `REPEAT=x ./run_all_dpct_cpu.sh`
    > Outputs are written to `output.sycl_gpu.txt` in `sycl-bench/build`

## Generating reports

> Tip: to remove old experiments, use `git clean -fxdn`, and remove the `-n` option to perform deletion

> Recommendation: rename `results(_sycl-bench).csv` to `results(_sycl-bench)-[host].csv` before commiting

### `CUDA`, `DPCT_CPU` and `DPCT_GPU`

- Run `python3 parse_log.py`
    - Outputs to `results.csv`

### `SYCL_CPU` and `SYCL_GPU`

- Run `python3 parse_log_sycl-bench.py`
    - Outputs to `results_sycl-bench.csv`

## Changing dataset sizes

1. Edit `../CUDA/utilities/common.mk` and add `-D<SIZE>_DATASET` to compile args
    - Dataset sizes are: `MINI_DATASET`, `SMALL_DATASET`, `STANDARD_DATASET`(default), `LARGE_DATASET`, `EXTRALARGE_DATASET`
2. Edit `gen_sizes.py` and set the `DATASET` variable
3. Run `gen_sizes.py` to export the sizes of the testcases to `sycl-bench`

