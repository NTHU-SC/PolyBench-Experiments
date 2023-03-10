#! /bin/bash

module purge
spack load cuda@11.8.0%intel@2021.8.0
module load dpct
export PATH=/home/yi/sycl_workspace/llvm/build/bin:$PATH
export LD_LIBRARY_PATH=/home/yi/sycl_workspace/llvm/build/lib:$LD_LIBRARY_PATH

source sycl-bench_sizes

for exe in ../sycl-bench/build/*.sycl_gpu.exe; do
    echo "[*] Running $exe"
    BASE=$(basename $exe .sycl_gpu.exe)
    SIZENAME="SIZE_$BASE"
    BENCHNAME="NAME_$BASE"
    ./$exe --no-verification --size=${!SIZENAME} | tee ../sycl-bench/build/${!BENCHNAME}.sycl_gpu.txt
    sleep 3
done