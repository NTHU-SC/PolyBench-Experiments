#! /bin/bash

source load_gpu.sh

source sycl-bench_sizes

REPEAT="${REPEAT:- 5}"

for exe in ../sycl-bench/build/*.sycl_gpu.exe; do
    echo "[*] Running $exe"
    BASE=$(basename $exe .sycl_gpu.exe)
    SIZENAME="SIZE_$BASE"
    BENCHNAME="NAME_$BASE"
    rm -f ../sycl-bench/build/${!BENCHNAME}.sycl_gpu.txt
    ./$exe --no-verification --size=${!SIZENAME} --num-runs="$REPEAT" \
        | tee ../sycl-bench/build/${!BENCHNAME}.sycl_gpu.txt
done