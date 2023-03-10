#! /bin/bash

module purge
module purge
module load compiler dpct

source sycl-bench_sizes

for exe in ../sycl-bench/build/*.sycl_cpu.exe; do
    echo "[*] Running $exe"
    BASE=$(basename $exe .sycl_cpu.exe)
    SIZENAME="SIZE_$BASE"
    BENCHNAME="NAME_$BASE"
    ./$exe --no-verification --size=${!SIZENAME} | tee ../sycl-bench/build/${!BENCHNAME}.sycl_cpu.txt
    sleep 3
done