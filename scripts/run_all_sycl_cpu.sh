#! /bin/bash

source load_cpu.sh

source sycl-bench_sizes

REPEAT="${REPEAT:- 5}"

for exe in ../sycl-bench/build/*.sycl_cpu.exe; do
    echo "[*] Running $exe"
    BASE=$(basename $exe .sycl_cpu.exe)
    SIZENAME="SIZE_$BASE"
    BENCHNAME="NAME_$BASE"
    rm -f ../sycl-bench/build/${!BENCHNAME}.sycl_cpu.txt
    ./$exe --no-verification --size=${!SIZENAME} --num-runs="$REPEAT" \
     | tee -a ../sycl-bench/build/${!BENCHNAME}.sycl_cpu.txt
done