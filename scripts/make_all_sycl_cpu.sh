#! /bin/bash

module purge
module load compiler dpct

mkdir -p ../sycl-bench/build

for src in ../sycl-bench/polybench/*.cpp; do
    echo "[*] Making $src"
    TARGET=../sycl-bench/build/$(basename $src .cpp).sycl_cpu.exe
    if [ -f $TARGET ]; then
        echo "[-] $src already built"
    else
        icpx --gcc-toolchain=$(dirname $(which gcc))/.. \
            -fsycl -O3 $src -I../sycl-bench/polybench/common -I../sycl-bench/include \
            -o $TARGET
    fi
done
