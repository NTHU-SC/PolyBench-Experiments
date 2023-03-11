#! /bin/bash

source load_gpu.sh

mkdir -p ../sycl-bench/build

for src in ../sycl-bench/polybench/*.cpp; do
    echo "[*] Making $src"
    TARGET=../sycl-bench/build/$(basename $src .cpp).sycl_gpu.exe
    if [ -f $TARGET ]; then
        echo "[-] $src already built"
    else
        clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_70 \
            -O3 $src -I../sycl-bench/polybench/common -I../sycl-bench/include \
            -o $TARGET
    fi
done
