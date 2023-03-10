#! /bin/bash

module purge
spack load cuda@11.8.0%intel@2021.8.0
module load dpct
export PATH=/home/yi/sycl_workspace/llvm/build/bin:$PATH
export LD_LIBRARY_PATH=/home/yi/sycl_workspace/llvm/build/lib:$LD_LIBRARY_PATH

for dir in $(cat targets); do
    echo "[*] Running $dir..."
    cd ../CUDA/$dir
    ./*.dpct_gpu.exe | tail | tee output.dpct_gpu.txt
    cd -
    sleep 3
done
