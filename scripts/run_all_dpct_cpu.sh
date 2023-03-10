#! /bin/bash

module purge
module load compiler dpct

for dir in $(cat targets); do
    echo "[*] Running $dir..."
    cd ../CUDA/$dir
    ./*.dpct_cpu.exe | tail | tee output.dpct_cpu.txt
    cd -
    sleep 3
done
