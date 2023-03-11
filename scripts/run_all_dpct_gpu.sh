#! /bin/bash

source load_gpu.sh

REPEAT="${REPEAT:- 5}"

for dir in $(cat targets); do
    echo "[*] Running $dir..."
    cd ../CUDA/$dir
    rm -f output.dpct_gpu.txt
    for i in $(seq $REPEAT); do
        echo "[-] Run $i for $dir"
        ./*.dpct_gpu.exe | tail | tee -a output.dpct_gpu.txt
    done
    cd -
done
