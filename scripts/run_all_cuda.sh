#! /bin/bash

source load_gpu.sh

REPEAT="${REPEAT:- 5}"

for dir in $(cat targets); do
    echo "[*] Running $dir..."
    cd ../CUDA/$dir
    rm -f output.cuda.txt
    for i in $(seq $REPEAT); do
        echo "[-] Run $i for $dir"
        ./*.cuda.exe | tail | tee -a output.cuda.txt
    done
    cd -
    sleep 3
done
