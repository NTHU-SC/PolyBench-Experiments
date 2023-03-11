#! /bin/bash

source load_cpu.sh

REPEAT="${REPEAT:- 5}"

for dir in $(cat targets); do
    echo "[*] Running $dir..."
    cd ../CUDA/$dir
    rm -f output.cuda.txt
    for i in $(seq $REPEAT); do
        echo "[-] Run $i for $dir"
        ./*.dpct_cpu.exe | tail | tee -a output.dpct_cpu.txt
    done
    cd -
    sleep 3
done
