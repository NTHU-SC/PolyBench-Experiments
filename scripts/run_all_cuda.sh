#! /bin/bash

module purge
spack load cuda@11.8.0%intel@2021.8.0

for dir in $(cat targets); do
    echo "[*] Running $dir..."
    cd ../CUDA/$dir
    ./*.cuda.exe | tail | tee output.cuda.txt
    cd -
    sleep 3
done
