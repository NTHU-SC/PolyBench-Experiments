#! /bin/bash

source load_gpu.sh

for dir in $(cat targets); do
    make -C ../CUDA/$dir dpct_gpu -j
done
