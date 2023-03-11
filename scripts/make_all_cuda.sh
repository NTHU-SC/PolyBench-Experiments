#! /bin/bash

source load_gpu.sh

for dir in $(cat targets); do
    make -C ../CUDA/$dir cuda -j
done
