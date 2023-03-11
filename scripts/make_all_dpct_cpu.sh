#! /bin/bash

source load_cpu.sh

for dir in $(cat targets); do
    make -C ../CUDA/$dir dpct_cpu -j
done
