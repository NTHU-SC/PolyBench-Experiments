#! /bin/bash

module purge
spack load cuda@11.8.0%intel@2021.8.0

for dir in $(cat targets); do
    make -C ../CUDA/$dir cuda -j
done
