#! /bin/bash

module purge
module load compiler dpct

for dir in $(cat targets); do
    make -C ../CUDA/$dir dpct_cpu -j
done
