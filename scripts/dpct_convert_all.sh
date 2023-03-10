#! /bin/bash

module purge
module load dpct

for dir in $(cat targets); do
    make -C ../CUDA/$dir convert -j
done
