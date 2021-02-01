#!/bin/bash

for k in 1 5 10
do
    for sim in 1 2
    do
        outdir="exp_k${k}_sim${sim}"
        echo $outdir
        mkdir -p $outdir
        ./build_kNN.sh examples/train.vectors.txt examples/test.vectors.txt \
            $k $sim $outdir/sys_output > $outdir/acc_file
    done
done
