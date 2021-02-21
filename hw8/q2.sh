#!/bin/sh

for idx in 1 2 3 4 5; do
    echo "Running model $idx"
    ./svm_classify.sh examples/test model.$idx sys.$idx
done
