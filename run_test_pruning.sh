#!/bin/bash
RUN="test_pruning.py"
result="result_pruning.dat"
echo "" > $result
for i in {0..660}; do
    echo $i   
    python $RUN $i >> $result
done
