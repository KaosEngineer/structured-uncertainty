#!/bin/bash

#for j in test test5 test6 test7 test8 test9 test10 test11 test12 test13 test14; do
for j in test-clean test-other ami-eval cv-ru cv-fr; do
rm -rf ${j};
mkdir ${j};
for i in $(seq 0 3); do
        egrep "^H-[0-9]+\s" results-m${i}-avg-${j} | sed -E "s/(^H-[0-9]+\s)-[0-9]\.[0-9]+\s/\1/" > ${j}/hypos_${i}.txt;
        egrep "^H-[0-9]+\s" results-m${i}-avg-${j} | sed -E "s/(^H-[0-9]+\s)-[0-9]\.[0-9]+\s//" > ${j}/whypos_${i}.txt;
        egrep '^H-[0-9]+'   results-m${i}-avg-${j} | awk '{print $1 "-X"}' > ${j}/ids_${i}.txt;
        paste -d " " ${j}/whypos_${i}.txt ${j}/ids_${i}.txt > ${j}/whypos_${i}.txt
done;
python ~/fairseq-py/examples/structured_uncertainty/assessment/compute_cross_metrics.py ${j} 4
done