#!/bin/bash

#for j in test test5 test7 test10 test12 test13 test14; do
for j in test-clean test-other ami-eval cv-ru cv-fr; do
#rm -rf ${j};
#mkdir ${j};
for i in $(seq 0 3); do
        egrep "^H-[0-9]+\s" results-m${i}-avg-${j} | sed -E "s/^H-[0-9]+\s-[0-9]\.[0-9]+\s//" > ${j}/foo_${i}.txt;
        egrep '^H-[0-9]+'   results-m${i}-avg-${j} | sed "s/H-//" | awk '{print "(" $1 "-X-X)"}' > ${j}/ids_${i}.txt;
        egrep '^H-[0-9]+'   results-m${i}-avg-${j} | awk '{print $1}' > ${j}/hids.txt;
        paste -d " " ${j}/foo_${i}.txt ${j}/ids_${i}.txt > ${j}/whypos_${i}.txt
        rm ${j}/foo_${i}.txt
done;
python ~/fairseq-py/examples/structured_uncertainty/assessment/compute_cross_metrics.py ${j} 4 --wer
paste -d " " ${j}/hids.txt ${j}/cross_wer_tmp.txt > ${j}/cross_wer.txt
done