#!/bin/bash
for i in $(seq 1 20)
do
  pcregrep -M -o1 '^.+BLAT_ECOLX\s([A-Z]\d*[A-Z])' folds/BLAT_ECOLX_Scratch.fasta_set_$i > mutListFolds/BLAT_ECOLX_mutList.fasta_set_$i
done

