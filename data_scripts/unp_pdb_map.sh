while read l
do

a=$(cat /scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/unp_pdb_map/${l}_unp_expPDB_map.tsv | wc -l)

b=$(cat /scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/unp_pdb_map/${l}_unp_afPDB_map.tsv | wc -l)
echo $l $a, $b

#cat /scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/unp_pdb_map_old/${l}_unp_expPDB_map.tsv | sort | uniq > /scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/unp_pdb_map/${l}_unp_expPDB_map.tsv

#cat /scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/unp_pdb_map_old/${l}_unp_afPDB_map.tsv | sort | uniq > /scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/structure/shin2021/unp_pdb_map/${l}_unp_afPDB_map.tsv


done < /scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets/sequence_list_setNames