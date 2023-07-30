import numpy as np
import re

work_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/pfam_32.0'
save_dir = '{}/pdbmap_sets'.format(work_dir)
pdbmap = np.loadtxt('{}/pdbmap'.format(work_dir),dtype='str',delimiter='\t')

uniq_pfam_list = np.unique(pdbmap[:,4])
count_list = []
for pf in uniq_pfam_list:
  print(pf)
  idxs = np.where(pdbmap[:,4] == pf)[0]
  uniprot_list = [pdbmap[i,5][:-1] for i in idxs]
  uniq_uni_list = np.unique(uniprot_list)
  count_list.append([pf[:-1],len(uniq_uni_list),','.join(uniq_uni_list)])

count_list = np.array(count_list)
count_list = count_list[np.argsort(count_list[:,1]),:]
np.savetxt('{}/uniprot_count.tsv'.format(save_dir),count_list,fmt='%s',delimiter='\t',header='total uniprot num:{}'.format(np.sum(count_list[:,1].astype(int))))
