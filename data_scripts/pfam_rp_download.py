import numpy as np
from prody import *
import re, sys

# paths
proj_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
data_dir = proj_dir+'/data_process/pfam'


#rund = [15, 35, 55, 75]
rund=[sys.argv[1]]
for i in rund:
  # load family acc
  pfam_info_list = np.loadtxt(data_dir+'/Pfam-A.rp{}.stat'.format(i),
      dtype='str', delimiter='\t')

  for idx in range(pfam_info_list.shape[0]):
  #for idx in range(2):
    pfam_acc = pfam_info_list[idx,1]
    pfam_dscp = pfam_info_list[idx,0]

    print('download seqs of pfam: {}, rp {}...'.format(pfam_acc, i), flush=True)
    # download msa in fasta format for this fam
    fetchPfamMSA(pfam_acc, alignment='rp{}'.format(i), 
                 format='fasta', order='tree', inserts='upper',
                 gaps=None, outname='pfam_rp{}_seqs/msa_{}'.format(i, pfam_acc))

