import numpy as np
import glob
'''
check the whether the vector is a qualified probability distribution
'''

# random generate 100 file indexes
file_idx = np.random.randint(1,18872, size=100)

for idx in file_idx:
  print('processing PF{:05d}'.format(idx))
  fl_nm=glob.glob('/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/pfam/Pfam-A.hmm_1d_profile/PF{:05d}*'.format(idx))
  if len(fl_nm) > 0:
    nega_ln_mat = np.loadtxt(fl_nm[0], dtype='float', delimiter=' ')
    prob_mat = 1 / np.exp(nega_ln_mat)
    sum_mat = np.sum(prob_mat, axis=-1)
    for ss in sum_mat:
      if np.abs(ss-1) > 0.0001:
        print('bad prob occurs in PF{:05d}'.format(idx))
  
