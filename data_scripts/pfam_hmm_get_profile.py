import numpy as np
import re

proj_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'

# load hmm_pfamAcc file
hmm_acc = np.loadtxt('{}/data_process/pfam/Pfam-A.hmm_dat_famAcc'.format(proj_dir), dtype='str', delimiter=' ')

prof_wrt_folder = '{}/data_process/pfam/Pfam-A.hmm_1d_profile'.format(proj_dir)
# loop through hmm file list
for i in range(hmm_acc.shape[0]):
  [hmm_file_dir, pfamAcc, clanAcc] = hmm_acc[i]
  print('process {} from {}'.format(hmm_file_dir, pfamAcc))
  # load whole hmm file into a string
  with open('{}/data_process/pfam/{}'.format(proj_dir, hmm_file_dir), 'r') as hmm_fl:
    hmm_str = hmm_fl.read()
  # extract info from hmm profile
  acc = re.findall(r"^ACC\s+(.+)\n", hmm_str, re.MULTILINE)[0]
  leng = re.findall(r"^LENG\s+(.+)\n", hmm_str, re.MULTILINE)[0]

  assert pfamAcc == acc

  aa_distri_str_list = re.findall(r"^ +\d+ +([\d. ]+\d) +\d+", hmm_str, re.MULTILINE)
  aa_distri_mat = np.array([re.split(r" +", dis_str) for dis_str in aa_distri_str_list])
 
  # write 1d profile to file
  np.savetxt('{}/{}'.format(prof_wrt_folder, pfamAcc), aa_distri_mat, fmt='%s',
      delimiter=' ', header='LENG:{}'.format(leng))
