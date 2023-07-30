'''
This script is to:
* split a fasta file containing seqs from multiple families to single files(one per family)
* count statistics of each family
'''
from prody import MSAFile, writeMSA
import numpy, os, re

proj_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/'
fam_clan_tsv = 'data_process/pfam/Pfam-A.clans.tsv'
fasta_fl_str = 'data_process/pfam/Pfam-A.fasta'
write_dir = 'data_process/pfam/Pfam-A.fasta_seqs'
stat_dir = 'data_process/pfam/Pfam-A.fasta.stat'

# params
msa_format='fasta'
msa_aligned=False

# load family list
fam_clan = numpy.loadtxt(proj_dir+fam_clan_tsv, dtype='str', delimiter='\t')
fam_list=fam_clan[:,0]

# open msa file
msa_stream = open(proj_dir+fasta_fl_str)

# statistics
count_arr = []
print('>>>start to split file', flush=True)

for fam in fam_list:
  print('>_{}'.format(fam), flush=True)
  msa_target = MSAFile(proj_dir+fasta_fl_str, filter=lambda lbl, seq: fam in lbl, aligned=msa_aligned, format=msa_format)
  print('>_writing msa...', flush=True)
  writeMSA(proj_dir+write_dir+'/{}.fasta'.format(fam), msa_target, format=msa_format)
  
  # check num of seqs in this msa and remove empty msa
  with open(proj_dir+write_dir+'/{}.fasta'.format(fam), 'r') as msa_fl:
    msa_str = msa_fl.read()
  find_header = re.findall(r'>', msa_str)
  num_seqs = len(find_header)
  # find pf acc with version number
  pfAcc = re.search("PF\d+\.\d+", msa_str).group(0)
  # remove empty file
  if num_seqs == 0:
    os.system('rm {}{}/{}.fasta'.format(proj_dir,write_dir,fam))
  else:
    count_arr.append([pfAcc, num_seqs])
print('Done', flush=True)

# save statistics to file
numpy.savetxt(proj_dir+stat_dir, count_arr, fmt='%s', delimiter='\t')

