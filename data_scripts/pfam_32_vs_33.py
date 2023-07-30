import numpy as np



#load pfam_clan
working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process'
pfam320_clan = np.loadtxt('{}/pfam_32.0/Pfam-A.clans.tsv'.format(working_dir),dtype='str',delimiter='\t')
pfam331_clan = np.loadtxt('{}/pfam_33.1/Pfam-A.clans.tsv'.format(working_dir),dtype='str',delimiter='\t')
#load rp stats
pfam331_rp15_stat = np.loadtxt('{}/pfam_33.1/stat_rp15/Pfam-A.rp15.stat'.format(working_dir),dtype='str',delimiter='\t')
pfam331_rp35_stat = np.loadtxt('{}/pfam_33.1/stat_rp35/Pfam-A.rp35.stat'.format(working_dir),dtype='str',delimiter='\t')
pfam331_rp55_stat = np.loadtxt('{}/pfam_33.1/stat_rp55/Pfam-A.rp55.stat'.format(working_dir),dtype='str',delimiter='\t')
pfam331_rp75_stat = np.loadtxt('{}/pfam_33.1/stat_rp75/Pfam-A.rp75.stat'.format(working_dir),dtype='str',delimiter='\t')
pfam331_stat = [pfam331_rp15_stat,pfam331_rp35_stat,pfam331_rp55_stat,pfam331_rp75_stat]
#newly added pfams in v33.1
new_pfam = np.sort(np.setdiff1d(pfam331_clan[:,0],pfam320_clan[:,0]))
new_pfam_withClan = []
new_pfam_rpseqNum = [0,0,0,0]
new_pfam_rpCount = [0,0,0,0]
for p in new_pfam:
  idx = np.where(pfam331_clan[:,0] == p)[0][0]
  clan = pfam331_clan[idx,1]
   #seq num
  idx_rp = []
  idx_rp.append(np.where(pfam331_rp15_stat[:,1]==p)[0])
  idx_rp.append(np.where(pfam331_rp35_stat[:,1]==p)[0])
  idx_rp.append(np.where(pfam331_rp55_stat[:,1]==p)[0])
  idx_rp.append(np.where(pfam331_rp75_stat[:,1]==p)[0])
  isInRp = []
  for i in range(len(idx_rp)):
    idx = idx_rp[i]
    rp_stat = pfam331_stat[i]
    if len(idx) > 0:
      isInRp.append(1)
      new_pfam_rpCount[i] += 1
      new_pfam_rpseqNum[i] += rp_stat[idx[0],-1] 
    else:
      isInRp.append(0)
  new_pfam_withClan.append([p,clan]+isInRp)


new_pfam_withClan = np.array(new_pfam_withClan)
uniq_clans = np.unique(new_pfam_withClan[:,1])
comm_line = 'uniq_clans:{}; pfam_count_in_rp:{};seq_num_in_rp:{}'.format(','.join(uniq_clans),new_pfam_rpCount,new_pfam_rpseqNum)

np.savetxt('{}/pfam_33.1/holdOut_sets/newPfams.csv'.format(working_dir),new_pfam_withClan,fmt='%s',delimiter=',',header=comm_line)

#newly added clans in v33.1
clan320_list = np.unique(pfam320_clan[:,1])
clan331_list = np.unique(pfam331_clan[:,1])

new_clan = np.sort(np.setdiff1d(clan331_list,clan320_list))
new_clan_withPfam = []
for c in new_clan:
  idxs = np.where(pfam331_clan[:,1] == c)[0]
  pfams = pfam331_clan[idxs,0]
  # check existance in rp and seq num
  #for fam in pfams:
  #  for rp_stat in pfam331_stat:
      
  new_clan_withPfam.append([c,';'.join(pfams)])
np.savetxt('{}/pfam_33.1/holdOut_sets/newClans.csv'.format(working_dir),new_clan_withPfam,fmt='%s',delimiter=',')

