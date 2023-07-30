import numpy as np
#from prody import *
import re, sys, json


'''
1. classify pfam entries based on Clan
'''
# paths
proj_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
data_dir = proj_dir+'/data_process/pfam_33.1'

def clan_anno(rp):
  clan_set = {}
  clan_set_seqNum = 0
  noClan_pfam_set = []
  # load whole pfam-clan pairs
  pfam_clan = np.loadtxt('{}/Pfam-A.clans.tsv'.format(data_dir),dtype='str',delimiter='\t')
  # load this rp's stat
  pfam_seqNum_count = np.loadtxt('{}/stat_rp{}/Pfam-A.rp{}.stat'.format(data_dir,rp,rp),dtype='str',delimiter='\t')
  #print(pfam_seqNum_count)
  for i in range(len(pfam_seqNum_count)):
    line = pfam_seqNum_count[i]
    #print(line)
    pfam_nm = line[1]
    pfam_nm_noVer = re.split(r'\.',pfam_nm)[0]
    tar_idx = np.where(pfam_clan[:,0]==pfam_nm_noVer)[0][0]
    clan_nm = pfam_clan[tar_idx,1]
    #pfam_nm = re.split(r'\.',pfam_nm_ver)[0]
    if len(clan_nm) == 0:
      tar_idx = np.where(pfam_seqNum_count[:,1]==pfam_nm)[0][0]
      noClan_pfam_set.append([pfam_nm, pfam_seqNum_count[tar_idx,3], pfam_seqNum_count[tar_idx,4]])
    else:
      if clan_nm not in clan_set.keys():
        clan_set[clan_nm] = []
        tar_idx = np.where(pfam_seqNum_count[:,1]==pfam_nm)[0][0]
        clan_set[clan_nm].append([pfam_nm, pfam_seqNum_count[tar_idx,3], pfam_seqNum_count[tar_idx,4]])
        clan_set_seqNum += int(pfam_seqNum_count[tar_idx,4])
      else:
        tar_idx = np.where(pfam_seqNum_count[:,1]==pfam_nm)[0][0]
        clan_set[clan_nm].append([pfam_nm, pfam_seqNum_count[tar_idx,3], pfam_seqNum_count[tar_idx,4]])
        clan_set_seqNum += int(pfam_seqNum_count[tar_idx,4])

  clan_entryNum = []
  for key, value in clan_set.items():
    value = np.array(value)
    num_entry = value.shape[0]
    uniq_types = np.unique(value[:,1])
    clan_entryNum.append([key,num_entry,';'.join(uniq_types)])
  clan_entryNum = np.array(clan_entryNum)
  clan_entryNum_all = np.sum(clan_entryNum[:,1].astype(np.int))
  
  noClan_pfam_set = np.array(noClan_pfam_set)
  noClan_SeqNum = np.sum(noClan_pfam_set[:,2].astype(np.int))
  noClan_pfamNum = len(noClan_pfam_set)
  noClan_pfam_set_sorted = noClan_pfam_set[noClan_pfam_set[:,2].astype(np.int).argsort()]
  np.savetxt('{}/stat_rp{}/pfam_noClan.csv'.format(data_dir,rp),noClan_pfam_set_sorted,fmt='%s',delimiter=',',header='seqNum_clan:{}, seqNum_noClan:{}, pfamNum_clan:{}, pfamNum_noClan:{}'.format(clan_set_seqNum, noClan_SeqNum,clan_entryNum_all,noClan_pfamNum))

  clan_entryNum_sorted = clan_entryNum[clan_entryNum[:,1].astype(np.int).argsort()]
  np.savetxt('{}/stat_rp{}/clan_entryNum.csv'.format(data_dir,rp),clan_entryNum_sorted,fmt='%s',delimiter=',',header='seqNum_clan:{}, seqNum_noClan:{}, pfamNum_clan:{}, pfamNum_noClan:{}'.format(clan_set_seqNum,noClan_SeqNum,clan_entryNum_all,noClan_pfamNum))

  #print(clan_set)
  with open('{}/stat_rp{}/clan_set.json'.format(data_dir,rp),'w') as fl:
    json.dump(clan_set, fl)

rp = sys.argv[1]
clan_anno(rp)

    
    
