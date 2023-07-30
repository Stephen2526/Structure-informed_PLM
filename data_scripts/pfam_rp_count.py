import numpy as np
import re, sys, json
'''
* extract 
'''

# path
proj_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
data_dir = proj_dir+'/data_process/pfam_32.0'

#rund = [15, 35, 55, 75]
var = sys.argv[1]
rund = [var]
def gather_stat(rund):
  for i in rund:
    print('Working on Pfam-A.rp{}...'.format(i))
    records = []
    with open(data_dir+'/Pfam-A.rp{}'.format(i), 'r', encoding='latin-1') as fl:
      for line in fl:
        ID = re.findall("^#=GF ID (.+)\n", line)
        AC = re.findall("^#=GF AC (.+)\n", line)
        DE = re.findall("^#=GF DE (.+)\n", line)
        TP = re.findall("^#=GF TP (.+)\n", line)
        SQ = re.findall("^#=GF SQ (.+)\n", line)
        if len(ID) > 0:
          print('Pfam - {}'.format(ID))
          rec = [ID[0]]
        elif len(AC) > 0:
          rec.append(AC[0])
        elif len(DE) > 0:
          rec.append(DE[0])
        elif len(TP) > 0:
          rec.append(TP[0])
        elif len(SQ) > 0:
          rec.append(SQ[0])
          #print('one record: {}'.format(rec))
          records.append(rec)

    # stat count
    records = np.array(records)
    total_num = records.shape[0]
    fields = records[:,3]
    field_uniq, fields_count = np.unique(fields, return_counts=True)
    seq_num = records[:,4].astype(np.int)
    seq_num_sum = sum(seq_num)
    # count of each type

    # comment line
    field_str = ''
    type_seq_num = []
    for j in range(len(field_uniq)):
      idx_list = np.where(records[:,3]==field_uniq[j])[0]
      seq_num = records[idx_list,4].astype(np.int)
      type_seq_num.append(np.sum(seq_num))
      field_str += '{}:{}/{}, '.format(field_uniq[j],fields_count[j],np.sum(seq_num))

    type_seq_num_all = np.sum(type_seq_num)
    #assert(type_seq_num_all == seq_num_sum)
    comm = '{}total_fam_num:{}, total_seq_num:{} =? type_seq_num_all:{}'.format(field_str,total_num,seq_num_sum,type_seq_num_all)
    #print(comm)
    
    # save to file
    
    np.savetxt(data_dir+'/stat_rp{}/Pfam-A.rp{}.stat'.format(i,i), records, fmt='%s', delimiter='\t',header=comm)

    # order by seq number, Fam iden
    rec_by_seqNum = records[np.argsort(records[:,-1])]
    rec_by_famIden = records[np.argsort(records[:,1])]
    np.savetxt(data_dir+'/stat_rp{}/Pfam-A.rp{}_sortBySeqsize.stat'.format(i,i),rec_by_seqNum,fmt='%s', delimiter='\t',header=comm)
    np.savetxt(data_dir+'/stat_rp{}/Pfam-A.rp{}_sortByFacc.stat'.format(i,i),rec_by_famIden,fmt='%s', delimiter='\t',header=comm)

def filter_stat(rund,len_thres):
  for i in rund:
    rp_stat = np.loadtxt(data_dir+'/stat_rp{}/Pfam-A.rp{}.stat'.format(i,i),dtype='str',delimiter='\t')
    pfam_clan = np.loadtxt('{}/Pfam-A.clans.tsv'.format(data_dir),dtype='str',delimiter='\t')
    clan_list = np.loadtxt('{}/Pfam-A.clan_list'.format(data_dir),dtype='str')
    clan_count = []
    
    for clan in clan_list:
      print(clan)
      tar_idx = np.where(pfam_clan[:,1]==clan)[0]
      tar_pfams = [pfam_clan[p,0] for p in tar_idx]
      # remove version
      rp_pfams = np.array([re.split(r'\.',pf)[0] for pf in rp_stat[:,1]])
      tar_idx = []
      for pf in tar_pfams:
        idx = np.where(rp_pfams==pf)[0]
        if len(idx) > 0:
          tar_idx.append(idx[0])
        else:
          pass
      if len(tar_idx) > 0:
        clan_count.append([clan,np.sum(rp_stat[tar_idx,-1].astype(int))])
    clan_count = np.array(clan_count)
    
    np.savetxt('{}/stat_rp{}/family_counts.tsv'.format(data_dir,i),rp_stat[:,[1,4]],fmt='%s',delimiter='\t',header='total pfam seq:{},total pfam num:{}'.format(np.sum(rp_stat[:,-1].astype(int)),rp_stat.shape[0]))
    np.savetxt('{}/stat_rp{}/clan_counts.tsv'.format(data_dir,i),clan_count,fmt='%s',delimiter='\t',header='total clan seq:{},total clan num:{}'.format(np.sum(clan_count[:,-1].astype(int)),clan_count.shape[0]))

    # filter by length
    if len_thres > 0:
      with open('{}/stat_rp{}/rawSeqLen_type.json'.format(data_dir,i),'r') as fl:
        rawSeqLen_type = json.load(fl)
      seqNum_byLen = []
      for key,value in rawSeqLen_type.items():
        value = np.array(value).astype(int)
        seqNum = len(np.where(value <= len_thres)[0])
        seqNum_byLen.append([key,seqNum])
      seqNum_byLen = np.array(seqNum_byLen).astype(str)
      np.savetxt('{}/stat_rp{}/seqNumType_filterBy{}.tsv'.format(data_dir,i,len_thres),seqNum_byLen,fmt='%s',delimiter='\t',header='total num:{}'.format(np.sum(seqNum_byLen[:,1].astype(int))))

def testSet_stat(rund,level,fl_dir):
  rp_rund = rund[0]
  fl_path = '{}/holdOut_sets/{}'.format(data_dir,fl_dir)
  holdOut_list = np.loadtxt(fl_path,dtype='str')
  seqNum_count = []
  if level=='pfam':
    stat_dt = np.loadtxt('{}/stat_rp{}/family_counts.tsv'.format(data_dir,rp_rund),dtype='str',delimiter='\t')
  elif level=='clan':
    stat_dt = np.loadtxt('{}/stat_rp{}/clan_counts.tsv'.format(data_dir,rp_rund),dtype='str',delimiter='\t')
  for iden in holdOut_list:
    tar_idx = np.where(np.array([re.split(r'\.',i)[0] for i in stat_dt[:,0]])==iden)[0]
    if len(tar_idx) > 0:
      seqNum_count.append([iden,int(stat_dt[tar_idx[0],1])])
    else:
      seqNum_count.append([iden,0])
  seqNum_count = np.array(seqNum_count)
  np.savetxt('{}/stat_rp{}/{}_stat.tsv'.format(data_dir,rp_rund,fl_dir),seqNum_count,fmt='%s',delimiter='\t',header='total num:{}'.format(np.sum(seqNum_count[:,1].astype(int))))

    

#filter_stat(rund,500)
testSet_stat(rund,'pfam','muta_pfam_large_set.txt')
testSet_stat(rund,'clan','muta_clan_large_set.txt')
testSet_stat(rund,'pfam','muta_pfam_small_set.txt')
testSet_stat(rund,'clan','muta_clan_small_set.txt')



