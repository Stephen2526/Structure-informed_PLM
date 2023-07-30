import numpy as np
from prody import *
import re, sys, os
import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import lmdb
import pickle as pkl
import time
import pandas as pd
import seaborn as sns

'''
1. save seqs in stockholm msa to unaligned seq in fasta format
2. collect unique characters occuring in msa
3. sequence length distribution (each type)
4. count and filter sequence with ambiguous tokens

For each entry, save pfam_iden, type, raw_seq_length,
aligned_seq_length
'''


def collect(rund):
  for i in rund:
    # define recorders
    rawSeqLen_type = {'Family':[],'Domain':[],'Motif':[],'Repeat':[],'Coiled-coil':[],'Disordered':[]}
    msaSeqLen_type = {'Family':[],'Domain':[],'Motif':[],'Repeat':[],'Coiled-coil':[],'Disordered':[]}
    rawSeqLen_pfam = {}
    rawSeqLen_clan = {'Orphan':[]}
    uniq_chars = [] #list to hold unique characters occuring in msa
    ambig_chars = ['B','Z','X','O','U','b','z','x','u']
    seqs_ambigChar = [] # list to store seqs containing ambiguous tokens
    
    
    # load family acc
    #pfam_info_list = np.loadtxt(data_dir+'/Pfam-A.rp{}.stat'.format(i), dtype='str', delimiter='\t')
    #new_label_list = [] # to store all labels
    #seq_list = [] # to store all seqs
    
    # load msa file list
    msa_files = np.loadtxt(data_dir+'/pfam_rp{}_seqs_files_famAcc'.format(i),dtype='str',delimiter=' ')
    #print(msa_files.shape)
    

    for idx in range(len(msa_files)):
    #for idx in range(1,2):
      #pfam_acc = pfam_info_list[idx,1]
      #pfam_dscp = pfam_info_list[idx,0]
      #print('download seqs of pfam: {}, rp {}...'.format(pfam_acc, i), flush=True)
      
      # download msa in fasta format for this fam
      #fetchPfamMSA(pfam_acc, alignment='rp{}'.format(i), 
      #             format='fasta', order='tree', inserts='upper',
      #             gaps=None, outname='tmp_msa')
      # modify comment line of msa file
      #msa = MSAFile('tmp_msa_rp{}.fasta'.format(i), aligned=False)
      msa_nm = msa_files[idx][0]
      pfam_nm = msa_files[idx][1]
      clan_nm = msa_files[idx][2]
      print('>>>{},{},{}'.format(msa_nm,pfam_nm, clan_nm))
      # initialize list to store seq len
      if pfam_nm not in rawSeqLen_pfam.keys():
        rawSeqLen_pfam[pfam_nm] = []
      if len(clan_nm) > 0 and clan_nm not in rawSeqLen_clan.keys():
        rawSeqLen_clan[clan_nm] = []
      # load msa file as a string
      msa_fl = open('{}/{}'.format(data_dir, msa_nm), 'r', encoding='latin-1')
      msa_content = msa_fl.readlines()
      # extract info from meta-header
      #ID, AC = '', ''
      for line in msa_content:
        '''
        ID_tmp = re.findall("^#=GF ID (.+)\n", line)
        AC_tmp = re.findall("^#=GF AC (.+)\n", line)
        if len(ID_tmp) > 0:
          ID = ID_tmp[0]
        if len(AC_tmp) > 0:
          AC = AC_tmp[0]
        if len(ID) > 0 and len(AC) > 0:
          break
        '''
        TP_tmp = re.findall("^#=GF TP (.+)\n", line)
        if len(TP_tmp) > 0:
          TP = TP_tmp[0]
          break
      print('type: {}'.format(TP))
      # initilize msa from stockholm file
      msa = MSAFile('{}/{}'.format(data_dir, msa_nm), format='Stockholm')
      for seq in msa:
        #extract from header
        res_idx = seq.getResnums()
        start_idx = res_idx[0]
        end_idx = res_idx[-1]
        label = seq.getLabel()
        uniP_id = label
        #new_label_list.append('{}/{}-{} {} {};{};'.format(label,start_idx,end_idx, uniP_id, AC, ID))
        pro_seq = ''.join(re.findall('[A-Za-z]', str(seq))).upper()
        #print(new_label_list[-1])
        #print(pro_seq)
        #seq_list.append(pro_seq)
        
        # store length
        msaSeq_len = len(str(seq))
        rawSeq_len = len(pro_seq)
        rawSeqLen_type[TP].append(rawSeq_len)
        msaSeqLen_type[TP].append(msaSeq_len)
        rawSeqLen_pfam[pfam_nm].append(rawSeq_len)
        if len(clan_nm) > 0:
          rawSeqLen_clan[clan_nm].append(rawSeq_len)
        else:
          rawSeqLen_clan['Orphan'].append(rawSeq_len)
        # collect uniq characters
        chars = list(set(str(seq)))
        for c in chars:
          if c not in uniq_chars:
            uniq_chars.append(c)
          if c in ambig_chars:
            seqs_ambigChar.append(['{}/{}/{}/{}-{}/{}'.format(pfam_nm,clan_nm,label,start_idx,end_idx,len(pro_seq)),pro_seq])
    
    # create new msa object
    #new_msa = MSA(seq_list, title='new_fasta', labels=new_label_list, aligned=False)
    # save to fasta file
    #writeMSA('Pfam-A.rp{}.fasta'.format(i), new_msa, aligned=False)
    '''
    print('writing to fasta file...') 
    with open(data_dir+'/Pfam-A.rp{}.fasta'.format(i), 'w') as fl:
      for seq_i in range(len(new_label_list)):
        fl.write('>{}\n'.format(new_label_list[seq_i]))
        fl.write(seq_list[seq_i]+'\n')
    '''

    # save counts and draw figures
    '''
    with open('{}/stat_rp{}/rawSeqLen_type.json'.format(data_dir,rund[0]), 'w') as fl:
      json.dump(rawSeqLen_type, fl)
    with open('{}/stat_rp{}/msaSeqLen_type.json'.format(data_dir,rund[0]), 'w') as fl:
      json.dump(msaSeqLen_type, fl)
    np.savetxt('{}/stat_rp{}/seqs_ambigChar'.format(data_dir,rund[0]),seqs_ambigChar,fmt='%s',delimiter=' ')
    uniq_chars.sort()
    np.savetxt('{}/stat_rp{}/uniq_chars'.format(data_dir,rund[0]),uniq_chars,fmt='%s')
    '''
    with open('{}/stat_rp{}/rawSeqLen_pfam.json'.format(data_dir,rund[0]), 'w') as fl:
      json.dump(rawSeqLen_pfam, fl)
    with open('{}/stat_rp{}/rawSeqLen_clan.json'.format(data_dir,rund[0]), 'w') as fl:
      json.dump(rawSeqLen_clan, fl)

def drawFig(rund):
  with open('{}/stat_rp{}/rawSeqLen_type.json'.format(data_dir,rund[0])) as fl:
    rawSeqLen_type = json.load(fl)
  with open('{}/stat_rp{}/msaSeqLen_type.json'.format(data_dir,rund[0])) as fl:
    msaSeqLen_type = json.load(fl)

  # individual figure
  '''
  fig_num = 0
  for key, value in rawSeqLen_type.items():
    fig = plt.figure(fig_num,figsize=(8,6),dpi=100)
    (n,bins,_) = plt.hist(value, bins=50)
    new_n = np.insert(n,0,0,axis=-1)
    np.savetxt('{}/stat_rp{}/rawSeqLen_type_{}_hist.csv'.format(data_dir,rund[0],key),np.column_stack((new_n,bins)),fmt='%s',delimiter=',')
    fig.savefig('{}/stat_rp{}/rawSeqLen_type_{}_hist.png'.format(data_dir,rund[0],key))
    fig_num += 1

  for key, value in msaSeqLen_type.items():
    fig = plt.figure(fig_num,figsize=(8,6),dpi=100)
    (n,bins,_) = plt.hist(value, bins=50)
    new_n = np.insert(n,0,0,axis=-1)
    np.savetxt('{}/stat_rp{}/msaSeqLen_type_{}_hist.csv'.format(data_dir,rund[0],key),np.column_stack((new_n,bins)),fmt='%s',delimiter=',')
    fig.savefig('{}/stat_rp{}/msaSeqLen_type_{}_hist.png'.format(data_dir,rund[0],key))
    fig_num += 1
  '''
  # one figure
  fig = plt.figure(0,figsize=(8,6),dpi=100)
  lab_list,val_list = [],[]

  for key, value in rawSeqLen_type.items():
    lab_list.append(key)
    val_list.append(value)
  #plt.yscale('log', nonposy='clip')
  #plt.yticks(np.arange(0, 1.04, 0.05))
  plt.hist(val_list,bins=50,density=False,histtype='step',fill=False,alpha=1,stacked=False,cumulative=False,label=lab_list)
  plt.legend()
  fig.savefig('{}/stat_rp{}/rawSeqLen_allType_hist.png'.format(data_dir,rund[0]))

  fig = plt.figure(1,figsize=(8,6),dpi=100)
  lab_list,val_list = [],[]
  for key, value in msaSeqLen_type.items():
    lab_list.append(key)  
    val_list.append(value)
  #plt.yscale('log', nonposy='clip')
  #plt.yticks(np.arange(0, 1.04, 0.05))
  plt.hist(val_list,bins=50,density=False,histtype='step',fill=False,alpha=1,stacked=False,cumulative=False,label=lab_list)
  plt.legend()
  fig.savefig('{}/stat_rp{}/msaSeqLen_allType_hist.png'.format(data_dir,rund[0]))

def merge_lmdb(working_dir,set_list):
  set_dir = 'seq_json_rp75'
  target_set_dir = 'seq_json_rp75_all'
  all_dataDir = '{}/{}/pfam_train_lenCut.lmdb'.format(working_dir,target_set_dir)
  map_size = (1024 * 100) * (2 ** 20) # 100G
  all_dataList = []
  count_exp = 0
  for one_set in set_list:
    print('processing {}-{}'.format(set_dir,one_set))
    dt_dir = '{}/{}/pfam_{}_lenCut.lmdb'.format(working_dir,set_dir,one_set)
    dt_env = lmdb.open(str(dt_dir), max_readers=1, readonly=True,
                       lock=False, readahead=False, meminit=False)
    with dt_env.begin(write=False) as txn:
      num_exp = pkl.loads(txn.get(b'num_examples'))
      print('>>num: {}'.format(num_exp))
      for idx in range(num_exp):
        item = pkl.loads(txn.get(str(idx).encode()))
        item['ori_set'] = one_set
        item['id'] = count_exp
        count_exp += 1
        all_dataList.append(item)
    dt_env.close()
  
  all_dataEnv = lmdb.open(all_dataDir, map_size=map_size)
  with all_dataEnv.begin(write=True) as txn:
    for i, entry in enumerate(all_dataList):
      txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i + 1))
  all_dataEnv.close()
  print('new set num: {}'.format(i+1))
  
def seq_lmdb_subtract(working_dir):
  sbtor_dataDir = '{}/seq_json_rp75/pfam_holdout_lenCut.lmdb'.format(working_dir)
  sbtee_dataDir = '{}/seq_json_rp15/pfam_holdout_lenCut.lmdb'.format(working_dir)

  sbtor_env = lmdb.open(str(sbtor_dataDir), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
  sbtee_env = lmdb.open(str(sbtee_dataDir), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)

  # collect sbtee's unpIden
  sbtee_unpIden = []
  with sbtee_env.begin(write=False) as txn:
    sbtee_num_examples = pkl.loads(txn.get(b'num_examples'))
    print('rp15 holdout num: {}'.format(sbtee_num_examples),flush=True)
    for idx in range(sbtee_num_examples):
      item = pkl.loads(txn.get(str(idx).encode()))
      sbtee_unpIden.append(item['unpIden'])
  sbtee_env.close()

  # assemble new set
  new_set = []
  start = time.time()
  with sbtor_env.begin(write=False) as txn:
    sbtor_num_examples = pkl.loads(txn.get(b'num_examples'))
    print('rp75 holdout num: {}'.format(sbtor_num_examples),flush=True)
    for idx in range(sbtor_num_examples):
      if idx % 1000 == 0:
        end = time.time()
        print('idx: {}, cost time: {}'.format(idx,end-start),flush=True)
        start = time.time()
      item = pkl.loads(txn.get(str(idx).encode()))
      if item['unpIden'] not in sbtee_unpIden:
        new_set.append(item)
  sbtor_env.close()

  newSet_wrtDir = '{}/seq_json_rp75/pfam_holdout-filrp15_lenCut.lmdb'.format(working_dir)
  map_size = (1024 * 40) * (2 ** 20) # 15G
  newSet_wrtEnv = lmdb.open(newSet_wrtDir, map_size=map_size)
  with newSet_wrtEnv.begin(write=True) as txn:
    for i, entry in enumerate(new_set):
      txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i + 1))
  newSet_wrtEnv.close()
  print('new rp75 holdout num: {}'.format(i+1))

def pretrain_rp_split_fig_ppl(working_dir,jsonFl_iden,rp_set):
  '''
  generate figs for pretrained models over rp-split set
    * perplexity curve along training epochs
  '''
  
  ## perplexity curve along training epochs (rp15)
  ece = []
  ppl = []
  acc = [[],[],[],[]] # acc top 1,3,5,10
  stepsPerEpoch = 3072
  #rp_set = 'rp15'
  print('**rp_set:{}**'.format(rp_set))
  epoch_dict={
    'rp15':[[9,19,29,39,49,59,69,79,89,99,109,119,129,139,149,159,169,179,189,199,209,219,229,239,249,259,269,279,289,299,309,319,329,339,349]],
    'rp35':[[9,19,29,39,49,59,69,79,89,99,109,119,129,139,149,159,169],[179,184]],
    'rp55':[[9,19,29,39,49,59,69,79,89,94]],
    'rp75':[[9,19,29,39,49,59,69,74]]
    }
  epoch_list = epoch_dict[rp_set]
  epoch_all_list = np.concatenate(epoch_list)
  mdl_dict={
    'rp15':['masked_language_modeling_transformer_21-03-18-21-37-07_715933'],
    'rp35':['masked_language_modeling_transformer_21-03-18-21-34-54_566349','masked_language_modeling_transformer_21-03-26-00-24-49_842837'],
    'rp55':['masked_language_modeling_transformer_21-03-24-21-57-23_683717'],
    'rp75':['masked_language_modeling_transformer_21-03-18-21-29-01_485133']
    }
  mdl_path = mdl_dict[rp_set]
  #rp15 [9,19,29,39,49,59,69,79,89,99,109,119,129,139,149,159,169,179,189,199,209,219,229,239,249,259,269,279,289,299,309,319,329,339,349]
  #rp55 [9,19,29,39,49,59,69,79,89,94]
  #rp75 [9,19,29,39,49,59,69,74]
  for m_i in range(len(mdl_path)):
    for epo in epoch_list[m_i]:
      with open('{}/{}/{}/{}_{}.json'.format(working_dir,rp_set,mdl_path[m_i],jsonFl_iden,epo),'r') as fl:
        metrics = json.load(fl)
      ece.append(metrics['lm_ece'])
      ppl.append(metrics['lm_ppl'])
      acc[0].append(metrics['accuracy'])
      acc[1].append(metrics['accuracy_top3'])
      acc[2].append(metrics['accuracy_top5'])
      acc[3].append(metrics['accuracy_top10'])
  # min, max
  print('ece: min:{}-{}; mean-std:{}-{}\nppl: min:{}-{}; mean-std:{}-{}'.format(epoch_all_list[np.argmin(ece)],min(ece),np.mean(ece),np.std(ece), 
                                                                                epoch_all_list[np.argmin(ppl)],min(ppl),np.mean(ppl),np.std(ppl)))
  print('acc:   max:{}-{}; mean-std:{}-{}\nacc-3: max:{}-{}; mean-std:{}-{}\nacc-5: max:{}-{}; mean-std:{}-{}\nacc-10:max:{}-{}; mean-std:{}-{}'.format(
                                                  epoch_all_list[np.argmax(acc[0])],max(acc[0]),np.mean(acc[0]),np.std(acc[0]),
                                                  epoch_all_list[np.argmax(acc[1])],max(acc[1]),np.mean(acc[1]),np.std(acc[1]),
                                                  epoch_all_list[np.argmax(acc[2])],max(acc[2]),np.mean(acc[2]),np.std(acc[2]),
                                                  epoch_all_list[np.argmax(acc[3])],max(acc[3]),np.mean(acc[3]),np.std(acc[3])))
  if not os.path.isdir('{}/{}/figures'.format(working_dir,rp_set)):
    os.mkdir('{}/{}/figures'.format(working_dir,rp_set))
  fig,ax = plt.subplots()
  plt.plot(epoch_all_list,ece,marker='.',label='ECE')
  plt.plot(epoch_all_list,ppl,marker='+',label='PPL')
  plt.legend()
  plt.savefig('{}/{}/figures/ece_ppl_epoch.png'.format(working_dir,rp_set))
  plt.close(fig)
  
  fig,ax = plt.subplots()
  acc_labels = ['accuracy','accuracy_top3','accuracy_top5','accuracy_top10']
  for i in range(4):
    plt.plot(epoch_all_list,acc[i],marker='.',label=acc_labels[i])
  plt.legend()
  plt.savefig('{}/{}/figures/acc_epoch.png'.format(working_dir,rp_set))
  plt.close(fig)


def pretrain_rp_split_fig_precision(working_dir,jsonFl_iden,rp_set_list):
  '''
  generate figs for pretrained models over rp-split set
    * precision curve along training epochs
  '''
  
  ## perplexity curve along training epochs (rp15)
  df_list = []
  stepsPerEpoch = 3072
  metric_lm_list = ['accuracy','accuracy_top3','accuracy_top5','accuracy_top10','lm_ece','lm_ppl']
  metric_prec_list = ['max_precision_all_5','max_precision_all_2','max_precision_all_1','max_precision_short_5','max_precision_short_2','max_precision_short_1','max_precision_medium_5','max_precision_medium_2','max_precision_medium_1','max_precision_long_5','max_precision_long_2','max_precision_long_1','mean_precision_all_5','mean_precision_all_2','mean_precision_all_1','mean_precision_short_5','mean_precision_short_2','mean_precision_short_1','mean_precision_medium_5','mean_precision_medium_2','mean_precision_medium_1','mean_precision_long_5','mean_precision_long_2','mean_precision_long_1','apc_precision_all_5','apc_precision_all_2','apc_precision_all_1','apc_precision_short_5','apc_precision_short_2','apc_precision_short_1','apc_precision_medium_5','apc_precision_medium_2','apc_precision_medium_1','apc_precision_long_5','apc_precision_long_2','apc_precision_long_1']
  #rp_set = 'rp15'
  epoch_dict={
    'rp15':[[9,19,29,39,49,59,69,79,89,99,109,119,129,139,149,159,169,179,189,199,209,219,229,239,249,259,269,279,289,299,309,319,329,339,349]],
    'rp35':[[9,19,29,39,49,59,69,79,89,99,109,119,129,139,149,159,169],[179,184]],
    'rp55':[[9,19,29,39,49,59,69,79,89,94]],
    'rp75':[[9,19,29,39,49,59,69,74]]
    }
  mdl_dict={
    'rp15':['masked_language_modeling_transformer_21-03-18-21-37-07_715933'],
    'rp35':['masked_language_modeling_transformer_21-03-18-21-34-54_566349','masked_language_modeling_transformer_21-03-26-00-24-49_842837'],
    'rp55':['masked_language_modeling_transformer_21-03-24-21-57-23_683717'],
    'rp75':['masked_language_modeling_transformer_21-03-18-21-29-01_485133']
    }
  epoch2step = {
      'rp15':3053,
      'rp35':7773,
      'rp55':11989,
      'rp75':16662
      }
  for rp_set in rp_set_list:
    print('**rp_set:{}**'.format(rp_set))
    mdl_path = mdl_dict[rp_set]
    epoch_list = epoch_dict[rp_set]
    for set_nm in ['holdout','valid']: 
      for m_i in range(len(mdl_path)):
        for epo in epoch_list[m_i]:
          case_list = []
          case_list.append(rp_set) # append 'rp_nm'
          case_list.append(epo) # append 'epoch'
          case_list.append(epo*epoch2step[rp_set]) # append step
          with open('{}/{}/{}/{}_{}_{}.json'.format(working_dir,rp_set,mdl_path[m_i],jsonFl_iden,set_nm,epo),'r') as fl:
            metrics = json.load(fl)
          
          for metric_nm in metric_lm_list:
            case_list.append(metrics[metric_nm]) # append lm metrics
          
          for metric_nm in metric_prec_list:
            name_split = re.split('_',metric_nm)
            range_nm = name_split[2]
            symm = name_split[0]
            topK = 'L/{}'.format(name_split[3])
            dt_set = set_nm
            metric_value = np.array(metrics[metric_nm])
            metric_indiv_mean = np.array(metrics[metric_nm+'_indiv_mean'])
            metric_indiv_std = np.array(metrics[metric_nm+'_indiv_std'])
            for lay in range(metric_value.shape[0]):
              for hea in range(metric_value.shape[1]):
                prec_case_list = []
                prec_case_list.extend(case_list)
                #print(len(case_list))
                layer_idx, head_idx = lay+1, hea+1
                prec = metric_value[lay,hea]
                prec_mean = metric_indiv_mean[lay,hea]
                prec_std = metric_indiv_std[lay,hea]
                prec_case_list.extend([prec,prec_mean,prec_std,layer_idx,head_idx,symm,range_nm,topK,dt_set])
                #print(len(prec_case_list))
                df_list.append(prec_case_list) 
  #print(np.array(df_list).shape)
  df = pd.DataFrame(df_list,columns=['rp_nm','epoch','step','acc','acc_top3','acc_top5','acc_top10','lm_ece','lm_ppl','precision','precision_mean','precision_std','layer_idx','head_idx','symm','range','topK','dt_set'])
  
  set_nm='holdout'
  # average across these epochs
  prec_ave_epochs = {
      'rp15': [329,339,349],
      'rp35': [129,139,149,159,169,179,184],
      'rp55': [89,94],
      'rp75': [59,69,74]
      }
  for rp_set in rp_set_list:
    '''
    # check min, max of lm metrics
    df_lm_uniq = df.drop_duplicates(subset=['rp_nm','epoch','acc','acc_top3','acc_top5','acc_top10','lm_ece','lm_ppl','dt_set'])
    df_lm = df_lm_uniq.loc[(df_lm_uniq["rp_nm"]==rp_set) & (df_lm_uniq["dt_set"]==set_nm)]
    print('**{} lm summary({}):>>'.format(rp_set,set_nm))
    print('>>across-epoch min ece: {}'.format(df_lm[df_lm['lm_ece']==df_lm['lm_ece'].min()].values.tolist()))
    print('>>across-epoch ece mean-std:{}-{}'.format(df_lm['lm_ece'].mean(),df_lm['lm_ece'].std()))
    
    print('>>across-epoch min ppl: {}'.format(df_lm[df_lm['lm_ppl']==df_lm['lm_ppl'].min()].values.tolist()))
    print('>>across-epoch ppl mean-std:{}-{}'.format(df_lm['lm_ppl'].mean(),df_lm['lm_ppl'].std()))
    
    print('>>across-epoch max acc: {}'.format(df_lm[df_lm['acc']==df_lm['acc'].max()].values.tolist()))
    print('>>across-epoch acc mean-std:{}-{}'.format(df_lm['acc'].mean(),df_lm['acc'].std()))
    
    print('>>across-epoch max acc_top3: {}'.format(df_lm[df_lm['acc_top3']==df_lm['acc_top3'].max()].values.tolist()))
    print('>>across-epoch acc_top3 mean-std:{}-{}'.format(df_lm['acc_top3'].mean(),df_lm['acc_top3'].std()))
    
    print('>>across-epoch max acc_top5: {}'.format(df_lm[df_lm['acc_top5']==df_lm['acc_top5'].max()].values.tolist()))
    print('>>across-epoch acc_top5 mean-std:{}-{}'.format(df_lm['acc_top5'].mean(),df_lm['acc_top5'].std()))

    print('>>across-epoch max acc_top10: {}'.format(df_lm[df_lm['acc_top10']==df_lm['acc_top10'].max()].values.tolist()))
    print('>>across-epoch acc_top10 mean-std:{}-{}'.format(df_lm['acc_top10'].mean(),df_lm['acc_top10'].std()))
    '''
    if not os.path.isdir('{}/{}/figures'.format(working_dir,rp_set)):
      os.mkdir('{}/{}/figures'.format(working_dir,rp_set))
    '''
    ## lm fig
    fig,ax = plt.subplots()
    plt.plot(df_lm['epoch'],df_lm['lm_ece'],marker='.',label='ECE')
    plt.plot(df_lm['epoch'],df_lm['lm_ppl'],marker='+',label='PPL')
    plt.legend()
    plt.savefig('{}/{}/figures/structure_holdout_ece_ppl_epoch.png'.format(working_dir,rp_set))
    plt.close(fig)
    
    fig,ax = plt.subplots()
    acc_labels = ['acc','acc_top3','acc_top5','acc_top10']
    for i in range(4):
      plt.plot(df_lm['epoch'],df_lm[acc_labels[i]],marker='.',label=acc_labels[i])
    plt.legend()
    plt.savefig('{}/{}/figures/structure_holdout_acc_epoch.png'.format(working_dir,rp_set))
    plt.close(fig)
    '''
    '''contact precision summary
        * all, short, medium, long
        * layer 1, 2, 3, 4
        * average across heads,
        * last epochs
    '''
    '''
    print('**{} contact precision summary ({})>>'.format(rp_set,set_nm))
    df_contact = df.loc[(df["rp_nm"]==rp_set) & (df["dt_set"]==set_nm) & (df['epoch'].isin(prec_ave_epochs[rp_set])) & (df["symm"]=='apc')]
    for k in [1,2,5]:
      for ran in ['all','short','medium','long']:
        print('>>{}; L{}: lay1 {}-{}, lay2 {}-{}, lay3 {}-{}, lay4 {}-{}'.format(ran,k,
               df_contact.loc[(df_contact["topK"]=='L/{}'.format(k)) & (df_contact['range']==ran) & (df_contact['layer_idx']==1)].shape[0],
               df_contact.loc[(df_contact["topK"]=='L/{}'.format(k)) & (df_contact['range']==ran) & (df_contact['layer_idx']==1)]["precision_mean"].mean(),
               df_contact.loc[(df_contact["topK"]=='L/{}'.format(k)) & (df_contact['range']==ran) & (df_contact['layer_idx']==2)].shape[0],
               df_contact.loc[(df_contact["topK"]=='L/{}'.format(k)) & (df_contact['range']==ran) & (df_contact['layer_idx']==2)]["precision_mean"].mean(),
               df_contact.loc[(df_contact["topK"]=='L/{}'.format(k)) & (df_contact['range']==ran) & (df_contact['layer_idx']==3)].shape[0], 
               df_contact.loc[(df_contact["topK"]=='L/{}'.format(k)) & (df_contact['range']==ran) & (df_contact['layer_idx']==3)]["precision_mean"].mean(),
               df_contact.loc[(df_contact["topK"]=='L/{}'.format(k)) & (df_contact['range']==ran) & (df_contact['layer_idx']==4)].shape[0],
               df_contact.loc[(df_contact["topK"]=='L/{}'.format(k)) & (df_contact['range']==ran) & (df_contact['layer_idx']==4)]["precision_mean"].mean()
               ))
    '''
  
  def to_percent(y, position):
    s = '{:0.2f}'.format(100*y)
    return s + '%'
  
  '''
  
  #precision figure
  #* x="layer_idx", y="precision_mean", hue="rp_nm", row="range", col="topK",
  #* box or bar
  
  sns.set(style="whitegrid", rc={"lines.linewidth": 1.0})
  range_togo_list = ['all','short','medium','long']
  topK_togo_list = ['1','2','5']
  symm_togo_list = ['apc','max','mean']
  for range_togo in ['all']:
    for topK_togo in ['1']:
      for symm_togo in symm_togo_list:
        filter_df = df.loc[(df["dt_set"]=='holdout') & (df["symm"]==symm_togo)]
        filter_df = filter_df.loc[((filter_df["rp_nm"]=='rp15') & (filter_df["epoch"].isin(prec_ave_epochs['rp15']))) | 
                                  ((filter_df["rp_nm"]=='rp35') & (filter_df["epoch"].isin(prec_ave_epochs['rp35']))) | 
                                  ((filter_df["rp_nm"]=='rp55') & (filter_df["epoch"].isin(prec_ave_epochs['rp55']))) |  
                                  ((filter_df["rp_nm"]=='rp75') & (filter_df["epoch"].isin(prec_ave_epochs['rp75'])))
                                  ]
        #print(filter_df)
        
        #gax = sns.catplot(x="layer_idx", y="precision_mean", hue="rp_nm",
        #                   row="range", col="topK", data=filter_df, kind='box',
        #                   height=6, aspect=1.5, palette=sns.color_palette("hls", 4),
        #                   order=[1,2,3,4],hue_order=['rp15','rp35','rp55','rp75'],
        #                   row_order=['all','short','medium','long'],col_order=['L/1','L/2','L/5'],
        #                   whis=10.0,width=0.8,legend=False)
        
        gax = sns.catplot(x="layer_idx", y="precision_mean", hue="rp_nm",
                           row="range", col="topK", data=filter_df, kind='bar',
                           height=6, aspect=1.5, palette=sns.color_palette("hls", 4),
                           order=[1,2,3,4],hue_order=['rp15','rp35','rp55','rp75'],
                           row_order=['all','short','medium','long'],col_order=['L/1','L/2','L/5'],
                           legend=False,ci='sd')

        ## background precision mean
        gax.map(plt.axhline,y=0.0306,color='red',ls='solid',lw=1.,label='all_bg')
        gax.map(plt.axhline,y=0.0552,color='red',ls='dotted',lw=1.,label='short_bg')
        gax.map(plt.axhline,y=0.0403,color='red',ls='dashed',lw=1.,label='medium_bg')
        gax.map(plt.axhline,y=0.0257,color='red',ls='dashdot',lw=1.,label='long_bg')
        
        gax.axes[2][2].legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        
        #gax.savefig('{}/results_to_keep/figures/{}/{}_logis_layer_{}.png'.format(working_dir,tar_fig_dir,range_togo,topK_togo))
        #gax.savefig('{}/results_to_keep/figures/{}/overall_logis_{}.png'.format(working_dir,tar_fig_dir,topK_togo))
        #gax.savefig('{}/results_to_keep/figures/{}/all_distri_nonRegLayer_{}_{}.png'.format(working_dir,tar_fig_dir,symm_togo,topK_togo))
        #gax.savefig('{}/results_to_keep/figures/{}/{}_layer_apc_{}.png'.format(working_dir,tar_fig_dir,range_togo,topK_togo))
        #gax.savefig('{}/{}/figures/struct_holdout_contact_{}_L{}_layer_precision_epoch.png'.format(working_dir,rp_set,symm_togo,topK_togo))
        gax.savefig('{}/figures/pretrain/struct_holdout_contact_{}_layer_precision_rpNm_bar_SD_aveG1M.png'.format(working_dir,symm_togo))
        plt.clf()
  '''

  '''
  ece-precision figure
  * lineplot
  * x="ece", y="precision_mean", hue="rp_nm", row="range", col="topK"
  * 4th layer 
  '''
  sns.set(style="ticks")
  range_togo_list = ['all','short','medium','long']
  topK_togo_list = ['1','2','5']
  symm_togo_list = ['apc','max','mean']
  for range_togo in range_togo_list:
    for topK_togo in topK_togo_list:
      for symm_togo in symm_togo_list:
        df_orderByEpoch = df.loc[(df["dt_set"]=='valid')
            & (df["symm"]==symm_togo) & (df["range"]==range_togo)
            & (df["topK"]=='L/'+topK_togo) & (df["layer_idx"]==4)]
        #.sort_values(['epoch','rp_nm','layer_idx','head_idx'],ascending=[True,True,True,True])
        #print(df_orderByEpoch.loc[df_orderByEpoch["epoch"]==9])
        #gax = sns.relplot(x="step", y="precision_mean", hue="rp_nm",
        #                  row="range", col="topK", data=df_orderByEpoch, kind='line',
        #                  height=6, aspect=1.5, palette=sns.color_palette("hls", 4),
        #                  hue_order=['rp15','rp35','rp55','rp75'],sort=True,
        #                  row_order=['all','short','medium','long'],col_order=['L/1','L/2','L/5'],
        #                  ci=None)
        #gax = sns.lineplot(x="step", y="precision_mean", hue="rp_nm",
        #                  data=df_orderByEpoch, palette=sns.color_palette("hls", 4),
        #                  hue_order=['rp15','rp35','rp55','rp75'],sort=True,
        #                  ci=None)
        gax = sns.lineplot(x="step", y="lm_ece", hue="rp_nm",
                          data=df_orderByEpoch, palette=sns.color_palette("hls", 4),
                          hue_order=['rp15','rp35','rp55','rp75'],sort=True,
                          ci=None)

        
        #gax.savefig('{}/results_to_keep/figures/{}/{}_logis_layer_{}.png'.format(working_dir,tar_fig_dir,range_togo,topK_togo))
        #gax.savefig('{}/results_to_keep/figures/{}/overall_logis_{}.png'.format(working_dir,tar_fig_dir,topK_togo))
        #gax.savefig('{}/results_to_keep/figures/{}/all_distri_nonRegLayer_{}_{}.png'.format(working_dir,tar_fig_dir,symm_togo,topK_togo))
        #gax.savefig('{}/results_to_keep/figures/{}/{}_layer_apc_{}.png'.format(working_dir,tar_fig_dir,range_togo,topK_togo))
        #gax.savefig('{}/{}/figures/struct_holdout_contact_{}_L{}_layer_precision_epoch.png'.format(working_dir,rp_set,symm_togo,topK_togo))
        plt.savefig('{}/figures/pretrain/struct_valid_contact_{}_{}_{}_step_lmECE_rpNm_line.png'.format(working_dir,symm_togo,range_togo,topK_togo))
        plt.clf()
  
if __name__ == '__main__':
  # paths
  proj_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  data_dir = proj_dir+'/data_process/pfam_32.0'
  #rund = [15, 35, 55, 75]
  #rund=[sys.argv[1]]

  #collect(rund)
  #merge_lmdb(data_dir,['train','valid','holdout'])
  #seq_lmdb_subtract(data_dir)
  #pretrain_rp_split_fig_ppl(proj_dir+'/results_to_keep','results_metrics_seq_json_rp75_holdout','rp75')
  pretrain_rp_split_fig_precision(proj_dir+'/results_to_keep','results_metrics_pdbmap_contactData',['rp15','rp35','rp55','rp75'])


