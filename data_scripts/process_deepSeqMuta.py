from codecs import ignore_errors
from encodings import search_function
import glob
import numpy as np
import scipy.stats as st
import re, collections, os, sys, umap
import pandas as pd
import json, lmdb
import pickle as pkl
from typing import Dict, List
import random
from collections import OrderedDict
import Bio.PDB
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib import colors, markers
from scipy.stats import spearmanr
from sklearn.metrics import auc,roc_curve,precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from pdbmap_process import asym_mapping, queryApi_pdbInfo, get_unp_pdb_seqIdx_mapping, unmodel_pdb_idx, check_valid_pos, kth_diag_indices, NumpyArrayEncoder
#from tensorboard.backend.event_processing import event_accumulator


def proc_ind():
  file_list = '{}/DeepSequenceMutaSet_flList'.format(working_dir)
  with open(file_list,'r') as fl:
    for one_fl in fl:
      print('processing {}'.format(one_fl[:-1]))
      mut_dt = np.loadtxt('{}/DeepSequenceMutaSet/{}'.format(working_dir,one_fl[:-1]),dtype='str',delimiter=',',skiprows=1)
      #print(mut_dt.shape)
      prim_seq_idx = {} #idx:wt_aa
      mut_list = mut_dt[:,1]
      
      for muta_one in mut_list:
        multi_muta = re.split(r':',muta_one)
        for muta in multi_muta:
          if len(muta) > 2:
            wt_aa = muta[0]
            mut_aa = muta[-1]
            mut_idx = int(muta[1:-1])
            if mut_idx not in prim_seq_idx.keys():
              prim_seq_idx[mut_idx] = [wt_aa]
            else:
              if wt_aa not in prim_seq_idx[mut_idx]:
                print('>>>More than 1 aa at one pos!!!')
              else:
                pass
      sorted_primSeqIdx = collections.OrderedDict(sorted(prim_seq_idx.items()))
      idx_list = list(sorted_primSeqIdx.keys())
      range_len = idx_list[-1]-idx_list[0]+1
      print('>>>start:{},end:{},range:{}'.format(idx_list[0],idx_list[-1],len(idx_list)))
      if range_len != len(idx_list):
        print('>>>Some aa not covered!!!')
        sorted_primSeqIdx_list = []
        for manul_idx in range(idx_list[0],idx_list[-1]+1):
          if manul_idx in idx_list:
            sorted_primSeqIdx_list.append([manul_idx,sorted_primSeqIdx[manul_idx][0]])
          else:
            sorted_primSeqIdx_list.append([manul_idx,'*'])
        sorted_primSeqIdx_list = np.array(sorted_primSeqIdx_list)
        #np.savetxt('{}/DeepSequenceMutaSet_priSeq/{}_idxSeqMiss.csv'.format(working_dir,one_fl[:-5]),sorted_primSeqIdx_list,fmt='%s',delimiter=',')
        #with open('{}/DeepSequenceMutaSet_priSeq/{}_seq.txt'.format(working_dir,one_fl[:-5]),'w') as fl:
        #  fl.write(''.join(sorted_primSeqIdx_list[:,1]))
      else:
        # generate primary seq
        sorted_primSeqIdx_list = []
        for idx,aa in sorted_primSeqIdx.items():
          sorted_primSeqIdx_list.append([idx,aa[0]])
        sorted_primSeqIdx_list = np.array(sorted_primSeqIdx_list)
        #np.savetxt('{}/DeepSequenceMutaSet_priSeq/{}_idxSeq.csv'.format(working_dir,one_fl[:-5]),sorted_primSeqIdx_list,fmt='%s',delimiter=',')
        #with open('{}/DeepSequenceMutaSet_priSeq/{}_seq.txt'.format(working_dir,one_fl[:-5]),'w') as fl:
        #  fl.write(''.join(sorted_primSeqIdx_list[:,1]))

def label_statistic(working_dir):
  file_list = '{}/DeepSequenceMutaSet_flList'.format(working_dir)
  mut_stat = {}
  with open(file_list,'r') as fl:
    for one_fl in fl:
      print('processing {}'.format(one_fl[:-1]))
      mut_stat[one_fl[:-1]] = []
      #mut_dt = np.loadtxt('{}/DeepSequenceMutaSet/{}'.format(working_dir,one_fl[:-1]),dtype='str',delimiter=',',skiprows=1)
      mut_labels = pd.read_csv('{}/DeepSequenceMutaSet/{}'.format(working_dir,one_fl[:-1]),index_col=0)
      for col_name, col_count in mut_labels.count().iteritems():
        mut_stat[one_fl[:-1]].append({col_name:col_count})


  with open('{}/label_count.json'.format(working_dir), 'w', encoding='utf-8') as f:
    json.dump(mut_stat, f, ensure_ascii=False, indent=2)

def mutant_label_distribution(working_dir):
  '''
  check label distribution of wt, single and multi-site mutation
  '''
  label_target = {}
  with open('{}/target_label_set.tsv'.format(working_dir),'r') as fl:
    for line in fl:
      line_split = re.split(',',line[:-1])
      label_target[line_split[0]] = line_split[1]

  # keys: set_nm, wt_seq, mutants, mut_seq, fitness, mutation_effect_prediction_vae_ensemble,mutation_effect_prediction_vae_1,mutation_effect_prediction_vae_2,mutation_effect_prediction_vae_3,mutation_effect_prediction_vae_4,mutation_effect_prediction_vae_5,mutation_effect_prediction_pairwise,mutation_effect_prediction_independent
  for set_nm, label_nm in label_target.items():
    mutant_count_dict = OrderedDict()
    print('>processing {}'.format(set_nm))
    os.system("echo '>processing {}' >> {}/mutant_label_dist.txt".format(set_nm,working_dir))
    #mut_dt = np.loadtxt('{}/DeepSequenceMutaSet/{}'.format(working_dir,one_fl[:-1]),dtype='str',delimiter=',',skiprows=1)
    mut_labels = pd.read_csv('{}/DeepSequenceMutaSet/{}.csv'.format(working_dir,set_nm),index_col=0)
    target_df = mut_labels[mut_labels[label_nm].notnull()]
    for idx, row in target_df.iterrows():
      target_muts = re.split(':',row['mutant']) # multi-site mutation delimited by ':'
      target_label = row[label_nm]
      
      if target_muts in ['wt', 'WT']:
        if 0 not in mutant_count_dict.keys():
          mutant_count_dict[0]=[]
          mutant_count_dict[0].append(target_label)
        else:
          mutant_count_dict[0].append(target_label)
      elif len(target_muts) == 1:
        one_mut = target_muts[0]
        wt_aa = one_mut[0]
        mut_aa = one_mut[-1]
        if wt_aa == mut_aa:
          if 0 not in mutant_count_dict.keys():
            mutant_count_dict[0]=[]
            mutant_count_dict[0].append(target_label)
          else:
            mutant_count_dict[0].append(target_label)
        else:
          if 1 not in mutant_count_dict.keys():
            mutant_count_dict[1]=[]
            mutant_count_dict[1].append(target_label)
          else:
            mutant_count_dict[1].append(target_label)
      elif len(target_muts) > 1:
        mut_num = len(target_muts)
        if mut_num not in mutant_count_dict.keys():
          mutant_count_dict[mut_num]=[]
          mutant_count_dict[mut_num].append(target_label)
        else:
          mutant_count_dict[mut_num].append(target_label)
    print('>>label value dist:')      
    os.system("echo '>>label value dist:' >> {}/mutant_label_dist.txt".format(working_dir))
    for k, v in mutant_count_dict.items():
      print('>**>{}-site mut: min {}; max {}; mean {}; sd {}'.format(k,np.amin(v),np.amax(v),np.mean(v),np.std(v)))
      os.system("echo '>**>{}-site mut: min {}; max {}; mean {}; sd {}' >> {}/mutant_label_dist.txt".format(k,np.amin(v),np.amax(v),np.mean(v),np.std(v),working_dir))

def mutant_count(working_dir):
  '''
  count wt-change, single-site, multi-site mutants
  '''
  label_target = {}
  with open('{}/target_label_set.tsv'.format(working_dir),'r') as fl:
    for line in fl:
      line_split = re.split(',',line[:-1])
      label_target[line_split[0]] = line_split[1]

  # keys: set_nm, wt_seq, mutants, mut_seq, fitness, mutation_effect_prediction_vae_ensemble,mutation_effect_prediction_vae_1,mutation_effect_prediction_vae_2,mutation_effect_prediction_vae_3,mutation_effect_prediction_vae_4,mutation_effect_prediction_vae_5,mutation_effect_prediction_pairwise,mutation_effect_prediction_independent
  for set_nm, label_nm in label_target.items():
    mutant_count_dict = OrderedDict({0:0,
                                     1:0})
    print('>processing {}'.format(set_nm))
    os.system("echo '>processing {}' >> {}/mutant_count.txt".format(set_nm,working_dir))
    #mut_dt = np.loadtxt('{}/DeepSequenceMutaSet/{}'.format(working_dir,one_fl[:-1]),dtype='str',delimiter=',',skiprows=1)
    mut_labels = pd.read_csv('{}/DeepSequenceMutaSet/{}.csv'.format(working_dir,set_nm),index_col=0)
    target_df = mut_labels[mut_labels[label_nm].notnull()]
    for idx, row in target_df.iterrows():
      target_muts = re.split(':',row['mutant']) # multi-site mutation delimited by ':'
      if target_muts in ['wt', 'WT']:
        mutant_count_dict[0] += 1
      elif len(target_muts) == 1:
        one_mut = target_muts[0]
        wt_aa = one_mut[0]
        mut_aa = one_mut[-1]
        if wt_aa == mut_aa:
          mutant_count_dict[0] += 1
        else:
          mutant_count_dict[1] += 1
      elif len(target_muts) > 1:
        mut_num = len(target_muts)
        if mut_num not in mutant_count_dict.keys():
          mutant_count_dict[mut_num] = 1
        else:
          mutant_count_dict[mut_num] += 1
    print('>>total rows: {}'.format(target_df.shape[0]))
    print('>>detailed counts:')
    os.system("echo '>>total rows: {}' >> {}/mutant_count.txt".format(target_df.shape[0], working_dir))
    os.system("echo '>>detailed counts:' >> {}/mutant_count.txt".format(working_dir))
    sum_total = 0
    for k, v in mutant_count_dict.items():
      print('>>{}:{}'.format(k,v))
      os.system("echo '>>{}:{}' >> {}/mutant_count.txt".format(k,v,working_dir))
      sum_total += v
    assert sum_total == target_df.shape[0]

def prepare_DeepSeq_mutations(working_dir: str = None,
                              onlyKeep_label_existing: bool = False,
                              save_json: bool = False):
  """
  Descriptions:
    * prepare mutant sequence and label sets for fitness evaluation

  Parameters:
    * onlyKeep_label_existing: bool; if true, only keep variants with existing fitness label
    * save_json: bool; if true, save a json verson of outputs

  Outputs:
    * saved dictionary data in lmdb format
  """
  label_target = {}
  with open('{}/target_label_set.csv'.format(working_dir),'r') as fl:
    for line in fl:
      line_split = re.split(',',line[:-1])
      label_target[line_split[0]] = line_split[1]
  
  priSeq_range = np.loadtxt('{}/DeepSequenceMutaSet_priSeq_idxRange'.format(working_dir),dtype='str',delimiter=',')
  priSeq_range_dict = {}
  for i in range(priSeq_range.shape[0]):
    priSeq_range_dict[priSeq_range[i][0]]=priSeq_range[i,1:]

  # keys: set_nm, wt_seq, mutants, mut_seq, fitness, mutation_effect_prediction_vae_ensemble,mutation_effect_prediction_vae_1,mutation_effect_prediction_vae_2,mutation_effect_prediction_vae_3,mutation_effect_prediction_vae_4,mutation_effect_prediction_vae_5,mutation_effect_prediction_pairwise,mutation_effect_prediction_independent
  all_num = 0
  for set_nm, label_nm in label_target.items():
    # mkdir dir
    if not os.path.isdir('{}/mutagenesisData/set_data_2/{}'.format(working_dir,set_nm)):
      os.mkdir('{}/mutagenesisData/set_data_2/{}'.format(working_dir,set_nm))

    set_dt_list = [] # list of jsons
    set_wt_list = [] # list to hold wt cases
    print('>processing {}'.format(set_nm))
    i_num = 0
    #mut_dt = np.loadtxt('{}/DeepSequenceMutaSet/{}'.format(working_dir,one_fl[:-1]),dtype='str',delimiter=',',skiprows=1)
    mut_labels = pd.read_csv('{}/DeepSequenceMutaSet/{}.csv'.format(working_dir,set_nm),index_col=0)
    
    with open('{}/DeepSequenceMutaSet_priSeq/{}_seq.txt'.format(working_dir,set_nm),'r') as fl:
      wt_seq = fl.read().replace('\n','')
    mut_seq_list = list(wt_seq)
    ## find rows(mutations) with non-NAN label(fitness) values
    if onlyKeep_label_existing:
      target_df = mut_labels[mut_labels[label_nm].notnull()]
    else:
      target_df = mut_labels
    ## loop over each row(mutant)
    for idx, row in target_df.iterrows():
      mut_seq_list = list(wt_seq) 
      target_muts = re.split(':',row['mutant']) # multi-site mutation delimited by ':'
      #if len(target_muts) > 1:
      #  continue
      mut_relative_idxs = []
      # replace aa to get mutant seq
      for one_mut in target_muts:
        if one_mut not in ['wt', 'WT']:
          wt_aa = one_mut[0]
          mut_aa = one_mut[-1]
          if wt_aa not in ['_','X'] and mut_aa not in ['_', 'X']:
            idx_aa = int(one_mut[1:-1])
            priSeq_startIdx = int(priSeq_range_dict[set_nm][0])
            priSeq_endIdx = int(priSeq_range_dict[set_nm][1])
            #print('idx_aa:{}; priSeq_startIdx:{}'.format(idx_aa, priSeq_startIdx))
            assert (idx_aa >= priSeq_startIdx) and (idx_aa <= priSeq_endIdx)
            assert wt_seq[idx_aa - priSeq_startIdx] == wt_aa
            mut_relative_idxs.append(idx_aa-priSeq_startIdx)
            ## mutant wt seq
            mut_seq_list[idx_aa - priSeq_startIdx] = mut_aa
      if ''.join(mut_seq_list) != wt_seq:
        one_json = {"set_nm": set_nm,
                    "wt_seq": wt_seq,
                    "seq_len": len(mut_seq_list),
                    "mutants": target_muts,
                    "mut_relative_idxs": mut_relative_idxs,
                    "mut_seq": ''.join(mut_seq_list),
                    "fitness": row[label_nm]}
        set_dt_list.append(one_json)
        i_num += 1
        all_num += 1
      else:
        wt_one_json = {"set_nm": set_nm,
                      "wt_seq": wt_seq,
                      "seq_len": len(wt_seq),
                      "mutants": target_muts,
                      "mut_relative_idxs": mut_relative_idxs,
                      "fitness": row[label_nm]}
        set_wt_list.append(wt_one_json)
        #print('wt mut:',target_muts)
    print('>->- {} examples for this set (exclude wt cases)'.format(i_num))
    
    # save data
    if save_json:
      #sample_dt = random.sample(set_dt_list, 5)
      #with open('{}/mutagenesisData/set_data/{}/{}_mut_samples.json'.format(working_dir,set_nm,set_nm),'w') as fl:
      #  json.dump(sample_dt,fl)
      with open('{}/mutagenesisData/set_data/{}/{}_mut_all.json'.format(working_dir, set_nm, set_nm),'w') as fl:
        json.dump(set_dt_list,fl)
      if len(set_wt_list) > 0:
        with open('{}/mutagenesisData/set_data/{}/{}_wt_all.json'.format(working_dir, set_nm, set_nm),'w') as fl:
          json.dump(set_wt_list,fl)
      
  
    map_size = (1024 * 15) * (2 ** 20) # 15G
    wrtEnv = lmdb.open('{}/mutagenesisData/set_data_2/{}/{}_mut_all.lmdb'.format(working_dir,set_nm,set_nm),map_size=map_size)
    with wrtEnv.begin(write=True) as txn:
      for i, entry in enumerate(set_dt_list):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(i+1))
    wrtEnv.close()
    
    if len(set_wt_list) > 0:
      wrtEnv = lmdb.open('{}/mutagenesisData/set_data_2/{}/{}_wt_all.lmdb'.format(working_dir,set_nm,set_nm),map_size=map_size)
      with wrtEnv.begin(write=True) as txn:
        for i, entry in enumerate(set_wt_list):
          txn.put(str(i).encode(), pkl.dumps(entry))
        txn.put(b'num_examples', pkl.dumps(i+1))
      wrtEnv.close()

    print('>saving data done for {}'.format(set_nm))
  print('>In total, {} mut cases'.format(all_num))

def change_variant_data_setNm(set_list: List = None):
  """change set_nm values in variant dataset
  """
  path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  map_size = (1024 * 15) * (2 ** 20) # 15G
  for set_name in set_list:
    env = lmdb.open(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_name}/{set_name}_mut_all.lmdb',map_size=map_size)
    with env.begin(write=True) as txn: 
      num_examples = pkl.loads(txn.get(b'num_examples'))
      for idx in range(num_examples):
        item = pkl.loads(txn.get(str(idx).encode()))
        item['set_nm'] = set_name
        txn.replace(str(idx).encode(),pkl.dumps(item))
  return NotImplementedError

def new_wtSeq(
      wtSeq_fasta: str = None,
      mut2add: List = None,
      save_fasta: str = None):
  """
  Description:
    apply a list of mutations(only missense) to a wt seq to generate new a wt seq
  """
  with open(f'{wtSeq_fasta}') as handle:
    for record in SeqIO.parse(handle,"fasta"):
      wtSeq = list(record.seq)
  outSeq = wtSeq
  for mut in mut2add:
    # missense
    refAA = mut[0]
    mutAA = mut[-1]
    pos = int(mut[1:-1])
    assert wtSeq[pos-1] == refAA, f'{wtSeq[pos-1]}@{pos-1} not match wt AA in mutation {refAA}'
    outSeq[pos-1] = mutAA
    record = SeqRecord(
                Seq(''.join(outSeq)),
                id="delta_RBD|delta RBD mutations introduced to wt covid spike protein")
    with open(f'{save_fasta}', "w") as output_handle:
      SeqIO.write(record, output_handle, "fasta")

def prepare_fitnessScan_inputs(
      wtSeq_fasta: str = None,
      seq_range: List = None,
      scan_range: List = None,
      scan_pos: List = None,
      save_json: bool = False,
      save_file: str = None):
  """
  Descriptions:
    Prepare input data for fitness landscape evaluation (predict fitness for all possible AA mutations)

  Parameters:
    * wtSeq_fasta: str; path of the file storing wt seq for target protein (fasta format)
    * seq_range: List; [start_idx,end_idx], define a segment as input seq (not whole seq)
    * scan_range: List; [start_idx,end_idx], region to do fitness scanning
    * scan_pos: List; positions to do fitness scanning
    * save_json: bool; if true, save a json version of outputs
  """
  subLineage_level1_mutations = [
    #alpha
    'L452R','E484K','S494P','D138H','F490S','P681R',
    #beta
    'L18F','P384L','E516Q',
    #gamma
    'Y144F','T470N','Q675H','N679K','P681H','P681R','C1235F',
    #omicron
    'T19I','L24S','V213G','S371F','T376A','D405N','R408S',
    #delta
    'K417N','E484Q','Q613H','T95I','A222V'
  ]

  subLineage_level2_mutations = [
    #gamma
    'H49Y','R246G','T284I',
    #omicron
    'R346K'
  ]

  with open(f'{wtSeq_fasta}') as handle:
    for record in SeqIO.parse(handle,"fasta"):
      wtSeq = str(record.seq)
  wtSeq_target = wtSeq[seq_range[0]-1:seq_range[1]]
  dt2save = []
  if scan_range is not None:
    for scan_i in range(scan_range[0],scan_range[1]+1):
      mut_relative_idxs = [scan_i-seq_range[0]]
      one_dt = {"wt_seq": wtSeq_target,
                "seq_len": len(wtSeq_target),
                "mut_relative_idxs": mut_relative_idxs,
                "mut_abso_idxs": [scan_i]}
      dt2save.append(one_dt)
  elif scan_pos is not None:
    for scan_i in scan_pos:
      mut_relative_idxs = [scan_i-seq_range[0]]
      one_dt = {"wt_seq": wtSeq_target,
                "seq_len": len(wtSeq_target),
                "mut_relative_idxs": mut_relative_idxs,
                "mut_abso_idxs": [scan_i]}
      dt2save.append(one_dt)
  if save_json:
    with open(f'{save_file}.json','w') as fl:
      json.dump(dt2save,fl)
  
  map_size = (1024 * 15) * (2 ** 20) # 15G
  wrtEnv = lmdb.open(f'{save_file}',map_size=map_size)
  with wrtEnv.begin(write=True) as txn:
    for i, entry in enumerate(dt2save):
      txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i+1))
  wrtEnv.close()

def split_files(
      file_prefix: str,
      format: str,
      split_num: int,):
  """Split large sets for quick inference
  """
  if format == 'lmdb':
    map_size = (1024 * 50) * (2 ** 20) # 15G
    input_env = lmdb.open(f'{file_prefix}.lmdb', readonly=True, lock=False, readahead=False, meminit=False)

    with input_env.begin(write=False) as read_txn:
      num_examples = pkl.loads(read_txn.get(b'num_examples'))
      num_valid = int(input_env.stat()['entries'])-1
      assert num_valid == num_examples
      print(f'total num: {num_examples}')
      subset_idx = np.split(np.arange(num_examples), split_num)
      for s in range(split_num):
        print(f'>write set {s}: {len(subset_idx[s])}')
        counter = 0
        write_env = lmdb.open(f'{file_prefix}_{s}.lmdb', map_size=map_size)
        with write_env.begin(write=True) as write_txn:
          for sub_idx in subset_idx[s]:
            item = pkl.loads(read_txn.get(str(sub_idx).encode()))
            write_txn.put(str(counter).encode(), pkl.dumps(item))
            counter += 1
          write_txn.put(b'num_examples', pkl.dumps(len(subset_idx[s])))
        write_env.close()
    input_env.close()

  return

def split_dataset_label_supervise(working_dir):
  '''
  * split-1 (single/mul-site mixed together)
    80% of each set gathered together as training set
    evaluate on remaining 20% for each set
  '''
  train_mut_list = []
  val_mut_list = []
  test_mut_list = []
  map_size = (1024 * 15) * (2 ** 20) # 15G

  set_list = np.loadtxt('{}/DeepSequenceMutaSet_flList'.format(working_dir),dtype='str')
  # loop through sets
  for set_nm in set_list:
    print('>process set: {}'.format(set_nm))
    test_mut_list_indi = []
    with open('{}/mutagenesisData/set_data/{}/{}_mut_all.json'.format(working_dir,set_nm,set_nm)) as fl:
      mut_dt_all = json.load(fl)
    # loop through examples
    for one_dt in mut_dt_all:
      prob = random.random()
      if prob < 0.8:
        prob /= 0.8
        if prob < 0.1:
          val_mut_list.append(one_dt)
        else:
          train_mut_list.append(one_dt)
      else:
        test_mut_list.append(one_dt)
        test_mut_list_indi.append(one_dt)

    # save test_individual
    wrtEnv = lmdb.open('{}/mutagenesisData/set_data/{}/{}_mut_holdout.lmdb'.format(working_dir,set_nm,set_nm),map_size=map_size)
    with wrtEnv.begin(write=True) as txn:
      for i, entry in enumerate(test_mut_list_indi):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(i+1))
    wrtEnv.close()
    print('>*>* set individual test num: {}'.format(len(test_mut_list_indi)))

  # save train, val, test set
  wrtEnv = lmdb.open('{}/mutagenesisData/mut_train.lmdb'.format(working_dir),map_size=map_size)
  with wrtEnv.begin(write=True) as txn:
      for i, entry in enumerate(train_mut_list):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(i+1))
  wrtEnv.close()
  
  wrtEnv = lmdb.open('{}/mutagenesisData/mut_valid.lmdb'.format(working_dir),map_size=map_size)
  with wrtEnv.begin(write=True) as txn:
      for i, entry in enumerate(val_mut_list):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(i+1))
  wrtEnv.close()

  wrtEnv = lmdb.open('{}/mutagenesisData/mut_holdout.lmdb'.format(working_dir),map_size=map_size)
  with wrtEnv.begin(write=True) as txn:
      for i, entry in enumerate(test_mut_list):
        txn.put(str(i).encode(), pkl.dumps(entry))
      txn.put(b'num_examples', pkl.dumps(i+1))
  wrtEnv.close()
  print('>In total, train num: {}, validation num: {}, test num: {}'.format(len(train_mut_list),len(val_mut_list),len(test_mut_list)))

def generate_contact_map(working_dir):
  '''
  generate comtact map from pdb structures
  '''
  
  cutoff = 8.0 # distance cutoff for contact
  # strcture coverage in uniprot indices
  wt_seq_struct_range = np.loadtxt('{}/wt_seq_structure/wt_structure.csv'.format(working_dir),dtype='str',delimiter=',',skiprows=1)
  noAsymId_pdb, noRsp_pdb, noAlign_pdb, noExpAtom_pdb, noValidReg_pdb, wtNotUnp = [], [], [], [], [], []
  packed_data = []
  bio_pdbList = Bio.PDB.PDBList()
  bio_pdbParser = Bio.PDB.PDBParser()
  bio_mmcifParser = Bio.PDB.FastMMCIFParser(QUIET=True)
  for l in range(wt_seq_struct_range.shape[0]):
    set_nm = wt_seq_struct_range[l,0]
    wtSeq_unp_start,wtSeq_unp_end = re.split('-',wt_seq_struct_range[l,1])
    wtSeq_unp_start,wtSeq_unp_end = int(wtSeq_unp_start),int(wtSeq_unp_end)
    if len(wt_seq_struct_range[l,2]) > 0:
      pdbId,pdbChain = re.split('-',wt_seq_struct_range[l,2])
      pdbId = pdbId.upper()
    else:
      pdbId,pdbChain = None,None
    if len(wt_seq_struct_range[l,3]) > 0:
      struc_unp_start,struc_unp_end = re.split('-',wt_seq_struct_range[l,3])
      struc_unp_start,struc_unp_end = int(struc_unp_start), int(struc_unp_end)
    else:
      struc_unp_start,struc_unp_end = None,None
    unpAcc = wt_seq_struct_range[l,4]
    print('{},{}-{},{}-{},{}-{}'.format(set_nm,wtSeq_unp_start,wtSeq_unp_end,pdbId,pdbChain,struc_unp_start,struc_unp_end))
    
    # load wt seq
    with open('{}/DeepSequenceMutaSet_priSeq/{}_seq.txt'.format(working_dir,set_nm),'r') as fl:
      wt_seq = fl.read().replace('\n','')
    
    if pdbId is not None:
      # mapping {auth_asym_id:asym_id}
      asym_mapping_dict = asym_mapping(pdbId)
      if asym_mapping_dict is None:
        noAsymId_pdb.append([set_nm,pdbId,pdbChain])
        # make a dumb asym_mapping_dict
        asym_mapping_dict = {pdbChain:pdbChain}
      # query pdb info
      res_flag,auth_pdbSeq_mapping,unp_seq,pdb_seq,aligned_regions,unobserved_residues,unobserved_atoms=queryApi_pdbInfo(working_dir,pdbId,asym_mapping_dict[pdbChain],unpAcc)
      if not res_flag: # no response
        noRsp_pdb.append([set_nm,pdbId,pdbChain])
        continue
      else:
        # fetch pdb file and generate pdb object
        pdb_flNm = bio_pdbList.retrieve_pdb_file(pdbId,pdir='tmp_download/{}_pdb'.format(set_nm),file_format='mmCif',overwrite=True)
        #pdb_struc = bio_pdbParser.get_structure(pdbId,'{}'.format(pdb_flNm))
        pdb_struc = bio_mmcifParser.get_structure(pdbId,'{}'.format(pdb_flNm))
        pdb_model = pdb_struc[0]
        #os.remove(pdb_flNm)

        # build residue id dict
        resiId_dict={}
        for resi_obj in pdb_model[pdbChain]:
          resiId_tuple = resi_obj.get_id()
          if resiId_tuple[2] != ' ':
            resiId_dict['{}{}'.format(resiId_tuple[1],resiId_tuple[2])] = resiId_tuple
          else:
            resiId_dict['{}'.format(resiId_tuple[1])] = resiId_tuple

        if aligned_regions is None: # this unpAcc not covered by this pdb
          # get unp_seq, do seq alignment
          noAlign_pdb.append([set_nm,pdbId,pdbChain])
          #unp_seq = query_unpSeq(unpAcc)
          #aligned_range = pdb_unp_align(pdb_seq,unp_seq)
          continue
        else:
          unp_pdb_seqIdx_mapping = get_unp_pdb_seqIdx_mapping(aligned_regions)
          unmodelResi_pdbIdxs,unmodelAtom_pdbIdxs = unmodel_pdb_idx(unobserved_residues,unobserved_atoms)
          valid_unpIdx_list = check_valid_pos(wtSeq_unp_start,wtSeq_unp_end,aligned_regions,unmodelResi_pdbIdxs,unp_pdb_seqIdx_mapping)
          if len(valid_unpIdx_list) > 0:
            # loop over multiple valid regions of the pdb
            print('>>val_region:{}-{},len:{}'.format(valid_unpIdx_list[0],valid_unpIdx_list[-1],len(valid_unpIdx_list)))
            # build json
            tmp_data_dict = {} 
            tmp_data_dict['unpAcc'] = unpAcc
            tmp_data_dict['pfamAcc'] = 'PF'
            # uniprot seq is used as target seq
            seq_tar = ""
            unpSeq_len = len(unp_seq)
            unp_pfam_range_maxUnp = range(wtSeq_unp_start, min(wtSeq_unp_end,unpSeq_len)+1) # index should not exceed unp seq length
            unp_pfam_range = range(wtSeq_unp_start, wtSeq_unp_end+1) # index should not exceed unp seq length
            unp_pfam_range_len = len(unp_pfam_range)
            assert len(wt_seq) == unp_pfam_range_len
            for unpIdx_i in unp_pfam_range_maxUnp:
              seq_tar += unp_seq[unpIdx_i-1]
            # wt seq verification
            # UBC9_HUMAN_Roth2017 (extra Y159 - as unmodeled residue)
            if not wt_seq == seq_tar and set_nm != 'UBC9_HUMAN_Roth2017':
              wtNotUnp.append(set_nm)
              continue
            tmp_data_dict['set_nm'] = set_nm
            tmp_data_dict['unp_pfam_range'] = '{}-{}'.format(unp_pfam_range[0],unp_pfam_range[-1])
            tmp_data_dict['target_seq'] = wt_seq
            tmp_data_dict['targetSeq_len'] = unp_pfam_range_len
            tmp_data_dict['best_pdb'] = pdbId
            tmp_data_dict['chain_id'] = pdbChain
            tmp_data_dict['valid_unpIdxs_len'] = len(valid_unpIdx_list)
            tmp_data_dict['valid_unpIdxs'] = valid_unpIdx_list
            
            valid_mask = [False]*unp_pfam_range_len
            valid_pos = [val_i - unp_pfam_range[0] for val_i in valid_unpIdx_list]
            for i in valid_pos:
              valid_mask[i] = True
            tmp_data_dict['valid_mask'] = valid_mask 
            
            # generate contact-map (with self-self, self-neighbor as 1)
            contact_mat = np.zeros((unp_pfam_range_len,unp_pfam_range_len))
            diagIdx = kth_diag_indices(contact_mat,[-1,0,1])
            contact_mat[diagIdx] = 1
            # loop rows and cols
            for valReg_row in range(unp_pfam_range_len):
              for valReg_col in range(unp_pfam_range_len):
                unp_idx_row = unp_pfam_range[valReg_row]
                unp_idx_col = unp_pfam_range[valReg_col]
                if unp_idx_row in valid_unpIdx_list and unp_idx_col in valid_unpIdx_list:
                  # get pdb natural seq idx(1-idxed), then author-defined index
                  resIdx_pdbSeq_row = unp_pdb_seqIdx_mapping[unp_idx_row]
                  resIdx_pdbSeq_col = unp_pdb_seqIdx_mapping[unp_idx_col]
                  resIdx_pdbAuth_row = str(auth_pdbSeq_mapping[resIdx_pdbSeq_row-1])
                  resIdx_pdbAuth_col = str(auth_pdbSeq_mapping[resIdx_pdbSeq_col-1])
                  if 'CB' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_row]]:
                    atomNm_row = 'CB'
                  elif 'CA' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_row]]:
                    atomNm_row = 'CA'
                  elif 'CB1' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_row]]:
                    atomNm_row = 'CB1'
                  elif 'CA1' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_row]]:
                    atomNm_row = 'CA1'
                  else:
                    '''
                    # debug
                    print(resiId_dict[resIdx_pdbAuth_row])
                    for atm in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_row]]:
                      print(atm.get_name)
                    '''
                    #raise Exception("No atoms CB,CA,CB1,CA1")
                    noExpAtom_pdb.append([set_nm,pdbId,pdbChain,resiId_dict[resIdx_pdbAuth_row]])
                    # set as no-contact if CB,CA,CB1,CA1 not exist
                    continue

                  if 'CB' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_col]]:
                    atomNm_col = 'CB'
                  elif 'CA' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_col]]:
                    atomNm_col = 'CA'
                  elif 'CB1' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_col]]:
                    atomNm_col = 'CB1'
                  elif 'CA1' in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_col]]:
                    atomNm_col = 'CA1'
                  else:
                    '''
                    # debug
                    print(resiId_dict[resIdx_pdbAuth_col])
                    for atm in pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_col]]:
                      print(atm.get_name)
                    '''
                    #raise Exception("No atoms CB,CA,CB1,CA1")
                    noExpAtom_pdb.append([set_nm,pdbId,pdbChain,resiId_dict[resIdx_pdbAuth_col]])
                    # set as no-contact if CB,CA,CB1,CA1 not exist
                    continue

                  #print('{},{}'.format(resiId_dict[resIdx_pdbAuth_row],resiId_dict[resIdx_pdbAuth_col]))
                  if pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_row]][atomNm_row]-pdb_model[pdbChain][resiId_dict[resIdx_pdbAuth_col]][atomNm_col] <= cutoff:
                    contact_mat[valReg_row][valReg_col] = 1
                  else:
                    continue
            #build json
            tmp_data_dict['contact-map'] = contact_mat
            packed_data.append(tmp_data_dict)
            print('>*ADDED*<')
          else:
            noValidReg_pdb.append([set_nm,pdbId,pdbChain])
    else: # use predicted structure
      # build json
      tmp_data_dict = {} 
      tmp_data_dict['pfamAcc'] = 'PF'
      tmp_data_dict['unpAcc'] = unpAcc
      unp_pfam_range = range(wtSeq_unp_start, wtSeq_unp_end+1) # index should not exceed unp seq length
      unp_pfam_range_len = len(unp_pfam_range)
      valid_unpIdx_list = unp_pfam_range
      assert len(wt_seq) == unp_pfam_range_len
      tmp_data_dict['set_nm'] = set_nm
      tmp_data_dict['unp_pfam_range'] = '{}-{}'.format(unp_pfam_range[0],unp_pfam_range[-1])
      tmp_data_dict['target_seq'] = wt_seq
      tmp_data_dict['targetSeq_len'] = unp_pfam_range_len
      tmp_data_dict['best_pdb'] = pdbId
      tmp_data_dict['chain_id'] = pdbChain
      tmp_data_dict['valid_unpIdxs_len'] = len(valid_unpIdx_list)
      tmp_data_dict['valid_unpIdxs'] = list(valid_unpIdx_list)
      
      valid_mask = [True]*unp_pfam_range_len
      tmp_data_dict['valid_mask'] = valid_mask 

      # generate contact-map (with self-self, self-neighbor as 1)
      contact_mat = np.zeros((unp_pfam_range_len,unp_pfam_range_len))
      diagIdx = kth_diag_indices(contact_mat,[-1,0,1])
      contact_mat[diagIdx] = 1
      chain_id = 'A'
      pdb_struc = bio_pdbParser.get_structure('model1','{}/wt_seq_structure/{}_trR_results/model1.pdb'.format(working_dir,set_nm))
      pdb_model = pdb_struc[0]
      # build residue id dict
      resiId_dict={}
      for resi_obj in pdb_model[chain_id]:
        resiId_tuple = resi_obj.get_id()
        if resiId_tuple[2] != ' ':
          resiId_dict['{}{}'.format(resiId_tuple[1],resiId_tuple[2])] = resiId_tuple
        else:
          resiId_dict['{}'.format(resiId_tuple[1])] = resiId_tuple

      # loop rows and cols
      for valReg_row in range(unp_pfam_range_len):
        for valReg_col in range(unp_pfam_range_len):
          resIdx_pdbAuth_row = str(valReg_row+1)
          resIdx_pdbAuth_col = str(valReg_col+1)
          if 'CB' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_row]]:
            atomNm_row = 'CB'
          elif 'CA' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_row]]:
            atomNm_row = 'CA'
          elif 'CB1' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_row]]:
            atomNm_row = 'CB1'
          elif 'CA1' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_row]]:
            atomNm_row = 'CA1'
          else:
            '''
            # debug
            print(resiId_dict[resIdx_pdbAuth_row])
            for atm in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_row]]:
              print(atm.get_name)
            '''
            #raise Exception("No atoms CB,CA,CB1,CA1")
            noExpAtom_pdb.append([set_nm,pdbId,pdbChain,resiId_dict[resIdx_pdbAuth_row]])
            # set as no-contact if CB,CA,CB1,CA1 not exist
            continue

          if 'CB' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]]:
            atomNm_col = 'CB'
          elif 'CA' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]]:
            atomNm_col = 'CA'
          elif 'CB1' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]]:
            atomNm_col = 'CB1'
          elif 'CA1' in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]]:
            atomNm_col = 'CA1'
          else:
            '''
            # debug
            print(resiId_dict[resIdx_pdbAuth_col])
            for atm in pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]]:
              print(atm.get_name)
            '''
            #raise Exception("No atoms CB,CA,CB1,CA1")
            noExpAtom_pdb.append([set_nm,pdbId,pdbChain,resiId_dict[resIdx_pdbAuth_row]])
            # set as no-contact if CB,CA,CB1,CA1 not exist
            continue

          #print('{},{}'.format(resiId_dict[resIdx_pdbAuth_row],resiId_dict[resIdx_pdbAuth_col]))
          if pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_row]][atomNm_row]-pdb_model[chain_id][resiId_dict[resIdx_pdbAuth_col]][atomNm_col] <= cutoff:
            contact_mat[valReg_row][valReg_col] = 1
          else:
            continue
      #build json
      tmp_data_dict['contact-map'] = contact_mat
      packed_data.append(tmp_data_dict)
      print('>*ADDED*<')
  print('>>Exception occurs: noRsp_pdb-{},noAlign_pdb-{},noValidReg_pdb-{},noAsymId_pdb-{},wtNotUnp-{}'.format(len(noRsp_pdb),len(noAlign_pdb),len(noValidReg_pdb),len(noAsymId_pdb),len(wtNotUnp)))
  # save all data
  print('>>total {} seqs'.format(len(packed_data)))
  with open('{}/wt_seq_structure/allData_lenCut_l8h500_wt.json'.format(working_dir),'w') as fl:
    json.dump(packed_data,fl,cls=NumpyArrayEncoder)
  
  wrtDir = '{}/wt_seq_structure/allData_lenCut_l8h500_wt.lmdb'.format(working_dir)
  map_size = (1024 * 10) * (2 ** 20) # 10G
  wrtEnv = lmdb.open(wrtDir, map_size=map_size)
  with wrtEnv.begin(write=True) as txn:
    for i, entry in enumerate(packed_data):
      txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i + 1))
  wrtEnv.close() 
  
  np.savetxt('{}/wt_seq_structure/contact_NoRsps.csv'.format(working_dir),noRsp_pdb,fmt='%s',delimiter=',')
  np.savetxt('{}/wt_seq_structure/contact_NoAligns.csv'.format(working_dir),noAlign_pdb,fmt='%s',delimiter=',')
  np.savetxt('{}/wt_seq_structure/contact_NoValids.csv'.format(working_dir),noValidReg_pdb,fmt='%s',delimiter=',')
  np.savetxt('{}/wt_seq_structure/contact_NoAsymIds.csv'.format(working_dir),noAsymId_pdb,fmt='%s',delimiter=',')
  np.savetxt('{}/wt_seq_structure/contact_NoExpAtoms.csv'.format(working_dir),noExpAtom_pdb,fmt='%s',delimiter=';')
  np.savetxt('{}/wt_seq_structure/contact_wtNotUnp.csv'.format(working_dir),wtNotUnp,fmt='%s',delimiter=',')

def mut_fig():
  # params
  working_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  data_name = 'mutagenesisData'
  config_set = {'con': ['_2_0'],
                'nonCon': ['_2_0'],
                'ce': ['_2_0'],
                'pretrain': ['']}
  init_epoch = '20'
  rp_set = 'rp15_all'
  set_list = np.loadtxt('{}/data_process/mutagenesis/DeepSequenceMutaSet_flList'.format(working_dir),dtype='str')
  set_order=['POL_HV1N5-CA_Ndungu2014','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Ostermeier2014','P84126_THETH_b0','BLAT_ECOLX_Palzkill2012','RL401_YEAST_Bolon2013','RASH_HUMAN_Kuriyan','B3VI55_LIPSTSTABLE','HG_FLU_Bloom2016','BG_STRSQ_hmmerbit','TIM_SULSO_b0','AMIE_PSEAE_Whitehead','BG505_env_Bloom2018','PABP_YEAST_Fields2013-singles','PABP_YEAST_Fields2013-doubles','TIM_THEMA_b0','KKA2_KLEPN_Mikkelsen2014','BF520_env_Bloom2018','YAP1_HUMAN_Fields2012-singles','MK01_HUMAN_Johannessen','UBC9_HUMAN_Roth2017','DLG4_RAT_Ranganathan2012','RL401_YEAST_Bolon2014','BLAT_ECOLX_Tenaillon2013','HSP82_YEAST_Bolon2016','RL401_YEAST_Fraser2016','PTEN_HUMAN_Fowler2018','GAL4_YEAST_Shendure2015','MTH3_HAEAESTABILIZED_Tawfik2015','IF1_ECOLI_Kishony','SUMO1_HUMAN_Roth2017','TPMT_HUMAN_Fowler2018','PA_FLU_Sun2015','BRCA1_HUMAN_BRCT','UBE4B_MOUSE_Klevit2013','HIS7_YEAST_Kondrashov2017','BRCA1_HUMAN_RING','B3VI55_LIPST_Whitehead2015','TPK1_HUMAN_Roth2017','parEparD_Laub2015_all','CALM1_HUMAN_Roth2017','POLG_HCVJF_Sun2014']
  df_list = []
  # load score in other modes
  for mode in ['con','nonCon', 'ce', 'pretrain']:
    for new_i in config_set[mode]:
      for test_set in ['holdout']:
        print('loading set: {}{}'.format(mode,new_i))
        log_fl = 'mutation_{}_{}_torch_eval_{}{}.{}.0.out'.format(rp_set,init_epoch,mode,new_i,test_set)
        print('>log file: {}'.format(log_fl))
        os.system("grep 'loading weights file' job_logs/{} | cut -d'/' -f12 > tmp_rec".format(log_fl))
        with open('tmp_rec', 'r') as f:
          tar_dir = f.read()[:-1]
        os.system("rm tmp_rec")
        print('>model dir:',tar_dir)
        print('>json file: results_metrics_{}_{}.json'.format(data_name,test_set))
        with open('{}/results_to_keep/{}/{}_mutation_models/{}/results_metrics_{}_{}.json'.format(working_dir,rp_set,rp_set,tar_dir,data_name,test_set),'r') as f:
          metric_json = json.load(f)
        for set_nm in set_list:
          data_num = metric_json[set_nm+'_num']
          mse = metric_json[set_nm+'_mse']
          spearmanr = metric_json[set_nm+'_spearmanr']
          df_list.append([mode,rp_set,init_epoch,new_i,test_set,set_nm,data_num,mse,np.abs(spearmanr)])
  df = pd.DataFrame(df_list,columns=['mode','rp_nm','init_epoch','config_set','test_set','set_nm','data_num','mse','spearmanr'])

  # draw point plot
  filter_df = df.loc[(df["test_set"]=='holdout') & (df["rp_nm"]=='rp15_all') & (df["init_epoch"]=='20')]
  sns.set(style="whitegrid", rc={"lines.linewidth": 1.0, 'figure.figsize':(120,80)}, font_scale=6)
  gax = sns.pointplot(x="set_nm",y="spearmanr",hue="mode",data=filter_df,join=False,scale=8,
                      ci=None,dodge=False,order=set_order,hue_order=['pretrain','ce','nonCon','con'])
  gax.set_xticklabels(gax.get_xticklabels(), rotation=270)
  tar_fig_dir = 'mut'
  if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir)):
    os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,tar_fig_dir))
  plt.savefig('{}/results_to_keep/figures/{}/spearmanr_all_sets.png'.format(working_dir,tar_fig_dir))
  plt.clf()

def mut_precision_fitnessSV_fig():
  '''
  * delta precision vs delta spearmanr
  * delta precision vs delta mse
  '''

  # params
  working_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  data_name = 'mutagenesisData'
  data_name_wt = 'wt_seq_structure_wt'
  config_set = {'con': ['_2_0'],
                'nonCon': ['_2_0'],
                'ce': ['_2_0'],
                'pretrain': ['_0']}
  init_epoch = '20'
  rp_set = 'rp15_all'
  set_list = np.loadtxt('{}/data_process/mutagenesis/DeepSequenceMutaSet_flList'.format(working_dir),dtype='str')
  set_order=['POL_HV1N5-CA_Ndungu2014','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Ostermeier2014','P84126_THETH_b0','BLAT_ECOLX_Palzkill2012','RL401_YEAST_Bolon2013','RASH_HUMAN_Kuriyan','B3VI55_LIPSTSTABLE','HG_FLU_Bloom2016','BG_STRSQ_hmmerbit','TIM_SULSO_b0','AMIE_PSEAE_Whitehead','BG505_env_Bloom2018','PABP_YEAST_Fields2013-singles','PABP_YEAST_Fields2013-doubles','TIM_THEMA_b0','KKA2_KLEPN_Mikkelsen2014','BF520_env_Bloom2018','YAP1_HUMAN_Fields2012-singles','MK01_HUMAN_Johannessen','UBC9_HUMAN_Roth2017','DLG4_RAT_Ranganathan2012','RL401_YEAST_Bolon2014','BLAT_ECOLX_Tenaillon2013','HSP82_YEAST_Bolon2016','RL401_YEAST_Fraser2016','PTEN_HUMAN_Fowler2018','GAL4_YEAST_Shendure2015','MTH3_HAEAESTABILIZED_Tawfik2015','IF1_ECOLI_Kishony','SUMO1_HUMAN_Roth2017','TPMT_HUMAN_Fowler2018','PA_FLU_Sun2015','BRCA1_HUMAN_BRCT','UBE4B_MOUSE_Klevit2013-singles','HIS7_YEAST_Kondrashov2017','BRCA1_HUMAN_RING','B3VI55_LIPST_Whitehead2015','TPK1_HUMAN_Roth2017','parEparD_Laub2015_all','CALM1_HUMAN_Roth2017','POLG_HCVJF_Sun2014']
  df_list = []
  # load scores
  # * apc precision@all,short,medium,long; L1,2,5
  # * spearmanr
  # * mse
  for mode in ['con','nonCon', 'ce', 'pretrain']:
    for new_i in config_set[mode]:
      for test_set in ['holdout']:
        print('**loading set: {}{}>>'.format(mode,new_i))
        log_fl = 'mutation_{}_{}_torch_eval_{}{}.{}.0.out'.format(rp_set,init_epoch,mode,new_i,test_set)
        print('>log file: {}'.format(log_fl))
        os.system("grep 'loading weights file' job_logs/{} | cut -d'/' -f12 > tmp_rec".format(log_fl))
        with open('tmp_rec', 'r') as f:
          tar_dir = f.read()[:-1]
        os.system("rm tmp_rec")
        print('>model dir:',tar_dir)
        print('>json file: results_metrics_{}_{}.json'.format(data_name,test_set))
        with open('{}/results_to_keep/{}/{}_mutation_models/{}/results_metrics_{}_{}.json'.format(working_dir,rp_set,rp_set,tar_dir,data_name,test_set),'r') as f:
          metric_json = json.load(f)
        with open('{}/results_to_keep/{}/{}_mutation_models/{}/results_metrics_{}.json'.format(working_dir,rp_set,rp_set,tar_dir,data_name_wt),'r') as f:
          metric_wt_json = json.load(f)
        for set_nm in set_list:
          data_num = metric_json[set_nm+'_num']
          mse = metric_json[set_nm+'_mse']
          spearmanr = metric_json[set_nm+'_spearmanr']
          for topK in ['1','2','5']:
            for ran in ['all','short','medium','long']:
              prec_arr = np.array(metric_wt_json['{}_apc_precision_{}_{}_wt'.format(set_nm,ran,topK)])
              for lay in range(prec_arr.shape[0]):
                for hea in range(prec_arr.shape[1]):
                  head_idx = hea + 1
                  layer_idx = lay + 1
                  prec_val = prec_arr[lay][hea]
                  df_list.append([mode,rp_set,init_epoch,new_i,test_set,set_nm,data_num,mse,np.abs(spearmanr),'L/'+topK,ran,layer_idx,head_idx,prec_val])
  df = pd.DataFrame(df_list,columns=['mode','rp_nm','init_epoch','config_set','test_set','set_nm','data_num','mse','spearmanr','topK','range','layer_idx','head_idx','precision'])

  # plot delta-prec vs delta-spearmanr
  score_dict = {}
  for mode_i in ['ce','con','nonCon']:
    for conf_i in config_set[mode_i]:
      for topK_i in ['1','2','5']:
        precS2plot = []
        precM2plot = []
        precL2plot = []
        spr2plot = []
        mse2plot = []
        for set_i in set_order:
          df_uniq = df.drop_duplicates(subset=['mode','rp_nm','init_epoch','config_set','test_set','set_nm','data_num','mse','spearmanr'])
          df_filter = df_uniq.loc[(df_uniq["test_set"]=='holdout') & 
                                  (df_uniq["rp_nm"]=='rp15_all') & 
                                  (df_uniq["init_epoch"]=='20') &
                                  (df_uniq["mode"]==mode_i) &
                                  (df_uniq["config_set"]==conf_i) &
                                  (df_uniq["set_nm"]==set_i)]
          #print(df_filter["spearmanr"].values[0])
          
          ## pretrain
          df_filter_pretrain = df_uniq.loc[(df_uniq["test_set"]=='holdout') & 
                                  (df_uniq["rp_nm"]=='rp15_all') & 
                                  (df_uniq["init_epoch"]=='20') &
                                  (df_uniq["mode"]=='pretrain') &
                                  (df_uniq["set_nm"]==set_i)]

          
          try:
            assert((len(df_filter.index) == 1) & (len(df_filter_pretrain.index)==1))
          except:
            Exception('{}\n{}'.format(df_filter,df_filter_pretrain))

          spr2plot.append(df_filter["spearmanr"].values[0] - df_filter_pretrain["spearmanr"].values[0])
          mse2plot.append(df_filter["mse"].values[0] - df_filter_pretrain["mse"].values[0])


          df_uniq = df.drop_duplicates(subset=['mode','rp_nm','init_epoch','config_set','test_set','set_nm','topK','range','layer_idx','head_idx','precision'])
          df_filter = df_uniq.loc[(df_uniq["test_set"]=='holdout') & 
                                  (df_uniq["rp_nm"]=='rp15_all') & 
                                  (df_uniq["init_epoch"]=='20') &
                                  (df_uniq["mode"]==mode_i) &
                                  (df_uniq["config_set"]==conf_i) &
                                  (df_uniq["set_nm"]==set_i) &
                                  (df_uniq["layer_idx"]==4) &
                                  (df_uniq["topK"]=='L/'+topK_i)]
          
          df_filter_pretrain = df_uniq.loc[(df_uniq["test_set"]=='holdout') & 
                                  (df_uniq["rp_nm"]=='rp15_all') & 
                                  (df_uniq["init_epoch"]=='20') &
                                  (df_uniq["mode"]=='pretrain') &
                                  (df_uniq["set_nm"]==set_i) &
                                  (df_uniq["layer_idx"]==4) &
                                  (df_uniq["topK"]=='L/'+topK_i)]


          #print(df_filter.loc[(df_filter["range"]=='long')]["precision"].replace(0.0,np.nan))
          #precS2plot.append(df_filter.loc[(df_filter["range"]=='short')]["precision"].replace(0.0,np.nan).mean())
          #precM2plot.append(df_filter.loc[(df_filter["range"]=='medium')]["precision"].replace(0.0,np.nan).mean())
          #precL2plot.append(df_filter.loc[(df_filter["range"]=='long')]["precision"].replace(0.0,np.nan).mean())

          try:
            assert(len(df_filter.loc[(df_filter["range"]=='short')].index) == 8)
            assert(len(df_filter_pretrain.loc[(df_filter_pretrain["range"]=='short')].index) == 8)
          except:
            Exception('check this spot')

          precS2plot.append(df_filter.loc[(df_filter["range"]=='short')]["precision"].mean()
                             - df_filter_pretrain.loc[(df_filter_pretrain["range"]=='short')]["precision"].mean())
          precM2plot.append(df_filter.loc[(df_filter["range"]=='medium')]["precision"].mean()
                             - df_filter_pretrain.loc[(df_filter_pretrain["range"]=='medium')]["precision"].mean())
          precL2plot.append(df_filter.loc[(df_filter["range"]=='long')]["precision"].mean()
                             - df_filter_pretrain.loc[(df_filter_pretrain["range"]=='long')]["precision"].mean())
        ## append to score_dict
        score_dict['{}_{}_{}_precS'.format(mode_i,conf_i,topK_i)]=precS2plot
        score_dict['{}_{}_{}_precM'.format(mode_i,conf_i,topK_i)]=precM2plot
        score_dict['{}_{}_{}_precL'.format(mode_i,conf_i,topK_i)]=precL2plot
        score_dict['{}_{}_{}_spr'.format(mode_i,conf_i,topK_i)]=spr2plot
        score_dict['{}_{}_{}_mse'.format(mode_i,conf_i,topK_i)]=mse2plot

        #print('spr2plot',len(spr2plot),spr2plot)
        #print('mse2plot',len(mse2plot),mse2plot)
        #print('precS2plot',len(precS2plot),precS2plot)
        #print('precM2plot',len(precM2plot),precM2plot)
        #print('precL2plot',len(precL2plot),precL2plot)
        
        ''' 
        ## fig: x-set_nm
        fig, host = plt.subplots(figsize=(30,15))
        par1 = host.twinx()
        par2 = host.twinx()

        #host.set_ylim(0, 2)
        #par1.set_ylim(0, 4)
        #par2.set_ylim(1, 65)

        host.set_xlabel("setName")
        host.set_ylabel("deltaPrecision")
        par1.set_ylabel("deltaSpearmanR")
        par2.set_ylabel("deltaMSE")

        x_tick = np.arange(1,2+1*(len(set_order)-1),1)
        p1, = host.plot(x_tick,precS2plot, marker='1', markersize=10, color='r', linestyle = 'None',label="short")
        p1_1, = host.plot(x_tick,precM2plot, marker='+', markersize=10, color=p1.get_color(), linestyle = 'None', label="medium")
        p1_2, = host.plot(x_tick,precL2plot, marker='^', markersize=10, color=p1.get_color(), linestyle = 'None', label="long")
        p2, = par1.plot(x_tick,spr2plot, color='g', marker='.', markersize=10, linestyle = 'None', label="spearmanR")
        p3, = par2.plot(x_tick,mse2plot, color='b', marker='.', markersize=10, linestyle = 'None', label="mse")

        host.axhline(y=0, color='r', linestyle='--')
        par1.axhline(y=0, color='g', linestyle='--')
        par2.axhline(y=0, color='b', linestyle='--')

        lns = [p1,p1_1,p1_2,p2,p3]
        host.legend(handles=lns, loc='best')

        # right, left, top, bottom
        par2.spines['right'].set_position(('outward', 60))
        
        # no x-ticks                 
        par2.xaxis.set_ticks([])
        
        # Sometimes handy, same for xaxis
        #par2.yaxis.set_ticks_position('right')
        
        # Move "Velocity"-axis to the left
        # par2.spines['left'].set_position(('outward', 60))
        # par2.spines['left'].set_visible(True)
        # par2.yaxis.set_label_position('left')
        # par2.yaxis.set_ticks_position('left')
        
        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        par2.yaxis.label.set_color(p3.get_color())
        
        # You can specify a rotation for the tick labels in degrees or with keywords.
        plt.xticks(x_tick)
        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.2)
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.15)

        # Adjust spacings w.r.t. figsize
        fig.tight_layout()
        # Alternatively: bbox_inches='tight' within the plt.savefig function 
        #                (overwrites figsize)
        
        host.grid(which='major', axis='x', linestyle='--')
        
        # Best for professional typesetting, e.g. LaTeX
        plt.savefig('{}/results_to_keep/figures/mut/delta_prec_R_mse_{}{}_L{}.png'.format(working_dir,mode_i,conf_i,topK_i))
        plt.clf()
        # For raster graphics use the dpi argument. E.g. '[...].png",dpi=200)'
        '''
        
        ## x, y both delta-score
        prec_ran_list = [precS2plot,precM2plot,precL2plot]
        ran_list = ['short','medium','long']
        for ran_i in range(3):
          ## R
          fig, host = plt.subplots()
          host.axhline(y=0, linestyle='--', color='gray')
          host.axvline(x=0, linestyle='--', color='gray')
          host.scatter(prec_ran_list[ran_i], spr2plot, s=1)
          for i, txt in enumerate(range(1,len(set_order)+1)):
            host.annotate(txt, (prec_ran_list[ran_i][i], spr2plot[i]), fontsize=7)
          plt.xlabel('delta-precision', fontsize=10)
          plt.ylabel('delta-R', fontsize=10)
          host.set_title('xyDelta_prec_R_{}{}_L{}_{}'.format(mode_i,conf_i,topK_i,ran_list[ran_i]))
          fig.savefig('{}/results_to_keep/figures/mut/xyDelta_prec_R_{}{}_L{}_{}.png'.format(working_dir,mode_i,conf_i,topK_i,ran_list[ran_i]))
          plt.close(fig)
          ## mse
          fig, host = plt.subplots()
          host.axhline(y=0, linestyle='--', color='gray')
          host.axvline(x=0, linestyle='--', color='gray')
          host.scatter(prec_ran_list[ran_i], mse2plot, s=1)
          for i, txt in enumerate(range(1,len(set_order)+1)):
            host.annotate(txt, (prec_ran_list[ran_i][i], mse2plot[i]), fontsize=7)
          plt.xlabel('delta-precision', fontsize=10)
          plt.ylabel('delta-mse', fontsize=10)
          host.set_title('xyDelta_prec_mse_{}{}_L{}_{}'.format(mode_i,conf_i,topK_i,ran_list[ran_i]))
          fig.savefig('{}/results_to_keep/figures/mut/xyDelta_prec_mse_{}{}_L{}_{}.png'.format(working_dir,mode_i,conf_i,topK_i,ran_list[ran_i]))
          plt.close(fig)
          
  '''
  ## con vs nonCon vs ce, out of loop
  for method_pair in ['con_nonCon','con_ce', 'nonCon_ce']:
    x_methd, y_methd = re.split(r'_',method_pair)
    for conf_x in config_set[x_methd]:
      for conf_y in config_set[y_methd]:
        print('{}{}; {}{}'.format(x_methd,conf_x,y_methd,conf_y))
        for topK_i in ['1', '2', '5']:
          fig, host = plt.subplots()
          host.scatter(score_dict['{}_{}_{}_spr'.format(x_methd,conf_x,topK_i)],
              score_dict['{}_{}_{}_spr'.format(y_methd,conf_y,topK_i)], s=1)
          for i, txt in enumerate(range(1,len(set_order)+1)):               
            host.annotate(txt, (score_dict['{}_{}_{}_spr'.format(x_methd,conf_x,topK_i)][i],
              score_dict['{}_{}_{}_spr'.format(y_methd,conf_y,topK_i)][i]), fontsize=7)
          plt.xlabel('delta-spr({}{})'.format(x_methd,conf_x), fontsize=10)
          plt.ylabel('delta-spr({}{})'.format(y_methd,conf_y), fontsize=10)
          lims = [
                  np.min([host.get_xlim(), host.get_ylim()]),  # min of both axes
                  np.max([host.get_xlim(), host.get_ylim()]),  # max of both axes
                 ]

          # now plot both limits against eachother
          host.plot(lims, lims, ls='--', color='gray', alpha=0.75, zorder=0)
          host.set_aspect('equal')
          host.set_xlim(lims)
          host.set_ylim(lims) 
          host.axhline(y=0, linestyle='--', color='gray', zorder=0)
          host.axvline(x=0, linestyle='--', color='gray', zorder=0)
          host.set_title('xyMethod_{}{}-{}{}_L{}'.format(x_methd,conf_x,y_methd,conf_y,topK_i))
          fig.savefig('{}/results_to_keep/figures/mut/xyMethod_{}{}-{}{}_L{}.png'.format(working_dir,x_methd,conf_x,y_methd,conf_y,topK_i))
          plt.close(fig)
  '''

def mut_precision_fitnessUNSV_fig():
  '''
  evaluation of unsupervise predicted fitness scores

  1st set of fitures
  structure awareness change vs fitness performance change
  * x-delta precision vs y-delta spearmanr
  * x-delta precision vs y-delta pearsonr
  
  2nd set of figures
  spearmanr of each set, compare across three AS methods, pretrain
  * x-mutagenesis set, y-spearmanr, hue-method
  
  3rd set of figures
  spearman of each set (pretrain vs con vs nonCon vs ce)
  * x-spearman of method 1,y-spearman of method 2, z-two method combination
  '''

  # params
  working_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  data_name_wt = 'wt_seq_structure_wt'
  config_set = {'con': ['_1_1'],
                'nonCon': ['_1_0'],
                'ce': ['_1_8'],
                'pretrain': ['_0']}
  init_epoch = '700'
  rp_set = 'rp15_all'
  rp_set_sub = '_1'
  set_list = np.loadtxt('{}/data_process/mutagenesis/DeepSequenceMutaSet_flList'.format(working_dir),dtype='str')
  set_order=['POL_HV1N5-CA_Ndungu2014','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Ostermeier2014','P84126_THETH_b0','BLAT_ECOLX_Palzkill2012','RL401_YEAST_Bolon2013','RASH_HUMAN_Kuriyan','B3VI55_LIPSTSTABLE','HG_FLU_Bloom2016','BG_STRSQ_hmmerbit','TIM_SULSO_b0','AMIE_PSEAE_Whitehead','BG505_env_Bloom2018','PABP_YEAST_Fields2013-singles','PABP_YEAST_Fields2013-doubles','TIM_THEMA_b0','KKA2_KLEPN_Mikkelsen2014','BF520_env_Bloom2018','YAP1_HUMAN_Fields2012-singles','MK01_HUMAN_Johannessen','UBC9_HUMAN_Roth2017','DLG4_RAT_Ranganathan2012','RL401_YEAST_Bolon2014','BLAT_ECOLX_Tenaillon2013','HSP82_YEAST_Bolon2016','RL401_YEAST_Fraser2016','PTEN_HUMAN_Fowler2018','GAL4_YEAST_Shendure2015','MTH3_HAEAESTABILIZED_Tawfik2015','IF1_ECOLI_Kishony','SUMO1_HUMAN_Roth2017','TPMT_HUMAN_Fowler2018','PA_FLU_Sun2015','BRCA1_HUMAN_BRCT','UBE4B_MOUSE_Klevit2013-singles','HIS7_YEAST_Kondrashov2017','BRCA1_HUMAN_RING','B3VI55_LIPST_Whitehead2015','TPK1_HUMAN_Roth2017','parEparD_Laub2015_all','CALM1_HUMAN_Roth2017','POLG_HCVJF_Sun2014']
  mode_blc_list = ['_blc', '_blc', '_blc_gm2.0', '']
  mode_list = ['con','nonCon', 'ce', 'pretrain']
  mdl_path_blc = '_blc'
  strethW = '1'
  n_layer = 12
  n_head = 12
  fig_save_dir = 'mut_{}{}@{}{}'.format(rp_set,rp_set_sub,init_epoch,mdl_path_blc)
  if not os.path.isdir('{}/results_to_keep/figures/{}'.format(working_dir,fig_save_dir)):
    os.mkdir('{}/results_to_keep/figures/{}'.format(working_dir,fig_save_dir))
  
  df_list = []
  # load scores
  # * apc precision@all,short,medium,long; L1,2,5
  # * spearmanr, pearsonr
  for mode_i in range(len(mode_list)):
    mode = mode_list[mode_i]
    mode_blc = mode_blc_list[mode_i]
    for new_i in config_set[mode]:
      print('**loading set: {}{}>>'.format(mode,new_i))
      log_fl = 'mutation_{}{}_{}_torch_eval_{}{}_wtPrecision.wt.0.out'.format(rp_set,rp_set_sub,init_epoch,mode,new_i)
      print('>log file: {}'.format(log_fl))
      
      if mode == 'pretrain':
        os.system("grep 'loading weights file' job_logs/archive_mutation_bert_eval/{} | cut -d'/' -f12 > tmp_rec".format(log_fl))
      else:
        os.system("grep 'loading weights file' job_logs/archive_mutation_bert_eval/{} | cut -d'/' -f13 > tmp_rec".format(log_fl))
      
      with open('tmp_rec', 'r') as f:
        tar_dir = f.read()[:-1]
      os.system("rm tmp_rec")
      print('>model dir:',tar_dir)
      if mode == 'pretrain':
        mdl_path = '{rp_set}_pretrain{rp_set_sub}_models'.format(rp_set=rp_set,rp_set_sub=rp_set_sub)
        wtPrec_fl = 'results_metrics_{}_{}'.format(data_name_wt,int(init_epoch)-1)
      else:
        mdl_path = '{rp_set}{rp_set_sub}_{init_epoch}_con_non_ce{mdl_path_blc}_strethWeight{strethW}_models/{mode}{mode_blc}'.format(
            rp_set=rp_set,rp_set_sub=rp_set_sub,init_epoch=init_epoch,mdl_path_blc=mdl_path_blc,strethW=strethW,mode=mode,mode_blc=mode_blc)
        wtPrec_fl = 'results_metrics_{}'.format(data_name_wt)
      ## load unSV fitness
      metric_unSVfit_json_dict = {}
      for set_nm in set_list:
        with open('{}/results_to_keep/{}/{}/{}/mutagenesis_fitness_unSV/{}.json'.format(working_dir,rp_set,mdl_path,tar_dir,set_nm,'r')) as f:
          metric_unSVfit_json = json.load(f)
        metric_unSVfit_json_dict[set_nm] = metric_unSVfit_json
      ## load wt precision
      with open('{}/results_to_keep/{}/{}/{}/{}.json'.format(working_dir,rp_set,mdl_path,tar_dir,wtPrec_fl),'r') as f:
        metric_wtPrec_json = json.load(f)
      for set_nm in set_list:
        data_num = metric_unSVfit_json_dict[set_nm]['mut_num']
        spearmanr = metric_unSVfit_json_dict[set_nm]['spearmanR']
        spearmanpval = metric_unSVfit_json_dict[set_nm]['spearmanPvalue']
        pearsonr = metric_unSVfit_json_dict[set_nm]['pearsonR']
        pearsonpval = metric_unSVfit_json_dict[set_nm]['pearsonPvalue']
        for topK in ['1','2','5']:
          for ran in ['all','short','medium','long']:
            prec_arr = np.array(metric_wtPrec_json['{}_apc_precision_{}_{}_wt'.format(set_nm,ran,topK)])
            for lay in range(prec_arr.shape[0]):
              for hea in range(prec_arr.shape[1]):
                head_idx = hea + 1
                layer_idx = lay + 1
                prec_val = prec_arr[lay][hea]
                df_list.append([mode,rp_set,init_epoch,new_i,set_nm,data_num,spearmanr,np.abs(spearmanr),spearmanpval,pearsonr,np.abs(pearsonr),pearsonpval,'L/'+topK,ran,layer_idx,head_idx,prec_val])
  df = pd.DataFrame(df_list,columns=['mode','rp_nm','init_epoch','config_set','set_nm','data_num','spearmanr_ori','spearmanr','spearmanpval','pearsonr_ori','pearsonr','pearsonpval','topK','range','layer_idx','head_idx','precision'])

  # plot delta-prec vs delta-spearmanr/pearsonr
  score_dict = {}
  fig_point_count = []
  for mode_i in ['ce','con','nonCon']:
    for conf_i in config_set[mode_i]:
      for topK_i in ['1','2','5']:
        # value: [pretrain, AS]
        precA2plot = []        
        precS2plot = []
        precM2plot = []
        precL2plot = []
        spr2plot = []
        pear2plot = []
        
        for set_i in set_order:
          ## unSVfit
          df_uniq = df.drop_duplicates(subset=['mode','rp_nm','init_epoch','config_set','set_nm','data_num','spearmanr_ori','spearmanr','spearmanpval','pearsonr_ori','pearsonr','pearsonpval'])
          # AS models
          df_filter_AS = df_uniq.loc[(df_uniq["rp_nm"]=='rp15_all') & 
                                  (df_uniq["init_epoch"]==init_epoch) &
                                  (df_uniq["mode"]==mode_i) &
                                  (df_uniq["config_set"]==conf_i) &
                                  (df_uniq["set_nm"]==set_i)]
          #print(df_filter["spearmanr"].values[0])
          
          ## pretrain
          df_filter_pretrain = df_uniq.loc[(df_uniq["rp_nm"]=='rp15_all') & 
                                           (df_uniq["init_epoch"]==init_epoch) &
                                           (df_uniq["mode"]=='pretrain') &
                                           (df_uniq["set_nm"]==set_i)]

          
          try:
            assert((len(df_filter_AS.index) == 1) & (len(df_filter_pretrain.index)==1))
          except:
            Exception('{}\n{}'.format(df_filter_AS,df_filter_pretrain))

          spr2plot.append([df_filter_pretrain["spearmanr"].values[0],df_filter_AS["spearmanr"].values[0]])
          pear2plot.append([df_filter_pretrain["pearsonr"].values[0],df_filter_AS["pearsonr"].values[0]])

          ## precision
          df_uniq = df.drop_duplicates(subset=['mode','rp_nm','init_epoch','config_set','set_nm','topK','range','layer_idx','head_idx','precision'])
          sel_last_lays = 3
          df_filter_AS = df_uniq.loc[(df_uniq["rp_nm"]=='rp15_all') & 
                                   (df_uniq["init_epoch"]==init_epoch) &
                                   (df_uniq["mode"]==mode_i) &
                                   (df_uniq["config_set"]==conf_i) &
                                   (df_uniq["set_nm"]==set_i) &
                                   (df_uniq["layer_idx"].isin(list(range(n_layer))[-sel_last_lays:])) &
                                   (df_uniq["topK"]=='L/'+topK_i)]
          
          df_filter_pretrain = df_uniq.loc[(df_uniq["rp_nm"]=='rp15_all') & 
                                  (df_uniq["init_epoch"]==init_epoch) &
                                  (df_uniq["mode"]=='pretrain') &
                                  (df_uniq["set_nm"]==set_i) &
                                  (df_uniq["layer_idx"].isin(list(range(n_layer))[-sel_last_lays:])) &
                                  (df_uniq["topK"]=='L/'+topK_i)]


          #print(df_filter.loc[(df_filter["range"]=='long')]["precision"].replace(0.0,np.nan))
          #precS2plot.append(df_filter.loc[(df_filter["range"]=='short')]["precision"].replace(0.0,np.nan).mean())
          #precM2plot.append(df_filter.loc[(df_filter["range"]=='medium')]["precision"].replace(0.0,np.nan).mean())
          #precL2plot.append(df_filter.loc[(df_filter["range"]=='long')]["precision"].replace(0.0,np.nan).mean())

          try:
            assert(len(df_filter_AS.loc[(df_filter_AS["range"]=='short')].index) == n_head*sel_last_lays)
            assert(len(df_filter_pretrain.loc[(df_filter_pretrain["range"]=='short')].index) == n_head*sel_last_lays)
          except:
            Exception('check spot 2')
          precA2plot.append([df_filter_pretrain.loc[(df_filter_pretrain["range"]=='all')]["precision"].mean(),df_filter_AS.loc[(df_filter_AS["range"]=='all')]["precision"].mean()])
          precS2plot.append([df_filter_pretrain.loc[(df_filter_pretrain["range"]=='short')]["precision"].mean(),df_filter_AS.loc[(df_filter_AS["range"]=='short')]["precision"].mean()])
          precM2plot.append([df_filter_pretrain.loc[(df_filter_pretrain["range"]=='medium')]["precision"].mean(),df_filter_AS.loc[(df_filter_AS["range"]=='medium')]["precision"].mean()])
          precL2plot.append([df_filter_pretrain.loc[(df_filter_pretrain["range"]=='long')]["precision"].mean(),df_filter_AS.loc[(df_filter_AS["range"]=='long')]["precision"].mean()])
          # ori value: as, pretrain
        precA2plot = np.array(precA2plot)
        precS2plot = np.array(precS2plot)
        precM2plot = np.array(precM2plot)
        precL2plot = np.array(precL2plot)
        spr2plot = np.array(spr2plot)
        pear2plot = np.array(pear2plot)
        #delta: AS-pretrain
        precA2plot_delta = precA2plot[:,1]-precA2plot[:,0]
        precS2plot_delta = precS2plot[:,1]-precS2plot[:,0]
        precM2plot_delta = precM2plot[:,1]-precM2plot[:,0]
        precL2plot_delta = precL2plot[:,1]-precL2plot[:,0]
        spr2plot_delta = spr2plot[:,1]-spr2plot[:,0]
        pear2plot_delta = pear2plot[:,1]-pear2plot[:,0]
        ## add to score_dict
        score_dict['{}_{}_{}_precA'.format(mode_i,conf_i,topK_i)]=precA2plot
        score_dict['{}_{}_{}_precS'.format(mode_i,conf_i,topK_i)]=precS2plot
        score_dict['{}_{}_{}_precM'.format(mode_i,conf_i,topK_i)]=precM2plot
        score_dict['{}_{}_{}_precL'.format(mode_i,conf_i,topK_i)]=precL2plot
        score_dict['{}_{}_{}_spr'.format(mode_i,conf_i,topK_i)]=spr2plot
        score_dict['{}_{}_{}_pear'.format(mode_i,conf_i,topK_i)]=pear2plot

        #print('spr2plot',len(spr2plot),spr2plot)
        #print('mse2plot',len(mse2plot),mse2plot)
        #print('precS2plot',len(precS2plot),precS2plot)
        #print('precM2plot',len(precM2plot),precM2plot)
        #print('precL2plot',len(precL2plot),precL2plot)
        
        ''' 
        ## fig: x-set_nm
        fig, host = plt.subplots(figsize=(30,15))
        par1 = host.twinx()
        par2 = host.twinx()

        #host.set_ylim(0, 2)
        #par1.set_ylim(0, 4)
        #par2.set_ylim(1, 65)

        host.set_xlabel("setName")
        host.set_ylabel("deltaPrecision")
        par1.set_ylabel("deltaSpearmanR")
        par2.set_ylabel("deltaMSE")

        x_tick = np.arange(1,2+1*(len(set_order)-1),1)
        p1, = host.plot(x_tick,precS2plot, marker='1', markersize=10, color='r', linestyle = 'None',label="short")
        p1_1, = host.plot(x_tick,precM2plot, marker='+', markersize=10, color=p1.get_color(), linestyle = 'None', label="medium")
        p1_2, = host.plot(x_tick,precL2plot, marker='^', markersize=10, color=p1.get_color(), linestyle = 'None', label="long")
        p2, = par1.plot(x_tick,spr2plot, color='g', marker='.', markersize=10, linestyle = 'None', label="spearmanR")
        p3, = par2.plot(x_tick,mse2plot, color='b', marker='.', markersize=10, linestyle = 'None', label="mse")

        host.axhline(y=0, color='r', linestyle='--')
        par1.axhline(y=0, color='g', linestyle='--')
        par2.axhline(y=0, color='b', linestyle='--')

        lns = [p1,p1_1,p1_2,p2,p3]
        host.legend(handles=lns, loc='best')

        # right, left, top, bottom
        par2.spines['right'].set_position(('outward', 60))
        
        # no x-ticks                 
        par2.xaxis.set_ticks([])
        
        # Sometimes handy, same for xaxis
        #par2.yaxis.set_ticks_position('right')
        
        # Move "Velocity"-axis to the left
        # par2.spines['left'].set_position(('outward', 60))
        # par2.spines['left'].set_visible(True)
        # par2.yaxis.set_label_position('left')
        # par2.yaxis.set_ticks_position('left')
        
        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        par2.yaxis.label.set_color(p3.get_color())
        
        # You can specify a rotation for the tick labels in degrees or with keywords.
        plt.xticks(x_tick)
        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.2)
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.15)

        # Adjust spacings w.r.t. figsize
        fig.tight_layout()
        # Alternatively: bbox_inches='tight' within the plt.savefig function 
        #                (overwrites figsize)
        
        host.grid(which='major', axis='x', linestyle='--')
        
        # Best for professional typesetting, e.g. LaTeX
        plt.savefig('{}/results_to_keep/figures/mut/delta_prec_R_mse_{}{}_L{}.png'.format(working_dir,mode_i,conf_i,topK_i))
        plt.clf()
        # For raster graphics use the dpi argument. E.g. '[...].png",dpi=200)'
        '''
        
        ##  * x-delta precision vs y-delta spearmanr
        ##  * x-delta precision vs y-delta pearsonr
        prec_ran_list = [precA2plot_delta,precS2plot_delta,precM2plot_delta,precL2plot_delta]
        ran_list = ['all','short','medium','long']
        for ran_i in range(len(ran_list)):
          ## spearmanr
          fig, host = plt.subplots()
          host.axhline(y=0, linestyle='--', color='gray')
          host.axvline(x=0, linestyle='--', color='gray')
          host.scatter(prec_ran_list[ran_i], spr2plot_delta, s=1)
          for i, txt in enumerate(range(1,len(set_order)+1)):
            host.annotate(txt, (prec_ran_list[ran_i][i], spr2plot_delta[i]), fontsize=8)
          plt.xlabel('delta-precision', fontsize=12)
          plt.ylabel('delta-spearmanr', fontsize=12)
          #host.set_title('xyDelta_prec_spear_{}{}_L{}_{}'.format(mode_i,conf_i,topK_i,ran_list[ran_i]))
          spear_figNm = 'xyDelta_prec_spearman_{}{}_L{}_{}'.format(mode_i,conf_i,topK_i,ran_list[ran_i])
          fig.savefig('{}/results_to_keep/figures/{}/{}.png'.format(working_dir,fig_save_dir,spear_figNm))
          plt.close(fig)
         
          ## spearmanr
          fig, host = plt.subplots()
          host.axhline(y=0, linestyle='--', color='gray')
          host.axvline(x=0, linestyle='--', color='gray')
          host.scatter(prec_ran_list[ran_i], pear2plot_delta, s=1)
          for i, txt in enumerate(range(1,len(set_order)+1)):
            host.annotate(txt, (prec_ran_list[ran_i][i], pear2plot_delta[i]), fontsize=8)
          plt.xlabel('delta-precision', fontsize=12)
          plt.ylabel('delta-spearmanr', fontsize=12)
          #host.set_title('xyDelta_prec_spear_{}{}_L{}_{}'.format(mode_i,conf_i,topK_i,ran_list[ran_i]))
          pears_figNm = 'xyDelta_prec_pearson_{}{}_L{}_{}'.format(mode_i,conf_i,topK_i,ran_list[ran_i])
          fig.savefig('{}/results_to_keep/figures/{}/{}.png'.format(working_dir,fig_save_dir,pears_figNm))
          plt.close(fig)
          
          ## count
          spear_tt = np.sum(np.logical_and((prec_ran_list[ran_i]>=0.),(spr2plot_delta>=0.)))
          spear_tf = np.sum(np.logical_and((prec_ran_list[ran_i]>=0.),(spr2plot_delta<0.)))
          spear_ft = np.sum(np.logical_and((prec_ran_list[ran_i]<0.),(spr2plot_delta>=0.)))
          spear_ff = np.sum(np.logical_and((prec_ran_list[ran_i]<0.),(spr2plot_delta<0.)))

          pears_tt = np.sum(np.logical_and((prec_ran_list[ran_i]>=0.),(pear2plot_delta>=0.)))
          pears_tf = np.sum(np.logical_and((prec_ran_list[ran_i]>=0.),(pear2plot_delta<0.)))
          pears_ft = np.sum(np.logical_and((prec_ran_list[ran_i]<0.),(pear2plot_delta>=0.)))
          pears_ff = np.sum(np.logical_and((prec_ran_list[ran_i]<0.),(pear2plot_delta<0.)))
          fig_point_count.append([spear_figNm,spear_tt,spear_tf,spear_ft,spear_ff])
          fig_point_count.append([pears_figNm,pears_tt,pears_tf,pears_ft,pears_ff])
        np.savetxt('{}/results_to_keep/figures/{}/figCount_deltaPrec_deltaCorr.csv'.format(working_dir,fig_save_dir),fig_point_count,fmt='%s',delimiter=',')


  ## con vs nonCon vs ce, 
  for method_pair in ['con_nonCon','con_ce','nonCon_ce','con_pretrain','nonCon_pretrain','ce_pretrain']:
    x_methd, y_methd = re.split(r'_',method_pair)
    for conf_x in config_set[x_methd]:
      for conf_y in config_set[y_methd]:
        print('{}{}; {}{}'.format(x_methd,conf_x,y_methd,conf_y))
        for topK_i in ['1', '2', '5']:
          for corr_nm in ['spr', 'pear']:
            if y_methd == 'pretrain':
              fig, host = plt.subplots()
              host.scatter(score_dict['{}_{}_{}_{}'.format(x_methd,conf_x,topK_i,corr_nm)][:,1],
                           score_dict['{}_{}_{}_{}'.format(x_methd,conf_x,topK_i,corr_nm)][:,0], s=1)
              for i, txt in enumerate(range(1,len(set_order)+1)):               
                host.annotate(txt,(score_dict['{}_{}_{}_{}'.format(x_methd,conf_x,topK_i,corr_nm)][i,1],
                                   score_dict['{}_{}_{}_{}'.format(x_methd,conf_x,topK_i,corr_nm)][i,0]),fontsize=8)
              plt.xlabel('{}'.format(x_methd), fontsize=12)
              plt.ylabel('{}'.format(y_methd), fontsize=12)
              lims = [
                      np.min([host.get_xlim(), host.get_ylim()]),  # min of both axes
                      np.max([host.get_xlim(), host.get_ylim()]),  # max of both axes
                     ]

              # now plot both limits against eachother
              host.plot(lims, lims, ls='--', color='gray', alpha=0.75, zorder=0)
              host.set_aspect('equal')
              host.set_xlim(lims)
              host.set_ylim(lims) 
              #host.axhline(y=0, linestyle='--', color='gray', zorder=0)
              #host.axvline(x=0, linestyle='--', color='gray', zorder=0)
              #host.set_title('xyMethod_{}{}-{}{}_L{}'.format(x_methd,conf_x,y_methd,conf_y,topK_i))
              fig.savefig('{}/results_to_keep/figures/{}/xyMethod_{}_{}{}-{}{}_L{}.png'.format(working_dir,fig_save_dir,corr_nm,x_methd,conf_x,y_methd,conf_y,topK_i))
              plt.close(fig)
            else:
              fig, host = plt.subplots()
              host.scatter(score_dict['{}_{}_{}_{}'.format(x_methd,conf_x,topK_i,corr_nm)][:,1],
                           score_dict['{}_{}_{}_{}'.format(y_methd,conf_y,topK_i,corr_nm)][:,1], s=1)
              for i, txt in enumerate(range(1,len(set_order)+1)):               
                host.annotate(txt,(score_dict['{}_{}_{}_{}'.format(x_methd,conf_x,topK_i,corr_nm)][i,1],
                                   score_dict['{}_{}_{}_{}'.format(y_methd,conf_y,topK_i,corr_nm)][i,1]),
                                   fontsize=8)
              plt.xlabel('{}'.format(x_methd), fontsize=12)
              plt.ylabel('{}'.format(y_methd), fontsize=12)
              lims = [
                      np.min([host.get_xlim(), host.get_ylim()]),  # min of both axes
                      np.max([host.get_xlim(), host.get_ylim()]),  # max of both axes
                     ]

              # now plot both limits against eachother
              host.plot(lims, lims, ls='--', color='gray', alpha=0.75, zorder=0)
              host.set_aspect('equal')
              host.set_xlim(lims)
              host.set_ylim(lims) 
              #host.axhline(y=0, linestyle='--', color='gray', zorder=0)
              #host.axvline(x=0, linestyle='--', color='gray', zorder=0)
              #host.set_title('xyMethod_{}{}-{}{}_L{}'.format(x_methd,conf_x,y_methd,conf_y,topK_i))
              fig.savefig('{}/results_to_keep/figures/{}/xyMethod_{}_{}{}-{}{}_L{}.png'.format(working_dir,fig_save_dir,corr_nm,x_methd,conf_x,y_methd,conf_y,topK_i))
              plt.close(fig)

def mut_epoch_fitnessUNSV_fig(working_dir: str = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark',
                              rpSet_list: List = ['rp15_all_1','rp15_all_2','rp15_all_3','rp15_all_4'],
                              pretrainMdlNm_list: List = ['masked_language_modeling_transformer_21-05-22-23-09-12_354660','masked_language_modeling_transformer_21-05-07-19-48-02_813698','masked_language_modeling_transformer_21-05-01-22-15-01_112209','masked_language_modeling_transformer_21-05-01-22-22-18_663653'],
                              mutaSetList_file: str = None,
                              epoch_list: int = None,
                              report_metric: str='spR'):
  """
  plot figure for pretrained models
    * x-epoch, y-spearmanR, unsupervised fitness evaluation
  """
  

  ## init a dict to store spearmanR
  '''
  print('>> load spearManR')
  spearR_dict = {}
  for mutaNm in muta_flList:
    print(f'>> {mutaNm}')
    spearR_dict[mutaNm] = []
    ## loop over each rp_mdl
    for rp_set in rpSet_list:
      print(f'>> {rp_set}')
      rp_split = re.split(r'_',rp_set)
      rpNm = f"{rp_split[0]}_{rp_split[1]}"
      mdl_subIdx = rp_split[2]
      mdl_list = os.popen(f"ls -d {working_dir}/results_to_keep/{rpNm}/{rpNm}_pretrain_{mdl_subIdx}_models/*").read().strip('\n').split('\n')
      spearR_rpSet_list = [] # hold spearmanR for one rp model
      for mdlNm in mdl_list:
        json_list = os.popen(f"ls -v {mdlNm}/mutagenesis_fitness_unSV/{mutaNm}*.json").read().strip('\n').split('\n')
        # loop over epochs in one folder
        for fl_json in json_list:
          with open(fl_json) as handle:
            metric_json = json.load(handle)
            spearR_rpSet_list.append(abs(metric_json['spearmanR']))
      spearR_dict[mutaNm].append(spearR_rpSet_list)
    
  ## save this json
  with open(f'{working_dir}/data_process/mutagenesis/pretrain_unSV_fitness_spearmanR.json','w') as handle:
    json.dump(spearR_dict,handle)
  '''
  '''
  ## load json
  with open(f'{working_dir}/data_process/mutagenesis/pretrain_unSV_fitness_spearmanR.json','r') as handle:
    spearR_dict = json.load(handle)
  ## draw figure
  print('>> fraw figures')
  for mutaNm in muta_flList:
    spearR_rpSet_list = spearR_dict[mutaNm]
    print(f'>>{mutaNm}:{np.amax(spearR_rpSet_list)}')
    fig, ax = plt.subplots()
    ax.plot(range(len(spearR_rpSet_list[0])),spearR_rpSet_list[0],label=rpSet_list[0])
    ax.plot(range(len(spearR_rpSet_list[1])),spearR_rpSet_list[1],label=rpSet_list[1])
    ax.plot(range(len(spearR_rpSet_list[2])),spearR_rpSet_list[2],label=rpSet_list[2])
    ax.plot(range(len(spearR_rpSet_list[3])),spearR_rpSet_list[3],label=rpSet_list[3])
    plt.legend()
    plt.savefig(f'{working_dir}/data_process/mutagenesis/figures/{mutaNm}_pretrainUnSVFit.png',dpi=300)
  '''
  return None

def mut_fitnessUNSV_ensemble(working_dir: str = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark',
                            rpSet_list: List = None,
                            pretrainMdlNm_list: List = None,
                            mutaSetList_file: str = None,
                            epoch_list: int = None,
                            report_metric: str=None,
                            pretrain_topk: int=None,
                            finetune_topk: int=None,
                            ci_alpha: float = 0.95):
  """
  Ensemble performance from models and print
  """
  #ensemble_name=f'Shin2021Data_PreBest{pretrain_topk}_FtBest{finetune_topk}'
  ensemble_name=f'PreBest{pretrain_topk}_FtBest{finetune_topk}'
  ## load muta set name list
  muta_pfam_clan_list = np.loadtxt(f'{working_dir}/data_process/mutagenesis/DeepSequenceMutaSet_pfam_clan', dtype='str',delimiter=',',skiprows=1)
  
  ## open file to save ensemble scores (all rp_bert)
  save_file_handle = open(f'{working_dir}/data_process/mutagenesis/results/bert_all_{ensemble_name}_{report_metric}.tsv','w')
  save_file_handle.write('mutaNm\tfamId\tmax_spearmanR\tmean_spearmanR\tstd_spearmanR\tCI_low\tCI_upper\n') # header

  for muta_i in range(len(muta_pfam_clan_list)):
    mutaNm = muta_pfam_clan_list[muta_i][0]
    shinNm = muta_pfam_clan_list[muta_i][-1]
    famId =  muta_pfam_clan_list[muta_i][1]

    ensemble_spM = []
    if len(shinNm) > 0:
      ## loop through bert model settings
      for rp_set_i in range(len(rpSet_list)):
        rp_set = rpSet_list[rp_set_i]
        rp_split = re.split(r'_',rp_set)
        rpNm = f"{rp_split[0]}_{rp_split[1]}"
        mdl_subIdx = rp_split[2]

        ## open file to save ensemble scores (each rp_bert)
        #save_file_handle = open(f'{working_dir}/data_process/mutagenesis/results/{rp_set}_{ensemble_name}_{report_metric}.tsv','w')
        #save_file_handle.write('mutaNm\tfamId\tmax_spearmanR\tmean_spearmanR\tstd_spearmanR\n') # header
    
      
        ## load json from pretrained models ##
        if pretrainMdlNm_list is not None:
          try:
            with open(f'results_to_keep/{rpNm}/{rpNm}_pretrain_{mdl_subIdx}_models/{pretrainMdlNm_list[rp_set_i]}/mutagenesis_fitness_unSV/{mutaNm}_{epoch}.json') as handle:
              metric_json = json.load(handle)
            print(f"{rpNm}_{mdl_subIdx};{mutaNm};{abs(metric_json['spearmanR'])}")
          except:
            pass
        ## load json from finetuned models ##
        if report_metric == 'spR':
          ## query ensemble epochs
          epoch_list = os.popen(f'grep {famId} {working_dir}/data_process/pfam_34.0/finetune_ensemble/{rp_set}_selection | cut -d\',\' -f2').read().strip('\n').split(' ')
          for setnm in epoch_list[:pretrain_topk]:
            epoch = setnm.split('_')[-1]
            #!! load spearmanR from finetuned models !!#
            
            ## get model dir from log file
            #log_file = f'{working_dir}/job_logs/archive_baseline_bert_eval/baseline_bert_{rp_set}_torch_eval.all_fit_{mutaNm}.reweighted.0.{epoch}.out'
            log_file = f'{working_dir}/job_logs/archive_baseline_bert_eval/baseline_bert_{rp_set}_torch_eval.{famId}_fit_{mutaNm}.reweightedFT.0.pre{epoch}.out'
            #log_file = f'{working_dir}/job_logs/archive_baseline_bert_eval/baseline_bert_{rp_set}_torch_eval.{famId}_fit_{mutaNm}.shin2021Data.reweightedFT.pre{epoch}.ftBest.0.out'
            
            mdl_dir = os.popen(f"grep 'loading weights file' {log_file} | awk -F 'results_to_keep/' '{{print $2}}' | cut -d'/' -f5").read().strip('\n')
            #ft_epoch = os.popen(f"grep 'loading weights file' {log_file} | cut -d'/' -f15 | cut -d'_' -f3 | cut -d'.' -f1").read().strip('\n')
            
            ## loop over finetuned epochs
            ft_epochs = os.popen(f'ls -v results_to_keep/{rpNm}/{rp_set}_ensemble_finetune_models/family_specific/{famId}_reweighted/{mdl_dir}/pytorch_model_* | tail -n2 | cut -d"/" -f7 | cut -d"_" -f3 | cut -d"." -f1').read().strip('\n').split('\n')
            #ft_epochs = os.popen(f'ls -v results_to_keep/{rpNm}/{rp_set}_ensemble_finetune_models/Shin2021Data/{shinNm}_reweighted/{mdl_dir}/pytorch_model_* | tail -n2 | cut -d"/" -f7 | cut -d"_" -f3 | cut -d"." -f1').read().strip('\n').split('\n')
            
            ft_epochs = [''] + [f'_{fte}' for fte in ft_epochs[::-1]]
            #ft_epochs = ['']
            for ftEpo in ft_epochs[:finetune_topk]:
              #json_dir=f'{working_dir}/results_to_keep/{rpNm}/{rp_set}_{epoch}_finetune_models/family_specific/{famId}_reweighted/{mdl_dir}/mutagenesis_fitness_unSV/{mutaNm}{ftEpo}.json'
              #json_dir=f'{working_dir}/results_to_keep/{rpNm}/{rp_set}_ensemble_finetune_models/Shin2021Data/{shinNm}_reweighted/{mdl_dir}/mutagenesis_fitness_unSV/{mutaNm}{ftEpo}.json'
              json_dir=f'{working_dir}/results_to_keep/{rpNm}/{rp_set}_ensemble_finetune_models/family_specific/{famId}_reweighted/{mdl_dir}/mutagenesis_fitness_unSV/{mutaNm}{ftEpo}.json'
              with open(json_dir) as handle:
                metric_json = json.load(handle)
              ensemble_spM.append(abs(metric_json['spearmanR']))
              #print(f"{rp_set};{mutaNm};Pre{epoch}Ft{ftEpo};{abs(metric_json['spearmanR'])}")
          
          ## each rp_bert
          #print(f">>{rp_set};{mutaNm};Pre{epoch}Ft{ftEpo};{np.amax(ensemble_spM)}/{np.mean(ensemble_spM)}/{np.std(ensemble_spM)}")
          # assert len(ensemble_spM) == 45
          # print(f">>{rp_set};{mutaNm};{len(ensemble_spM)}")
          # save_file_handle.write(f'{mutaNm}\t{famId}\t{np.amax(ensemble_spM)}\t{np.mean(ensemble_spM)}\t{np.std(ensemble_spM)}\n')
        elif report_metric == 'ppl':
          #!! load spearmanR from finetuned models !!#
          ## get model dir from log file
          '''
          log_file = f'{working_dir}/job_logs/archive_baseline_bert_eval/baseline_bert_{rp_set}_torch_eval.all_lm_{famId}.reweighted.0.{epoch}.out'
          if os.path.isfile(log_file):
            mdl_dir = os.popen(f"grep 'loading weights file' {log_file} | cut -d'/' -f14").read().strip('\n')
            eval_epoch = os.popen(f"grep 'loading weights file' {log_file} | cut -d'/' -f15 | cut -d'_' -f3 | cut -d'.' -f1").read().strip('\n')
            with open(f'{working_dir}/results_to_keep/{rpNm}/{rp_set}_{epoch}_finetune_models/family_specific/{famId}_reweighted/{mdl_dir}/results_metrics_finetune_datasets_{famId}_{eval_epoch}.json') as handle:
              metric_json = json.load(handle)
            print(f"{rpNm}_{mdl_subIdx};{famId};{mutaNm};{metric_json['lm_ece']},{metric_json['accuracy']}")
          '''
          log_file = f'{working_dir}/job_logs/archive_baseline_bert_eval/baseline_bert_{rp_set}_torch_eval.all_lm_{famId}.pretrained.0.{epoch}.out'
          if os.path.isfile(log_file):
            mdl_dir = os.popen(f"grep 'loading weights file' {log_file} | cut -d'/' -f12").read().strip('\n')
            eval_epoch = os.popen(f"grep 'loading weights file' {log_file} | cut -d'/' -f13 | cut -d'_' -f3 | cut -d'.' -f1").read().strip('\n')
            with open(f'{working_dir}/results_to_keep/{rpNm}/{rpNm}_pretrain_{mdl_subIdx}_models/{mdl_dir}/results_metrics_finetune_datasets_{famId}_{eval_epoch}.json') as handle:
              metric_json = json.load(handle)
            #print(f"{rpNm}_{mdl_subIdx};{famId};{mutaNm};{metric_json['lm_ece']},{metric_json['accuracy']}")
            print(f"{metric_json['lm_ece']}")
      
      ## all rp_bert
      print(f"{mutaNm},{len(ensemble_spM)}")
      assert len(ensemble_spM) == 45
      ## confidence interval
      ci_interval = st.t.interval(ci_alpha, len(ensemble_spM)-1, loc=np.mean(ensemble_spM), scale=st.sem(ensemble_spM))
      if len(ensemble_spM) > 0:
        save_file_handle.write(f'{mutaNm}\t{famId}\t{np.amax(ensemble_spM)}\t{np.mean(ensemble_spM)}\t{np.std(ensemble_spM)}\t{ci_interval[0]}\t{ci_interval[1]}\n')
      else:
        save_file_handle.write(f'{mutaNm}\t{famId}\t\t\t\n')  
  save_file_handle.close()

def mut_set_spearmanR_fig(working_dir: str = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark',
                          curve_list: List = None,
                          plot_title: str = None):
  ## load SOTA spearmanR
  with open(f"{working_dir}/data_process/mutagenesis/results/mutation_final.pkl", "rb") as f:
    out = pkl.load(f)    
  
  score_sets={}
  score_sets['HMM']=out['hmm:'].copy()
  score_sets['Independent']=out['Independent'].copy()
  score_sets['Evmutation']=out['Evmutation'].copy()
  score_sets['DeepSequence']=out['DeepSequence'].copy()
  score_sets['Shin_2021']=out['shin2021_emb:'].copy()
  score_sets['Yue_2021']=out['our_emb:'].copy()


  ## load mutagenesis set names (Shin2021 order)
  mutaNm_list_Shin2021 = np.loadtxt(f'{working_dir}/data_process/mutagenesis/results/mutaNm_Shin2021.tsv',dtype='str',delimiter='\t',skiprows=1)
  mutaNm_list_extra = np.loadtxt(f'{working_dir}/data_process/mutagenesis/results/mutaNm_deepSeq_extra.tsv',dtype='str',delimiter='\t',skiprows=1)
  
  ## load my spearmanR (by Shin2021's order)
  # 'B1_rp15_RW_FT': 'rp15_all_1_PreBest3_FtBest3_spR',
  # 'B2_rp15_RW_FT': 'rp15_all_2_PreBest3_FtBest3_spR',
  # 'B3_rp15_RW_FT': 'rp15_all_3_PreBest3_FtBest3_spR',
  # 'B4_rp15_RW_FT': 'rp15_all_4_PreBest3_FtBest3_spR',
  # 'B1_rp75_RW_FT': 'rp75_all_1_PreBest3_FtBest3_spR',
  mySet_dict = {'ENSM_RW_FT_PfamUniKB': 'bert_all_PreBest3_FtBest3_spR',
                'ENSM_RW_FT_ShinData': 'bert_all_Shin2021Data_PreBest3_FtBest3_spR'}
  ## confidence interval for my models
  myConfIntv_dict = {}
  for myKey, myVal in mySet_dict.items():
    score_sets[myKey] = []
    myConfIntv_dict[myKey] = []
    for line in range(mutaNm_list_Shin2021.shape[0]):
      myMutaNm = mutaNm_list_Shin2021[line,1]
      ## get max(3),mean(4),ci_low(6),ci_upper(7) spearmanR
      spm_score = os.popen(f"grep '{myMutaNm}' {working_dir}/data_process/mutagenesis/results/{myVal}.tsv | cut -f4").read().strip('\n')
      ci_low = os.popen(f"grep '{myMutaNm}' {working_dir}/data_process/mutagenesis/results/{myVal}.tsv | cut -f6").read().strip('\n')
      ci_upper = os.popen(f"grep '{myMutaNm}' {working_dir}/data_process/mutagenesis/results/{myVal}.tsv | cut -f7").read().strip('\n')
      score_sets[myKey].append(float(spm_score))
      myConfIntv_dict[myKey].append((float(ci_upper)-float(ci_low))/2.)
    # for line in range(mutaNm_list_extra.shape[0]):
    #   myMutaNm = mutaNm_list_extra[line,0]
    #   ## get max(3), mean(4) spearmanR
    #   spm_score = os.popen(f"grep '{myMutaNm}' {working_dir}/data_process/mutagenesis/results/{myVal}.tsv | cut -f3").read().strip('\n')
    #   if len(spm_score) == 0:
    #     print(f'{myMutaNm},{myVal}')
    #   score_sets[myKey].append(float(spm_score))
  
  '''
  #>>> abalaton <<<#
  ## numerical comparision
  bertSet_numTop = [0] * len(curve_list)
  for i in range(len(mutaNm_list_Shin2021)+len(mutaNm_list_extra)):
    score2compare = []
    for cur in curve_list:
      score2compare.append(score_sets[cur][i])
    bestIdx = np.argmax(score2compare)
    bertSet_numTop[bestIdx] += 1
  for i in range(len(curve_list)):
    print(f'{curve_list[i]}:{bertSet_numTop[i]}')
  
  ## set up color
  color_abla={}
  color_abla['B1_rp15_RW_FT']= 'navy'
  color_abla['B2_rp15_RW_FT']= 'dodgerblue'
  color_abla['B3_rp15_RW_FT']= 'cyan'
  color_abla['B4_rp15_RW_FT']= 'lightblue'
  color_abla['B1_rp75_RW_FT']= 'darkred'
  color_abla['All_RW_FT'] = 'orange'

  ## muta set name for 42 sets
  mutaNm42 = []
  for i in range(len(mutaNm_list_Shin2021)):
    mutaNm42.append(mutaNm_list_Shin2021[i,2])
  for i in range(len(mutaNm_list_extra)):
    mutaNm42.append(mutaNm_list_extra[i,1])
  ## plot figure for ablation study
  plt.figure()
  x=np.linspace(0, 30, len(mutaNm_list_Shin2021)+len(mutaNm_list_extra))
  for setNm in curve_list:
      plt.scatter(x, score_sets[setNm], label=setNm, s=100, color=color_abla[setNm])

  for i in x:
      plt.plot(np.linspace(i,i,10), np.linspace(-0.1,1,10), linestyle='--', color='gray', alpha=0.3)

  plt.ylabel("SpearmanR: pred vs exp", fontsize=25)
  plt.ylim([-0.0,0.9])
  plt.xlim([-1,31])
 
  for i in range(len(x)):
      plt.text(x[i]+0.05, -0.01, mutaNm42[i], rotation=50, fontsize=14, verticalalignment='top', horizontalalignment='right')

  plt.xticks(x, [])
  plt.legend(loc="best", fontsize=18)
  plt.yticks(fontsize=18)
  fig =plt.gcf()
  fig.set_size_inches(18.5, 10.5)
  plt.tight_layout()
  plt.savefig(f'{working_dir}/data_process/mutagenesis/figures/{plot_title}.png', format='png', dpi=400)
  #plt.savefig(title+".eps", format='eps')
  '''
  
  #>>> SOTA <<<#
  ## numerical comparision
  bertSet_numTop = [0] * len(curve_list)
  for i in range(len(mutaNm_list_Shin2021)):
    score2compare = []
    for cur in curve_list:
      score2compare.append(score_sets[cur][i])
    bestIdx = np.argmax(score2compare)
    bertSet_numTop[bestIdx] += 1
  for i in range(len(curve_list)):
    print(f'{curve_list[i]}:{bertSet_numTop[i]}')
  
  vsShin2021 = 0
  for i in range(len(mutaNm_list_Shin2021)):
    if score_sets['ENSM_RW_FT_ShinData'][i] > score_sets['Shin_2021'][i]:
      vsShin2021 += 1
  print(f'ENSM_RW_FT_ShinData better than Shin2021:{vsShin2021}')
  
  vsYue2021 = 0
  for i in range(len(mutaNm_list_Shin2021)):
    if score_sets['ENSM_RW_FT_ShinData'][i] > score_sets['Yue_2021'][i]:
      vsYue2021 += 1
  print(f'ENSM_RW_FT_ShinData better than Yue2021:{vsYue2021}')

  vsDeepSeq = 0
  for i in range(len(mutaNm_list_Shin2021)):
    if score_sets['ENSM_RW_FT_ShinData'][i] > score_sets['DeepSequence'][i]:
      vsDeepSeq += 1
  print(f'ENSM_RW_FT_ShinData better than DeepSequence:{vsDeepSeq}')

  ## set up color
  color_assign={}
  color_assign['DeepSequence']= 'black'
  color_assign['Evmutation']= 'dimgrey'
  color_assign['Independent']= 'darkgrey'
  color_assign['HMM']= 'lightgrey'
  color_assign['Shin_2021']= 'blue'
  color_assign['Yue_2021']= 'orange'
  color_assign['ENSM_RW_FT_PfamUniKB']= 'orangered'
  color_assign['ENSM_RW_FT_ShinData']= 'darkred'


  ## muta set name for 42 sets
  mutaNm35 = []
  for i in range(len(mutaNm_list_Shin2021)):
    mutaNm35.append(mutaNm_list_Shin2021[i,2])
  ## plot figure for ablation study
  plt.figure()
  x=np.linspace(0, 30, len(mutaNm_list_Shin2021))
  for setNm in curve_list:
      plt.scatter(x, score_sets[setNm], label=setNm, s=40, color=color_assign[setNm], alpha=0.7)
      ## plot error bar(95% CI) for my models
      if setNm in myConfIntv_dict.keys():
        plt.errorbar(x, score_sets[setNm], yerr=myConfIntv_dict[setNm], fmt='none', ecolor=color_assign[setNm], capsize=4.0, alpha=0.7)
  for i in x:
      plt.plot(np.linspace(i,i,10), np.linspace(-0.1,1,10), linestyle='--', color='gray', alpha=0.3)

  plt.ylabel("SpearmanR: pred vs exp", fontsize=25)
  plt.ylim([-0.0,0.8])
  plt.xlim([-1,31])
 
  for i in range(len(x)):
      plt.text(x[i]+0.05, -0.01, mutaNm35[i], rotation=50, fontsize=14, verticalalignment='top', horizontalalignment='right')

  plt.xticks(x, [])
  plt.legend(loc="upper right", fontsize=13)
  plt.yticks(fontsize=18)
  fig =plt.gcf()
  fig.set_size_inches(18.5, 10.5)
  plt.tight_layout()
  plt.savefig(f'{working_dir}/data_process/mutagenesis/figures/{plot_title}.png', format='png', dpi=800)
  #plt.savefig(f'{working_dir}/data_process/mutagenesis/figures/{plot_title}.eps', format='eps')

def collect_fitness_ratioScores(working_dir: str = None,
                                rpSet_list: str = None,
                                pretrainMdlNm_list: List = None,
                                mutaSetList_file: str = None,
                                epoch_list: int = None,
                                pretrain_topk: int = None,
                                finetune_topk: int = None,
                                mutaSetIdx_list: List = None):
  """
  gather unsupervised predicted mutation score of different model into one file
  """
  
  ## load mapping between mutagenesis_setNm, shin2021_setNm, pfam_id
  muta_pfam_clan_list = np.loadtxt(f'{working_dir}/data_process/mutagenesis/DeepSequenceMutaSet_pfam_clan', dtype='str',delimiter=',',skiprows=1)
  
  ## load mutagenesis lable names
  #muta_label_names = np.loadtxt(f'{working_dir}/data_process/mutagenesis/target_label_set.tsv', dtype='str',delimiter='\t')

  #for muta_i in range(len(muta_pfam_clan_list)):
  for muta_i in mutaSetIdx_list:
    muta_i = int(muta_i)
    mutaNm = muta_pfam_clan_list[muta_i][0]
    shinNm = muta_pfam_clan_list[muta_i][-1]
    famId =  muta_pfam_clan_list[muta_i][1]
    print(f'>process {mutaNm}')
    raw_score_files = [] # to store file path of each model's raw predited scores
    
    ## only process Shin2021 included dataset
    if len(shinNm) > 0:
      ## open file to save predicted ratio (all rp_bert)
      save_file_handle = open(f'{working_dir}/data_process/mutagenesis/results/raw_predictions/{mutaNm}_seqMdl_famFinetuned.csv','w')
      ## assemble header
      muta_label_names = os.popen(f'grep {mutaNm} {working_dir}/data_process/mutagenesis/target_label_set.tsv | cut -f3').read().strip('\n').split(',')
      header = ['mutant'] + muta_label_names

      ## loop through bert model settings (e.g. rp15_all_1)
      for rp_set_i in range(len(rpSet_list)):
        rp_set = rpSet_list[rp_set_i] 
        rp_split = re.split(r'_',rp_set)
        rpNm = f"{rp_split[0]}_{rp_split[1]}"
        mdl_subIdx = rp_split[2]
      
        ## 1. load json from pretrained models (only do print)##
        if pretrainMdlNm_list is not None:
          try:
            with open(f'results_to_keep/{rpNm}/{rpNm}_pretrain_{mdl_subIdx}_models/{pretrainMdlNm_list[rp_set_i]}/mutagenesis_fitness_unSV/{mutaNm}_{epoch}.json') as handle:
              metric_json = json.load(handle)
            print(f"{rpNm}_{mdl_subIdx};{mutaNm};{abs(metric_json['spearmanR'])}")
          except:
            pass
        ## 2. load json from finetuned models ##
        ## query ensemble pretrained epoch numbers (ordered by ppl evaluation of each family)
        epoch_list = os.popen(f'grep {famId} {working_dir}/data_process/pfam_34.0/finetune_ensemble/{rp_set}_selection | cut -d\',\' -f2').read().strip('\n').split(' ')
        epoch_topk = 0
        for setnm in epoch_list[:pretrain_topk]:
          epoch_topk += 1
          epoch = setnm.split('_')[-1]
          
          ## get model dir from any finetune log file (e.g. Best fintuned epoch's log)
          #log_file = f'{working_dir}/job_logs/archive_baseline_bert_eval/baseline_bert_{rp_set}_torch_eval.all_fit_{mutaNm}.reweighted.0.{epoch}.out'
          #log_file = f'{working_dir}/job_logs/archive_baseline_bert_eval/baseline_bert_{rp_set}_torch_eval.{famId}_fit_{mutaNm}.reweightedFT.0.pre{epoch}.out'
          log_file = f'{working_dir}/job_logs/archive_baseline_bert_eval/baseline_bert_{rp_set}_torch_eval.{famId}_fit_{mutaNm}.shin2021Data.reweightedFT.pre{epoch}.ftBest.0.out'
          
          mdl_dir = os.popen(f"grep 'loading weights file' {log_file} | awk -F 'results_to_keep/' '{{print $2}}' | cut -d'/' -f5").read().strip('\n')
          #ft_epoch = os.popen(f"grep 'loading weights file' {log_file} | cut -d'/' -f15 | cut -d'_' -f3 | cut -d'.' -f1").read().strip('\n')
          
          ## loop over finetuned epochs (order from first to last one)
          #ft_epochs = os.popen(f'ls -v results_to_keep/{rpNm}/{rp_set}_ensemble_finetune_models/family_specific/{famId}_reweighted/{mdl_dir}/pytorch_model_* | tail -n2 | cut -d"/" -f7 | cut -d"_" -f3 | cut -d"." -f1').read().strip('\n').split('\n')
          ft_epochs = os.popen(f'ls -v results_to_keep/{rpNm}/{rp_set}_ensemble_finetune_models/Shin2021Data/{shinNm}_reweighted/{mdl_dir}/pytorch_model_* | tail -n2 | cut -d"/" -f7 | cut -d"_" -f3 | cut -d"." -f1').read().strip('\n').split('\n')
          
          ## add best epoch to beginning and reverse order(last to first)
          ft_epochs = [''] + [f'_{fte}' for fte in ft_epochs[::-1]]
          ft_epochs_topk = 0
          for ftEpo in ft_epochs[:finetune_topk]:
            ft_epochs_topk += 1
            ## model name for header
            header += [f'{rp_set}_pt{epoch_topk}_ft{ft_epochs_topk}']

            #csv_dir=f'{working_dir}/results_to_keep/{rpNm}/{rp_set}_{epoch}_finetune_models/family_specific/{famId}_reweighted/{mdl_dir}/mutagenesis_fitness_unSV/{mutaNm}{ftEpo}.csv'
            csv_dir=f'{working_dir}/results_to_keep/{rpNm}/{rp_set}_ensemble_finetune_models/Shin2021Data/{shinNm}_reweighted/{mdl_dir}/mutagenesis_fitness_unSV/{mutaNm}{ftEpo}.csv'
            #csv_dir=f'{working_dir}/results_to_keep/{rpNm}/{rp_set}_ensemble_finetune_models/family_specific/{famId}_reweighted/{mdl_dir}/mutagenesis_fitness_unSV/{mutaNm}{ftEpo}.csv'
            raw_score_files.append(csv_dir)
      ## loop each mutant and write exp score and prediction score to one whole file
      mut_labels_df = pd.read_csv(f'{working_dir}/data_process/mutagenesis/DeepSequenceMutaSet/{mutaNm}.csv',index_col=0)
      ##!! dataset from DeepSeq and Shin2021 have duplicating rows
      mut_labels_df.drop_duplicates(subset=['mutant'], keep='first', inplace=True, ignore_index=True)
      ## write header
      header_str = ','.join(header)
      save_file_handle.write(f'{header_str}\n')
      for row_idx, row in mut_labels_df.iterrows():
        mutant_name = row['mutant']
        if re.match(r"[_xX]\d+[A-Za-z]|[A-Za-z]\d+[_xX]",mutant_name):
          print(mutant_name)
          continue
        row2write = [mutant_name]
        for mut_label in muta_label_names:
          if np.isnan(row[mut_label]):
            row2write.append('')
          else:
            row2write.append(str(row[mut_label]))
        for raw_file in raw_score_files:
          ## preditions have duplicating rows(mutants), pick first one
          raw_score = os.popen(f"grep {mutant_name} {raw_file} | uniq | head -n1 | cut -d',' -f2").read().strip('\n')
          row2write.append(raw_score)
        row2write_str = ','.join(row2write)
        # print(row2write_str)
        # if '\n' in row2write_str:
        #   input()
        save_file_handle.write(f'{row2write_str}\n')
      save_file_handle.close()
      ## reload as dataframe and add model mean columns
      rawScore_df = pd.read_csv(f'{working_dir}/data_process/mutagenesis/results/raw_predictions/{mutaNm}_seqMdl_famFinetuned.csv')
      all_ensemble_names = []
      for rp_set_i in range(len(rpSet_list)):
          rp_set = rpSet_list[rp_set_i]
          rp_ensemble_names = []
          for pt in range(3):
            for ft in range(3):
              rp_ensemble_names.append(f'{rp_set}_pt{pt+1}_ft{ft+1}')
              all_ensemble_names.append(f'{rp_set}_pt{pt+1}_ft{ft+1}')
          rawScore_df[f'{rp_set}_mean'] = rawScore_df[rp_ensemble_names].mean(axis=1)
      rawScore_df[f'pretrain_seqMdl_mean'] = rawScore_df[all_ensemble_names].mean(axis=1)
      rawScore_df.to_csv(f'{working_dir}/data_process/mutagenesis/results/raw_predictions/{mutaNm}_seqMdl_famFinetuned.csv',index=False)

def add_new_prediction(working_dir: str,
                       rpSet_list: str,
                       epoch_list: int,
                       setNm_list: List,
                       pretrainMdlNm_dict: dict):
  """Add new prediction of fitness as columns in csv files under data_process/mutagenesis/results/raw_predictions
  
  """
  # load each individual prediction files
  ## load mapping between mutagenesis_setNm, shin2021_setNm, pfam_id
  muta_pfam_clan_list = np.loadtxt(f'{working_dir}/data_process/mutagenesis/DeepSequenceMutaSet_pfam_clan', dtype='str',delimiter=',',skiprows=1)
  
  ## load mutagenesis lable names
  #muta_label_names = np.loadtxt(f'{working_dir}/data_process/mutagenesis/target_label_set.tsv', dtype='str',delimiter='\t')

  #for muta_i in range(len(muta_pfam_clan_list)):
  for muta_i in range(muta_pfam_clan_list.shape[0]):
    mutaNm = muta_pfam_clan_list[muta_i][0]
    shinNm = muta_pfam_clan_list[muta_i][-1]
    famId =  muta_pfam_clan_list[muta_i][1]
    if len(shinNm) == 0:
      continue
    if len(setNm_list) > 0 and shinNm not in setNm_list:
      continue
    print(f'>process {mutaNm}')
    raw_score_files = [] # to store file path of each model's raw predited scores
    header = []
    ## only process Shin2021 included dataset
    if len(shinNm) > 0:
      ## loop through bert model settings (e.g. rp15_all_1)
      for rp_set_i in range(len(rpSet_list)):
        rp_set = rpSet_list[rp_set_i] 
        rp_split = re.split(r'_',rp_set)
        rpNm = f"{rp_split[0]}_{rp_split[1]}"
        mdl_subIdx = rp_split[2]
        
        
        for epoch in epoch_list:
          ## load json from pretrained models ##
          try:
            with open(f'results_to_keep/{rpNm}/{rpNm}_pretrain_{mdl_subIdx}_models/{pretrainMdlNm_dict[rp_set]}/mutagenesis_fitness_unSV/{mutaNm}_{epoch}.json') as handle:
              metric_json = json.load(handle)
            print(f"{rpNm}_{mdl_subIdx};{mutaNm};{abs(metric_json['spearmanR'])}")
          except:
            pass

          ## load json from finetuned models ##
          ## get model dir from any finetune log file (e.g. Best fintuned epoch's log)
          #log_file = f'{working_dir}/job_logs/archive_baseline_bert_eval/baseline_bert_{rp_set}_torch_eval.all_fit_{mutaNm}.reweighted.0.{epoch}.out'
          #log_file = f'{working_dir}/job_logs/archive_baseline_bert_eval/baseline_bert_{rp_set}_torch_eval.{famId}_fit_{mutaNm}.reweightedFT.0.pre{epoch}.out'
          log_file = f'{working_dir}/job_logs/archive_baseline_bert_eval/baseline_bert_{rp_set}_torch_eval.{famId}_fit_{mutaNm}.shin2021Data.nonReweightedFT.pre{epoch}.ftBest.2.out'

          if not os.path.isfile(log_file):
            continue

          mdl_dir = os.popen(f"grep 'loading weights file' {log_file} | awk -F 'results_to_keep/' '{{print $2}}' | cut -d'/' -f5").read().strip('\n')
          #ft_epoch = os.popen(f"grep 'loading weights file' {log_file} | cut -d'/' -f15 | cut -d'_' -f3 | cut -d'.' -f1").read().strip('\n')

          ## loop over finetuned epochs (order from first to last one)
          #ft_epochs = os.popen(f'ls -v results_to_keep/{rpNm}/{rp_set}_ensemble_finetune_models/family_specific/{famId}_reweighted/{mdl_dir}/pytorch_model_* | tail -n2 | cut -d"/" -f7 | cut -d"_" -f3 | cut -d"." -f1').read().strip('\n').split('\n')
          ft_epochs = os.popen(f'ls -v results_to_keep/{rpNm}/{rp_set}_ensemble_finetune_models/Shin2021Data/{shinNm}_nonReweighted/{mdl_dir}/pytorch_model_* | cut -d"/" -f7 | cut -d"_" -f3 | cut -d"." -f1').read().strip('\n').split('\n')
          
          ## add best epoch to beginning and reverse order(last to first)
          ft_epochs = [''] + [f'_{fte}' for fte in ft_epochs[::-1]]
          for ftEpo in ft_epochs[0:1]:
            if ftEpo == '':
              ftEpo_name = 'Best'
            else:
              ftEpo_name = ftEpo[1:]
            ## model name for header
            header.append(f'{rp_set}_s100_nonReweight_pt{epoch}_ft{ftEpo_name}')

            #csv_dir=f'{working_dir}/results_to_keep/{rpNm}/{rp_set}_{epoch}_finetune_models/family_specific/{famId}_reweighted/{mdl_dir}/mutagenesis_fitness_unSV/{mutaNm}{ftEpo}.csv'
            csv_dir=f'{working_dir}/results_to_keep/{rpNm}/{rp_set}_ensemble_finetune_models/Shin2021Data/{shinNm}_nonReweighted/{mdl_dir}/mutagenesis_fitness_unSV/{mutaNm}{ftEpo}.csv'
            #csv_dir=f'{working_dir}/results_to_keep/{rpNm}/{rp_set}_ensemble_finetune_models/family_specific/{famId}_reweighted/{mdl_dir}/mutagenesis_fitness_unSV/{mutaNm}{ftEpo}.csv'
            raw_score_files.append(csv_dir)
    mut_values = pd.read_csv(f'{working_dir}/data_process/mutagenesis/results/raw_predictions/{mutaNm}_seqMdl_famFinetuned.csv')
    
    for col_name, col_file in zip(header,raw_score_files):
      new_df = pd.read_csv(col_file, names=['mutant',col_name])
      mut_values = mut_values.merge(new_df,how='left',on='mutant')
    new_pred_mean = mut_values[header].mean(axis=1).to_list()
    mut_values['pretrain_seqMdl_mean_s100_nonReweight'] = new_pred_mean
    ## resave
    mut_values.to_csv(f'{working_dir}/data_process/mutagenesis/results/raw_predictions/{mutaNm}_seqMdl_famFinetuned.csv',index=False)

def add_new_mean(working_dir: str = None,
                 shin_muta_path: str = None,
                 col_list: List = None,
                 new_col_name: str = None):
  """ HMM predictions in Shin2021 dataset have no mean column for HMM
      This script is to add a column for HMM mean scores
  
  mutation_effect_prediction_vae_ensemble,mutation_effect_prediction_vae_1,mutation_effect_prediction_vae_2,mutation_effect_prediction_vae_3,mutation_effect_prediction_vae_4,mutation_effect_prediction_vae_5,mutation_effect_prediction_pairwise,mutation_effect_prediction_independent,hmm_effect_prediction_mean,hmm_effect_prediction_symfrac0.5,hmm_effect_prediction_symfrac0.7,mutation_effect_prediction_forward_rseed-11_channels-24,mutation_effect_prediction_forward_rseed-11_channels-48,mutation_effect_prediction_forward_rseed-22_channels-24,mutation_effect_prediction_forward_rseed-22_channels-48,mutation_effect_prediction_forward_rseed-33_channels-24,mutation_effect_prediction_forward_rseed-33_channels-48,mutation_effect_prediction_reverse_rseed-11_channels-24,mutation_effect_prediction_reverse_rseed-11_channels-48,mutation_effect_prediction_reverse_rseed-22_channels-24,mutation_effect_prediction_reverse_rseed-22_channels-48,mutation_effect_prediction_reverse_rseed-33_channels-24,mutation_effect_prediction_reverse_rseed-33_channels-48,mutation_effect_prediction_forward_mean_channels-24,mutation_effect_prediction_reverse_mean_channels-24,mutation_effect_prediction_forward_mean_channels-48,mutation_effect_prediction_reverse_mean_channels-48,mutation_effect_prediction_all_mean_channels-24,mutation_effect_prediction_all_mean_channels-48,mutation_effect_prediction_all_mean_autoregressive

      * HMM prediction columns
        hmm_effect_prediction_symfrac0.5
        hmm_effect_prediction_symfrac0.7
      * autoregressive model forward
        mutation_effect_prediction_forward_mean_channels-24
        mutation_effect_prediction_forward_mean_channels-48
      * autoregressive model reverse
        mutation_effect_prediction_reverse_mean_channels-24
        mutation_effect_prediction_reverse_mean_channels-48

  """
  ## load mutagenesis lable names
  muta_label_names = np.loadtxt(f'{working_dir}/data_process/mutagenesis/target_label_set_Shin2021.tsv', dtype='str',delimiter='\t',skiprows=1)
  for ii in range(len(muta_label_names)):
    pred_file = muta_label_names[ii][0]
    muta_setNm = muta_label_names[ii][-1]
    print(f'>>muta_setNm: {muta_setNm}<<')
    mut_values = pd.read_csv(f'{shin_muta_path}/output_processed/{pred_file}')
    new_mean = mut_values[col_list].mean(axis=1)
    mut_values.insert(mut_values.columns.get_loc(col_list[0]), new_col_name, new_mean)
    ## resave
    mut_values.to_csv(f'{shin_muta_path}/output_processed/{pred_file}',index=False)

"""
Analysis of sequence model for fitness prediction

* start vs end vs middle variants
* Single-site mutation vs multiple-site mutations
* protein: single domain vs multi-domain
* variants: covered by domain fine-tune seq vs not covered (for domain fine-tuned model)

"""

def analysis_fine_region(
      working_dir: str = None,
      shin_muta_path: str = None,
      exp_label_set: str = None,
      sota_mdl_compare: List = None,
      my_mdl_compare: List = None,
      mdls_draw: List = None,
      mdls_diff_draw: List = None,
      fig_subNm: str = None,
      save_score: bool = False,
      draw_figure: bool = True,
      nume_analysis: bool = True,
      nume_mdl_pairs: List = None,
      minNum_for_rank: int = 10,
      num_pos_bin: int = 5):
  """
  Positional analysis at finer level for fitness prediction.
  
  Bin mutant positions into $num_pos_bin bins and analysis spearmanR across each bin

  """
  mdl_name_convert = {
      'mutation_effect_prediction_vae_ensemble': 'DeepSequence',
      'mutation_effect_prediction_pairwise': 'Evmutation',
      'mutation_effect_prediction_independent': 'Independent',
      'hmm_effect_prediction_mean': 'HMM',
      'mutation_effect_prediction_all_mean_autoregressive': 'Shin2021',
      'mutation_effect_prediction_forward_mean_channels-48': 'Shin2021_fw_c48',
      'mutation_effect_prediction_reverse_mean_channels-48': 'Shin2021_rv_c48',
      'mutation_effect_prediction_forward_mean_channels-24': 'Shin2021_fw_c24',
      'mutation_effect_prediction_reverse_mean_channels-24': 'Shin2021_rv_c24',
      'mutation_effect_prediction_forward_mean': 'Shin2021_fw',
      'mutation_effect_prediction_reverse_mean': 'Shin2021_rv',
      'pretrain_seqMdl_mean': 'OurSeqMdl',
      'rp75_all_1_mean': 'OurSeqMdl_B1_rp75',
      'rp15_all_1_mean': 'OurSeqMdl_B1_rp15',
      'rp15_all_2_mean': 'OurSeqMdl_B2_rp15',
      'rp15_all_3_mean': 'OurSeqMdl_B3_rp15',
      'rp15_all_4_mean': 'OurSeqMdl_B4_rp15'
    }

  ## load mutagenesis lable names
  muta_label_names = np.loadtxt(f'{working_dir}/data_process/mutagenesis/target_label_set_Shin2021.tsv', dtype='str',delimiter='\t',skiprows=1)
  if save_score:
    spearman_scores_region_bin = {}
    mdl_list = my_mdl_compare + sota_mdl_compare
    for mdl in mdl_list:
      spearman_scores_region_bin[mdl] = []
      for bin_i in range(num_pos_bin):
        spearman_scores_region_bin[mdl].append([])
    for ii in range(len(muta_label_names)):
      pred_file = muta_label_names[ii][0]
      muta_setNm = muta_label_names[ii][-1]
      print(f'>>muta_setNm: {muta_setNm}<<')
      if muta_setNm in ['PABP_YEAST_Fields2013-doubles']: ## pass higher order mutations
        continue
      if exp_label_set == 'deepsequence':
        target_label = muta_label_names[ii][2] ## DeepSequence provided exp label name
      elif exp_label_set == 'yue':
        target_label = muta_label_names[ii][1] ## Yue provided exp label name
      ## load sota predictions
      mut_values = pd.read_csv(f'{shin_muta_path}/output_processed/{pred_file}')
      ## remove duplicating rows(mutants)
      mut_values.drop_duplicates(subset=['mutant'], keep='first', inplace=True, ignore_index=True)
      ## remove wild-type cases
      pure_mut_values = mut_values[~(mut_values.mutant.str.match(r"([A-Za-z])\d+\1|wt|WT"))]
      ## select rows: vae null start; vae null end, middle 
      exp_notnull_df = pure_mut_values[pure_mut_values[target_label].notnull()]
      pred_notnull_df = exp_notnull_df[exp_notnull_df['mutation_effect_prediction_all_mean_autoregressive'].notnull()]
      ## filter single mutations
      single_mut_df = pred_notnull_df[pred_notnull_df['mutant'].str.count(':') == 0]
      ## order by index
      singleMut_pos = single_mut_df.mutant.str.extract(r"[A-Za-z](?P<digit>\d+)[A-Za-z]")
      singleMut_pos.digit = singleMut_pos.digit.astype(int)
      singleMut_posSort_idx = singleMut_pos.sort_values(by=['digit'],kind='mergesort').index.to_numpy()
      singleMut_posSort_idxSplit = np.array_split(singleMut_posSort_idx,num_pos_bin)
      
      ## load mutagenesis predictions of my model (find the same mutant set compared to Shin2021)
      my_mut_values = pd.read_csv(f'{working_dir}/data_process/mutagenesis/results/raw_predictions/{muta_setNm}_seqMdl_famFinetuned.csv')
      ## remove duplicating rows(mutants)
      my_mut_values.drop_duplicates(subset=['mutant'], keep='first', inplace=True, ignore_index=True)
      ## remove wild-type cases
      my_pure_mut_values = my_mut_values[~(my_mut_values.mutant.str.match(r"([A-Za-z])\d+\1|wt|WT"))]
      # select mutants which are in pred_start_df
      my_singleMut_df = my_pure_mut_values.merge(single_mut_df,on=['mutant'],suffixes=(None, '_y'))
      ## order by index
      my_singleMut_pos = my_singleMut_df['mutant'].str.extract(r"[A-Za-z](?P<digit>\d+)[A-Za-z]")
      my_singleMut_pos.digit = my_singleMut_pos.digit.astype(int)
      my_singleMut_posSort_idx = my_singleMut_pos.sort_values(by=['digit'],kind='mergesort').index.to_numpy()
      my_singleMut_posSort_idxSplit = np.array_split(my_singleMut_posSort_idx,num_pos_bin)
      for bin_i in range(num_pos_bin):
        ## spearman for sota
        singleMut_target_df = single_mut_df[single_mut_df.index.isin(singleMut_posSort_idxSplit[bin_i])]
        for mdl in sota_mdl_compare:
          if len(singleMut_target_df.index) >= minNum_for_rank:
            spearman_scores_region_bin[mdl][bin_i].append(st.spearmanr(singleMut_target_df[target_label].to_numpy(),singleMut_target_df[mdl].to_numpy()).correlation)
          else:
            spearman_scores_region_bin[mdl][bin_i].append(0.)
        ## spearman for my models
        my_singleMut_target_df = my_singleMut_df[my_singleMut_df.index.isin(my_singleMut_posSort_idxSplit[bin_i])]
        for mdl in my_mdl_compare:
          if len(my_singleMut_target_df.index) >= minNum_for_rank:
            spearman_scores_region_bin[mdl][bin_i].append(st.spearmanr(my_singleMut_target_df[target_label].to_numpy(),my_singleMut_target_df[mdl].to_numpy()).correlation)
          else:
            spearman_scores_region_bin[mdl][bin_i].append(0.)
    with open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/fineInterval_{num_pos_bin}_region.json','w') as f:
      json.dump(spearman_scores_region_bin, f) 
  else:
    ## load spearmanr scores
    with open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/fineInterval_{num_pos_bin}_region.json') as f:
      spearman_scores_region_bin= json.load(f)
  if nume_analysis:
    nume_ana_write = open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/fineInterval_{num_pos_bin}_nume.csv','w')
    for mdl_pair in nume_mdl_pairs:
      for bin_i in range(num_pos_bin):
        score_set1,score_set2 = [],[]
        mdl_host, mdl_guest = mdl_pair[0], mdl_pair[1]
        score_host = spearman_scores_region_bin[mdl_host][bin_i]
        score_guest = spearman_scores_region_bin[mdl_guest][bin_i]
        for i in range(len(score_host)):
          ## remove cases that have no mutations
          if score_host[i] == 0. and score_guest[i] == 0.:
            continue
          else:
            score_set1.append(score_host[i])
            score_set2.append(score_guest[i])
        better_count = np.sum((np.abs(score_set1) - np.abs(score_set2)) >=0.)
        ave_diff = np.mean(np.abs(score_set1) - np.abs(score_set2))
        median_diff = np.median(np.abs(score_set1) - np.abs(score_set2))
        ks_twoSide = st.ks_2samp(np.abs(score_set1), np.abs(score_set2), alternative='two-sided')
        ks_less = st.ks_2samp(np.abs(score_set1), np.abs(score_set2), alternative='less')
        ks_greater = st.ks_2samp(np.abs(score_set1), np.abs(score_set2), alternative='greater')
        #ind_ttest = st.ttest_ind(np.abs(score_set1), np.abs(score_set2), equal_var=False)
        #ind_ttest_stat, ind_ttest_p = ind_ttest.statistic, ind_ttest.pvalue
        rel_ttest = st.ttest_rel(np.abs(score_set1), np.abs(score_set2))
        nume_ana_write.write(f'{mdl_name_convert[mdl_pair[0]]}-{mdl_name_convert[mdl_pair[1]]}-{bin_i},{better_count}/{len(score_set1)},t-rel:{rel_ttest.statistic}/{rel_ttest.pvalue},ks_2s:{ks_twoSide.statistic}/{ks_twoSide.pvalue},ks_less:{ks_less.statistic}/{ks_less.pvalue},ks_greater:{ks_greater.statistic}/{ks_greater.pvalue},ave diff:{ave_diff},median diff:{median_diff}\n')
    nume_ana_write.close()
  ## draw figures
  if draw_figure:
    ## set up color
    color_assign={
      'DeepSequence': 'black',
      'Evmutation': 'darkgrey',
      'Independent': 'tan',
      'HMM': 'lime',
      'Shin2021': 'blue',
      'Shin2021_fw_c48': 'blue',
      'Shin2021_rv_c48': 'blue',
      'Shin2021_fw_c24': 'blue',
      'Shin2021_rv_c24': 'blue',
      'Shin2021_fw': 'blue',
      'Shin2021_rv': 'blue',
      'Yue2021': 'orange',
      'OurSeqMdl': 'darkred',
      'OurSeqMdl_B1_rp75': 'darkred',
      'OurSeqMdl_B1_rp15': 'darkred',
      'OurSeqMdl_B2_rp15': 'darkred',
      'OurSeqMdl_B3_rp15': 'darkred',
      'OurSeqMdl_B4_rp15': 'darkred',
    }
    marker_assign={
      'DeepSequence': ".",
      'Evmutation': ".",
      'Independent': ".",
      'HMM': ".",
      'Shin2021': ".",
      'Shin2021_fw_c48': '^',
      'Shin2021_rv_c48': 'v',
      'Shin2021_fw_c24': '<',
      'Shin2021_rv_c24': '>',
      'Shin2021_fw': '>',
      'Shin2021_rv': '<',
      'Yue2021': ".",
      'OurSeqMdl': ".",
      'OurSeqMdl_B1_rp75': '+',
      'OurSeqMdl_B1_rp15': '1',
      'OurSeqMdl_B2_rp15': '2',
      'OurSeqMdl_B3_rp15': '3',
      'OurSeqMdl_B4_rp15': '4',
    }

    ## only contain 35 sets
    mutaNm_list_Shin2021 = np.loadtxt(f'{working_dir}/data_process/mutagenesis/results/mutaNm_Shin2021.tsv',dtype='str',delimiter='\t',skiprows=1)
    mutaNm35 = []
    for i in range(len(mutaNm_list_Shin2021)):
      if mutaNm_list_Shin2021[i,1] not in ['PABP_YEAST_Fields2013-doubles']:
        mutaNm35.append(mutaNm_list_Shin2021[i,2])
    mutaNm35_len = len(mutaNm35)
    ## convert scores into dataframe
    spearman_list4df = []
    spearman_diff_list4df = []
    for mdl_nm, score_list in spearman_scores_region_bin.items():
      for bin_i in range(num_pos_bin):
        for muta_i in range(mutaNm35_len):
          spearman_list4df.append([np.abs(score_list[bin_i][muta_i]),bin_i,mdl_name_convert[mdl_nm],mutaNm35[muta_i]])
    for mdl_pair in nume_mdl_pairs:
      for bin_i in range(num_pos_bin):
        for muta_i in range(mutaNm35_len):
          mdl_host, mdl_guest = mdl_pair[0], mdl_pair[1]
          score_host = spearman_scores_region_bin[mdl_host][bin_i][muta_i]
          score_guest = spearman_scores_region_bin[mdl_guest][bin_i][muta_i]
          ## remove cases that have no mutations
          if score_host == 0. and score_guest == 0.:
            continue
          else:
            spearman_diff_list4df.append([np.abs(score_host)-np.abs(score_guest),bin_i,f'{mdl_name_convert[mdl_host]}-{mdl_name_convert[mdl_guest]}',mutaNm35[muta_i]])
    spearman_df = pd.DataFrame(spearman_list4df,columns=['spearmanR','region','model','mutaSet'])
    spearman_diff_df = pd.DataFrame(spearman_diff_list4df,columns=['spearmanR-diff','region','model-pair','mutaSet'])
    filter_df = spearman_df.loc[(spearman_df['spearmanR'] > 0) & (spearman_df['model'].isin(mdls_draw))]
    filter_diff_df = spearman_diff_df.loc[(spearman_diff_df['model-pair'].isin(mdls_diff_draw))]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x="region", y="spearmanR", hue="model",
                    data=filter_df, order=range(num_pos_bin),hue_order=mdls_draw)
    #sns.swarmplot(x="region", y="spearmanR", hue="model", data=filter_df, dodge=True, size=2, color=".25")
    plt.savefig(f'{working_dir}/data_process/mutagenesis/results/result_analysis/figures/{fig_subNm}_{num_pos_bin}_box.png', format='png', dpi=800)
    plt.clf()
    ## score diff figure
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x="region", y="spearmanR-diff", hue="model-pair",
                data=filter_diff_df, order=range(num_pos_bin),hue_order=mdls_diff_draw)
    #sns.swarmplot(x="region", y="spearmanR-diff", hue="model-pair", data=filter_diff_df, dodge=True, size=2, color=".25")
    plt.savefig(f'{working_dir}/data_process/mutagenesis/results/result_analysis/figures/{fig_subNm}_{num_pos_bin}_box_diff.png', format='png', dpi=800)
    plt.clf()

def analysis_region(
      working_dir: str = None,
      shin_muta_path: str = None,
      exp_label_set: str = None,
      sota_mdl_compare: List = None,
      my_mdl_compare: List = None,
      mdls_draw: List = None,
      regions_draw: List = None,
      fig_subNm: str = None,
      save_score: bool = False,
      draw_figure: bool = True,
      domain_group: bool = False,
      nume_analysis: bool = True,
      nume_mdl_pairs: List = None,
      minNum_for_rank: int = 10):
  """
    positional analysis for fitness prediction
  """
  mdl_name_convert = {
      'mutation_effect_prediction_vae_ensemble': 'DeepSequence',
      'mutation_effect_prediction_pairwise': 'Evmutation',
      'mutation_effect_prediction_independent': 'Independent',
      'hmm_effect_prediction_mean': 'HMM',
      'mutation_effect_prediction_all_mean_autoregressive': 'Shin2021',
      'mutation_effect_prediction_forward_mean_channels-48': 'Shin2021_fw_c48',
      'mutation_effect_prediction_reverse_mean_channels-48': 'Shin2021_rv_c48',
      'mutation_effect_prediction_forward_mean_channels-24': 'Shin2021_fw_c24',
      'mutation_effect_prediction_reverse_mean_channels-24': 'Shin2021_rv_c24',
      'mutation_effect_prediction_forward_mean': 'Shin2021_fw',
      'mutation_effect_prediction_reverse_mean': 'Shin2021_rv',
      'pretrain_seqMdl_mean': 'OurSeqMdl_Mean_FT',
      'rp75_all_1_mean': 'OurSeqMdl_B1_rp75_FT',
      'rp15_all_1_mean': 'OurSeqMdl_B1_rp15_FT',
      'rp15_all_2_mean': 'OurSeqMdl_B2_rp15_FT',
      'rp15_all_3_mean': 'OurSeqMdl_B3_rp15_FT',
      'rp15_all_4_mean': 'OurSeqMdl_B4_rp15_FT',
      'pretrain_seqMdl_mean_s100_reweight': 'OurSeqMdl_e100_rwt',
      'pretrain_seqMdl_mean_s100_nonReweight': 'OurSeqMdl_e100_nrwt',
      'rp75_all_1_pre_224': 'OurSeqMdl_B1_rp75_PRE',
      'rp15_all_1_pre_729': 'OurSeqMdl_B1_rp75_PRE',
      'rp15_all_2_pre_729': 'OurSeqMdl_B1_rp75_PRE',
      'rp15_all_3_pre_729': 'OurSeqMdl_B1_rp75_PRE',
      'rp15_all_4_pre_729': 'OurSeqMdl_B1_rp75_PRE',
      'pre_only_mean': 'OurSeqMdl_Mean_PRE',
    }

  ## load mutagenesis lable names
  muta_label_names = np.loadtxt(f'{working_dir}/data_process/mutagenesis/target_label_set_Shin2021.tsv', dtype='str',delimiter='\t',skiprows=1)
  if save_score:
    spearman_scores_all = {}
    spearman_scores_start = {}
    spearman_scores_middle = {}
    spearman_scores_end = {}
    spearman_scores_not_middle = {}
    spearman_scores_middle_vaeNull = {}

    mdl_list = my_mdl_compare + sota_mdl_compare
    for mdl in mdl_list:
      spearman_scores_all[mdl] = []
      spearman_scores_start[mdl] = []
      spearman_scores_middle[mdl] = []
      spearman_scores_end[mdl] = []
      spearman_scores_not_middle[mdl] = []
      spearman_scores_middle_vaeNull[mdl] = []
    ## for debug
    debug_file = open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/start_middle_end_nums','w')
    debug_file.write(f'muta_setNm,my_start_df,start_df,my_middle_df,middle_df,my_end_df,end_df,my_mid_vaeN_df,mid_vaeN_df\n')
    for ii in range(len(muta_label_names)):
      pred_file = muta_label_names[ii][0]
      muta_setNm = muta_label_names[ii][-1]
      print(f'>>muta_setNm: {muta_setNm}<<')
      #if muta_setNm in ['HIS7_YEAST_Kondrashov2017']:
      #  continue
      if exp_label_set == 'deepsequence':
        target_label = muta_label_names[ii][2] ## DeepSequence provided exp label name
      elif exp_label_set == 'yue':
        target_label = muta_label_names[ii][1] ## Yue provided exp label name
      mut_values = pd.read_csv(f'{shin_muta_path}/output_processed/{pred_file}')
      ## remove duplicating rows(mutants)
      mut_values.drop_duplicates(subset=['mutant'], keep='first', inplace=True, ignore_index=True)
      ## remove wild-type cases
      pure_mut_values = mut_values[~(mut_values.mutant.str.match(r"([A-Za-z])\d+\1|wt|WT"))]
      ## select rows: vae null start; vae null end, middle 
      exp_notnull_df = pure_mut_values[pure_mut_values[target_label].notnull()]
      pred_notnull_df = exp_notnull_df[exp_notnull_df['mutation_effect_prediction_all_mean_autoregressive'].notnull()]
      pred_vae_notnull_idx = exp_notnull_df[exp_notnull_df['mutation_effect_prediction_vae_ensemble'].notnull()].index.tolist()
      pred_start_df = pred_notnull_df[pred_notnull_df.index < pred_vae_notnull_idx[0]] # vae null start (might be empty)
      pred_middle_df = pred_notnull_df[(pred_notnull_df.index.isin(pred_vae_notnull_idx))]
      pred_not_middle_df = pred_notnull_df[~(pred_notnull_df.index.isin(pred_vae_notnull_idx))]
      pred_end_df = pred_notnull_df[pred_notnull_df.index > pred_vae_notnull_idx[-1]] # vae null tail (might be empty)
      pred_middle_vaeNull_df = pred_notnull_df[~(pred_notnull_df.index.isin(pred_vae_notnull_idx)) & (pred_notnull_df.index >= pred_vae_notnull_idx[0]) & (pred_notnull_df.index <= pred_vae_notnull_idx[-1])]
      ## calculate spearmanr of SOTA
      for mdl in sota_mdl_compare:
        if len(pred_notnull_df.index) >= minNum_for_rank:
          spearman_scores_all[mdl].append(st.spearmanr(pred_notnull_df[target_label].to_numpy(),pred_notnull_df[mdl].to_numpy()).correlation)
        else:
          spearman_scores_all[mdl].append(0.)
        if len(pred_start_df.index) >= minNum_for_rank:
          spearman_scores_start[mdl].append(st.spearmanr(pred_start_df[target_label].to_numpy(),pred_start_df[mdl].to_numpy()).correlation)
        else:
          spearman_scores_start[mdl].append(0.)
        if len(pred_middle_df.index) >= minNum_for_rank:
          spearman_scores_middle[mdl].append(st.spearmanr(pred_middle_df[target_label].to_numpy(),pred_middle_df[mdl].to_numpy()).correlation)
        else:
          spearman_scores_middle[mdl].append(0.)
        if len(pred_end_df.index) >= minNum_for_rank:
          spearman_scores_end[mdl].append(st.spearmanr(pred_end_df[target_label].to_numpy(),pred_end_df[mdl].to_numpy()).correlation)
        else:
          spearman_scores_end[mdl].append(0.)
        if len(pred_not_middle_df.index) >= minNum_for_rank:
          spearman_scores_not_middle[mdl].append(st.spearmanr(pred_not_middle_df[target_label].to_numpy(),pred_not_middle_df[mdl].to_numpy()).correlation)
        else:
          spearman_scores_not_middle[mdl].append(0.)
        if len(pred_middle_vaeNull_df.index) >= minNum_for_rank:
          spearman_scores_middle_vaeNull[mdl].append(st.spearmanr(pred_middle_vaeNull_df[target_label].to_numpy(),pred_middle_vaeNull_df[mdl].to_numpy()).correlation)
        else:
          spearman_scores_middle_vaeNull[mdl].append(0.)
      
      ## load mutagenesis predictions of my model (find the same mutant set compared to Shin2021)
      my_mut_values = pd.read_csv(f'{working_dir}/data_process/mutagenesis/results/raw_predictions/{muta_setNm}_seqMdl_famFinetuned.csv')
      ## remove duplicating rows(mutants)
      my_mut_values.drop_duplicates(subset=['mutant'], keep='first', inplace=True, ignore_index=True)
      ## remove wild-type cases
      my_pure_mut_values = my_mut_values[~(my_mut_values.mutant.str.match(r"([A-Za-z])\d+\1|wt|WT"))]
      # select mutants which are in pred_start_df
      my_pred_notnull_df = my_pure_mut_values.merge(pred_notnull_df,on=['mutant'],suffixes=(None, '_y'))
      my_pred_start_df = my_pure_mut_values.merge(pred_start_df,on=['mutant'],suffixes=(None, '_y'))
      my_pred_end_df = my_pure_mut_values.merge(pred_end_df,on=['mutant'],suffixes=(None, '_y'))
      my_pred_middle_df = my_pure_mut_values.merge(pred_middle_df,on=['mutant'],suffixes=(None, '_y'))
      my_pred_not_middle_df = my_pure_mut_values.merge(pred_not_middle_df,on=['mutant'],suffixes=(None, '_y'))
      my_pred_middle_vaeNull_df = my_pure_mut_values.merge(pred_middle_vaeNull_df,on=['mutant'],suffixes=(None, '_y'))
      for mdl in my_mdl_compare:
        if len(my_pred_notnull_df.index) >= minNum_for_rank:
          spearman_scores_all[mdl].append(st.spearmanr(my_pred_notnull_df[target_label].to_numpy(),my_pred_notnull_df[mdl].to_numpy()).correlation)
        else:
          spearman_scores_all[mdl].append(0.)
        if len(my_pred_start_df.index) >= minNum_for_rank:
          spearman_scores_start[mdl].append(st.spearmanr(my_pred_start_df[target_label].to_numpy(),my_pred_start_df[mdl].to_numpy()).correlation)
        else:
          spearman_scores_start[mdl].append(0.)
        if len(my_pred_middle_df.index) >= minNum_for_rank:
          spearman_scores_middle[mdl].append(st.spearmanr(my_pred_middle_df[target_label].to_numpy(),my_pred_middle_df[mdl].to_numpy()).correlation)
        else:
          spearman_scores_middle[mdl].append(0.)
        if len(my_pred_end_df.index) >= minNum_for_rank:
          spearman_scores_end[mdl].append(st.spearmanr(my_pred_end_df[target_label].to_numpy(),my_pred_end_df[mdl].to_numpy()).correlation)
        else:
          spearman_scores_end[mdl].append(0.)
        if len(my_pred_not_middle_df.index) >= minNum_for_rank:
          spearman_scores_not_middle[mdl].append(st.spearmanr(my_pred_not_middle_df[target_label].to_numpy(),my_pred_not_middle_df[mdl].to_numpy()).correlation)
        else:
          spearman_scores_not_middle[mdl].append(0.)
        if len(my_pred_middle_vaeNull_df.index) >= minNum_for_rank:
          spearman_scores_middle_vaeNull[mdl].append(st.spearmanr(my_pred_middle_vaeNull_df[target_label].to_numpy(),my_pred_middle_vaeNull_df[mdl].to_numpy()).correlation)
        else:
          spearman_scores_middle_vaeNull[mdl].append(0.)
      ## load spearman scores of pretrained model
      preSet_list = ['rp75_all_1','rp15_all_1','rp15_all_2','rp15_all_3','rp15_all_4']
      preEpoch_list = [224,729,729,729,729]
      preMdl_list = ['masked_language_modeling_transformer_21-08-23-02-59-06_850428','masked_language_modeling_transformer_21-05-22-23-09-12_354660','masked_language_modeling_transformer_21-05-07-19-48-02_813698','masked_language_modeling_transformer_21-05-01-22-15-01_112209','masked_language_modeling_transformer_21-05-01-22-22-18_663653']

      for pre_pair in zip(preSet_list,preEpoch_list,preMdl_list):
        with open(f'results_to_keep/{pre_pair[0][0:-2]}/{pre_pair[0][0:-2]}_pretrain_{pre_pair[0][-1]}_models/{pre_pair[2]}/mutagenesis_fitness_unSV/{muta_setNm}_{pre_pair[1]}.json') as handle:
          metric_json = json.load(handle)
          if f'{pre_pair[0]}_pre_{pre_pair[1]}' not in spearman_scores_middle.keys():
            spearman_scores_middle[f'{pre_pair[0]}_pre_{pre_pair[1]}'] = [abs(metric_json['spearmanR'])]
          else:
            spearman_scores_middle[f'{pre_pair[0]}_pre_{pre_pair[1]}'].append(abs(metric_json['spearmanR']))
      ## pretrain mean
      spearman_scores_middle['pre_only_mean'] = np.mean(np.vstack([spearman_scores_middle[f'{pre_pair[0]}_pre_{pre_pair[1]}'] for pre_pair in zip(preSet_list,preEpoch_list)]),axis=0).tolist()

      debug_file.write(f'{muta_setNm},{len(my_pred_start_df)},{len(pred_start_df)},{len(my_pred_middle_df)},{len(pred_middle_df)},{len(my_pred_end_df)},{len(pred_end_df)},{len(my_pred_not_middle_df)},{len(pred_not_middle_df)}\n')
    debug_file.close()
    ## save spearmanr scores
    with open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/all_region.json','w') as f:
      json.dump(spearman_scores_all, f)
    with open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/start_region.json','w') as f:
      json.dump(spearman_scores_start, f)
    with open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/middle_region.json','w') as f:
      json.dump(spearman_scores_middle, f)
    with open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/end_region.json','w') as f:
      json.dump(spearman_scores_end, f)
    with open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/not_middle_region.json','w') as f:
      json.dump(spearman_scores_not_middle, f)
    with open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/middle_vaeNull_region.json','w') as f:
      json.dump(spearman_scores_middle_vaeNull, f)
  else:
    ## load spearmanr scores
    with open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/all_region.json') as f:
      spearman_scores_all= json.load(f)
    with open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/start_region.json') as f:
      spearman_scores_start= json.load(f)
    with open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/middle_region.json') as f:
      spearman_scores_middle = json.load(f)
    with open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/end_region.json') as f:
      spearman_scores_end = json.load(f)
    with open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/not_middle_region.json') as f:
      spearman_scores_not_middle = json.load(f)
    with open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/middle_vaeNull_region.json') as f:
      spearman_scores_middle_vaeNull = json.load(f)
  
  data2draw = {'all': spearman_scores_all, 
               'start': spearman_scores_start, 
               'middle': spearman_scores_middle,
               'end': spearman_scores_end,
               'not_middle': spearman_scores_not_middle,
               'middle_vaeNull': spearman_scores_middle_vaeNull}
  ## numerical analysis
  ## * count (*/40 better) 
  ## * p-value (paired t-test)
  if nume_analysis:
    for region_nm in regions_draw:
      nume_ana_write = open(f'{working_dir}/data_process/mutagenesis/results/result_analysis/{region_nm}_nume.csv','w')
      score_sets = data2draw[region_nm]
      for mdl_pair in nume_mdl_pairs:
        score_set1,score_set2 = [],[]
        for i in range(len(score_sets[mdl_pair[0]])):
          ## remove cases that have no mutations
          if score_sets[mdl_pair[0]][i] == 0. and score_sets[mdl_pair[1]][i] == 0.:
            continue
          else:
            score_set1.append(score_sets[mdl_pair[0]][i])
            score_set2.append(score_sets[mdl_pair[1]][i])
        better_count = np.sum((np.abs(score_set1) - np.abs(score_set2)) >=0.)
        ave_diff = np.mean(np.abs(score_set1) - np.abs(score_set2))
        median_diff = np.median(np.abs(score_set1) - np.abs(score_set2))
        ks_twoSide = st.ks_2samp(np.abs(score_set1), np.abs(score_set2), alternative='two-sided')
        ks_less = st.ks_2samp(np.abs(score_set1), np.abs(score_set2), alternative='less')
        ks_greater = st.ks_2samp(np.abs(score_set1), np.abs(score_set2), alternative='greater')
        #ind_ttest = st.ttest_ind(np.abs(score_set1), np.abs(score_set2), equal_var=False)
        #ind_ttest_stat, ind_ttest_p = ind_ttest.statistic, ind_ttest.pvalue
        rel_ttest = st.ttest_rel(np.abs(score_set1), np.abs(score_set2))
        nume_ana_write.write(f'{mdl_name_convert[mdl_pair[0]]},{mdl_name_convert[mdl_pair[1]]},{better_count}/{len(score_set1)},t-rel:{rel_ttest.statistic}/{rel_ttest.pvalue},ks_2s:{ks_twoSide.statistic}/{ks_twoSide.pvalue},ks_less:{ks_less.statistic}/{ks_less.pvalue},ks_greater:{ks_greater.statistic}/{ks_greater.pvalue},ave diff:{ave_diff},median diff:{median_diff}\n')
      nume_ana_write.close()
  ## draw figures
  if draw_figure:
    ## set up color
    color_assign={
      'DeepSequence': 'black',
      'Evmutation': 'darkgrey',
      'Independent': 'tan',
      'HMM': 'lime',
      'Shin2021': 'blue',
      'Shin2021_fw_c48': 'blue',
      'Shin2021_rv_c48': 'blue',
      'Shin2021_fw_c24': 'blue',
      'Shin2021_rv_c24': 'blue',
      'Shin2021_fw': 'blue',
      'Shin2021_rv': 'blue',
      'Yue2021': 'orange',
      'OurSeqMdl': 'darkred',
      'OurSeqMdl_B1_rp75': 'darkred',
      'OurSeqMdl_B1_rp15': 'darkred',
      'OurSeqMdl_B2_rp15': 'darkred',
      'OurSeqMdl_B3_rp15': 'darkred',
      'OurSeqMdl_B4_rp15': 'darkred',
      'OurSeqMdl_e100_rwt': 'lightcoral',
      'OurSeqMdl_e100_nrwt': 'orangered'
    }
    marker_assign={
      'DeepSequence': ".",
      'Evmutation': ".",
      'Independent': ".",
      'HMM': ".",
      'Shin2021': ".",
      'Shin2021_fw_c48': '^',
      'Shin2021_rv_c48': 'v',
      'Shin2021_fw_c24': '<',
      'Shin2021_rv_c24': '>',
      'Shin2021_fw': '>',
      'Shin2021_rv': '<',
      'Yue2021': ".",
      'OurSeqMdl': ".",
      'OurSeqMdl_B1_rp75': '+',
      'OurSeqMdl_B1_rp15': '1',
      'OurSeqMdl_B2_rp15': '2',
      'OurSeqMdl_B3_rp15': '3',
      'OurSeqMdl_B4_rp15': '4',
      'OurSeqMdl_e100_rwt': '.',
      'OurSeqMdl_e100_nrwt': '.'
    }

    ## only contain 35 sets
    mutaNm_list_Shin2021 = np.loadtxt(f'{working_dir}/data_process/mutagenesis/results/mutaNm_Shin2021.tsv',dtype='str',delimiter='\t',skiprows=1)
    mutaNm35 = []
    for i in range(len(mutaNm_list_Shin2021)):
      #if mutaNm_list_Shin2021[i,1] not in ['HIS7_YEAST_Kondrashov2017']:
      mutaNm35.append(mutaNm_list_Shin2021[i,2])
    mutaNm35_len = len(mutaNm35)

    if fig_subNm == 'three_region': # three region figure: start, middle, end
      ## convert scores into dataframe
      spearman_list4df = []
      for reg_name in ['start','middle','end']:
        spearman_score_region = data2draw[reg_name]
        for mdl_nm, score_list in spearman_score_region.items():
          for muta_i in range(mutaNm35_len):
            spearman_list4df.append([np.abs(score_list[muta_i]),reg_name,mdl_name_convert[mdl_nm],mutaNm35[muta_i]])
      spearman_df = pd.DataFrame(spearman_list4df,columns=['spearmanR','region','model','mutaSet'])
      filter_df = spearman_df.loc[(spearman_df['spearmanR'] > 0) & (spearman_df['model'].isin(['Shin2021_fw','Shin2021_rv','HMM','OurSeqMdl']))]
      sns.set_theme(style="whitegrid")
      ax = sns.catplot(x="region", y="spearmanR", hue="model",
                      data=filter_df, kind="violin", height=4, aspect=1.5,
                      order=['start','middle','end'],hue_order=['Shin2021_fw','Shin2021_rv','HMM','OurSeqMdl'])
      ax.savefig(f'{working_dir}/data_process/mutagenesis/results/result_analysis/figures/{fig_subNm}_violin.png', format='png', dpi=800)
      plt.clf()
    elif fig_subNm in ['clean', 'crowd']:
      for set_nm in regions_draw:
        score_sets = data2draw[set_nm]
        ## remove proteins having no mutations for this region
        score_sets_new = {}
        mutaNm35_new = []
        mutaIdx = []
        for i in range(mutaNm35_len):
          value_oneSet = []
          for mdlNm in mdls_draw:
            value_oneSet.append(score_sets[mdlNm][i])
          if np.mean(value_oneSet) == 0.:
            continue
          else:
            mutaIdx.append(i)
            mutaNm35_new.append(mutaNm35[i])
        for mdlNm in mdls_draw:
          score_sets_new[mdlNm] = []
          for i in mutaIdx:
            score_sets_new[mdlNm].append(score_sets[mdlNm][i])    
        if not domain_group: 
          ## plot figures (absolute version)
          plt.figure()
          x=np.linspace(0, round(30*len(mutaNm35_new)/mutaNm35_len), len(mutaNm35_new))
          for setNm in mdls_draw:
            short_setNm = mdl_name_convert[setNm]
            ## check if NAN in list
            has_nan = True if True in np.isnan(np.array(score_sets[setNm])) else False
            if not has_nan:
              plt.scatter(x, np.abs(score_sets_new[setNm]), label=short_setNm, s=80, marker=marker_assign[short_setNm], color=color_assign[short_setNm], alpha=0.8)
            ## plot error bar(95% CI) for my models
            # if setNm in myConfIntv_dict.keys():
            #   plt.errorbar(x, score_sets[setNm], yerr=myConfIntv_dict[setNm], fmt='none', ecolor=color_assign[setNm], capsize=4.0, alpha=0.7)
          ## draw vertical line
          for i in x:
            plt.plot(np.linspace(i,i,10), np.linspace(-0.1,1,10), linestyle='--', color='gray', alpha=0.3)
          ## draw 0-horizontal line
          plt.plot([-1,round(30*len(mutaNm35_new)/mutaNm35_len)+1],[0,0], linestyle='--', color='gray', alpha=0.3)
          
          plt.ylabel("SpearmanR: pred vs exp", fontsize=25)
          plt.ylim([-0.05,1.0])
          plt.xlim([-1,round(30*len(mutaNm35_new)/mutaNm35_len)+1])
        
          for i in range(len(x)):
              plt.text(x[i]+0.05, -0.06, mutaNm35_new[i], rotation=50, fontsize=14, verticalalignment='top', horizontalalignment='right')

          plt.xticks(x, [])
          plt.legend(loc="upper right", fontsize=13)
          plt.yticks(fontsize=18)
          fig =plt.gcf()
          fig.set_size_inches(18.5, 10.5)
          plt.tight_layout()
          plt.savefig(f'{working_dir}/data_process/mutagenesis/results/result_analysis/figures/{set_nm}_abs_{fig_subNm}.png', format='png', dpi=800)
          
          ###!!! plot figures (true value version) !!!###
          plt.figure()
          x=np.linspace(0, round(30*len(mutaNm35_new)/mutaNm35_len), len(mutaNm35_new))
          for setNm in mdls_draw:
            short_setNm = mdl_name_convert[setNm]
            ## check if NAN in list
            has_nan = True if True in np.isnan(np.array(score_sets[setNm])) else False
            if not has_nan:
              plt.scatter(x, score_sets_new[setNm], label=short_setNm, s=80, marker=marker_assign[short_setNm], color=color_assign[short_setNm], alpha=0.8)
            ## plot error bar(95% CI) for my models
            # if setNm in myConfIntv_dict.keys():
            #   plt.errorbar(x, score_sets[setNm], yerr=myConfIntv_dict[setNm], fmt='none', ecolor=color_assign[setNm], capsize=4.0, alpha=0.7)
          for i in x:
            plt.plot(np.linspace(i,i,10), np.linspace(-1,1,10), linestyle='--', color='gray', alpha=0.3)
          ## draw 0-horizontal line
          plt.plot([-1,round(30*len(mutaNm35_new)/mutaNm35_len)+1],[0,0], linestyle='--', color='gray', alpha=0.3)

          plt.ylabel("SpearmanR: pred vs exp", fontsize=25)
          plt.ylim([-0.4,1.0])
          plt.xlim([-1,round(30*len(mutaNm35_new)/mutaNm35_len)+1])
        
          for i in range(len(x)):
              plt.text(x[i]+0.05, -0.41, mutaNm35_new[i], rotation=50, fontsize=14, verticalalignment='top', horizontalalignment='right')

          plt.xticks(x, [])
          plt.legend(loc="upper right", fontsize=13)
          plt.yticks(fontsize=18)
          fig =plt.gcf()
          fig.set_size_inches(18.5, 10.5)
          plt.tight_layout()
          plt.savefig(f'{working_dir}/data_process/mutagenesis/results/result_analysis/figures/{set_nm}_real_{fig_subNm}.png', format='png', dpi=800)
        else:
          ###!!! flag for single/multi domain !!!###
          sig_mul_domain = []
          for i in mutaIdx:
            #if mutaNm_list_Shin2021[i,1] not in ['HIS7_YEAST_Kondrashov2017']:
            sig_mul_domain.append(mutaNm_list_Shin2021[i,-1])
          sig_mul_domain = np.array(sig_mul_domain)
          sig_idx = np.argwhere(sig_mul_domain == '1')
          mul_idx = np.argwhere(sig_mul_domain == '0')
          reorder_idx = np.concatenate([sig_idx,mul_idx]).ravel()
          ###!!! plot figures (absolute version) !!!###
          plt.figure()
          x=np.linspace(0, round(30*len(mutaNm35_new)/mutaNm35_len), len(mutaNm35_new))
          for setNm in mdls_draw:
            short_setNm = mdl_name_convert[setNm]
            ## check if NAN in list
            has_nan = True if True in np.isnan(np.array(score_sets[setNm])) else False
            if not has_nan:
              plt.scatter(x, np.abs(np.array(score_sets_new[setNm])[reorder_idx]), label=short_setNm, s=80, marker=marker_assign[short_setNm], color=color_assign[short_setNm], alpha=0.8)
            ## plot error bar(95% CI) for my models
            # if setNm in myConfIntv_dict.keys():
            #   plt.errorbar(x, score_sets[setNm], yerr=myConfIntv_dict[setNm], fmt='none', ecolor=color_assign[setNm], capsize=4.0, alpha=0.7)
          ## draw vertical line
          for i in x:
            plt.plot(np.linspace(i,i,10), np.linspace(-0.1,1,10), linestyle='--', color='gray', alpha=0.3)
          ## draw 0-horizontal line
          plt.plot([-1,round(30*len(mutaNm35_new)/mutaNm35_len)+1],[0,0], linestyle='--', color='gray', alpha=0.3)  
          ## a vertical line splittig single-multi domain groups
          pos_x = np.mean([x[len(sig_idx)-1],x[len(sig_idx)]])
          plt.plot(np.linspace(pos_x,pos_x,10), np.linspace(-0.1,1,10), linestyle='--', color='red', alpha=0.3)

          plt.ylabel("SpearmanR: pred vs exp", fontsize=25)
          plt.ylim([-0.05,1.0])
          plt.xlim([-1,round(30*len(mutaNm35_new)/mutaNm35_len)+1])
        
          for i in range(len(x)):
              plt.text(x[i]+0.05, -0.06, mutaNm35_new[reorder_idx[i]], rotation=50, fontsize=14, verticalalignment='top', horizontalalignment='right')

          plt.xticks(x, [])
          plt.legend(loc="upper right", fontsize=13)
          plt.yticks(fontsize=18)
          fig =plt.gcf()
          fig.set_size_inches(18.5, 10.5)
          plt.tight_layout()
          plt.savefig(f'{working_dir}/data_process/mutagenesis/results/result_analysis/figures/{set_nm}_abs_{fig_subNm}_groupDomain.png', format='png', dpi=800)
          ###!!! plot figures (true value version) !!!###
          plt.figure()
          x=np.linspace(0, round(30*len(mutaNm35_new)/mutaNm35_len), len(mutaNm35_new))
          for setNm in mdls_draw:
            short_setNm = mdl_name_convert[setNm]
            ## check if NAN in list
            has_nan = True if True in np.isnan(np.array(score_sets[setNm])) else False
            if not has_nan:
              plt.scatter(x, np.array(score_sets_new[setNm])[reorder_idx], label=short_setNm, s=80, marker=marker_assign[short_setNm], color=color_assign[short_setNm], alpha=0.8)
            ## plot error bar(95% CI) for my models
            # if setNm in myConfIntv_dict.keys():
            #   plt.errorbar(x, score_sets[setNm], yerr=myConfIntv_dict[setNm], fmt='none', ecolor=color_assign[setNm], capsize=4.0, alpha=0.7)
          for i in x:
            plt.plot(np.linspace(i,i,10), np.linspace(-1,1,10), linestyle='--', color='gray', alpha=0.3)
          ## draw 0-horizontal line
          plt.plot([-1,round(30*len(mutaNm35_new)/mutaNm35_len)+1],[0,0], linestyle='--', color='gray', alpha=0.3)  
          ## a vertical line splittig single-multi domain groups
          pos_x = np.mean([x[len(sig_idx)-1],x[len(sig_idx)]])
          plt.plot(np.linspace(pos_x,pos_x,10), np.linspace(-0.1,1,10), linestyle='--', color='red', alpha=0.3)

          plt.ylabel("SpearmanR: pred vs exp", fontsize=25)
          plt.ylim([-0.4,1.0])
          plt.xlim([-1,round(30*len(mutaNm35_new)/mutaNm35_len)+1])
        
          for i in range(len(x)):
              plt.text(x[i]+0.05, -0.41, mutaNm35_new[reorder_idx[i]], rotation=50, fontsize=14, verticalalignment='top', horizontalalignment='right')

          plt.xticks(x, [])
          plt.legend(loc="upper right", fontsize=13)
          plt.yticks(fontsize=18)
          fig =plt.gcf()
          fig.set_size_inches(18.5, 10.5)
          plt.tight_layout()
          plt.savefig(f'{working_dir}/data_process/mutagenesis/results/result_analysis/figures/{set_nm}_real_{fig_subNm}_groupDomain.png', format='png', dpi=800)

    elif fig_subNm in ['stacked_bar']:
      r_cutoff = [0.2,0.4,0.6,1.0]
      for set_nm in regions_draw:
        score_sets = data2draw[set_nm]
        ## remove proteins having no mutations for this region
        score_sets_new = {}
        mutaNm35_new = []
        mutaIdx = []
        for i in range(mutaNm35_len):
          value_oneSet = []
          for mdlNm in mdls_draw:
            value_oneSet.append(score_sets[mdlNm][i])
          if np.mean(value_oneSet) == 0.:
            continue
          else:
            mutaIdx.append(i)
            mutaNm35_new.append(mutaNm35[i])
        for mdlNm in mdls_draw:
          score_sets_new[mdlNm] = []
          for i in mutaIdx:
            score_sets_new[mdlNm].append(score_sets[mdlNm][i])
        score_bin_count = []
        mdlNm_new = []
        for mdlNm in mdls_draw:
          mdlNm_new.append(mdl_name_convert[mdlNm])
          score_bin_count.append([np.where(np.array(score_sets_new[mdlNm])<r_cutoff[0],True,False).sum(),np.where((np.array(score_sets_new[mdlNm])>=r_cutoff[0]) & (np.array(score_sets_new[mdlNm])<r_cutoff[1]),True,False).sum(),np.where((np.array(score_sets_new[mdlNm])>=r_cutoff[1]) & (np.array(score_sets_new[mdlNm])<r_cutoff[2]),True,False).sum(),np.where(np.array(score_sets_new[mdlNm])>=r_cutoff[2],True,False).sum()])
        score_bin_count = np.array(score_bin_count)
        print(score_bin_count)
        x_pos=np.linspace(0, len(mdlNm_new)/1.5+0.5, len(mdlNm_new))
        bar_bin_labels = ['low','medium-','medium+','high']
        fig = plt.figure(figsize=(18,10))
        ax = plt.subplot(111)
        plt.rcParams.update({'font.size': 14})
        for i in range(len(bar_bin_labels)):
          if i == 0:
            plt.bar(x_pos,score_bin_count[:,i],width=0.4,label=f'{bar_bin_labels[i]}(<{r_cutoff[i]})')  
          else:
            y_bottom_arr = score_bin_count[:,0:i].reshape(-1,i)
            y_bottom = np.sum(y_bottom_arr,axis=-1)
            plt.bar(x_pos,score_bin_count[:,i],bottom=y_bottom,width=0.4,label=f'{bar_bin_labels[i]}(<{r_cutoff[i]})')
          pos_y_arr = np.copy(score_bin_count[:,0:i+1].reshape(-1,i+1)).astype(float)
          pos_y_arr[:,-1] = pos_y_arr[:,-1] / 2.
          for xpos, ypos, yval in zip(x_pos, np.sum(pos_y_arr,axis=-1), score_bin_count[:,i]):
            plt.text(xpos, ypos, yval, ha="center", va="center")
        
        for i in range(len(x_pos)):
          plt.text(x_pos[i]+0.05, -0.2, mdlNm_new[i], rotation=15, fontsize=14, verticalalignment='top', horizontalalignment='right')
        plt.xticks(x_pos, [])
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.ylim([0,42])
        #plt.legend(loc="upper right", fontsize=13)
        plt.savefig(f'{working_dir}/data_process/mutagenesis/results/result_analysis/figures/{set_nm}_abs_{fig_subNm}_sota_1.png', format='png', dpi=800)

def analysis_mutOrder(
      working_dir: str = None,
      shin_muta_path: str = None,
      exp_label_set: str = None,
      sota_mdl_compare: List = None,
      my_mdl_compare: List = None,
      mdls_draw: List = None,
      region_name: str = None,
      sets_draw: List = None,
      fig_subNm: str = None,
      count_sites: bool = None,
      draw_figure: bool = True,
      minNum_for_rank: int = 10):
  """
    single-site vs multiple-site variants (HMM vs Shin2021 vs Yue2021(pending) vs MySeqMdl)
  """
  ## load mutagenesis lable names
  muta_label_names = np.loadtxt(f'{working_dir}/data_process/mutagenesis/target_label_set_Shin2021.tsv', dtype='str',delimiter='\t',skiprows=1)
  if count_sites:
    for ii in range(len(muta_label_names)):
      pred_file = muta_label_names[ii][0]
      muta_setNm = muta_label_names[ii][-1]
      print(f'>>muta_setNm: {muta_setNm}<<')
      # if muta_setNm in ['HIS7_YEAST_Kondrashov2017']:
      #   continue
      mut_values = pd.read_csv(f'{shin_muta_path}/output_processed/{pred_file}')
      ## remove duplicating rows(mutants)
      mut_values.drop_duplicates(subset=['mutant'], keep='first', inplace=True, ignore_index=True)
      ## remove wild-type cases
      pure_mut_values = mut_values[~(mut_values.mutant.str.match(r"([A-Za-z])\d+\1|wt|WT"))]
      mutant_site_count = pure_mut_values['mutant'].str.count(':').to_numpy()
      mut_siteType, mut_siteNum = np.unique(mutant_site_count+1, return_counts=True)
      print(f'{list(zip(mut_siteType,mut_siteNum))}')

  spearman_scores = {}
  for setNm in sets_draw:
    print(f'>>spearmanR calculation')
    print(f'>>muta_setNm: {setNm}<<')
    spearman_scores[setNm] = {}
    muta_label_list = os.popen(f"grep '{setNm}' {working_dir}/data_process/mutagenesis/target_label_set_Shin2021.tsv").read().strip('\n').split('\t')
    if exp_label_set == 'deepsequence':
      target_label = muta_label_list[2] ## DeepSequence provided exp label name
    elif exp_label_set == 'yue':
      target_label = muta_label_list[1] ## Yue provided exp label name
    mut_values = pd.read_csv(f'{shin_muta_path}/output_processed/{muta_label_list[0]}')
    ## remove duplicating rows(mutants)
    mut_values.drop_duplicates(subset=['mutant'], keep='first', inplace=True, ignore_index=True)
    ## remove wild-type cases
    pure_mut_values = mut_values[~(mut_values.mutant.str.match(r"([A-Za-z])\d+\1|wt|WT"))]

    ## take care of region
    exp_notnull_df = pure_mut_values[pure_mut_values[target_label].notnull()]
    pred_notnull_df = exp_notnull_df[exp_notnull_df['mutation_effect_prediction_all_mean_autoregressive'].notnull()]
    pred_vae_notnull_idx = exp_notnull_df[exp_notnull_df['mutation_effect_prediction_vae_ensemble'].notnull()].index.tolist()
    if region_name == 'all':
      pred_target_df = pred_notnull_df
    elif region_name == 'start':
      pred_target_df = pred_notnull_df[pred_notnull_df.index < pred_vae_notnull_idx[0]] # vae null start (might be empty)
    elif region_name == 'middle':
      pred_target_df = pred_notnull_df[(pred_notnull_df.index.isin(pred_vae_notnull_idx))]
    elif region_name == 'not_middle':
      pred_target_df = pred_notnull_df[~(pred_notnull_df.index.isin(pred_vae_notnull_idx))]
    elif region_name == 'end':
      pred_target_df = pred_notnull_df[pred_notnull_df.index > pred_vae_notnull_idx[-1]] # vae null tail (might be empty)

    mutant_site_count = pred_target_df['mutant'].str.count(':')
    mut_siteType, mut_siteNum = np.unique(mutant_site_count.to_numpy()+1, return_counts=True) # returned values are sorted (small to large)
    spearman_scores[setNm]['site_type'] = []
    ## load mutagenesis predictions of my model (find the same mutant set compared to Shin2021)
    my_mut_values = pd.read_csv(f'{working_dir}/data_process/mutagenesis/results/raw_predictions/{setNm}_seqMdl_famFinetuned.csv')
    ## remove duplicating rows(mutants)
    my_mut_values.drop_duplicates(subset=['mutant'], keep='first', inplace=True, ignore_index=True)
    ## remove wild-type cases
    my_pure_mut_values = my_mut_values[~(my_mut_values.mutant.str.match(r"([A-Za-z])\d+\1|wt|WT"))]
    for mdl in sota_mdl_compare:
      spearman_scores[setNm][mdl] = []
      ## loop over mutant-site types from 1-site to n-site
      for siteTyIdx in range(len(mut_siteType)):
        mut_order = mut_siteType[siteTyIdx]
        #print(f'>>> mut_order: {mut_order}')
        ## get sota spearmanr
        sele_mut_rows = pred_target_df[pred_target_df['mutant'].str.count(':') == mut_siteType[siteTyIdx]-1]
        mut_rows_zero = sele_mut_rows[sele_mut_rows[target_label] == 0.]
        if len(sele_mut_rows.index) >= minNum_for_rank or len(mut_rows_zero.index) == len(sele_mut_rows):
          spr = st.spearmanr(sele_mut_rows[target_label].to_numpy(),sele_mut_rows[mdl].to_numpy(),nan_policy='omit').correlation
          if not np.isnan(spr):
            spearman_scores[setNm][mdl].append(spr)
            if mut_order not in spearman_scores[setNm]['site_type']:
              spearman_scores[setNm]['site_type'].append(mut_order)
    for mdl in my_mdl_compare:
      spearman_scores[setNm][mdl] = []
      ## loop over mutant-site types from 1-site to n-site
      for siteTyIdx in range(len(mut_siteType)):
        ## get my model spearmanr
        sele_mut_rows = pred_target_df[pred_target_df['mutant'].str.count(':') == mut_siteType[siteTyIdx]-1]
        mut_rows_zero = sele_mut_rows[sele_mut_rows[target_label] == 0.]
        my_sele_mut_rows = my_pure_mut_values.merge(sele_mut_rows,on=['mutant'],suffixes=(None, '_y'))
        if len(sele_mut_rows.index) >= minNum_for_rank or len(mut_rows_zero.index) == len(sele_mut_rows):
          spr = st.spearmanr(my_sele_mut_rows[target_label].to_numpy(),my_sele_mut_rows[mdl].to_numpy(),nan_policy='omit').correlation
          if not np.isnan(spr):
            spearman_scores[setNm][mdl].append(spr)

  if draw_figure:
    print('Draw figure')
    mdl_name_convert = {
      'mutation_effect_prediction_vae_ensemble': 'DeepSequence',
      'mutation_effect_prediction_pairwise': 'Evmutation',
      'mutation_effect_prediction_independent': 'Independent',
      'hmm_effect_prediction_mean': 'HMM',
      'mutation_effect_prediction_all_mean_autoregressive': 'Shin2021',
      'mutation_effect_prediction_forward_mean_channels-48': 'Shin2021_fw_c48',
      'mutation_effect_prediction_reverse_mean_channels-48': 'Shin2021_rv_c48',
      'mutation_effect_prediction_forward_mean_channels-24': 'Shin2021_fw_c24',
      'mutation_effect_prediction_reverse_mean_channels-24': 'Shin2021_rv_c24',
      'mutation_effect_prediction_forward_mean': 'Shin2021_fw',
      'mutation_effect_prediction_reverse_mean': 'Shin2021_rv',
      'pretrain_seqMdl_mean': 'OurSeqMdl',
      'rp75_all_1_mean': 'OurSeqMdl_B1_rp75',
      'rp15_all_1_mean': 'OurSeqMdl_B1_rp15',
      'rp15_all_2_mean': 'OurSeqMdl_B2_rp15',
      'rp15_all_3_mean': 'OurSeqMdl_B3_rp15',
      'rp15_all_4_mean': 'OurSeqMdl_B4_rp15'
    }
    
    ## set up color
    color_assign={
      'DeepSequence': 'black',
      'Evmutation': 'darkgrey',
      'Independent': 'tan',
      'HMM': 'lime',
      'Shin2021': 'blue',
      'Shin2021_fw_c48': 'blue',
      'Shin2021_rv_c48': 'blue',
      'Shin2021_fw_c24': 'blue',
      'Shin2021_rv_c24': 'blue',
      'Shin2021_fw': 'blue',
      'Shin2021_rv': 'blue',
      'Yue2021': 'orange',
      'OurSeqMdl': 'darkred',
      'OurSeqMdl_B1_rp75': 'darkred',
      'OurSeqMdl_B1_rp15': 'darkred',
      'OurSeqMdl_B2_rp15': 'darkred',
      'OurSeqMdl_B3_rp15': 'darkred',
      'OurSeqMdl_B4_rp15': 'darkred',
    }
    marker_assign={
      'DeepSequence': ".",
      'Evmutation': ".",
      'Independent': ".",
      'HMM': ".",
      'Shin2021': ".",
      'Shin2021_fw_c48': '^',
      'Shin2021_rv_c48': 'v',
      'Shin2021_fw_c24': '<',
      'Shin2021_rv_c24': '>',
      'Shin2021_fw': '>',
      'Shin2021_rv': '<',
      'Yue2021': ".",
      'OurSeqMdl': ".",
      'OurSeqMdl_B1_rp75': '+',
      'OurSeqMdl_B1_rp15': '1',
      'OurSeqMdl_B2_rp15': '2',
      'OurSeqMdl_B3_rp15': '3',
      'OurSeqMdl_B4_rp15': '4',
    }

    ## plot figures
    #for set_nm, score_sets in data2draw.items():
    for muta_setNm in sets_draw:
      print(f'>>muta_setNm: {muta_setNm}<<')
      score_sets = spearman_scores[muta_setNm]
      mut_siteType = spearman_scores[muta_setNm]['site_type']
      plt.figure()
      x=np.linspace(0, len(mut_siteType), len(mut_siteType))
      for setNm in mdls_draw:
        short_setNm = mdl_name_convert[setNm]
        ## check if NAN in list
        has_nan = True if True in np.isnan(np.array(score_sets[setNm])) else False
        if not has_nan:
          plt.scatter(x, np.abs(score_sets[setNm]), label=short_setNm, s=80, marker=marker_assign[short_setNm], color=color_assign[short_setNm], alpha=0.7)
        ## plot error bar(95% CI) for my models
        # if setNm in myConfIntv_dict.keys():
        #   plt.errorbar(x, score_sets[setNm], yerr=myConfIntv_dict[setNm], fmt='none', ecolor=color_assign[setNm], capsize=4.0, alpha=0.7)
      for i in x:
        plt.plot(np.linspace(i,i,10), np.linspace(-0.1,1,10), linestyle='--', color='gray', alpha=0.3)

      plt.ylabel(f"{muta_setNm} SpearmanR", fontsize=20)
      plt.ylim([-0.0,1.0])
      plt.xlim([-0.25,len(mut_siteType)+0.25])
    
      for i in range(len(x)):
          plt.text(x[i]+0.005, -0.01, mut_siteType[i], rotation=0, fontsize=14, verticalalignment='top', horizontalalignment='right')

      plt.xticks(x, [])
      plt.legend(loc="upper right", fontsize=13)
      plt.yticks(fontsize=18)
      fig =plt.gcf()
      fig.set_size_inches(18.5, 10.5)
      plt.tight_layout()
      plt.savefig(f'{working_dir}/data_process/mutagenesis/results/result_analysis/figures/{muta_setNm}_{fig_subNm}.png', format='png', dpi=800)  

def multitask_fitness_analysis(
      family_list: List,
      figure_list: List
      ):
  """fitness prediction performance analysis over multitask models
  """
  root_path = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  balance_mode_list = ['ub', 'cb', 'lcb']
  mp_layer_list=['noMP','gatv2'] #'noMP' 'gatv2' 'gine' 'transformer'
  graph_strategy_dict={
    'noMP': ['noGS'],
    'gatv2': ['knnDistCut','distCut','knn','full','random','sequential'] # 'knnDistCut' 'distCut' 'knn' 'full' 'random' 'sequential'
  }
  eval_epoch_list = [79,99,'best']
  eval_epoch_kw_dict = {79:'_79',99:'_99','best':''}
  metric_score_df = pd.DataFrame(columns=['muta_set','balance_mode','mp_type','graph_strategy','mp_gs','eval_epoch','aa_ppl','spearman_r','spearman_p_value','pearson_r','pearson_p_value'])
  # mapping between family name and mutagenesis set name
  fam_muta_mapping = pd.read_csv(f'{root_path}/data_process/mutagenesis/DeepSequenceMutaSet_pfam_clan',decimal=',',header=0)
  mutaSet_dict = {}
  for fam in family_list:
    muta_targets = fam_muta_mapping.loc[fam_muta_mapping['Shin2021_set'] == fam]['setNM'].values.tolist()
    for mutS in muta_targets:
      mutaSet_dict[mutS] = fam
  
  # gather prediction and GT labels
  # loop over muta sets
  for mutS,fam in mutaSet_dict.items():
    for blc_mode in balance_mode_list:
      for mp in mp_layer_list:
        for gs in graph_strategy_dict[mp]:
          for eval_epoch in eval_epoch_list:
            if blc_mode == 'ub':
              model_id=os.popen(f"grep 'loading weights file' job_logs/archive_multitask_models_eval/multitask_eval_bert_rp75_all_1_best.shin2021Data.reweighted.2.*.GraStra_{gs}.Knn_20.DistCut_12.0.MP_{mp}.epoch_{eval_epoch}.mutaSet_{mutS}.out | awk -F '{fam}/' '{{print $2}}'| cut -d'/' -f1 | tr -d '\r'").read().strip('\n')
            elif blc_mode == 'cb':
              model_id=os.popen(f"grep 'loading weights file' job_logs/archive_multitask_models_eval/multitask_eval_bert_rp75_all_1_best.shin2021Data.reweighted.2.*.GraStra_{gs}.Knn_20.DistCut_12.0.MP_{mp}.ClassBalance.epoch_{eval_epoch}.mutaSet_{mutS}.out | awk -F '{fam}/' '{{print $2}}'| cut -d'/' -f1 | tr -d '\r'").read().strip('\n')
            elif blc_mode == 'lcb':
              model_id=os.popen(f"grep 'loading weights file' job_logs/archive_multitask_models_eval/multitask_eval_bert_rp75_all_1_best.shin2021Data.reweighted.2.*.GraStra_{gs}.Knn_20.DistCut_12.0.MP_{mp}.L_aa1.0_ss0.33_rsa0.33_dist0.33.epoch_{eval_epoch}.mutaSet_{mutS}.out | awk -F '{fam}/' '{{print $2}}'| cut -d'/' -f1 | tr -d '\r'").read().strip('\n')
            print(f'{mutS},{blc_mode},{mp},{gs},{eval_epoch},{model_id}')
            with open(f'{root_path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/predictions/{model_id}/{mutS}{eval_epoch_kw_dict[eval_epoch]}_metrics.json') as f:
              metric_json = json.load(f)
            row2append = pd.DataFrame([{'muta_set':mutS,'balance_mode':blc_mode,'mp_type':mp,'graph_strategy':gs,'mp_gs':f'{mp}_{gs}','eval_epoch':eval_epoch,'aa_ppl':metric_json['aa_ppl'],'spearman_r':metric_json['spearmanR'],'spearman_p_value':metric_json['spearmanPvalue'],'pearson_r':metric_json['pearsonR'],'pearson_p_value':metric_json['pearsonPvalue']}])
            metric_score_df = pd.concat([metric_score_df, row2append], ignore_index=True)

  # draw figures
  for fig_name in figure_list:
    if fig_name == '3_balance_epoch_graphDef_bar':
      r_cutoff = [0.1,0.2,0.3,0.6,1.0]
      bar_bin_labels = ['L1','L2','L3','L4','L5']
      for blc_mode in balance_mode_list:
        for eval_epoch in eval_epoch_list:
          score_bin_count = []
          mdlNm_list = []
          for mp in mp_layer_list:
            for gs in graph_strategy_dict[mp]:
              target_df = metric_score_df.loc[(metric_score_df['balance_mode'] == blc_mode) & (metric_score_df['mp_type'] == mp) & (metric_score_df['graph_strategy'] == gs) & (metric_score_df['eval_epoch'] == eval_epoch)]
              mdlNm_list.append(f'{mp}-{gs}')
              pred_spearman_r = target_df['spearman_r'].to_list()
              score_bin_count.append([np.where(np.absolute(pred_spearman_r)<r_cutoff[0],True,False).sum(),
                                      np.where((np.absolute(pred_spearman_r)>=r_cutoff[0]) & (np.absolute(pred_spearman_r)<r_cutoff[1]),True,False).sum(),
                                      np.where((np.absolute(pred_spearman_r)>=r_cutoff[1]) & (np.absolute(pred_spearman_r)<r_cutoff[2]),True,False).sum(),
                                      np.where((np.absolute(pred_spearman_r)>=r_cutoff[2]) & (np.absolute(pred_spearman_r)<r_cutoff[3]),True,False).sum(),
                                      np.where(np.absolute(pred_spearman_r)>=r_cutoff[3],True,False).sum()])
          score_bin_count = np.array(score_bin_count)
          #print(score_bin_count)
          x_pos=np.linspace(0, len(mdlNm_list)/1.5+0.5, len(mdlNm_list))
          fig = plt.figure(figsize=(18,10))
          ax = plt.subplot(111)
          plt.rcParams.update({'font.size': 14})
          for i in range(len(bar_bin_labels)):
            if i == 0:
              plt.bar(x_pos,score_bin_count[:,i],width=0.4,label=f'{bar_bin_labels[i]}(<{r_cutoff[i]})')  
            else:
              y_bottom_arr = score_bin_count[:,0:i].reshape(-1,i)
              y_bottom = np.sum(y_bottom_arr,axis=-1)
              plt.bar(x_pos,score_bin_count[:,i],bottom=y_bottom,width=0.4,label=f'{bar_bin_labels[i]}(<{r_cutoff[i]})')
            pos_y_arr = np.copy(score_bin_count[:,0:i+1].reshape(-1,i+1)).astype(float)
            pos_y_arr[:,-1] = pos_y_arr[:,-1] / 2.
            for xpos, ypos, yval in zip(x_pos, np.sum(pos_y_arr,axis=-1), score_bin_count[:,i]):
              plt.text(xpos, ypos, yval, ha="center", va="center")
          
          for i in range(len(x_pos)):
            plt.text(x_pos[i]+0.05, -0.2, mdlNm_list[i], rotation=15, fontsize=14, verticalalignment='top', horizontalalignment='right')
          plt.xticks(x_pos, [])
          # Shrink current axis by 20%
          box = ax.get_position()
          ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

          # Put a legend to the right of the current axis
          ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
          #plt.ylim([0,42])
          #plt.legend(loc="upper right", fontsize=13)
          plt.savefig(f'{root_path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/stackedBar_{blc_mode}_{eval_epoch}.png', format='png', dpi=800)
    elif fig_name == '3_balance_epoch_graphDef_violin':
      for blc_mode in balance_mode_list:
        for eval_epoch in eval_epoch_list:
          target_df = metric_score_df.loc[(metric_score_df['balance_mode'] == blc_mode) & (metric_score_df['eval_epoch'] == eval_epoch)]
          fig, ax = plt.subplots(figsize=(9, 8))
          sns.set(style = 'whitegrid')
          ## violin plot
          ax = sns.violinplot(x="mp_gs", y="spearman_r", data=target_df)
          _ = plt.xticks(rotation=30)
          plt.ylim(top=0.6)
          plt.savefig(f'{root_path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/violin_{blc_mode}_{eval_epoch}.png', format='png', dpi=800)
      for gs in ['knnDistCut','distCut','knn','full','random','sequential','noGS']:
        target_df = metric_score_df.loc[(metric_score_df['graph_strategy'] == gs) & (metric_score_df['eval_epoch'].isin([99,'best']))]
        fig, ax = plt.subplots()
        sns.set(style = 'whitegrid')
        ## violin plot
        ax = sns.violinplot(x="balance_mode", y="spearman_r", hue="eval_epoch", data=target_df,split=True)
        #_ = plt.xticks(rotation=30)
        plt.ylim(top=0.6)
        plt.savefig(f'{root_path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/violin_{gs}_99_best.png', format='png', dpi=800)
  return None

def fitScan_figure(
      data_path: str = None,
      mdl_sets: List = None,
      figure_name: str = None,
      topPos_num: int = 10):
  """
  Plot figures for fitness scanning analysis

  Params:

  Outputs:
  """
  allAA_ids = [12,19,10,7,6,15,18,21,20,5,9,4,23,13,11,14,17,25,8,24]
  allAA_char = ["K","R","H","E","D","N","Q","T","S","C","G","A","V","L","I","M","P","Y","F","W"]
  delta_var = np.array([[484,"Q"],[417,"N"]],dtype=object)
  omicron_BA2_var = np.array([[371,'F'],[373,'P'],[375,'F'],[376,'A'],[405,'N'],[408,'S'],[417,'N'],[440,'K'],[477,'N'],[478,'K'],[484,'A'],[493,'R'],[498,'R'],[501,'Y'],[505,'H']],dtype=object)
  omicron_BA1_var = np.array([[371,'L'],[373,'P'],[375,'F'],[417,'N'],[440,'K'],[446,'S'],[477,'N'],[478,'K'],[484,'A'],[493,'R'],[496,'S'],[498,'R'],[501,'Y'],[505,'H']],dtype=object)
  wtSeq = 'ASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYRYRLFRKSNLKPFERDISTEIYQAGSKPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCG'
  all_scores = []
  ## load predicted scores
  for mdl_id in mdl_sets:
    fit_score = np.loadtxt(f'{data_path}/deltaRBD_mutScan_predFit_{mdl_id}_best.csv',dtype='float',delimiter=',')
    fit_score_posSort = fit_score[np.argsort(fit_score[:,0])]
    all_scores.append(fit_score_posSort)
  all_scores = np.array(all_scores)
  pos_list = all_scores[0,:,0].astype(int)
  score_arr = all_scores[:,:,1:] 
  mdl_ave = np.mean(score_arr,axis=0) #[n_pos,20]
  mdl_pos_ave = np.mean(mdl_ave,axis=-1)
  mdl_pos_med = np.median(mdl_ave,axis=-1)
  
  ## plot figure
  if figure_name == 'pos_wise':
    marker_size = 4
    plt.figure(figsize=(40,5))
    plt.plot(pos_list,mdl_pos_ave)
    y_min,y_max = plt.gca().get_ylim()
    plt.plot(delta_var[:,0],[y_min]*len(delta_var[:,0]),linestyle='None',color='lime',marker='o',label='delta+_var',markersize=marker_size)
    plt.plot(omicron_BA1_var[:,0],[y_min+0.5]*len(omicron_BA1_var[:,0]),linestyle='None',color='orange',marker='o',label='omicro_BA1_var',markersize=marker_size)
    plt.plot(omicron_BA2_var[:,0],[y_min+1]*len(omicron_BA2_var[:,0]),linestyle='None',color='cyan',marker='o',label='omicron_BA2_var',markersize=marker_size)
    top_pos = pos_list[np.argsort(mdl_pos_ave)[-1*topPos_num:]]
    for hl_pos in top_pos:
      plt.axvline(x=hl_pos,color='red')
    plt.xticks(pos_list, rotation=90)
    plt.tick_params(axis='x', which='major', labelsize=9)
    plt.gca().set_xlim([pos_list[0]-1,pos_list[-1]+1])
    plt.grid()
    plt.legend()
    plt.savefig(f'{data_path}/deltaRBD_mutScan_posWise_ave.png')
    plt.clf()

    plt.figure(figsize=(40,5))
    plt.plot(pos_list,mdl_pos_med)
    y_min,y_max = plt.gca().get_ylim()
    plt.plot(delta_var[:,0],[y_min]*len(delta_var[:,0]),linestyle = 'None',color='lime',marker='o',label='delta+_var',markersize=marker_size)
    plt.plot(omicron_BA1_var[:,0],[y_min+0.5]*len(omicron_BA1_var[:,0]),linestyle = 'None',color='orange',marker='o',label='omicron_BA1_var',markersize=marker_size)
    plt.plot(omicron_BA2_var[:,0],[y_min+1]*len(omicron_BA2_var[:,0]),linestyle = 'None',color='cyan',marker='o',label='omicron_BA2_var',markersize=marker_size)
    top_pos = pos_list[np.argsort(mdl_pos_med)[-1*topPos_num:]]
    for hl_pos in top_pos:
      plt.axvline(x=hl_pos,color='red')
    plt.xticks(pos_list, rotation=90)
    plt.tick_params(axis='x', which='major', labelsize=9)
    plt.gca().set_xlim([pos_list[0]-1,pos_list[-1]+1])
    plt.grid()
    plt.legend()
    plt.savefig(f'{data_path}/deltaRBD_mutScan_posWise_median.png')
    plt.clf()
  if figure_name == 'mut_wise':
    fig = plt.figure(figsize=(48,5))
    ax = plt.gca()
    divnorm=colors.TwoSlopeNorm(vmin=np.amin(mdl_ave), vcenter=0., vmax=np.amax(mdl_ave))
    im = ax.matshow(mdl_ave.T,cmap=plt.cm.bwr,aspect="auto",norm=divnorm)
    
    ax.set_xticks(np.arange(len(pos_list)))
    ax.set_xticklabels([f'{wtSeq[i]}{pos_list[i]}' for i in range(len(pos_list))])
    ax.set_yticks(np.arange(20))
    ax.set_yticklabels(allAA_char)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=90, ha="right", va="center", rotation_mode="anchor")
    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(pos_list), 1), minor=True)
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    # cell annotation
    for x in range(mdl_ave.shape[0]): #n_pos
      ## order muts for one position
      mut_sort = np.argsort(np.argsort(mdl_ave[x,:]))
      for y in range(mdl_ave.shape[1]): #20
        plt.text(x, y, f'{allAA_char[y]}{19-mut_sort[y]+1}',
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=6)
    # mark cell
    for var_i in omicron_BA1_var:
      var_pos = var_i[0] - pos_list[0]
      var_aa = np.where(np.array(allAA_char) == var_i[1])[0][0]
      rect = plt.Rectangle((var_pos-.5, var_aa-.4), 1,1, fill=False, color="orange", linewidth=1)
      ax.add_patch(rect)
    for var_i in omicron_BA2_var:
      var_pos = var_i[0] - pos_list[0]
      var_aa = np.where(np.array(allAA_char) == var_i[1])[0][0]
      rect = plt.Rectangle((var_pos-.5, var_aa-.6), 1,1, fill=False, color="cyan", linewidth=1)
      ax.add_patch(rect)
    for var_i in delta_var:
      var_pos = var_i[0] - pos_list[0]
      var_aa = np.where(np.array(allAA_char) == var_i[1])[0][0]
      rect = plt.Rectangle((var_pos-.5, var_aa-.5), 1,1, fill=False, color="lime", linewidth=1)
      ax.add_patch(rect)
    
    
    fig.colorbar(im)
    plt.savefig(f'{data_path}/deltaRBD_mutScan_mutWise.png',dpi=300)
    plt.clf()


def pos_select_figure(data_path: str,
                      mdl_sets: List,
                      figure_name: str,
                      var_set: str):
  allAA_char = ["K","R","H","E","D","N","Q","T","S","C","G","A","V","L","I","M","P","Y","F","W"]
  delta_BA2_var = np.array([[19,"T","R","I"],[142,"G","D","D"],[156,"E","G","E"],[452,"L","R","L"],[478,"T","K","K"],[614,"D","G","G"],[681,"P","R","H"],[950,"D","N","D"]],dtype=object)
  all_scores = []
  ## load predicted scores
  for mdl_id in mdl_sets:
    mdl_scores = []
    for region in ['RBD','S1C','S1N','S2']:
      fit_score = np.loadtxt(f'{data_path}/{var_set}{region}_mutScan_predFit_{mdl_id}_best.csv',dtype='float',delimiter=',')
      fit_score_posSort = fit_score[np.argsort(fit_score[:,0])]
      mdl_scores.extend(fit_score_posSort)
    mdl_scores = np.array(mdl_scores)
    mdl_scores = mdl_scores[np.argsort(mdl_scores[:,0])]
    all_scores.append(mdl_scores)
  
  all_scores = np.array(all_scores) 
  
  pos_list = all_scores[0,:,0].astype(int)
  score_arr = all_scores[:,:,1:] #[n_mdl,n_pos,20]
  mdl_ave = np.mean(score_arr,axis=0) #[n_pos,20]
  mdl_pos_ave = np.mean(mdl_ave,axis=-1) #[n_pos]
  mdl_pos_med = np.median(mdl_ave,axis=-1) #[n_pos]
  mdl_ave_pos_sele = mdl_ave[delta_BA2_var[:,0].astype(int)-1,:]
   
  ## draw figure
  tar_arr_list = [mdl_ave_pos_sele]
  tar_arr_name = ['ensemble']
  tar_arr_name.extend(mdl_sets)
  for mdl_i in range(len(mdl_sets)):
    tar_arr_list.append(score_arr[mdl_i][delta_BA2_var[:,0].astype(int)-1,:])

  for tar_i in range(len(tar_arr_list)):
    tar_arr = tar_arr_list[tar_i]
    fig = plt.figure(figsize=(10,8))
    ax = plt.gca()
    divnorm=colors.TwoSlopeNorm(vmin=np.amin(tar_arr), vcenter=0., vmax=np.amax(tar_arr))
    im = ax.matshow(tar_arr.T,cmap=plt.cm.bwr,aspect="auto",norm=divnorm)
    
    ax.set_xticks(np.arange(delta_BA2_var.shape[0]))
    ax.set_xticklabels([f'{delta_BA2_var[i,3]}({delta_BA2_var[i,1]}){delta_BA2_var[i,0]}' for i in range(delta_BA2_var.shape[0])])
    ax.set_yticks(np.arange(20))
    ax.set_yticklabels(allAA_char)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=45, ha="right", va="center", rotation_mode="anchor")
    # Minor ticks
    ax.set_xticks(np.arange(-.5, delta_BA2_var.shape[0], 1), minor=True)
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    # cell annotation
    for x in range(delta_BA2_var.shape[0]): #n_pos
      ## order muts for one position
      mut_sort = np.argsort(np.argsort(tar_arr[x,:]))
      for y in range(mdl_ave.shape[1]): #20
        plt.text(x, y, f'{allAA_char[y]}{19-mut_sort[y]+1}',
                  horizontalalignment='center',
                  verticalalignment='center',
                  fontsize=6)
    # mark cell
    for var_i in range(delta_BA2_var.shape[0]):
        var_aa = np.where(np.array(allAA_char) == delta_BA2_var[var_i,2])[0][0]
        rect = plt.Rectangle((var_i-.4, var_aa-.4), 0.8,0.8, fill=False, color="lime", linewidth=1)
        ax.add_patch(rect)
    
    fig.colorbar(im)
    plt.savefig(f'{data_path}/BA2_deltaVar_{tar_arr_name[tar_i]}_mutScan_mutWise.png',dpi=300)
    plt.clf()

def fitScan_topKacc_lineage(data_path: str = None,
                            mdl_sets: List = None):
  start_RBD = 348
  end_RBD = 526
  voc_list = ['alpha','beta','gamma','delta','omicron']
  # lineage specific mutations
  lineage_specific_mutations = {
    "alpha": ['E484K','F490S','S494P'],
    "beta": ['P384L','E516Q'],
    "gamma": ['T470P'],
    "delta": ['K417N','E484Q'],
    "omicron": ['S371F','S375F','T376A','D405N','R408S','F486V','Y505H'],
    "wt": []
  }
  lineage_specific_mutations_list = []
  lineage_specific_selected_position = {}

  for var_key in voc_list:
    lineage_specific_mutations_list.extend(lineage_specific_mutations[var_key])
    if len(lineage_specific_mutations[var_key]) > 0:
      lineage_specific_selected_position[var_key] = sorted([int(mut[1:-1]) for mut in lineage_specific_mutations[var_key]])
    else:
      lineage_specific_selected_position[var_key] = []
    lineage_specific_mutations['wt'].extend(lineage_specific_mutations[var_key])
  lineage_specific_selected_position['wt'] = sorted([int(mut[1:-1]) for mut in lineage_specific_mutations['wt']])

  lineage_nonSpecific_mutations = {
    "alpha": ['L452R'],
    "beta": [],
    "gamma": [],
    "delta": [],
    "omicron": ['L452R'],
    "wt": []
  }
  lineage_nonSpecific_mutations_list = []
  lineage_nonSpecific_selected_position = {}
  for var_key in voc_list:
    lineage_nonSpecific_mutations_list.extend(lineage_nonSpecific_mutations[var_key])
    if len(lineage_nonSpecific_mutations[var_key]) > 0:
      lineage_nonSpecific_selected_position[var_key] = sorted([int(mut[1:-1]) for mut in lineage_nonSpecific_mutations[var_key]])
    else:
      lineage_nonSpecific_selected_position[var_key] = []
    lineage_nonSpecific_mutations['wt'].extend(lineage_nonSpecific_mutations[var_key])
  lineage_nonSpecific_selected_position['wt'] = sorted([int(mut[1:-1]) for mut in lineage_nonSpecific_mutations['wt']])
  
  ## pos->var mapping
  pos2variant_dict = {}
  mutPos_gt_list = []
  for p in range(start_RBD,end_RBD+1):
    pos2variant_dict[p] = []
    for voc in ['alpha','beta','gamma','delta','omicron']:
      if p in lineage_specific_selected_position[voc] or p in lineage_nonSpecific_selected_position[voc]:
        pos2variant_dict[p].append(voc)
        if p not in mutPos_gt_list:
          mutPos_gt_list.append(p)
      else:
        pos2variant_dict[p].append('wt')
             
  allAA_ids = [12,19,10,7,6,15,18,21,20,5,9,4,23,13,11,14,17,25,8,24]
  allAA_char = ["K","R","H","E","D","N","Q","T","S","C","G","A","V","L","I","M","P","Y","F","W"]
  
  with open(f'{data_path}/wtSeq_P0DTC2.fasta') as handle:
    for record in SeqIO.parse(handle,"fasta"):
      wtSeq = list(record.seq)
  wtRBD_seq = wtSeq[start_RBD-1:end_RBD]
  
  ensemble_score_arr = {}
  for pare in lineage_specific_mutations.keys():
    all_scores = []
    for mdl_id in mdl_sets:
      fit_score = np.loadtxt(f'{data_path}/{pare}RBD_mutScan_predFit_{mdl_id}_best.csv',dtype='float',delimiter=',')
      fit_score_posSort = fit_score[np.argsort(fit_score[:,0])]
      all_scores.append(fit_score_posSort)
    all_scores = np.array(all_scores) # [n_model, n_pos, 20]
    pos_list = all_scores[0,:,0].astype(int)
    score_arr = all_scores[:,:,1:]
    mdl_ave = np.mean(score_arr,axis=0) # model emsembles [n_pos,20]
    mdl_pos_ave = np.mean(mdl_ave,axis=-1)
    mdl_pos_med = np.median(mdl_ave,axis=-1)
    ensemble_score_arr[pare] = np.vstack((mdl_ave.reshape(1,-1,20),score_arr)) # [n_model+1, n_pos, 20]
    
  print('################## Mutation wise ########################')
  print('lineage_set;model;parent;precision;recall;f1;EF;REF')
  lineage_mutations_list = [lineage_specific_mutations, lineage_nonSpecific_mutations]
  lineage_mutations_name_list = ['lin_spe', 'lin_nonSpe']
  for mdl_i in range(len(mdl_sets)+1):
    mdl_nm = mdl_sets[mdl_i-1] if mdl_i > 0 else 'ensemble'
    for lineage_mut_idx in range(len(lineage_mutations_list)):
      lineage_mut_dict = lineage_mutations_list[lineage_mut_idx]
      lineage_mut_nm = lineage_mutations_name_list[lineage_mut_idx]
      ### variant parents
      top_1 = 0
      top_3 = 0
      top_5 = 0
      for var_key, var_val in lineage_mut_dict.items():
        if var_key == 'wt':
          continue
        score_set = ensemble_score_arr[var_key][mdl_i] # [n_pos, 20]
        for mut in var_val:
          nature_aa = mut[0].upper()
          mut_aa = mut[-1].upper()
          site = int(mut[1:-1])
          pred_vec = score_set[np.where(pos_list == site)[0][0],:]
          pred_vec[allAA_char.index(nature_aa)] = -np.inf
          pred_vec = np.argsort(pred_vec)
          if allAA_char[pred_vec[-1]] == mut_aa:
            top_1 += 1
            top_3 += 1
            top_5 += 1
          elif allAA_char[pred_vec[-2]] == mut_aa or allAA_char[pred_vec[-3]] == mut_aa:
            top_3 += 1
            top_5 += 1
          elif allAA_char[pred_vec[-4]] == mut_aa or allAA_char[pred_vec[-5]] == mut_aa:
            top_5 += 1
      top_1_precision = top_1/(1*len(lineage_specific_selected_position['wt']))
      top_3_precision = top_3/(3*len(lineage_specific_selected_position['wt']))
      top_5_precision = top_5/(5*len(lineage_specific_selected_position['wt']))
      top_1_recall = top_1/len(lineage_specific_mutations['wt'])
      top_3_recall = top_3/len(lineage_specific_mutations['wt'])
      top_5_recall = top_5/len(lineage_specific_mutations['wt'])
      top_1_f1 = 2*top_1_precision*top_1_recall/(top_1_precision+top_1_recall) if (top_1_precision+top_1_recall) > 0 else 0
      top_3_f1 = 2*top_3_precision*top_3_recall/(top_3_precision+top_3_recall) if (top_3_precision+top_3_recall) > 0 else 0
      top_5_f1 = 2*top_5_precision*top_5_recall/(top_5_precision+top_5_recall) if (top_5_precision+top_5_recall) > 0 else 0
      top_1_EF = top_1_precision * (19*len(lineage_specific_selected_position['wt'])/len(lineage_specific_mutations['wt']))
      top_3_EF = top_3_precision * (19*len(lineage_specific_selected_position['wt'])/len(lineage_specific_mutations['wt']))
      top_5_EF = top_5_precision * (19*len(lineage_specific_selected_position['wt'])/len(lineage_specific_mutations['wt']))
      top_1_REF = 100*top_1/(min(len(lineage_specific_mutations['wt']),1*len(lineage_specific_selected_position['wt'])))
      top_3_REF = 100*top_3/(min(len(lineage_specific_mutations['wt']),3*len(lineage_specific_selected_position['wt'])))
      top_5_REF = 100*top_5/(min(len(lineage_specific_mutations['wt']),5*len(lineage_specific_selected_position['wt'])))
      print(f'{lineage_mut_nm:>10};{mdl_nm:>10};variant;{top_1_precision:.3f}/{top_3_precision:.3f}/{top_5_precision:.3f};{top_1_recall:.3f}/{top_3_recall:.3f}/{top_5_recall:.3f};{top_1_f1:.3f}/{top_3_f1:.3f}/{top_5_f1:.3f};{top_1_EF:.3f}/{top_3_EF:.3f}/{top_5_EF:.3f};{top_1_REF:.3f}/{top_3_REF:.3f}/{top_5_REF:.3f}')

      ### WT parents
      top_1 = 0
      top_3 = 0
      top_5 = 0
      for var_key, var_val in lineage_mut_dict.items():
        if var_key != 'wt':
          continue
        score_set = ensemble_score_arr[var_key][mdl_i] # [n_pos, 20]
        for mut in var_val:
          nature_aa = mut[0].upper()
          mut_aa = mut[-1].upper()
          site = int(mut[1:-1])
          pred_vec = score_set[np.where(pos_list == site)[0][0],:]
          pred_vec[allAA_char.index(nature_aa)] = -np.inf
          pred_vec = np.argsort(pred_vec)
          if allAA_char[pred_vec[-1]] == mut_aa:
            top_1 += 1
            top_3 += 1
            top_5 += 1
          elif allAA_char[pred_vec[-2]] == mut_aa or allAA_char[pred_vec[-3]] == mut_aa:
            top_3 += 1
            top_5 += 1
          elif allAA_char[pred_vec[-4]] == mut_aa or allAA_char[pred_vec[-5]] == mut_aa:
            top_5 += 1
      top_1_precision = top_1/(1*len(lineage_specific_selected_position['wt']))
      top_3_precision = top_3/(3*len(lineage_specific_selected_position['wt']))
      top_5_precision = top_5/(5*len(lineage_specific_selected_position['wt']))
      top_1_recall = top_1/len(lineage_specific_mutations['wt'])
      top_3_recall = top_3/len(lineage_specific_mutations['wt'])
      top_5_recall = top_5/len(lineage_specific_mutations['wt'])
      top_1_f1 = 2*top_1_precision*top_1_recall/(top_1_precision+top_1_recall) if (top_1_precision+top_1_recall) > 0 else 0
      top_3_f1 = 2*top_3_precision*top_3_recall/(top_3_precision+top_3_recall) if (top_3_precision+top_3_recall) > 0 else 0
      top_5_f1 = 2*top_5_precision*top_5_recall/(top_5_precision+top_5_recall) if (top_5_precision+top_5_recall) > 0 else 0
      top_1_EF = top_1_precision * (19*len(lineage_specific_selected_position['wt'])/len(lineage_specific_mutations['wt']))
      top_3_EF = top_3_precision * (19*len(lineage_specific_selected_position['wt'])/len(lineage_specific_mutations['wt']))
      top_5_EF = top_5_precision * (19*len(lineage_specific_selected_position['wt'])/len(lineage_specific_mutations['wt']))
      top_1_REF = 100*top_1/(min(len(lineage_specific_mutations['wt']),1*len(lineage_specific_selected_position['wt'])))
      top_3_REF = 100*top_3/(min(len(lineage_specific_mutations['wt']),3*len(lineage_specific_selected_position['wt'])))
      top_5_REF = 100*top_5/(min(len(lineage_specific_mutations['wt']),5*len(lineage_specific_selected_position['wt'])))
      print(f'{lineage_mut_nm:>10};{mdl_nm:>10};wt;{top_1_precision:.3f}/{top_3_precision:.3f}/{top_5_precision:.3f};{top_1_recall:.3f}/{top_3_recall:.3f}/{top_5_recall:.3f};{top_1_f1:.3f}/{top_3_f1:.3f}/{top_5_f1:.3f};{top_1_EF:.3f}/{top_3_EF:.3f}/{top_5_EF:.3f};{top_1_REF:.3f}/{top_3_REF:.3f}/{top_5_REF:.3f}')
  print('**********************************')

  print('################## Position Prediction ########################')
  print('parent;model;pos_topK;pos_mode;TP_count;precision;recall;f1;EF;REF')
  aa_ave_nan_dict = {}
  aa_max_nan_dict = {}
  aa_median_nan_dict = {}
  for var_key, sco_arr in ensemble_score_arr.items():
    for pos in range(sco_arr.shape[1]):
      nature_aa = wtRBD_seq[pos]
      sco_arr[:,pos,allAA_char.index(nature_aa)] = np.nan    
    aa_ave_nan = np.nanmean(score_arr,axis=-1) #[n_mdl,pos]
    aa_max_nan = np.nanmax(score_arr,axis=-1) #[n_mdl,pos]
    aa_median_nan = np.nanmedian(score_arr, axis=-1) #[n_mdl,pos]
    mdl_emsb_nan = np.nanmean(score_arr,axis=0) #[pos,20]
    aa_ave_mdlEm_nan = np.nanmean(mdl_emsb_nan,axis=-1) #[pos,]
    aa_max_mdlEm_nan = np.nanmax(mdl_emsb_nan,axis=-1)
    aa_median_mdlEm_nan = np.median(mdl_emsb_nan,axis=-1)
    aa_ave_nan_dict[var_key] = np.vstack((aa_ave_mdlEm_nan.reshape(1,-1),aa_ave_nan))
    aa_max_nan_dict[var_key] = np.vstack((aa_max_mdlEm_nan.reshape(1,-1),aa_max_nan))
    aa_median_nan_dict[var_key] = np.vstack((aa_median_mdlEm_nan.reshape(1,-1),aa_median_nan))
  
  ## position score from lineage input (max version)
  pos2score_lineage_max_arr = [] # [pos,model]
  for p in range(start_RBD,end_RBD+1):
    score2collect = [] # [variant,model]
    for nm in pos2variant_dict[p]:
      score2collect.append(aa_max_nan_dict[nm][:,p-start_RBD])
    pos2score_lineage_max_arr.append(np.nanmax(score2collect,axis=0)) # [model,]
  pos2score_lineage_max_arr = np.array(pos2score_lineage_max_arr)

  ## position score from wt input (max version)
  pos2score_wt_max_arr = [] # [pos,model]
  for p in range(start_RBD,end_RBD+1):
    pos2score_wt_max_arr.append(aa_max_nan_dict['wt'][:,p-start_RBD]) # [model,]
  pos2score_wt_max_arr = np.array(pos2score_wt_max_arr)

  aa_nan_list = [pos2score_lineage_max_arr,pos2score_wt_max_arr]
  aa_nan_nm_list = ['lineage', 'wt']
  posRBD_list = list(range(start_RBD,end_RBD+1))
  for topk in [10,20,50]:
    top_count_all = []
    for aa_nan_idx in range(len(aa_nan_nm_list)):
      aa_nan = aa_nan_list[aa_nan_idx]
      aa_nan_nm = aa_nan_nm_list[aa_nan_idx]
      top_count_mdl = []
      for mdl_i in range(aa_nan.shape[1]):
        aa_mdl_nan = aa_nan[:,mdl_i]
        top_pos_idx = np.argsort(aa_mdl_nan)[::-1][:topk]
        top_pos_list = [posRBD_list[tpi] for tpi in top_pos_idx]
        top_count = 0
        for mut_pos in mutPos_gt_list:
          if mut_pos in top_pos_list:
            top_count += 1
        pos_precision = top_count/topk
        pos_recall = top_count/len(mutPos_gt_list)
        pos_f1 = 2*pos_precision*pos_recall/(pos_precision+pos_recall) if (pos_precision+pos_recall) > 0 else 0
        pos_EF = pos_precision * (len(posRBD_list)/len(mutPos_gt_list))
        pos_REF = 100*top_count/(min(len(mutPos_gt_list),len(posRBD_list)))
        top_count_mdl.append([top_count,pos_precision,pos_recall,pos_f1,pos_EF,pos_REF])
      top_count_all.append(top_count_mdl)
      for p_idx in range(aa_nan.shape[1]):
        mdl_nm = mdl_sets[p_idx-1] if p_idx > 0 else 'ensemble'
        score_print = ';'.join(["%.3f" % s for s in top_count_all[0][p_idx]])
        print(f'{aa_nan_nm};{mdl_nm};{topk};max;{score_print}')
  print('**********************************')

  print('################## Given topK Pos, Mutation Results ########################')
  print('pos_mode;pos_topK;lineage_set_nm;model;parent;precision;recall;f1;EF;REF')
  for topk in [10,20]:
    ## lineage topK position
    for mdl_i in range(pos2score_lineage_max_arr.shape[1]):
      mdl_nm = mdl_sets[mdl_i-1] if mdl_i > 0 else 'ensemble'
      aa_mdl_nan = pos2score_lineage_max_arr[:,mdl_i]
      top_pos_idx = np.argsort(aa_mdl_nan)[::-1][:topk]
      top_pos_list = [posRBD_list[tpi] for tpi in top_pos_idx]
      for lineage_mut_idx in range(len(lineage_mutations_list)):
        lineage_mut_dict = lineage_mutations_list[lineage_mut_idx]
        lineage_mut_nm = lineage_mutations_name_list[lineage_mut_idx]
        top_1, top_3, top_5 = 0, 0, 0
        mut_num = 0
        ### variant parents
        for var_key, var_val in lineage_mut_dict.items():
          if var_key == 'wt':
            continue
          score_set = ensemble_score_arr[var_key][mdl_i] # [n_pos, 20]
          for mut in var_val:
            nature_aa = mut[0].upper()
            mut_aa = mut[-1].upper()
            site = int(mut[1:-1])
            if site in top_pos_list:
              mut_num += 1
              pred_vec = score_set[np.where(pos_list == site)[0][0],:]
              pred_vec[allAA_char.index(nature_aa)] = -np.inf
              pred_vec = np.argsort(pred_vec)
              if allAA_char[pred_vec[-1]] == mut_aa:
                top_1 += 1
                top_3 += 1
                top_5 += 1
              elif allAA_char[pred_vec[-2]] == mut_aa or allAA_char[pred_vec[-3]] == mut_aa:
                top_3 += 1
                top_5 += 1
              elif allAA_char[pred_vec[-4]] == mut_aa or allAA_char[pred_vec[-5]] == mut_aa:
                top_5 += 1
      top_1_precision = top_1/(1*len(top_pos_list))
      top_3_precision = top_3/(3*len(top_pos_list))
      top_5_precision = top_5/(5*len(top_pos_list))
      top_1_recall = top_1/mut_num if mut_num > 0 else 0
      top_3_recall = top_3/mut_num if mut_num > 0 else 0
      top_5_recall = top_5/mut_num if mut_num > 0 else 0
      top_1_f1 = 2*top_1_precision*top_1_recall/(top_1_precision+top_1_recall) if (top_1_precision+top_1_recall) > 0 else 0
      top_3_f1 = 2*top_3_precision*top_3_recall/(top_3_precision+top_3_recall) if (top_3_precision+top_3_recall) > 0 else 0
      top_5_f1 = 2*top_5_precision*top_5_recall/(top_5_precision+top_5_recall) if (top_5_precision+top_5_recall) > 0 else 0
      top_1_EF = top_1_precision * (19*len(top_pos_list)/mut_num) if mut_num > 0 else 0
      top_3_EF = top_3_precision * (19*len(top_pos_list)/mut_num) if mut_num > 0 else 0
      top_5_EF = top_5_precision * (19*len(top_pos_list)/mut_num) if mut_num > 0 else 0
      top_1_REF = 100*top_1/(min(mut_num,1*len(top_pos_list))) if mut_num > 0 else 0
      top_3_REF = 100*top_3/(min(mut_num,3*len(top_pos_list))) if mut_num > 0 else 0
      top_5_REF = 100*top_5/(min(mut_num,5*len(top_pos_list))) if mut_num > 0 else 0
      
      print(f'lineage_pos;{topk};{lineage_mut_nm:>10};{mdl_nm:>10};variant;{top_1_precision:.3f}/{top_3_precision:.3f}/{top_5_precision:.3f};{top_1_recall:.3f}/{top_3_recall:.3f}/{top_5_recall:.3f};{top_1_f1:.3f}/{top_3_f1:.3f}/{top_5_f1:.3f};{top_1_EF:.3f}/{top_3_EF:.3f}/{top_5_EF:.3f};{top_1_REF:.3f}/{top_3_REF:.3f}/{top_5_REF:.3f}')

    ## wt topK position
    for mdl_i in range(pos2score_wt_max_arr.shape[1]):
      mdl_nm = mdl_sets[mdl_i-1] if mdl_i > 0 else 'ensemble'
      aa_mdl_nan = pos2score_wt_max_arr[:,mdl_i]
      top_pos_idx = np.argsort(aa_mdl_nan)[::-1][:topk]
      top_pos_list = [posRBD_list[tpi] for tpi in top_pos_idx]
      for lineage_mut_idx in range(len(lineage_mutations_list)):
        lineage_mut_dict = lineage_mutations_list[lineage_mut_idx]
        lineage_mut_nm = lineage_mutations_name_list[lineage_mut_idx]
        top_1, top_3, top_5 = 0, 0, 0
        mut_num = 0
        ### wt parents
        for var_key, var_val in lineage_mut_dict.items():
          if var_key != 'wt':
            continue
          score_set = ensemble_score_arr[var_key][mdl_i] # [n_pos, 20]
          for mut in var_val:
            nature_aa = mut[0].upper()
            mut_aa = mut[-1].upper()
            site = int(mut[1:-1])
            if site in top_pos_list:
              mut_num += 1
              pred_vec = score_set[np.where(pos_list == site)[0][0],:]
              pred_vec[allAA_char.index(nature_aa)] = -np.inf
              pred_vec = np.argsort(pred_vec)
              if allAA_char[pred_vec[-1]] == mut_aa:
                top_1 += 1
                top_3 += 1
                top_5 += 1
              elif allAA_char[pred_vec[-2]] == mut_aa or allAA_char[pred_vec[-3]] == mut_aa:
                top_3 += 1
                top_5 += 1
              elif allAA_char[pred_vec[-4]] == mut_aa or allAA_char[pred_vec[-5]] == mut_aa:
                top_5 += 1
      top_1_precision = top_1/(1*len(top_pos_list))
      top_3_precision = top_3/(3*len(top_pos_list))
      top_5_precision = top_5/(5*len(top_pos_list))
      top_1_recall = top_1/mut_num if mut_num > 0 else 0
      top_3_recall = top_3/mut_num if mut_num > 0 else 0
      top_5_recall = top_5/mut_num if mut_num > 0 else 0
      top_1_f1 = 2*top_1_precision*top_1_recall/(top_1_precision+top_1_recall) if (top_1_precision+top_1_recall) > 0 else 0
      top_3_f1 = 2*top_3_precision*top_3_recall/(top_3_precision+top_3_recall) if (top_3_precision+top_3_recall) > 0 else 0
      top_5_f1 = 2*top_5_precision*top_5_recall/(top_5_precision+top_5_recall) if (top_5_precision+top_5_recall) > 0 else 0
      top_1_EF = top_1_precision * (19*len(top_pos_list)/mut_num) if mut_num > 0 else 0
      top_3_EF = top_3_precision * (19*len(top_pos_list)/mut_num) if mut_num > 0 else 0
      top_5_EF = top_5_precision * (19*len(top_pos_list)/mut_num) if mut_num > 0 else 0
      top_1_REF = 100*top_1/(min(mut_num,1*len(top_pos_list))) if mut_num > 0 else 0
      top_3_REF = 100*top_3/(min(mut_num,3*len(top_pos_list))) if mut_num > 0 else 0
      top_5_REF = 100*top_5/(min(mut_num,5*len(top_pos_list))) if mut_num > 0 else 0
      
      print(f'wt_pos;{topk};{lineage_mut_nm:>10};{mdl_nm:>10};wt;{top_1_precision:.3f}/{top_3_precision:.3f}/{top_5_precision:.3f};{top_1_recall:.3f}/{top_3_recall:.3f}/{top_5_recall:.3f};{top_1_f1:.3f}/{top_3_f1:.3f}/{top_5_f1:.3f};{top_1_EF:.3f}/{top_3_EF:.3f}/{top_5_EF:.3f};{top_1_REF:.3f}/{top_3_REF:.3f}/{top_5_REF:.3f}')
  print('**********************************')
  


def fitScan_topKacc(data_path: str,
                    mdl_sets: List,
                    parent: str):
  
  # RBD domain mutations
  '''
  mutations = [
    ## Alpha
    'N501Y',
    'E484K', 'N501Y',
    'L452R', 'N501Y',
    'S494P', 'N501Y',
    'F490S', 'N501Y',
    ## Beta
    'K417N', 'E484K', 'N501Y',
    'K417N', 'E484K', 'N501Y', 'E516Q',
    'P384L', 'K417N', 'E484K', 'N501Y',
    ## Gamma
    'K417T', 'E484K', 'N501Y',
    'K417T', 'E484K', 'N501Y', 'T470N',
    ## Delta
    'L452R', 'T478K',
    'L452R', 'T478K', 'E484Q',
    'K417N', 'L452R', 'T478K',
    ## Omicron
    # B.1.1.529
    'S371L', 'S373P', 'K417N', 'N440K', 'G446S', 'S477N', 'T478K', 'E484A', 'Q493R', 'G496S', 'Q498R', 'N501Y', 
    # BA.1
    'S371L', 'S373P', 'S375F', 'K417N', 'N440K', 'G446S', 'S477N', 'T478K', 'E484A', 'Q493R', 'G496S', 'Q498R', 'N501Y', 'Y505H',
    # BA.2
    'S371F', 'S373P', 'S375F', 'T376A', 'D405N', 'R408S', 'K417N', 'N440K', 'S477N', 'T478K', 'E484A', 'Q493R', 'Q498R', 'N501Y', 'Y505H',
    # BA.4
    'S371F', 'S373P', 'S375F', 'T376A', 'D405N', 'R408S', 'K417N', 'N440K', 'L452R', 'S477N', 'T478K', 'E484A', 'F486V', 'Q498R', 'N501Y', 'Y505H',
  ]
  '''
  #'''
  mutations = [
    #BA.4
    'S371F','S373P','S375F','T376A','D405N','R408S','K417N','N440K','L452R','S477N','T478K','E484A','F486V','Q498R','N501Y','Y505H',
    #BA.5
    'S371F','S373P','S375F','T376A','D405N','R408S','K417N','N440K','L452R','S477N','T478K','E484A','F486V','Q498R','N501Y','Y505H',
  ]
  #'''
  mutations_set = set(mutations)
  selected_position = sorted(set([int(mut[1:-1]) for mut in mutations_set]))
  allAA_ids = [12,19,10,7,6,15,18,21,20,5,9,4,23,13,11,14,17,25,8,24]
  allAA_char = ["K","R","H","E","D","N","Q","T","S","C","G","A","V","L","I","M","P","Y","F","W"]
  
  with open(f'{data_path}/wtSeq_P0DTC2.fasta') as handle:
    for record in SeqIO.parse(handle,"fasta"):
      wtSeq = list(record.seq)
  wtRBD_seq = wtSeq[348-1:526]
  
  all_scores = []
  for mdl_id in mdl_sets:
    fit_score = np.loadtxt(f'{data_path}/{parent}RBD_mutScan_predFit_{mdl_id}_best.csv',dtype='float',delimiter=',')
    fit_score_posSort = fit_score[np.argsort(fit_score[:,0])]
    all_scores.append(fit_score_posSort)
  all_scores = np.array(all_scores)
  pos_list = all_scores[0,:,0].astype(int)
  score_arr = all_scores[:,:,1:]
  mdl_ave = np.mean(score_arr,axis=0) # model emsembles [n_pos,20]
  mdl_pos_ave = np.mean(mdl_ave,axis=-1)
  mdl_pos_med = np.median(mdl_ave,axis=-1)
  ensemble_score_arr = np.vstack((mdl_ave.reshape(1,-1,20),score_arr))

  mut_dict = {}
  nature_dict = {}
  for mut in mutations_set:
    nature_aa = mut[0]
    mut_aa = mut[-1]
    site = int(mut[1:-1])
    if not site in mut_dict.keys():
      nature_dict[site] = nature_aa
      mut_dict[site] = []
    mut_dict[site].append(mut_aa)
  '''
  print('################## Position wise ########################')
  top_1 = 0
  top_3 = 0
  top_5 = 0
  top1_pos = []
  top3_pos = []
  top5_pos = []
  for i,posi in enumerate(selected_position):
      mut_list = [aa.upper() for aa in mut_dict[posi]]
      nature_aa = nature_dict[posi].upper()
      pred_vec = mdl_ave[np.where(pos_list == posi)[0][0],:]
      pred_vec[allAA_char.index(nature_aa)] = -np.inf
      pred_vec = np.argsort(pred_vec)
      if allAA_char[pred_vec[-1]] in mut_list:
          top_1 += 1
          top_3 += 1
          top_5 += 1
          top1_pos.append(posi)
      elif allAA_char[pred_vec[-2]] in mut_list or allAA_char[pred_vec[-3]] in mut_list:
          top_3 += 1 
          top_5 += 1
          top3_pos.append(posi)
      elif allAA_char[pred_vec[-4]] in mut_list or allAA_char[pred_vec[-5]] in mut_list:
          top_5 += 1
          top5_pos.append(posi)
  print(f'Top 1: {top_1 / len(selected_position)}; {top1_pos}')
  print(f'Top 3: {top_3 / len(selected_position)}; {top3_pos}')
  print(f'Top 5: {top_5 / len(selected_position)}; {top5_pos}')
  print('**********************************')
  '''  
  print('################## Mutation wise ########################')
  print('mdl;precision;recall;f1;EF;REF;Mut')
  for mdl_i in range(ensemble_score_arr.shape[0]):
    score_set = ensemble_score_arr[mdl_i]
    mdl_nm = mdl_sets[mdl_i-1] if mdl_i > 0 else 'ensemble'
    top_1,top_3,top_5 = 0,0,0
    top1_muts,top3_muts,top5_muts = [],[],[]
    for mut in mutations_set:
      nature_aa = mut[0].upper()
      mut_aa = mut[-1].upper()
      site = int(mut[1:-1])
      pred_vec = score_set[np.where(pos_list == site)[0][0],:]
      pred_vec[allAA_char.index(nature_aa)] = -np.inf
      pred_vec = np.argsort(pred_vec)
      if allAA_char[pred_vec[-1]] == mut_aa:
        top_1 += 1
        top_3 += 1
        top_5 += 1
        top1_muts.append(mut)
        top3_muts.append(mut)
        top5_muts.append(mut)
      elif allAA_char[pred_vec[-2]] == mut_aa or allAA_char[pred_vec[-3]] == mut_aa:
        top_3 += 1
        top_5 += 1
        top3_muts.append(mut)
        top5_muts.append(mut)
      elif allAA_char[pred_vec[-4]] == mut_aa or allAA_char[pred_vec[-5]] == mut_aa:
        top_5 += 1
        top5_muts.append(mut)
    top_1_precision = top_1/(1*len(selected_position))
    top_3_precision = top_3/(3*len(selected_position))
    top_5_precision = top_5/(5*len(selected_position))
    top_1_recall = top_1/len(mutations_set)
    top_3_recall = top_3/len(mutations_set)
    top_5_recall = top_5/len(mutations_set)
    top_1_f1 = 2*top_1_precision*top_1_recall/(top_1_precision+top_1_recall) if (top_1_precision+top_1_recall) > 0 else 0
    top_3_f1 = 2*top_3_precision*top_3_recall/(top_3_precision+top_3_recall) if (top_3_precision+top_3_recall) > 0 else 0
    top_5_f1 = 2*top_5_precision*top_5_recall/(top_5_precision+top_5_recall) if (top_5_precision+top_5_recall) > 0 else 0
    top_1_EF = top_1_precision * (19*len(selected_position)/len(mutations_set))
    top_3_EF = top_3_precision * (19*len(selected_position)/len(mutations_set))
    top_5_EF = top_5_precision * (19*len(selected_position)/len(mutations_set))
    top_1_REF = 100*top_1/(min(len(mutations_set),1*len(selected_position)))
    top_3_REF = 100*top_3/(min(len(mutations_set),3*len(selected_position)))
    top_5_REF = 100*top_5/(min(len(mutations_set),5*len(selected_position)))
    print(f'{mdl_nm:>10};{top_1_precision:.3f}/{top_3_precision:.3f}/{top_5_precision:.3f};{top_1_recall:.3f}/{top_3_recall:.3f}/{top_5_recall:.3f};{top_1_f1:.3f}/{top_3_f1:.3f}/{top_5_f1:.3f};{top_1_EF:.3f}/{top_3_EF:.3f}/{top_5_EF:.3f};{top_1_REF:.3f}/{top_3_REF:.3f}/{top_5_REF:.3f};{top1_muts};{top3_muts};{top5_muts}')
  print('**********************************')
  
  print('################## Position Prediction ########################')
  print('model;pos_topK;pos_mode;TP_count;precision;recall;f1;EF;REF')
  for m in range(score_arr.shape[0]):
    mdl_score = score_arr[m]
    for pos in range(mdl_score.shape[0]):
      nature_aa = wtRBD_seq[pos]
      score_arr[m,pos,allAA_char.index(nature_aa)] = np.nan
  mdl_emsb_nan = np.nanmean(score_arr,axis=0) #[pos,20]
  aa_ave_nan = np.nanmean(score_arr,axis=-1) #[n_mdl,pos]
  aa_max_nan = np.nanmax(score_arr,axis=-1)
  aa_median_nan = np.nanmedian(score_arr, axis=-1)
  aa_ave_mdlEm_nan = np.nanmean(mdl_emsb_nan,axis=-1) #[pos,]
  aa_max_mdlEm_nan = np.nanmax(mdl_emsb_nan,axis=-1)
  aa_median_mdlEm_nan = np.median(mdl_emsb_nan,axis=-1)
  
  #aa_mdlEm_nan_list = [aa_ave_mdlEm_nan, aa_max_mdlEm_nan, aa_median_mdlEm_nan]
  aa_nan_list = [np.vstack((aa_ave_mdlEm_nan.reshape(1,-1),aa_ave_nan)),
                 np.vstack((aa_max_mdlEm_nan.reshape(1,-1),aa_max_nan)),
                 np.vstack((aa_median_mdlEm_nan.reshape(1,-1),aa_median_nan))]

  mutPos_gt_list = nature_dict.keys()
  posRBD_list = list(range(348,526+1))
  for topk in [10,20,50]:
    top_count_all = []
    for aa_nan in aa_nan_list:
      top_count_mdl = []
      for mdl_i in range(aa_nan.shape[0]):
        aa_mdl_nan = aa_nan[mdl_i]
        top_pos_idx = np.argsort(aa_mdl_nan)[::-1][:topk]
        top_pos_list = [posRBD_list[tpi] for tpi in top_pos_idx]
        top_count = 0
        for mut_pos in mutPos_gt_list:
          if mut_pos in top_pos_list:
            top_count += 1
        pos_precision = top_count/topk
        pos_recall = top_count/len(selected_position)
        pos_f1 = 2*pos_precision*pos_recall/(pos_precision+pos_recall) if (pos_precision+pos_recall) > 0 else 0
        pos_EF = pos_precision * (len(posRBD_list)/len(selected_position))
        pos_REF = 100*top_count/(min(len(selected_position),len(posRBD_list)))
        top_count_mdl.append([top_count,pos_precision,pos_recall,pos_f1,pos_EF,pos_REF])
      top_count_all.append(top_count_mdl)
    
    for p_idx in range(aa_nan.shape[0]):
      mdl_nm = mdl_sets[p_idx-1] if p_idx > 0 else 'ensemble'
      ave_score_print = ';'.join(["%.3f" % s for s in top_count_all[0][p_idx]])
      max_score_print = ';'.join(["%.3f" % s for s in top_count_all[1][p_idx]])
      median_score_print = ';'.join(["%.3f" % s for s in top_count_all[2][p_idx]])
      print(f'{mdl_nm};{topk};max;{max_score_print}')
      print(f'{mdl_nm};{topk};ave;{ave_score_print}')
      print(f'{mdl_nm};{topk};median;{median_score_print}')
  print('**********************************')

  print('################## Given topK Pos, Mutation Results ########################')
  print('model;pos_topK;TP_count;precision;recall;f1;EF;REF')
  combo_aa_max_nan = np.vstack((aa_max_mdlEm_nan.reshape(1,-1),aa_max_nan))
  for topk in [10,20]:
    ## loop single model and ensemble prediction
    for mdl_i in range(combo_aa_max_nan.shape[0]):
      aa_mdl_nan = combo_aa_max_nan[mdl_i]
      top_pos_idx = np.argsort(aa_mdl_nan)[::-1][:topk]
      top_pos_list = [posRBD_list[tpi] for tpi in top_pos_idx]
      top_1, top_3, top_5 = 0, 0, 0
      mut_num = 0
      mut_list = []
      for i,posi in enumerate(top_pos_list):
        if posi in mut_dict.keys():
          mut_list = [aa.upper() for aa in mut_dict[posi]]
          mut_num += len(mut_list)
          nature_aa = nature_dict[posi].upper()
          pred_vec = mdl_ave[np.where(pos_list == posi)[0][0],:]
          pred_vec[allAA_char.index(nature_aa)] = -np.inf
          pred_vec = np.argsort(pred_vec)
          if allAA_char[pred_vec[-1]] in mut_list:
            top_1 += 1
            top_3 += 1
            top_5 += 1
          elif allAA_char[pred_vec[-2]] in mut_list or allAA_char[pred_vec[-3]] in mut_list:
            top_3 += 1 
            top_5 += 1
          elif allAA_char[pred_vec[-4]] in mut_list or allAA_char[pred_vec[-5]] in mut_list:
            top_5 += 1
      ## precision, recall, f1, EF, REF
      top_1_precision = top_1/(1*len(top_pos_list))
      top_3_precision = top_3/(3*len(top_pos_list))
      top_5_precision = top_5/(5*len(top_pos_list))
      top_1_recall = top_1/mut_num
      top_3_recall = top_3/mut_num
      top_5_recall = top_5/mut_num
      top_1_f1 = 2*top_1_precision*top_1_recall/(top_1_precision+top_1_recall) if (top_1_precision+top_1_recall) > 0 else 0
      top_3_f1 = 2*top_3_precision*top_3_recall/(top_3_precision+top_3_recall) if (top_3_precision+top_3_recall) > 0 else 0
      top_5_f1 = 2*top_5_precision*top_5_recall/(top_5_precision+top_5_recall) if (top_5_precision+top_5_recall) > 0 else 0
      top_1_EF = top_1_precision * (19*len(top_pos_list)/mut_num)
      top_3_EF = top_3_precision * (19*len(top_pos_list)/mut_num)
      top_5_EF = top_5_precision * (19*len(top_pos_list)/mut_num)
      top_1_REF = 100*top_1/(min(mut_num,1*len(top_pos_list)))
      top_3_REF = 100*top_3/(min(mut_num,3*len(top_pos_list)))
      top_5_REF = 100*top_5/(min(mut_num,5*len(top_pos_list)))
      mdl_nm = mdl_sets[mdl_i-1] if mdl_i > 0 else 'ensemble'
      print(f'{mdl_nm:>10};{topk};{top_1}/{top_3}/{top_5};{top_1_precision:.3f}/{top_3_precision:.3f}/{top_5_precision:.3f};{top_1_recall:.3f}/{top_3_recall:.3f}/{top_5_recall:.3f};{top_1_f1:.3f}/{top_3_f1:.3f}/{top_5_f1:.3f};{top_1_EF:.3f}/{top_3_EF:.3f}/{top_5_EF:.3f};{top_1_REF:.3f}/{top_3_REF:.3f}/{top_5_REF:.3f}')
  print('**********************************')

def fireprot_family_query():
  """Curate family infomation about fireprotDB
  """
  path = "/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process"
  # experiment_id,protein_name,uniprot_id,pdb_id,chain,position,wild_type,mutation,ddG,dTm,is_curated,type,derived_type,interpro_families,conservation,is_essential,correlated_positions,is_back_to_consensus,secondary_structure,asa,is_in_catalytic_pocket,is_in_tunnel_bottleneck,b_factor,method,method_details,technique,technique_details,pH,tm,notes,publication_doi,publication_pubmed,hsw_job_id,datasets,sequence
  df_stability = pd.read_csv(f"{path}/stability/fireprot/fireprotdb_results_0829.csv",header=0,quotechar='"',converters={'uniprot_id':str,'sequence':str})
  print('>finish loading stability csv')
  # uniprot_acc     seq_version     crc64   md5     pfamA_acc       seq_start       seq_end
  df_pfam_unp = pd.read_csv(f"{path}/pfam_35.0/Pfam-A.regions.uniprot.tsv.gz",delimiter='\t',header=0,compression='gzip',chunksize=10000)
  print('>finish loading pfam_unp mapping')
  # Pfam accession, clan accession, clan ID, Pfam ID, Pfam description
  df_pfam_clan = pd.read_csv(f"{path}/pfam_35.0/Pfam-A.clans.tsv.gz",delimiter='\t',header=0,compression='gzip')
  print('>finish loading pfam_clan mapping')

  ## find records with existing uniprot_id or sequence
  sele_df = df_stability.loc[(df_stability['sequence'].str.len()!=0)]
  unp_id_collect = list(set(sele_df.loc[sele_df['uniprot_id'].str.len()!=0]['uniprot_id'].to_list()))
  sequence_collect = list(set(sele_df.loc[sele_df['sequence'].str.len()!=0]['sequence'].to_list()))

  fasta_seq_list = []
  for idx, row in sele_df.iterrows():
    SeqRecord(Seq(row['sequence']),id=f"{row['experiment_id']}|{row['protein_name']}|{row['uniprot_id']}")
    fasta_seq_list.append(SeqRecord)
  SeqIO.write(fasta_seq_list, f"{path}/stability/fireprot/seq.fasta", "fasta")

  ## run hmmscan (hmmscan on web: https://www.ebi.ac.uk/Tools/hmmer/search/hmmscan)
  #os.system(f"")

  ## curate pfam ids
  df_fam_query = pd.DataFrame()
  for chunk in df_pfam_unp:
    df_chunk = pd.DataFrame(chunk)
    tmp_df = df_chunk.loc[df_chunk['uniprot_acc'].isin(unp_id_collect)]
    if len(tmp_df) > 0:
      df_fam_query = df_fam_query.append(tmp_df)
  fam_info = []
  unp_has_fam = []
  for idx, row in df_fam_query.iterrows():
    unp_has_fam.append(row['uniprot_acc'])
    fam_info.append([row['uniprot_acc'],row['pfamA_acc'],row['seq_start'],row['seq_end']])
  print(f'unp with no pfam: {list(set(unp_id_collect)-set(unp_has_fam))}')
  np.savetxt(f'{path}/stability/fireprot/fam_set.csv',fam_info,fmt='%s',delimiter=',',header='uniprot_acc,pfamA_acc,seq_start,seq_end')


def find_family_across_datasets():
  """
  curate commonly existing families across fitness, stability and clinical set thourgh uniprot ids
  """
  path = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  fitness_fam_info = 'data_process/mutagenesis/DeepSequenceMutaSet_reference_file.csv'
  fireprot_fam_info = 'data_process/stability/fireprot/fam_set.csv'
  clinic_fam_info = 'data_process/clinical/EVE_data/variant_protein_list.csv'
  proteingym_fam_info = 'data_process/mutagenesis/proteingym/ProteinGym_reference_file_substitutions.csv'
  proteingym_map = 'data_process/mutagenesis/proteingym/proteingym_unp_name_id.tsv'

  fitness_fam_info_df = pd.read_csv(f'{path}/{fitness_fam_info}',delimiter=',',header=0)
  fireprot_fam_info_df = pd.read_csv(f'{path}/{fireprot_fam_info}',delimiter=',',header=0)
  clinic_fam_info_df = pd.read_csv(f'{path}/{clinic_fam_info}',delimiter=',',names=['Unp_name'])
  proteingym_fam_info_df = pd.read_csv(f'{path}/{proteingym_fam_info}',delimiter=',',header=0)
  proteingym_map_df = pd.read_csv(f'{path}/{proteingym_map}',delimiter='\t',header=0)

  cross_fam_info_df = []
  for i, row in fitness_fam_info_df.iterrows():
    target_unp_name = row['Unp_name']
    target_unp_id = row['Unp_id']
    search_fireprot_df = fireprot_fam_info_df.loc[fireprot_fam_info_df['uniprot_acc'] == target_unp_id]
    search_clinic_df = clinic_fam_info_df.loc[clinic_fam_info_df['Unp_name'] == target_unp_name]
    in_fireprot, in_clinic = 0,0
    if len(search_fireprot_df) > 0:
      in_fireprot = 1
    if len(search_clinic_df) > 0:
      in_clinic = 1
    cross_fam_info_df.append([target_unp_name,target_unp_id,in_fireprot,in_clinic])

  cross_fam_info_df = pd.DataFrame(cross_fam_info_df,columns=['unp_name','unp_id','in_fireprot','in_clinvar']).drop_duplicates()
  cross_fam_info_df.to_csv(f'{path}/data_process/mutagenesis/DeepSequenceMutaSet_fireprot_clinvar.csv',sep=',',index=False)

  ## proteingym
  cross_fam_info_df = []
  for i, row in proteingym_fam_info_df.iterrows():
    target_unp_name = row['UniProt_ID']
    target_unp_id = proteingym_map_df.loc[proteingym_map_df['Entry Name'] == target_unp_name]['Entry'][0]
    search_fireprot_df = fireprot_fam_info_df.loc[fireprot_fam_info_df['uniprot_acc'] == target_unp_id]
    search_clinic_df = clinic_fam_info_df.loc[clinic_fam_info_df['Unp_name'] == target_unp_name]
    in_fireprot, in_clinic = 0,0
    if len(search_fireprot_df) > 0:
      in_fireprot = 1
    if len(search_clinic_df) > 0:
      in_clinic = 1
    cross_fam_info_df.append([target_unp_name,target_unp_id,in_fireprot,in_clinic])

  cross_fam_info_df = pd.DataFrame(cross_fam_info_df,columns=['unp_name','unp_id','in_fireprot','in_clinvar']).drop_duplicates()
  cross_fam_info_df.to_csv(f'{path}/data_process/mutagenesis/proteingym/proteingym_fireprot_clinvar.csv',sep=',',index=False)

  return None

def fireprot_fam_label_data(fam_unp_list: List):
  """generate label data for families
  """
  path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process'
  # experiment_id,protein_name,uniprot_id,pdb_id,chain,position,wild_type,mutation,ddG,dTm,is_curated,type,derived_type,interpro_families,conservation,is_essential,correlated_positions,is_back_to_consensus,secondary_structure,asa,is_in_catalytic_pocket,is_in_tunnel_bottleneck,b_factor,method,method_details,technique,technique_details,pH,tm,notes,publication_doi,publication_pubmed,hsw_job_id,datasets,sequence(wt)
  df_stability = pd.read_csv(f"{path}/stability/fireprot/fireprotdb_results_0829.csv",header=0,quotechar='"',converters={'uniprot_id':str,'sequence':str,'b_factor':str,'method':str,'method_details':str,'technique':str,'technique_details':str,'tm':str,'notes':str}).drop_duplicates()
  
  # unp_name and id mapping
  unp_name_id_mapping = {
    'P11436': 'AMIE_PSEAE',
    'B3VI55': 'B3VI55_LIPST',
    'P62593': 'BLAT_ECOLX',
    'P38398': 'BRCA1_HUMAN_BRCT',
    'P12497': 'POL_HV1N5',
    'Q56319': 'TIM_THEMA',
    'P46937': 'YAP1_HUMAN'
  }
  map_size = (1024 * 15) * (2 ** 20) # 15G
  #'set_nm', 'wt_seq', 'seq_len', 'mutants', 'mut_relative_idxs', 'mut_seq', 'fitness'
  for fam in fam_unp_list:
    print(f'{unp_name_id_mapping[fam]}')
    # load domain wt seq
    with open(f'{path}/mutagenesis/DeepSequenceMutaSet_priSeq/{unp_name_id_mapping[fam]}.fasta') as handle:
      for record in SeqIO.parse(handle,"fasta"):
        wt_domain_seqStr = str(record.seq)
        range_str = re.split('\|',str(record.id))[-1]
        range_list = re.split('-',range_str)
        domain_start_idx, domain_end_idx = int(range_list[0]), int(range_list[-1])
    print(f'-> domain start-end : {domain_start_idx}-{domain_end_idx}')
    target_df = df_stability.loc[df_stability['uniprot_id'] == fam]
    # ddG labeled set
    target_df_ddG = target_df.loc[target_df['ddG'].notnull()]
    if len(target_df_ddG) > 0:
      data2save = []
      # mutation set
      mut_df = target_df_ddG[['position','wild_type','mutation']].drop_duplicates()
      for i, row in mut_df.iterrows():
        target_mut_df = target_df_ddG.loc[(target_df_ddG['position'] == row['position']) & (target_df_ddG['wild_type'] == row['wild_type']) & (target_df_ddG['mutation'] == row['mutation'])]
        seq_str = target_mut_df.iloc[0]['sequence']
        pos = int(row['position'])
        wt_aa = row['wild_type']
        mut_aa = row['mutation']
        mut_str = f'{wt_aa}{pos}{mut_aa}'
        if pos < domain_start_idx or pos > domain_end_idx:
          print(f'-> variant out of domain: {mut_str}')
          continue
        relative_idx_domain = pos - domain_start_idx
        assert wt_domain_seqStr[relative_idx_domain] == wt_aa
        assert seq_str[pos-1] == wt_aa
        domain_seq_aa_list = list(wt_domain_seqStr)
        domain_seq_aa_list[relative_idx_domain] = mut_aa
        domain_seq_mut_str = ''.join(domain_seq_aa_list)
        label_score = target_mut_df['ddG'].mean()
        data2save.append({
          'set_nm':unp_name_id_mapping[fam],
          'wt_seq':wt_domain_seqStr,
          'seq_len':len(wt_domain_seqStr),
          'mutants':[mut_str],
          'mut_relative_idxs':[relative_idx_domain],
          'mut_seq':domain_seq_mut_str,
          'fitness':label_score*-1}) ## negative ddG, stabilizing - positive
      print(f'-> ddG {len(data2save)}')
      # save lmdb
      wrtEnv = lmdb.open(f'{path}/stability/fireprot/label_data/{unp_name_id_mapping[fam]}_ddG.lmdb',map_size=map_size)
      with wrtEnv.begin(write=True) as txn:
        for i, entry in enumerate(data2save):
          txn.put(str(i).encode(), pkl.dumps(entry))
        txn.put(b'num_examples', pkl.dumps(i+1))
      wrtEnv.close()

    ## dTm labeled set
    target_df_dTm = target_df.loc[target_df['dTm'].notnull()]
    if len(target_df_dTm) > 0:
      data2save = []
      # mutation set
      mut_df = target_df_dTm[['position','wild_type','mutation']].drop_duplicates()
      for i, row in mut_df.iterrows():
        target_mut_df = target_df_dTm.loc[(target_df_dTm['position'] == row['position']) & (target_df_dTm['wild_type'] == row['wild_type']) & (target_df_dTm['mutation'] == row['mutation'])]
        seq_str = target_mut_df.iloc[0]['sequence']
        pos = row['position']
        wt_aa = row['wild_type']
        mut_aa = row['mutation']
        mut_str = f'{wt_aa}{pos}{mut_aa}'
        if pos < domain_start_idx or pos > domain_end_idx:
          print(f'-> variant out of domain: {mut_str}')
          continue
        relative_idx_domain = pos - domain_start_idx
        assert wt_domain_seqStr[relative_idx_domain] == wt_aa
        assert seq_str[pos-1] == wt_aa
        domain_seq_aa_list = list(wt_domain_seqStr)
        domain_seq_aa_list[relative_idx_domain] = mut_aa
        domain_seq_mut_str = ''.join(domain_seq_aa_list)
        label_score = target_mut_df['dTm'].mean()
        data2save.append({
          'set_nm':unp_name_id_mapping[fam],
          'wt_seq':wt_domain_seqStr,
          'seq_len':len(wt_domain_seqStr),
          'mutants':[mut_str],
          'mut_relative_idxs':[relative_idx_domain],
          'mut_seq':domain_seq_mut_str,
          'fitness':label_score})
      print(f'-> dTm {len(data2save)}')
      # save lmdb
      wrtEnv = lmdb.open(f'{path}/stability/fireprot/label_data/{unp_name_id_mapping[fam]}_dTm.lmdb',map_size=map_size)
      with wrtEnv.begin(write=True) as txn:
        for i, entry in enumerate(data2save):
          txn.put(str(i).encode(), pkl.dumps(entry))
        txn.put(b'num_examples', pkl.dumps(i+1))
      wrtEnv.close()
      
  return None

def thermostability_dataset(root_path):
  """
  process thermo-stability dataset
  """
  data_path = "data_process/thermostability"
  unpAcc_null_count = 0
  target_datas = []
  unpAcc_list = []
  target_cellline = 'HepG2'
  with open(f'{root_path}/{data_path}/full_dataset.json') as f:
    full_json = json.load(f)
  for rec in full_json:
    unp_acc = rec.get('uniprotAccession')
    run_name = rec.get('runName')
    melt_point = rec.get('meltingPoint')
    nor_melt_point = rec.get('quantNormMeltingPoint')
    seq = rec.get('sequence')
    if run_name is not None and run_name == target_cellline:
      if unp_acc is None or len(unp_acc) == 0:
        unpAcc_null_count += 1
      else:
        unpAcc_list.append(unp_acc)
      target_datas.append(rec)
  unpAcc_list = list(set(unpAcc_list))
  ## save target datas
  with open(f'{root_path}/{data_path}/{target_cellline}_dataset.json','w') as fw:
    json.dump(target_datas,fw)
      
  ## query pfam families
  mapping_df = pd.read_csv(f'{root_path}/data_process/uniprot/Pfam-A.regions.uniprot.tsv',delimiter='\t',header=0)
  selected_rows = mapping_df.loc[(mapping_df['uniprot_acc'].isin(unpAcc_list))]
  pfam_info_list = []
  uniq_pfam_fams = []
  for idx, row in selected_rows.iterrows():
    pfam_info_list.append([row['uniprot_acc'],row['pfamA_acc'],row['seq_start'],row['seq_end']])
    uniq_pfam_fams.append(row['pfamA_acc'])
  uniq_pfam_fams = list(set(uniq_pfam_fams))

  print(f'cell-line: {target_cellline}, # proteins {len(unpAcc_list)}, # family {len(uniq_pfam_fams)}')

  return None

def cdna_display_dataset(task: str = None):
  """process ddG data from cdna-display paper
  """
  path = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process'
  pfam_pdbmap = pd.read_csv(f'{path}/pfam_35.0/pdbmap',delimiter="\t",names=['pdb_id','chain_id','pdb_range','pfam_name','pfam_id','unp_id','unp_range'])
  df_stability = pd.read_csv(f"{path}/stability/cdna-display/Processed_K50_dG_datasets/K50_dG_Dataset1_Dataset2.csv",header=0,converters={'WT_cluster':str,'match_aaseq':str})
  df_sub = df_stability[['name','aa_seq_full','aa_seq','mut_type','WT_name','WT_cluster','dG_ML','ddG_ML','Stabilizing_mut']]
  wt_cluster_list = df_sub['WT_cluster'].unique().tolist()
  #wt_cluster_list.sort()
  if task == 'pfam_pdbmap':
    '''collect family info for each wt cluster based on pfam_pdbmap'''
    cluster_name_map = []
    for wt_cluster in wt_cluster_list:
      cluster_var_df = df_sub.loc[(df_sub['WT_cluster'] == wt_cluster) & (df_sub['ddG_ML'] != '-') & (df_sub['mut_type'] != 'wt')].reset_index().drop_duplicates()
      cluster_df = df_sub.loc[df_sub['WT_cluster'] == wt_cluster].reset_index().drop_duplicates()
      wt_name_list = cluster_df['WT_name'].unique().tolist()
      pfam_id_collect = []
      wt_info_list = []
      for wt_name in wt_name_list:
        if wt_cluster.isnumeric():
          pdb_id = wt_name.split('.')[0].split('_')[-1]
          pdb_var_df = cluster_var_df.loc[cluster_var_df['WT_name'] == wt_name].reset_index().drop_duplicates()
          target_pfam = pfam_pdbmap.loc[pfam_pdbmap['pdb_id'] == f'{pdb_id};']
          if len(target_pfam) > 0:
            pfam_id_collect.extend(target_pfam['pfam_id'].to_list())
            wt_info_list.append([wt_name,'_'.join([ele.replace(';','') for ele in target_pfam['pfam_id'].to_list()]),len(pdb_var_df)])
          else: ## synthesized domains
            wt_info_list.append([wt_name,'',len(pdb_var_df)])
        else:
          pdb_var_df = cluster_var_df.loc[cluster_var_df['WT_name'] == wt_name].reset_index().drop_duplicates()
          wt_info_list.append([wt_name,'',len(pdb_var_df)])
      wt_info_list.sort(key=lambda x:x[2],reverse=True)
      if len(pfam_id_collect) > 0:
        unique, counts = np.unique(pfam_id_collect, return_counts=True)
        pfam_ids_str = ';'.join([f"{u.strip(';')}:{c}" for u,c in zip(unique,counts)])
      else:
        pfam_ids_str = None
      cluster_name_map.append([wt_cluster,len(cluster_var_df),pfam_ids_str,';'.join([':'.join([str(e) for e in ele]) for ele in wt_info_list])])
    cluster_name_df = pd.DataFrame(cluster_name_map,columns=['WT_cluster','num_var_with_ddG','pfam_list','WT_name_list'])
    cluster_name_df = cluster_name_df.sort_values('num_var_with_ddG',ascending=False)
    cluster_name_df.to_csv(f'{path}/stability/cdna-display/Processed_K50_dG_datasets/D1D2_cluster_name_map.csv',index=False)
  elif task == 'collect_wt_seq':
    seq_record_list = []
    for wt_cluster in wt_cluster_list:
      cluster_df = df_sub.loc[df_sub['WT_cluster'] == wt_cluster].reset_index().drop_duplicates()
      wt_name_list = cluster_df['WT_name'].unique().tolist()
      for wt_name in wt_name_list:
        wt_seq_list = cluster_df.loc[(cluster_df['WT_name'] == wt_name) & (cluster_df['mut_type'].str.lower() == 'wt')]['aa_seq'].unique().tolist()
        if len(wt_seq_list) > 1:
          print(f'wt_cluster:{wt_cluster}; wt_name:{wt_name}')
        else:
          seq_record = SeqRecord(
              Seq(wt_seq_list[0].upper()),
              id=f'{wt_cluster}|{wt_name}',
              description='')
          seq_record_list.append(seq_record)
    with open(f'{path}/stability/cdna-display/Processed_K50_dG_datasets/D1D2_wtSeq.fasta', "w") as output_handle:
      SeqIO.write(seq_record_list, output_handle, "fasta")
  elif task == 'hmmscan':
    hmmscan_out_path = f'{path}/stability/cdna-display/hmmscan_results/D1D2_wtSeq_hmmscan.domtblout'
    ## load hmmscan output to dataframe
    # target name,accession,tlen,query name,accession,qlen,E-value,score,bias,#,of,c-Evalue,i-Evalue,score,bias,h_from,h_to,a_from,a_to,e_from,e_to,acc,description,of,target
    hmmscan_output_df = pd.read_csv(hmmscan_out_path, header=0, comment='#', delim_whitespace=True)
    cluster_name_map = []
    for wt_cluster in wt_cluster_list:
      cluster_var_df = df_sub.loc[(df_sub['WT_cluster'] == wt_cluster) & (df_sub['ddG_ML'] != '-') & (df_sub['mut_type'] != 'wt')].reset_index().drop_duplicates()
      cluster_df = df_sub.loc[df_sub['WT_cluster'] == wt_cluster].reset_index().drop_duplicates()
      wt_name_list = cluster_df['WT_name'].unique().tolist()
      wt_info_list = []
      pfam_id_collect = []
      for wt_name in wt_name_list:
        pdb_var_df = cluster_var_df.loc[cluster_var_df['WT_name'] == wt_name].reset_index().drop_duplicates()
        target_pfams = hmmscan_output_df.loc[hmmscan_output_df['query_name'] == f'{wt_cluster}|{wt_name}'].reset_index().drop_duplicates()
        if len(target_pfams) == 0:
          print(f'{wt_cluster}|{wt_name}')
        else:
          # pick the pfam with lowest E-value
          tar_pfam = target_pfams.iloc[target_pfams['E-value'].astype(float).idxmin()]['pfam_accession'].split('.')[0]
          pfam_id_collect.append(tar_pfam)
          wt_info_list.append([wt_name,tar_pfam,len(pdb_var_df)])
      wt_info_list.sort(key=lambda x:x[2],reverse=True)
      if len(pfam_id_collect) > 0:
        unique, counts = np.unique(pfam_id_collect, return_counts=True)
        pfam_ids_str = ';'.join([f"{u.strip(';')}:{c}" for u,c in zip(unique,counts)])
      else:
        pfam_ids_str = None
      cluster_name_map.append([wt_cluster,len(cluster_var_df),pfam_ids_str,';'.join([':'.join([str(e) for e in ele]) for ele in wt_info_list])])
    cluster_name_df = pd.DataFrame(cluster_name_map,columns=['WT_cluster','num_var_with_ddG','pfam_list','WT_name_list'])
    cluster_name_df = cluster_name_df.sort_values('num_var_with_ddG',ascending=False)
    cluster_name_df.to_csv(f'{path}/stability/cdna-display/Processed_K50_dG_datasets/D1D2_cluster_name_map_hmmscan.csv',index=False)
  else:
    Exception(f'undefined task: {task}')
  return None

def proteingym_domain_seq():
  '''generate seq file for proteins in proteingym dataset
  '''
  path = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  progym_subs_file = f'{path}/data_process/proteingym/ProteinGym_reference_file_substitutions.csv'
  progym_subs_df = pd.read_csv(progym_subs_file, header=0, sep=',')
  seq_rec_list = []
  for i, row in progym_subs_df.iterrows():
    dms_id = row['DMS_id']
    target_seq = row['target_seq']
    msa_start = int(row['MSA_start'])
    msa_end = int(row['MSA_end'])
    seq_rec = SeqRecord(
              Seq(target_seq[msa_start-1:msa_end]),
              id=dms_id,
              description='')
    seq_rec_list.append(seq_rec)
  with open(f'{path}/data_process/proteingym/protengym_wt_seq.fasta', "w") as output_handle:
      SeqIO.write(seq_rec_list, output_handle, "fasta")
  return None

def petase_mutations():
  """prepare mutation designs of Petase into lmdb file for evaluation
  """
  root_path = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  petase_muts_path = f'{root_path}/data_process/petase/PETase_design.npy'
  petase_dict = np.load(petase_muts_path,allow_pickle=True).item()
  #wt_seq = petase_dict['WT']
  wt_seq = 'MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS'
  ## load mutation list
  mutation_df = pd.read_csv(f'{root_path}/data_process/petase/mutations.csv',header=0,sep=',')
  
  set_dt_list = []
  ## loop over PETase_design.npy
  # for mut_str, mut_seq in petase_dict.items():
  #   if mut_str in ['WT']:
  #     continue
  #   print(f'>{mut_str}')
  #   target_muts = re.split('_',mut_str)
  #   mut_relative_idxs = []
  #   # get mut relative pos in the seq
  #   for seq_i in range(len(wt_seq)):
  #     if wt_seq[seq_i] != mut_seq[seq_i]:
  #       mut_relative_idxs.append(seq_i)
  #   assert len(mut_relative_idxs) == len(target_muts)
  #   one_json = {"set_nm": 'petase',
  #               "wt_seq": wt_seq,
  #               "seq_len": len(mut_seq),
  #               "mutants": target_muts,
  #               "mut_relative_idxs": mut_relative_idxs,
  #               "mut_seq": mut_seq,
  #               "fitness": 1.0}
  #   set_dt_list.append(one_json)
  
  wt_seq_list = list(wt_seq)
  for i, row in mutation_df.iterrows():
    mut_name = row['mut_name']
    print(mut_name)
    var_list = row['var_list'].split('+')
    mut_seq_list = list(wt_seq)
    mut_relative_idxs = []
    for var_str in var_list:
      wt_aa = var_str[0]
      mut_aa = var_str[-1]
      mut_idx = int(var_str[1:-1])
      #print(wt_seq_list[mut_idx-1],wt_aa)
      assert wt_seq_list[mut_idx-1] == wt_aa
      mut_relative_idxs.append(mut_idx-1)
      mut_seq_list[mut_idx-1] = mut_aa
    mut_seq_str = ''.join(mut_seq_list)
    print(mut_relative_idxs)
    one_json = {"set_nm": 'petase_mla',
                "wt_seq": wt_seq,
                "seq_len": len(mut_seq_str),
                "mutants": var_list,
                "mut_relative_idxs": mut_relative_idxs,
                "mut_seq": mut_seq_str,
                "fitness": 1.0}
    set_dt_list.append(one_json)
  # save data
  map_size = (1024 * 15) * (2 ** 20) # 15G
  wrtEnv = lmdb.open(f'{root_path}/data_process/petase/petase_mut_mla.lmdb',map_size=map_size)
  with wrtEnv.begin(write=True) as txn:
    for i, entry in enumerate(set_dt_list):
      txn.put(str(i).encode(), pkl.dumps(entry))
    txn.put(b'num_examples', pkl.dumps(i+1))
  wrtEnv.close()
  
  print(f'>In total, {len(set_dt_list)} mut cases')
  return None

def deepSeqSet_wt_seq_file():
  """Generate a sequence file (e.g. fasta) containing all wt sequences in DeepSequence Mutagenesis Set.
  Such file will be used for structure query process
  """
  path = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  deepSeqSet_fam_info = 'data_process/mutagenesis/DeepSequenceMutaSet_reference_file.csv'
  deepSeqSet_seqIdx_info = 'data_process/mutagenesis/DeepSequenceMutaSet_priSeq_idxRange'
  seq_save_path='data_process/mutagenesis/DeepSequenceMutaSet_priSeq'
  deepSeqSet_fam_df = pd.read_csv(f'{path}/{deepSeqSet_fam_info}',decimal=',',header=0)
  deepSeqSet_fam_unp_df = deepSeqSet_fam_df[['Shin2021_set','Unp_name','Unp_id','Target_seq','Start','End']].drop_duplicates()
  for idx, row in deepSeqSet_fam_unp_df.iterrows():
    record = SeqRecord(
                Seq(row['Target_seq']),
                id=f"{row['Shin2021_set']}|{row['Unp_id']}|{row['Unp_name']}|{row['Start']}-{row['End']}",
                description="", name="")
    with open(f"{path}/{seq_save_path}/{row['Shin2021_set']}.fasta", 'w') as output_handle:
      SeqIO.write([record], output_handle, "fasta")
  return None

def rho_auroc_prc(
      fit_true: np.ndarray,
      fit_pred: np.ndarray,
      bina_cutoff: float,
      sign_reverse: bool):
  """Calculate spearman's rho, AUROC and AUPRC
  """
  mut_y_gt = (fit_true >= bina_cutoff).astype(np.int8) # positive class 1 if > bina_cutoff
  rho_score,rho_p_value = spearmanr(fit_true,fit_pred)
  if sign_reverse:
    rho_score = abs(rho_score)
    mut_y_gt = (fit_true < bina_cutoff).astype(np.int8) # positive class 1 if <= bina_cutoff
  fpr, tpr, roc_thresholds = roc_curve(mut_y_gt, fit_pred)
  precision, recall, prc_thresholds = precision_recall_curve(mut_y_gt, fit_pred)
  auroc = auc(fpr, tpr)
  auprc = auc(recall, precision)
  return rho_score, rho_p_value, auroc, auprc

def sota_compare_use_esm_file(load_processed_df: bool = True, fig_name_list: List=None):
  """compare to sota models. Use predictions from ESM
  """
  path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  ## load sota predictions
  #header: protein_name,mutant,gt,MSA Transformer,PSSM,EVMutation (published),DeepSequence (published) - single,DeepSequence (published),ESM-1v (zero shot),ESM-1v (+further training),DeepSequence (replicated),EVMutation (replicated),ProtBERT-BFD,TAPE,UniRep,ESM-1b
  sota_pred_df = pd.read_csv(f'{path}/data_process/mutagenesis/esm_predictions/raw_with_wavenet_df.csv',delimiter=',',header=0)
  fitness_fam_info_df = pd.read_csv(f'{path}/data_process/mutagenesis/DeepSequenceMutaSet_reference_file.csv',delimiter=',',header=0)
  #'AMIE_PSEAE_Whitehead','B3VI55_LIPSTSTABLE','B3VI55_LIPST_Whitehead2015','BG_STRSQ_hmmerbit','BLAT_ECOLX_Ostermeier2014','BLAT_ECOLX_Palzkill2012','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Tenaillon2013','BRCA1_HUMAN_BRCT','BRCA1_HUMAN_RING','CALM1_HUMAN_Roth2017','DLG4_RAT_Ranganathan2012','GAL4_YEAST_Shendure2015','HIS7_YEAST_Kondrashov2017','HSP82_YEAST_Bolon2016','IF1_ECOLI_Kishony','KKA2_KLEPN_Mikkelsen2014','MK01_HUMAN_Johannessen','MTH3_HAEAESTABILIZED_Tawfik2015','P84126_THETH_b0','PABP_YEAST_Fields2013-doubles','PABP_YEAST_Fields2013-singles','PTEN_HUMAN_Fowler2018','RASH_HUMAN_Kuriyan','RL401_YEAST_Bolon2013','RL401_YEAST_Bolon2014','RL401_YEAST_Fraser2016','SUMO1_HUMAN_Roth2017','TIM_SULSO_b0','TIM_THEMA_b0','TPK1_HUMAN_Roth2017','TPMT_HUMAN_Fowler2018','UBC9_HUMAN_Roth2017','UBE4B_MOUSE_Klevit2013-singles','YAP1_HUMAN_Fields2012-singles'
  #'BF520_env_Bloom2018','BG505_env_Bloom2018','HG_FLU_Bloom2016','PA_FLU_Sun2015','POLG_HCVJF_Sun2014','POL_HV1N5-CA_Ndungu2014'
  #parEparD_Laub2015_all
  set_names = ['AMIE_PSEAE_Whitehead','B3VI55_LIPSTSTABLE','B3VI55_LIPST_Whitehead2015','BG_STRSQ_hmmerbit','BLAT_ECOLX_Ostermeier2014','BLAT_ECOLX_Palzkill2012','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Tenaillon2013','BRCA1_HUMAN_BRCT','BRCA1_HUMAN_RING','CALM1_HUMAN_Roth2017','DLG4_RAT_Ranganathan2012','GAL4_YEAST_Shendure2015','HIS7_YEAST_Kondrashov2017','HSP82_YEAST_Bolon2016','IF1_ECOLI_Kishony','KKA2_KLEPN_Mikkelsen2014','MK01_HUMAN_Johannessen','MTH3_HAEAESTABILIZED_Tawfik2015','P84126_THETH_b0','PABP_YEAST_Fields2013-doubles','PABP_YEAST_Fields2013-singles','PTEN_HUMAN_Fowler2018','RASH_HUMAN_Kuriyan','RL401_YEAST_Bolon2013','RL401_YEAST_Bolon2014','RL401_YEAST_Fraser2016','SUMO1_HUMAN_Roth2017','TIM_SULSO_b0','TIM_THEMA_b0','TPK1_HUMAN_Roth2017','TPMT_HUMAN_Fowler2018','UBC9_HUMAN_Roth2017','UBE4B_MOUSE_Klevit2013-singles','YAP1_HUMAN_Fields2012-singles']

  sign_reverse_set_names = ['BLAT_ECOLX_Palzkill2012','MK01_HUMAN_Johannessen']
  sota_mdl_names = ['PSSM','EVMutation (published)','DeepSequence (published)','Wavenet','MSA Transformer','ESM-1v (zero shot)','ESM-1v (+further training)','ESM-1b','ProtBERT-BFD','TAPE','UniRep']
  epoch_joint_ft_list = ['best',0,5,11,17,23,29,35,41,47,53,59]
  epoch_seq_ft_list = ['best',0,3,7,11,15,19,23,27,31,35,39]
  lambda_list = [0.0,0.5,2.0,20.0] #0.0,0.01,0.1,0.33,0.5,1.0,2.0,5.0,10.0,20.0
  if load_processed_df:
    metric_score_multitask_allMSASeq_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_multitaskFT_allMSASeq.csv',sep=',',header=0)
    pre_task_valid_multitask_allMSASeq_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/pre_task_valid_multitaskFT_allMSASeq.csv',sep=',',header=0)
    #metric_score_multitask_allMSASeq_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_multitaskFT_allMSASeq_DM_noSeqNb.csv',sep=',',header=0)
    #pre_task_valid_multitask_allMSASeq_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/pre_task_valid_multitaskFT_allMSASeq_DM_noSeqNb.csv',sep=',',header=0)
    metric_score_multitask_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_multitaskFT.csv',sep=',',header=0)
    pre_task_valid_multitask_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/pre_task_valid_multitaskFT.csv',sep=',',header=0)
    metric_score_seq_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_seqFT.csv',sep=',',header=0)
    metric_score_sota_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_sota.csv',sep=',',header=0)
  else:
    ## seq+struct finetune allMSASeq models
    metric_score_list = []
    mdl_path_list = [] # to store model paths, cols: 'prot_nm','mut_nm','lmd','mdl_path','mdl_path_1'
    mdl_path_extra = False
    pre_task_valid_score_list = [] # to store score on valid set for pre-train tasks
    for set_nm in set_names:
      bina_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM']==set_nm,:].iloc[0]['DMS_binarization_cutoff']
      sign_reverse = set_nm in sign_reverse_set_names
      ## get protein name
      prot_nm = fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,:].iloc[0]['Shin2021_set']
      ## load validation mutations
      valid_mut_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/valid_mutation.csv',header=0)
      fit_gt_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/mut_fit_gt.csv',header=0)
      valid_fit_gt_df = fit_gt_df.loc[fit_gt_df['mut_name'].isin(valid_mut_df['mut_name']) & (fit_gt_df['fitness'].notnull())]
      for lmd in lambda_list:
        ## get model path
        local_mdl_path_list = []
        log_files = glob.glob(f'{path}/job_logs/archive_multitask_models/*{prot_nm}*ss{lmd}_*lm_multitask_symm_fast.allMSASeq.o*')
        assert len(log_files) > 0
        log_files_extra = glob.glob(f'{path}/job_logs/archive_multitask_models/*{prot_nm}*ss{lmd}_*lm_multitask_symm_fast.allMSASeq.l*')
        if len(log_files_extra) > 0:
          mdl_path_extra = True
        for fl in log_files + log_files_extra:
          extra_mdl_path = os.popen(f"grep -a 'Saving model checkpoint' {fl} | grep -E -o 'seq_structure_multi_task.*[0-9]{{6}}' | uniq").read().strip('\n')
          local_mdl_path_list.append(extra_mdl_path)
        mdl_path_list.append([prot_nm,set_nm,lmd]+local_mdl_path_list)
        ## get pre-train validation scores
        pre_task_valid_score_df = joint_model_select_best_CK(prot_nm=prot_nm,prot_mut_nm=set_nm,lmd=lmd,mdl_path_list=local_mdl_path_list)
        pre_task_valid_score_list.append(pre_task_valid_score_df)
        ## loop epoch
        for epoch in epoch_joint_ft_list:
          print(f'seq+structFT (allMSASeq),{set_nm},{lmd},{epoch}')
          ## get model name
          model_path = os.popen(f"grep -E -o 'seq_structure_multi_task.*[0-9]{{6}}' {path}/job_logs/archive_multitask_models_eval/*ss{lmd}_*epoch_{epoch}.*{set_nm}*multitask_fitness_UNsupervise_mutagenesis.lm_multitask_symm_fast.allMSASeq.o* | uniq").read().strip('\n')
          ## load my predicted fitness scores
          my_pred_df = pd.read_csv(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/predictions/{model_path}/{set_nm}_{epoch}_wtstruct_rawScores.csv',names=['mutant','myMdl']).drop_duplicates().groupby(['mutant'],as_index=False)['myMdl'].mean()
          my_pred_df['epoch'] = epoch
          my_pred_df['lmd'] = lmd
          my_pred_df['protein_name'] = set_nm
          ## metrics calculation on validation mutations
          valid_merge_df = valid_fit_gt_df.merge(my_pred_df, left_on=['mut_name'], right_on=['mutant'], how='inner')
          rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(valid_merge_df['fitness'].to_numpy().reshape(-1),valid_merge_df['myMdl'].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
          one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,'valid']
          metric_score_list.append(one_score_case)
          ## test mutations, join on common variants
          set_sota_pred_df = sota_pred_df.loc[(sota_pred_df['protein_name'] == set_nm) & (sota_pred_df['gt'].notnull()) & (~sota_pred_df['mutant'].isin(valid_mut_df['mut_name']))]
          sota_myMdl_joint_df = set_sota_pred_df.merge(my_pred_df, on=['mutant','protein_name'], how='inner')
          assert len(set_sota_pred_df)-len(sota_myMdl_joint_df) == 0
          rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_joint_df['gt'].to_numpy().reshape(-1),sota_myMdl_joint_df['myMdl'].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
          one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,'test']
          metric_score_list.append(one_score_case)
    ## save fit eval score df
    metric_score_multitask_allMSASeq_df = pd.DataFrame(metric_score_list,columns=['protein_name','lambda','epoch','rho','rho_P','auroc','auprc','eval_set'])
    metric_score_multitask_allMSASeq_df.to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_multitaskFT_allMSASeq.csv',index=False,header=True)
    ## save model path df
    # if mdl_path_extra:
    #   pd.DataFrame(mdl_path_list,columns=['prot_nm','mut_nm','lmd','mdl_path','mdl_path_1']).to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/model_path_multitaskFT_allMSASeq.csv',index=False,header=True)
    # else:
    #   pd.DataFrame(mdl_path_list,columns=['prot_nm','mut_nm','lmd','mdl_path']).to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/model_path_multitaskFT_allMSASeq.csv',index=False,header=True)
    ## save valid scores for pre-train tasks
    pre_task_valid_multitask_allMSASeq_df = pd.concat(pre_task_valid_score_list)
    pre_task_valid_multitask_allMSASeq_df.to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/pre_task_valid_multitaskFT_allMSASeq.csv',index=False,header=True)

    ## seq+structure finetuned models
    metric_score_list = []
    mdl_path_list = [] # to store model paths, cols: 'prot_nm','mut_nm','lmd','mdl_path','mdl_path_1'
    pre_task_valid_score_list = [] # to store score on valid set for pre-train tasks
    for set_nm in set_names:
      prot_nm = fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,:].iloc[0]['Shin2021_set']
      bina_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM']==set_nm,:].iloc[0]['DMS_binarization_cutoff']
      sign_reverse = set_nm in sign_reverse_set_names
      ## load validation mutations
      valid_mut_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/valid_mutation.csv',header=0)
      fit_gt_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/mut_fit_gt.csv',header=0)
      valid_fit_gt_df = fit_gt_df.loc[fit_gt_df['mut_name'].isin(valid_mut_df['mut_name']) & (fit_gt_df['fitness'].notnull())]
      for lmd in lambda_list:
        ## get model path
        local_mdl_path_list = []
        log_files = glob.glob(f'{path}/job_logs/archive_multitask_models/*{prot_nm}*ss{lmd}_*lm_multitask_symm_fast.allData.o*')
        assert len(log_files) > 0
        log_files_extra = glob.glob(f'{path}/job_logs/archive_multitask_models/*{prot_nm}*ss{lmd}_*lm_multitask_symm_fast.allData.l*')
        for fl in log_files + log_files_extra:
          extra_mdl_path = os.popen(f"grep -a 'Saving model checkpoint' {fl} | grep -E -o 'seq_structure_multi_task.*[0-9]{{6}}' | uniq").read().strip('\n')
          local_mdl_path_list.append(extra_mdl_path)
        mdl_path_list.append([prot_nm,set_nm,lmd]+local_mdl_path_list)
        ## get pre-train validation scores
        pre_task_valid_score_df = joint_model_select_best_CK(prot_nm=prot_nm,prot_mut_nm=set_nm,lmd=lmd,mdl_path_list=local_mdl_path_list)
        pre_task_valid_score_list.append(pre_task_valid_score_df)
        for epoch in epoch_joint_ft_list:
          print(f'seq+structFT,{set_nm},{lmd},{epoch}')
          ## get model name
          model_path = os.popen(f"grep -E -o 'seq_structure_multi_task.*[0-9]{{6}}' {path}/job_logs/archive_multitask_models_eval/*ss{lmd}_*epoch_{epoch}.*{set_nm}*multitask_fitness_UNsupervise_mutagenesis.lm_multitask_symm_fast.allData.o* | uniq").read().strip('\n')
          ## load my predicted fitness scores
          my_pred_df = pd.read_csv(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/predictions/{model_path}/{set_nm}_{epoch}_wtstruct_rawScores.csv',names=['mutant','myMdl']).drop_duplicates().groupby(['mutant'],as_index=False)['myMdl'].mean()
          my_pred_df['epoch'] = epoch
          my_pred_df['lmd'] = lmd
          my_pred_df['protein_name'] = set_nm
          ## metrics calculation on validation mutations
          valid_merge_df = valid_fit_gt_df.merge(my_pred_df, left_on=['mut_name'], right_on=['mutant'], how='inner')
          rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(valid_merge_df['fitness'].to_numpy().reshape(-1),valid_merge_df['myMdl'].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
          one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,'valid']
          metric_score_list.append(one_score_case)
          ## test mutations, join on common variants
          set_sota_pred_df = sota_pred_df.loc[(sota_pred_df['protein_name'] == set_nm) & (sota_pred_df['gt'].notnull()) & (~sota_pred_df['mutant'].isin(valid_mut_df['mut_name']))]
          sota_myMdl_joint_df = set_sota_pred_df.merge(my_pred_df, on=['mutant','protein_name'], how='inner')
          assert len(set_sota_pred_df)-len(sota_myMdl_joint_df) == 0
          rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_joint_df['gt'].to_numpy().reshape(-1),sota_myMdl_joint_df['myMdl'].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
          one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,'test']
          metric_score_list.append(one_score_case)
    metric_score_multitask_df = pd.DataFrame(metric_score_list,columns=['protein_name','lambda','epoch','rho','rho_P','auroc','auprc','eval_set'])
    metric_score_multitask_df.to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_multitaskFT.csv',index=False,header=True)
    ## save model path df
    # pd.DataFrame(mdl_path_list,columns=['prot_nm','mut_nm','lmd','mdl_path','mdl_path_1']).to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/model_path_multitaskFT.csv',index=False,header=True)
    ## save valid scores for pre-train tasks
    pre_task_valid_multitask_df = pd.concat(pre_task_valid_score_list)
    pre_task_valid_multitask_df.to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/pre_task_valid_multitaskFT.csv',index=False,header=True)
    
    ## seq finetuned models (rho is calcualted over test mutation set)
    metric_score_list = []
    for set_nm in set_names:
      valid_mut_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/valid_mutation.csv',header=0)
      bina_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM']==set_nm,:].iloc[0]['DMS_binarization_cutoff']
      sign_reverse = set_nm in sign_reverse_set_names
      for epoch in epoch_seq_ft_list:
        for seq_mode in ['structSeq', 'allSeq']:
          print(f'seqFT,{set_nm},{epoch},{seq_mode}')
          ## get model name
          model_path_list = os.popen(f"grep -E -o 'masked_language_modeling_transformer.*[0-9]{{6}}' {path}/job_logs/archive_baseline_bert_eval/baseline_bert_mutation_fitness_UNsupervise_mutagenesis_rp15_all_2_729.seq_finetune.reweighted.{seq_mode}.TotalEpoch40.Interval*.{epoch}.*.{set_nm}.seqMaxL512.classWno.out | uniq").read().strip('\n').split('\n')
          assert len(model_path_list) == 1
          model_path = model_path_list[0]
          ## load my preds
          my_pred_df = pd.read_csv(f'{path}/eval_results/mutation_fitness_UNsupervise_mutagenesis/predictions/{model_path}/{set_nm}_{epoch}_rawScores.csv',names=['mutant','myMdl']).drop_duplicates().groupby(['mutant'],as_index=False)['myMdl'].mean()
          my_pred_df['epoch'] = epoch
          my_pred_df['protein_name'] = set_nm
          my_pred_df['seq_mode'] = seq_mode
          ## join on common variants
          set_sota_pred_df = sota_pred_df.loc[(sota_pred_df['protein_name'] == set_nm) & (sota_pred_df['gt'].notnull()) & (~sota_pred_df['mutant'].isin(valid_mut_df['mut_name']))]
          sota_myMdl_joint_df = set_sota_pred_df.merge(my_pred_df, on=['mutant','protein_name'], how='inner')
          assert len(set_sota_pred_df)-len(sota_myMdl_joint_df) == 0
          rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_joint_df['gt'].to_numpy().reshape(-1),sota_myMdl_joint_df['myMdl'].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
          one_score_case = [set_nm,epoch,seq_mode,rho_score,rho_p_value,auroc,auprc]
          metric_score_list.append(one_score_case)
    metric_score_seq_df = pd.DataFrame(metric_score_list,columns=['protein_name','epoch','seq_mode','rho','rho_P','auroc','auprc'])
    metric_score_seq_df.to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_seqFT.csv',index=False,header=True)

    ## sota models (rho is calcualted over test mutation set)
    metric_score_list = []
    for set_nm in set_names:
      bina_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM']==set_nm,:].iloc[0]['DMS_binarization_cutoff']
      sign_reverse = set_nm in sign_reverse_set_names
      valid_mut_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/valid_mutation.csv',header=0)
      set_sota_pred_df = sota_pred_df.loc[(sota_pred_df['protein_name'] == set_nm) & (sota_pred_df['gt'].notnull()) & (~sota_pred_df['mutant'].isin(valid_mut_df['mut_name']))]
      for sota_mdl in sota_mdl_names:
        print(f'sota,{set_nm},{sota_mdl}')
        rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(set_sota_pred_df['gt'].to_numpy().reshape(-1),set_sota_pred_df[sota_mdl].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
        one_score_case = [set_nm,sota_mdl,rho_score,rho_p_value,auroc,auprc]
        metric_score_list.append(one_score_case)
    metric_score_sota_df = pd.DataFrame(metric_score_list,columns=['protein_name','mdl_name','rho','rho_P','auroc','auprc'])
    metric_score_sota_df.to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_sota.csv',index=False,header=True)

  def lambda_transform(x):
    if x == -2:
      return -0.3
    elif x == -1:
      return -0.2
    elif x == 0:
      return -0.1
    elif x > 0:
      return np.log(1+x)

  if 'hist' in fig_name_list:
    ## score histogram
    wanted_percentiles = [20, 40, 50, 60, 80]
    fitness_fam_info_df = pd.read_csv(f'{path}/data_process/mutagenesis/DeepSequenceMutaSet_reference_file.csv',delimiter=',',header=0)
    fitness_fam_info_df['first_cutoff'] = fitness_fam_info_df['DMS_binarization_cutoff']
    fitness_fam_info_df['second_cutoff'] = fitness_fam_info_df['DMS_binarization_cutoff']
    for set_nm in set_names:
      fit_gt_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/mut_fit_gt.csv',header=0)
      target_fit_gt_df = fit_gt_df.loc[fit_gt_df['fitness'].notnull(),:]
      fit_median = target_fit_gt_df['fitness'].median()
      print(f'{set_nm} fitness median: {fit_median}')
      fit_arr = target_fit_gt_df['fitness'].to_numpy()
      fit_arr.sort()
      percentile_sx = [fit_arr[int(len(fit_arr) * p / 100)] for p in wanted_percentiles]
      binarize_method = fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,'DMS_binarization_method'].iloc[0]
      binarize_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,'DMS_binarization_cutoff'].iloc[0]
      if binarize_method == 'median':
        fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,'first_cutoff'] = percentile_sx[1]
        fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,'second_cutoff'] = percentile_sx[3]
        plt.figure(figsize=(12,8))
        ax = plt.gca()
        #sns.set_style("whitegrid")
        sns.set(style="whitegrid", rc={"lines.linewidth": 1.2})
        sns.histplot(data=target_fit_gt_df, x='fitness', kde=True)
        for xp in percentile_sx:
          ax.axvline(xp, color='crimson')
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(percentile_sx)
        ax2.set_xticklabels(percentile_sx, rotation=45, color='crimson')
        plt.savefig(f'eval_results/multitask_fitness_UNsupervise_mutagenesis_structure/figures/set_wise/{set_nm}_gt_hist.png',dpi=600,bbox_inches='tight')
        plt.clf()
        plt.close()
      elif binarize_method == 'manual':
        cutoff_idx = np.searchsorted(fit_arr, binarize_cutoff, side='right')
        ten_pc_num = int(len(fit_arr)*0.1)
        percentile_sx = [fit_arr[cutoff_idx-ten_pc_num],fit_arr[cutoff_idx],fit_arr[cutoff_idx+ten_pc_num]]
        fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,'first_cutoff'] = fit_arr[cutoff_idx-ten_pc_num]
        fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,'second_cutoff'] = fit_arr[cutoff_idx+ten_pc_num]
        plt.figure(figsize=(12,8))
        ax = plt.gca()
        #sns.set_style("whitegrid")
        sns.set(style="whitegrid", rc={"lines.linewidth": 1.2})
        sns.histplot(data=target_fit_gt_df, x='fitness', kde=True)
        for xp in percentile_sx:
          ax.axvline(xp, color='crimson')
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(percentile_sx)
        ax2.set_xticklabels(percentile_sx, rotation=45, color='crimson')
        plt.savefig(f'eval_results/multitask_fitness_UNsupervise_mutagenesis_structure/figures/set_wise/{set_nm}_gt_hist.png',dpi=600,bbox_inches='tight')
        plt.clf()
        plt.close()
    fitness_fam_info_df.to_csv(f'{path}/data_process/mutagenesis/DeepSequenceMutaSet_reference_file_new.csv',index=False,header=True)
  if 'mt_epoch_track' in fig_name_list:
    plt.figure(figsize=(12,8))
    #sns.set_style("whitegrid")
    sns.set(style="whitegrid", rc={"lines.linewidth": 1.2})
    #sns.lineplot(data=metric_score_multitask_df, x="epoch", y="rho", hue="protein_name", err_style="bars", errorbar=("sd"), palette='Paired',linewidth=3)
    gax = sns.pointplot(data=metric_score_multitask_df.loc[metric_score_multitask_df['eval_set']=='valid'], x="epoch", y="rho", hue="protein_name", join=True, errorbar=('sd'), dodge=True, palette='Paired')
    plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0, fontsize=12)
    plt.savefig('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/mt_epoch_track.png',dpi=600,bbox_inches='tight')
    plt.clf()
  if 'seq_epoch_track' in fig_name_list:
    plt.figure(figsize=(12,8))
    sns.set_style("whitegrid")
    sns.lineplot(data=metric_score_seq_df, x="epoch", y="rho", hue="protein_name", palette='Paired',linewidth=3)
    plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0, fontsize=12)
    plt.savefig('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/seq_epoch_track.png',dpi=600,bbox_inches='tight')
    plt.clf()
  if 'mt_lambda_track' in fig_name_list:
    target_multitask_df = metric_score_multitask_df.loc[(metric_score_multitask_df['epoch'] > 30) & (metric_score_multitask_df['eval_set'] == 'test')]
    metric_score_seq_df['lambda'] = -1
    metric_score_seq_df.loc[metric_score_seq_df['epoch'] == 0,'lambda'] = -2
    ## seq pretrained(epoch:0); finetuned model(epoch >30)
    target_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['epoch'] > 30) | (metric_score_seq_df['epoch'] == 0)]
    target_df = pd.concat([target_multitask_df,target_seq_df])
    target_df['lambda_trans']=target_df['lambda'].apply(lambda x: lambda_transform(x))
    plt.figure(figsize=(12,8))
    sns.set_style("whitegrid")
    gax = sns.lineplot(data=target_df, x="lambda_trans", y="rho", hue="protein_name", err_style="bars", errorbar=("sd"), palette='Paired',linewidth=1.5)
    gax.set(xticks=[-0.3,-0.2,-0.1]+np.log(1+np.array(lambda_list[1:])).tolist())
    gax.set(xticklabels=[-2,-1,0]+lambda_list[1:])
    plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0, fontsize=12)
    plt.savefig('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/mt_lambda_track.png',dpi=600,bbox_inches='tight')
    plt.clf()
  if 'mt_protein_lambda_track' in fig_name_list:
    target_multitask_df = metric_score_multitask_df.loc[(metric_score_multitask_df['epoch'] > 30) & (metric_score_multitask_df['eval_set'] == 'test')]
    metric_score_seq_df['lambda'] = -1
    metric_score_seq_df.loc[metric_score_seq_df['epoch'] == 0,'lambda'] = -2
    ## seq pretrained(epoch:0); finetuned model(epoch >30)
    target_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['epoch'] > 30) | (metric_score_seq_df['epoch'] == 0)]
    target_df = pd.concat([target_multitask_df,target_seq_df])
    fig, ax = plt.subplots(figsize=(14,8))
    sns.set_style("whitegrid")
    gax = sns.pointplot(data=target_df, x="protein_name", y="rho", hue="lambda", join=False, errorbar=None, dodge=True, palette='Paired') #('pi',100)
    gax.set_xticklabels(gax.get_xticklabels(), rotation=45, ha='right')
    ax.xaxis.grid(True) # Show the vertical gridlines
    plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0, fontsize=12)
    plt.savefig('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/mt_protein_lambda_track_noErrorBar.png',dpi=600,bbox_inches='tight')
    plt.clf()
  if 'sota_protein_track' in fig_name_list:
    epoch_filter_df = metric_score_multitask_df[['protein_name','lambda','epoch','rho']].loc[metric_score_multitask_df['epoch'] > 30]
    #'protein_name','lambda','epoch','rho','rho_P'
    target_df = epoch_filter_df.groupby(['protein_name','lambda'],as_index=False)['rho'].mean().groupby(['protein_name'],as_index=False)['rho'].max()
    target_df['mdl_name'] = 'MyMdl (joint finetune)'
    target_pre_seq_df = metric_score_seq_df[['protein_name','epoch','rho']].loc[metric_score_seq_df['epoch'] == 0,:]
    target_pre_seq_df['mdl_name'] = 'MyMdl (pretrain)'
    target_ft_seq_df = metric_score_seq_df[['protein_name','epoch','rho']].loc[metric_score_seq_df['epoch'] > 30,:].groupby(['protein_name'],as_index=False)['rho'].mean()
    target_ft_seq_df['mdl_name'] = 'MyMdl (seq finetune)'
    metric_score_all_df = pd.concat([metric_score_sota_df[['protein_name','rho','mdl_name']],target_df,target_pre_seq_df,target_ft_seq_df])
    ave_sota_df = metric_score_all_df.groupby(['mdl_name'],as_index=False)['rho'].mean()
    print(ave_sota_df.to_string())
    fig, ax = plt.subplots(figsize=(14,8))
    #sns.set_style("whitegrid")
    gax = sns.pointplot(data=metric_score_all_df, x="protein_name", y="rho", hue="mdl_name", join=False, errorbar=None, dodge=False, palette='tab20')
    gax.set_xticklabels(gax.get_xticklabels(), rotation=45, ha='right')
    ax.xaxis.grid(True) # Show the vertical gridlines
    plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0, fontsize=8)
    plt.savefig('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/sota_protein_track.png',dpi=600,bbox_inches='tight')
    plt.clf()
  if 'sota_protein_track_valid_best' in fig_name_list:
    ###>>> joint-model, allMSASeq, fit-valid <<<###
    test_score_allMSASeq_list = []
    for metric_nm in ['rho', 'auroc', 'auprc']:
      valid_best_idx = metric_score_multitask_allMSASeq_df.loc[metric_score_multitask_allMSASeq_df['eval_set'] == 'valid'].groupby(['protein_name'],as_index=False)[metric_nm].transform(max) == metric_score_multitask_allMSASeq_df.loc[metric_score_multitask_allMSASeq_df['eval_set'] == 'valid'][metric_nm].to_frame()
      valid_best_df = metric_score_multitask_allMSASeq_df.iloc[valid_best_idx.loc[valid_best_idx[metric_nm]==True].index].drop_duplicates(subset=['protein_name'], keep='last')
      test_score_allMSASeq_df = valid_best_df.merge(metric_score_multitask_allMSASeq_df.loc[metric_score_multitask_allMSASeq_df['eval_set'] == 'test'], on=['protein_name','lambda','epoch'], how='inner',suffixes=('_valid','_test'))
      ## output best (lambda, epoch) selected by validation set
      print(f'***allMSASeq, fit_valid_best, {metric_nm}: best lambda, epoch***')
      print(test_score_allMSASeq_df[['protein_name','lambda','epoch']].to_string())
      if metric_nm == 'rho':
        test_score_allMSASeq_df[['protein_name','lambda','epoch']].to_csv(f'eval_results/embedding_analysis/setups/SI_FT_all_fitValid.csv',index=False)
      #print(test_score_allMSASeq_df['lambda'].value_counts().to_string())
      #print(test_score_allMSASeq_df['epoch'].value_counts().to_string())

      test_score_allMSASeq_df= test_score_allMSASeq_df[['protein_name',f'{metric_nm}_test']].rename(columns={f'{metric_nm}_test': metric_nm})
      test_score_allMSASeq_df['mdl_name'] = 'Our SI-pLM (seq_all;fit_valid_best)'
      test_score_allMSASeq_list.append(test_score_allMSASeq_df)
    ## merge dfs of three metrics
    test_all_score_allMSASeq_df = test_score_allMSASeq_list[0].merge(test_score_allMSASeq_list[1],on=['protein_name','mdl_name'])
    test_all_score_allMSASeq_df = test_all_score_allMSASeq_df.merge(test_score_allMSASeq_list[2],on=['protein_name','mdl_name'])

    ###>>> joint-model, allMSASeq, pre_task_valid <<<###
    # keys in pre_task_valid_multitask_allMSASeq_df: 'prot_nm','mut_nm','lambda','epoch','valid_score'
    #* option 1: compare cross lambda and epoch using (val_aa_ppl+val_ss_ppl_val_ras_ppl+val_dm_ppl)
    '''
    pre_task_valid_best_idx = pre_task_valid_multitask_allMSASeq_df.groupby(['mut_nm'],as_index=False)['valid_score'].transform(min) == pre_task_valid_multitask_allMSASeq_df['valid_score'].to_frame()
    valid_best_df = pre_task_valid_multitask_allMSASeq_df.iloc[pre_task_valid_best_idx.loc[pre_task_valid_best_idx['valid_score']==True].index].rename(columns={'mut_nm':'protein_name'}).drop_duplicates(subset=['protein_name'], keep='last')
    test_score_allMSASeq_preTask_df = valid_best_df.merge(metric_score_multitask_allMSASeq_df.loc[metric_score_multitask_allMSASeq_df['eval_set'] == 'test'], on=['protein_name','lambda','epoch'])
    print('***allMSASeq, pre_task_best: best lambda, epoch***')
    #print(test_score_allMSASeq_preTask_df[['protein_name','lambda','epoch']].to_string())
    print(test_score_allMSASeq_preTask_df['lambda'].value_counts().to_string())
    #print(test_score_allMSASeq_preTask_df['epoch'].value_counts().to_string())
    
    test_score_allMSASeq_preTask_df= test_score_allMSASeq_preTask_df[['protein_name','rho','auroc','auprc']]
    test_score_allMSASeq_preTask_df['mdl_name'] = 'Our SI-pLM (seq_all;pre_valid_best)'
    '''
    #* option 2: best checkpoint is used within lambda, fitness validation set is used to select cross lambda 
    test_score_allMSASeq_list = []
    lmdBest_metric_score_multitask_allMSASeq_df = metric_score_multitask_allMSASeq_df.loc[(metric_score_multitask_allMSASeq_df['eval_set'] == 'valid') & (metric_score_multitask_allMSASeq_df['epoch'] == 'best'),:]
    for metric_nm in ['rho', 'auroc', 'auprc']:
      valid_best_idx = lmdBest_metric_score_multitask_allMSASeq_df.groupby(['protein_name'],as_index=False)[metric_nm].transform(max) == lmdBest_metric_score_multitask_allMSASeq_df[metric_nm].to_frame()
      valid_best_df = metric_score_multitask_allMSASeq_df.iloc[valid_best_idx.loc[valid_best_idx[metric_nm]==True].index].drop_duplicates(subset=['protein_name'], keep='last')
      test_score_allMSASeq_df = valid_best_df.merge(metric_score_multitask_allMSASeq_df.loc[metric_score_multitask_allMSASeq_df['eval_set'] == 'test'], on=['protein_name','lambda','epoch'], how='inner',suffixes=('_valid','_test'))
      ## output best (lambda, epoch) selected by validation set
      print(f'***allMSASeq, pre_valid_best, {metric_nm}: best lambda, epoch***')
      print(test_score_allMSASeq_df[['protein_name','lambda','epoch']].to_string())
      if metric_nm == 'rho':
        test_score_allMSASeq_df[['protein_name','lambda','epoch']].to_csv(f'eval_results/embedding_analysis/setups/SI_FT_all_preValid.csv',index=False)
      #print(test_score_allMSASeq_df['lambda'].value_counts().to_string())
      #print(test_score_allMSASeq_df['epoch'].value_counts().to_string())

      test_score_allMSASeq_df= test_score_allMSASeq_df[['protein_name',f'{metric_nm}_test']].rename(columns={f'{metric_nm}_test': metric_nm})
      test_score_allMSASeq_df['mdl_name'] = 'Our SI-pLM (seq_all;pre_valid_best)'
      test_score_allMSASeq_list.append(test_score_allMSASeq_df)
    ## merge dfs of three metrics
    test_score_allMSASeq_preTask_df = test_score_allMSASeq_list[0].merge(test_score_allMSASeq_list[1],on=['protein_name','mdl_name'])
    test_score_allMSASeq_preTask_df = test_score_allMSASeq_preTask_df.merge(test_score_allMSASeq_list[2],on=['protein_name','mdl_name'])
   

    ###>>>  joint-model, seq w/ structure; fit-valid <<<###
    test_score_list = []
    for metric_nm in ['rho', 'auroc', 'auprc']:
      valid_best_idx = metric_score_multitask_df.loc[metric_score_multitask_df['eval_set'] == 'valid'].groupby(['protein_name'],as_index=False)[metric_nm].transform(max) == metric_score_multitask_df.loc[metric_score_multitask_df['eval_set'] == 'valid'][metric_nm].to_frame()
      valid_best_df = metric_score_multitask_df.iloc[valid_best_idx.loc[valid_best_idx[metric_nm]==True].index].drop_duplicates(subset=['protein_name'], keep='last')
      test_score_df = valid_best_df.merge(metric_score_multitask_df.loc[metric_score_multitask_df['eval_set'] == 'test'], on=['protein_name','lambda','epoch'], how='inner',suffixes=('_valid','_test'))
      ## output best (lambda, epoch) selected by validation set
      print(f'***seq w/ structure, fit_valid_best, {metric_nm}: best lambda, epoch***')
      print(test_score_df[['protein_name','lambda','epoch']].to_string())
      if metric_nm == 'rho':
        test_score_df[['protein_name','lambda','epoch']].to_csv(f'eval_results/embedding_analysis/setups/SI_FT_sub_fitValid.csv',index=False)
      #print(test_score_df['lambda'].value_counts().to_string())
      #print(test_score_df['epoch'].value_counts().to_string())

      test_score_df= test_score_df[['protein_name',f'{metric_nm}_test']].rename(columns={f'{metric_nm}_test': metric_nm})
      test_score_df['mdl_name'] = 'Our SI-pLM (seqWStruct;fit_valid_best)'
      test_score_list.append(test_score_df)
    ## merge dfs of three metrics
    test_all_score_df = test_score_list[0].merge(test_score_list[1],on=['protein_name','mdl_name'])
    test_all_score_df = test_all_score_df.merge(test_score_list[2],on=['protein_name','mdl_name'])
    
    ## joint-model, seq w/ structure, pre_task_valid
    #'prot_nm','mut_nm','lambda','epoch','valid_score'
    #* option 1: compare cross lambda and epoch using (val_aa_ppl+val_ss_ppl_val_ras_ppl+val_dm_ppl)
    '''
    pre_task_valid_best_idx = pre_task_valid_multitask_df.groupby(['mut_nm'],as_index=False)['valid_score'].transform(min) == pre_task_valid_multitask_df['valid_score'].to_frame()
    valid_best_df = pre_task_valid_multitask_df.iloc[pre_task_valid_best_idx.loc[pre_task_valid_best_idx['valid_score']==True].index].rename(columns={'mut_nm':'protein_name'}).drop_duplicates(subset=['protein_name'], keep='last')
    test_score_preTask_df = valid_best_df.merge(metric_score_multitask_df.loc[metric_score_multitask_df['eval_set'] == 'test'], on=['protein_name','lambda','epoch'])
    print('***seq w/ structure, pre_task_best: best lambda, epoch***')
    #print(test_score_preTask_df[['protein_name','lambda','epoch']].to_string())
    print(test_score_preTask_df['lambda'].value_counts().to_string())
    #print(test_score_preTask_df['epoch'].value_counts().to_string())
    test_score_preTask_df= test_score_preTask_df[['protein_name','rho','auroc','auprc']]
    test_score_preTask_df['mdl_name'] = 'Our SI-pLM (seqWStruct;pre_valid_best)'
    '''
    #* option 2: best checkpoint is used within lambda, fitness validation set is used to select cross lambda 
    test_score_list = []
    lmdBest_metric_score_multitask_df = metric_score_multitask_df.loc[(metric_score_multitask_df['eval_set'] == 'valid') & (metric_score_multitask_df['epoch'] == 'best'),:]
    for metric_nm in ['rho', 'auroc', 'auprc']:
      valid_best_idx = lmdBest_metric_score_multitask_df.groupby(['protein_name'],as_index=False)[metric_nm].transform(max) == lmdBest_metric_score_multitask_df[metric_nm].to_frame()
      valid_best_df = metric_score_multitask_df.iloc[valid_best_idx.loc[valid_best_idx[metric_nm]==True].index].drop_duplicates(subset=['protein_name'], keep='last')
      test_score_df = valid_best_df.merge(metric_score_multitask_df.loc[metric_score_multitask_df['eval_set'] == 'test'], on=['protein_name','lambda','epoch'], how='inner',suffixes=('_valid','_test'))
      ## output best (lambda, epoch) selected by validation set
      print(f'***seq w/ structure, pre_valid_best, {metric_nm}: best lambda, epoch***')
      print(test_score_df[['protein_name','lambda','epoch']].to_string())
      if metric_nm == 'rho':
        test_score_df[['protein_name','lambda','epoch']].to_csv(f'eval_results/embedding_analysis/setups/SI_FT_sub_preValid.csv',index=False)
      #print(test_score_df['lambda'].value_counts().to_string())
      #print(test_score_df['epoch'].value_counts().to_string())

      test_score_df= test_score_df[['protein_name',f'{metric_nm}_test']].rename(columns={f'{metric_nm}_test': metric_nm})
      test_score_df['mdl_name'] = 'Our SI-pLM (seqWStruct;pre_valid_best)'
      test_score_list.append(test_score_df)
    ## merge dfs of three metrics
    test_score_preTask_df = test_score_list[0].merge(test_score_list[1],on=['protein_name','mdl_name'])
    test_score_preTask_df = test_score_preTask_df.merge(test_score_list[2],on=['protein_name','mdl_name'])

    ## seq pretrained model
    target_pre_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['epoch'] == '0') & (metric_score_seq_df['seq_mode'] == 'allSeq'),:][['protein_name','epoch','rho','auroc','auprc']].groupby(['protein_name'],as_index=False)[['rho','auroc','auprc']].mean()
    target_pre_seq_df['mdl_name'] = 'Our pLM (PT)'
    
    ## allSeq finetune
    target_ft_all_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['epoch'] == 'best') & (metric_score_seq_df['seq_mode'] == 'allSeq'),:][['protein_name','epoch','rho','auroc','auprc']].groupby(['protein_name'],as_index=False)[['rho','auroc','auprc']].mean()
    target_ft_all_seq_df['mdl_name'] = 'Our pLM (FT;seq_all)'

    ## structSeq finetune
    target_ft_struct_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['epoch'] == 'best') & (metric_score_seq_df['seq_mode'] == 'structSeq'),:][['protein_name','epoch','rho','auroc','auprc']].groupby(['protein_name'],as_index=False)[['rho','auroc','auprc']].mean()
    target_ft_struct_seq_df['mdl_name'] = 'Our pLM (FT;seqWStruct)'
    
    metric_score_all_df = pd.concat([metric_score_sota_df[['protein_name','mdl_name','rho','auroc','auprc']],test_all_score_allMSASeq_df,test_score_allMSASeq_preTask_df,test_all_score_df,test_score_preTask_df,target_pre_seq_df,target_ft_all_seq_df,target_ft_struct_seq_df])
    
    ## performance grouped by MSA-depth; structure-ratio
    # MSA-depth: Neff/L
    #   low: < 10.0; medium: 10.0-60.0; high: >60.0
    # structure-ratio
    #   low: < 50.0%; medium: 50.0%-90.0%; high: >90.0%
    msa_depth_cutoffs = [0.0,10.0,60.0,1000.0]
    struct_ratio_cutoffs = [0.0,50.0,90.0,100.0]
    range_names = ['low','medium','high']
    group_msa_ave_list = []
    group_structRatio_ave_list = []
    for msa_depth_i in range(3):
      set_msa_depth = fitness_fam_info_df.loc[(fitness_fam_info_df['setNM']).isin(set_names) & (fitness_fam_info_df['Neff_div_L'].astype(float) > msa_depth_cutoffs[msa_depth_i]) & (fitness_fam_info_df['Neff_div_L'].astype(float) <= msa_depth_cutoffs[msa_depth_i+1]),'setNM'].tolist()
      ave_sota_df = metric_score_all_df.loc[metric_score_all_df['protein_name'].isin(set_msa_depth),:].groupby(['mdl_name'],as_index=False)['rho'].mean().sort_values(by=['rho'],ascending=False) 
      print(f'***MSA Depth: {range_names[msa_depth_i]}, {len(set_msa_depth)}***')
      print(ave_sota_df.to_string())
      ave_sota_df['msa_depth'] = range_names[msa_depth_i]
      group_msa_ave_list.append(ave_sota_df)
    

    for struct_ratio_i in range(3):
      set_struct_ratio = fitness_fam_info_df.loc[(fitness_fam_info_df['setNM']).isin(set_names) & (fitness_fam_info_df['struct_ratio'].astype(float) > struct_ratio_cutoffs[struct_ratio_i]) & (fitness_fam_info_df['struct_ratio'].astype(float) <= struct_ratio_cutoffs[struct_ratio_i+1]),'setNM'].tolist()
      ave_sota_df = metric_score_all_df.loc[metric_score_all_df['protein_name'].isin(set_struct_ratio),:].groupby(['mdl_name'],as_index=False)['rho'].mean().sort_values(by=['rho'],ascending=False)
      print(f'***Struct Ratio: {range_names[struct_ratio_i]}, {len(set_struct_ratio)}***')
      print(ave_sota_df.to_string())
      ave_sota_df['ratio_depth'] = range_names[struct_ratio_i]
      group_structRatio_ave_list.append(ave_sota_df)

    ## set-wise score diff
    #'Our pLM (PT)','Our pLM (FT;seqWStruct)','Our pLM (FT;seq_all)','Our SI-pLM (seqWStruct;pre_valid_best)','Our SI-pLM (seqWStruct;fit_valid_best)','Our SI-pLM (seq_all;pre_valid_best)','Our SI-pLM (seq_all;fit_valid_best)']
    pre_joint_df = pd.DataFrame()
    pre_joint_df = metric_score_all_df.loc[metric_score_all_df['mdl_name'] == 'Our pLM (PT)',['protein_name','rho']].rename(columns={'rho':'rho_1'})
    pre_joint_df = pre_joint_df.merge(metric_score_all_df.loc[metric_score_all_df['mdl_name'] == 'Our SI-pLM (seq_all;fit_valid_best)',['protein_name','rho']],on=['protein_name'])
    pre_joint_df['delta_pre_joint'] = pre_joint_df['rho'] - pre_joint_df['rho_1']
    print('***delta pLM (PT) to SI-pLM (seq_all;fit_valid_best)***')
    print(pre_joint_df[['protein_name','delta_pre_joint']].sort_values(by='delta_pre_joint',ascending=False).to_string())
    print('******')

    pre_joint_df = pd.DataFrame()
    pre_joint_df = metric_score_all_df.loc[metric_score_all_df['mdl_name'] == 'Our pLM (PT)',['protein_name','rho']].rename(columns={'rho':'rho_1'})
    pre_joint_df = pre_joint_df.merge(metric_score_all_df.loc[metric_score_all_df['mdl_name'] == 'Our pLM (FT;seq_all)',['protein_name','rho']],on=['protein_name'])
    pre_joint_df['delta_pre_joint'] = pre_joint_df['rho'] - pre_joint_df['rho_1']
    print('***delta pLM (PT) to pLM (FT;seq_all)***')
    print(pre_joint_df[['protein_name','delta_pre_joint']].sort_values(by='delta_pre_joint',ascending=False).to_string())
    print('******')

    pre_joint_df = pd.DataFrame()
    pre_joint_df = metric_score_all_df.loc[metric_score_all_df['mdl_name'] == 'Our pLM (FT;seq_all)',['protein_name','rho']].rename(columns={'rho':'rho_1'})
    pre_joint_df = pre_joint_df.merge(metric_score_all_df.loc[metric_score_all_df['mdl_name'] == 'Our SI-pLM (seq_all;fit_valid_best)',['protein_name','rho']],on=['protein_name'])
    pre_joint_df['delta_pre_joint'] = pre_joint_df['rho'] - pre_joint_df['rho_1']
    print('***delta pLM (FT;seq_all) to Our SI-pLM (seq_all;fit_valid_best)***')
    print(pre_joint_df[['protein_name','delta_pre_joint']].sort_values(by='delta_pre_joint',ascending=False).to_string())
    print('******')

    pre_joint_df = pd.DataFrame()
    pre_joint_df = metric_score_all_df.loc[metric_score_all_df['mdl_name'] == 'Our pLM (FT;seq_all)',['protein_name','auroc']].rename(columns={'auroc':'auroc_1'})
    pre_joint_df = pre_joint_df.merge(metric_score_all_df.loc[metric_score_all_df['mdl_name'] == 'Our SI-pLM (seq_all;fit_valid_best)',['protein_name','auroc']],on=['protein_name'])
    pre_joint_df['delta_pre_joint'] = pre_joint_df['auroc'] - pre_joint_df['auroc_1']
    print('***delta pLM (FT;seq_all) to Our SI-pLM (seq_all;fit_valid_best); AUROC***')
    print(pre_joint_df[['protein_name','delta_pre_joint']].sort_values(by='delta_pre_joint',ascending=False).to_string())
    print('******')

    pre_joint_df = pd.DataFrame()
    pre_joint_df = metric_score_all_df.loc[metric_score_all_df['mdl_name'].isin(['DeepSequence (published)','MSA Transformer','ESM-1v (+further training)','Wavenet','Our SI-pLM (seq_all;fit_valid_best)']),['protein_name','mdl_name','rho']].sort_values(by=['protein_name','rho'],ascending=False).groupby(by='protein_name',as_index=False).head(3)
    print('***set-wise rank for five top models***')
    print(pre_joint_df.to_string())
    print('******')


    ## average rho
    ave_sota_df = metric_score_all_df.groupby(['mdl_name'],as_index=False)['rho'].mean().sort_values(by=['rho'],ascending=False)
    print('***average rho***')
    print(ave_sota_df.to_string())
    ave_sota_df['msa_depth'] = 'all'
    group_msa_ave_list.append(ave_sota_df)
    new_ave_sota_df = ave_sota_df.rename(columns={'msa_depth':'ratio_depth'})
    group_structRatio_ave_list.append(new_ave_sota_df)
    
    ## select models better then joint modeling
    rho_joint_model= ave_sota_df.loc[ave_sota_df['mdl_name'] == 'Our SI-pLM (seq_all;fit_valid_best)','rho'].iloc[0]
    better_mdls = ave_sota_df.loc[ave_sota_df['rho'] > rho_joint_model,'mdl_name'].tolist()

    ## statistical testing
    print('***wilcoxon test***')
    for compar_mdl_nm in sota_mdl_names:
      if compar_mdl_nm in better_mdls:
        res = st.wilcoxon(metric_score_all_df.loc[metric_score_all_df['mdl_name']=='Our SI-pLM (seq_all;fit_valid_best)','rho'].to_numpy(),metric_score_all_df.loc[metric_score_all_df['mdl_name']==compar_mdl_nm,'rho'].to_numpy(),alternative='less')
        print(f'less than {compar_mdl_nm}? p-value: {res.pvalue:.6f}')
      else:
        res = st.wilcoxon(metric_score_all_df.loc[metric_score_all_df['mdl_name']=='Our SI-pLM (seq_all;fit_valid_best)','rho'].to_numpy(),metric_score_all_df.loc[metric_score_all_df['mdl_name']==compar_mdl_nm,'rho'].to_numpy(),alternative='greater')
        print(f'greater than {compar_mdl_nm}? p-value: {res.pvalue:.6f}')

    ## average AUC
    for metric_nm in ['auroc','auprc']:
      ave_sota_df = metric_score_all_df.groupby(['mdl_name'],as_index=False)[metric_nm].mean().sort_values(by=[metric_nm],ascending=False)
      print(f'***average {metric_nm}***')
      print(ave_sota_df.to_string())
      ave_sota_df['msa_depth'] = 'all'
      group_msa_ave_list.append(ave_sota_df)
      new_ave_sota_df = ave_sota_df.rename(columns={'msa_depth':'ratio_depth'})
      group_structRatio_ave_list.append(new_ave_sota_df)

    ## print in latex table friendly way
    group_msa_ave_df = pd.concat(group_msa_ave_list,axis=0)
    print('>>MSA depth latex table format')
    for mdl_nm in group_msa_ave_df['mdl_name'].unique().tolist():
      mdl_group_msa_ave_df = group_msa_ave_df.loc[(group_msa_ave_df['mdl_name'] == mdl_nm),:].copy()
      mdl_group_msa_ave_df['msa_depth'] = pd.Categorical(mdl_group_msa_ave_df['msa_depth'], categories=["low", "medium", "high", "all"])
      score_string = ''
      for s in mdl_group_msa_ave_df.loc[~mdl_group_msa_ave_df['rho'].isnull(),:].sort_values('msa_depth')['rho'].tolist():
        score_string += f'&{s:.3f}'
      auroc = mdl_group_msa_ave_df.loc[(mdl_group_msa_ave_df['msa_depth']=='all') & (~mdl_group_msa_ave_df['auroc'].isnull()),'auroc'].iloc[0]
      auprc = mdl_group_msa_ave_df.loc[(mdl_group_msa_ave_df['msa_depth']=='all') & (~mdl_group_msa_ave_df['auprc'].isnull()),'auprc'].iloc[0]
      score_string += f"&{auroc:.3f}&{auprc:.3f}"
      #print(f"{mdl_nm}:{score_string}")
    group_structRatio_ave_df = pd.concat(group_structRatio_ave_list,axis=0)
    print('>>Structure ratio latex table format')
    for mdl_nm in group_structRatio_ave_df['mdl_name'].unique().tolist():
      mdl_group_structRatio_ave_df = group_structRatio_ave_df.loc[(group_structRatio_ave_df['mdl_name'] == mdl_nm),:].copy()
      mdl_group_structRatio_ave_df['ratio_depth'] = pd.Categorical(mdl_group_structRatio_ave_df['ratio_depth'], categories=["low", "medium", "high","all"])
      score_string = ''
      for s in mdl_group_structRatio_ave_df.loc[~mdl_group_structRatio_ave_df['rho'].isnull(),:].sort_values('ratio_depth')['rho'].tolist():
        score_string += f'&{s:.3f}'
      auroc = mdl_group_structRatio_ave_df.loc[(mdl_group_structRatio_ave_df['ratio_depth']=='all') & (~mdl_group_structRatio_ave_df['auroc'].isnull()),'auroc'].iloc[0]
      auprc = mdl_group_structRatio_ave_df.loc[(mdl_group_structRatio_ave_df['ratio_depth']=='all') & (~mdl_group_structRatio_ave_df['auprc'].isnull()),'auprc'].iloc[0]
      score_string += f"&{auroc:.3f}&{auprc:.3f}"
      #print(f"{mdl_nm}:{score_string}")
    
    ordered_set_names = metric_score_all_df.loc[metric_score_all_df['mdl_name']=='Our SI-pLM (seq_all;fit_valid_best)',:].sort_values('rho',ascending=False)['protein_name'].tolist()
    my_mdl_names = ['Our pLM (PT)','Our pLM (FT;seqWStruct)','Our pLM (FT;seq_all)','Our SI-pLM (seqWStruct;pre_valid_best)','Our SI-pLM (seqWStruct;fit_valid_best)','Our SI-pLM (seq_all;pre_valid_best)','Our SI-pLM (seq_all;fit_valid_best)']
    my_mdl_top_names = ['Our SI-pLM (seq_all;fit_valid_best)']
    fam_sota_mdl_names = ['PSSM','EVMutation (published)','Wavenet','DeepSequence (published)']
    fam_sota_sub_mdl_names = ['Wavenet','DeepSequence (published)']
    align_sota_sub_mdl_names = ['DeepSequence (published)','MSA Transformer']
    
    plm_sota_mdl_names = ['TAPE','UniRep','ProtBERT-BFD','ESM-1b','ESM-1v (zero shot)','ESM-1v (+further training)','MSA Transformer']
    plm_sota_sub_mdl_names = ['ESM-1v (zero shot)','ESM-1v (+further training)','MSA Transformer']
    nonAlign_sota_sub_mdl_names = ['ESM-1v (+further training)','Wavenet']
    metric_score_all_df['mdl_type'] = 'our'
    metric_score_all_df.loc[metric_score_all_df['mdl_name'].isin(align_sota_sub_mdl_names),'mdl_type'] = 'align'
    metric_score_all_df.loc[metric_score_all_df['mdl_name'].isin(nonAlign_sota_sub_mdl_names),'mdl_type'] = 'nonAlign'
    
    ## order sets by rho
    # mdl_order = my_mdl_names + sota_mdl_names
    # fig, ax = plt.subplots(figsize=(14,8))
    # #sns.set_style("whitegrid")
    # gax = sns.pointplot(data=metric_score_all_df, x="protein_name", y="rho", hue="mdl_name", order=ordered_set_names,join=False, errorbar=None, dodge=False, palette='tab20',hue_order=mdl_order)
    # gax.set_xticklabels(gax.get_xticklabels(), rotation=45, ha='right')
    # ax.xaxis.grid(True) # Show the vertical gridlines
    # plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0, fontsize=8)
    # plt.savefig('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/sota_protein_track_best_valid.png',dpi=600,bbox_inches='tight')
    # plt.clf()
    
    ## order sets by rho, model grouping by type
    #style_order=['fam','pLM','our'],markers={'fam':'.','pLM':'d','our':'*'}
    # fig, ax = plt.subplots(figsize=(14,8))
    # #sns.set_style("whitegrid")
    # sns.pointplot(data=metric_score_all_df.loc[metric_score_all_df['mdl_type']=='align',:], x="protein_name", y="rho", hue="mdl_name", order=ordered_set_names, palette='tab10', hue_order=fam_sota_mdl_names, errorbar=None, dodge=False, join=False, markers='+', scale=0.7)
    # sns.pointplot(data=metric_score_all_df.loc[metric_score_all_df['mdl_type']=='pLM',:], x="protein_name", y="rho", hue="mdl_name", order=ordered_set_names, palette='tab10', hue_order=plm_sota_mdl_names, errorbar=None, dodge=False, join=False, markers='x', scale=0.7)
    # gax = sns.pointplot(data=metric_score_all_df.loc[metric_score_all_df['mdl_type']=='our',:], x="protein_name", y="rho", hue="mdl_name", order=ordered_set_names, palette='tab10', hue_order=my_mdl_names, errorbar=None, dodge=False, join=False, markers='*', scale=0.7)
    # gax.set_xticklabels(gax.get_xticklabels(), rotation=45, ha='right')
    # ax.xaxis.grid(True) # Show the vertical gridlines
    # plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0, fontsize=8)
    # #plt.setp(gax.lines, zorder=100)
    # plt.setp(gax.collections, zorder=100, label="")
    # plt.savefig('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/sota_protein_track_best_valid_group.png',dpi=600,bbox_inches='tight')
    # plt.clf()

    ## order sets by rho, model grouping by type, my best modal
    metric_score_all_df.loc[metric_score_all_df['mdl_name'].isin(my_mdl_top_names),'mdl_type'] = 'our_sub'
    metric_score_all_df.loc[metric_score_all_df['mdl_name'].isin(align_sota_sub_mdl_names),'mdl_type'] = 'align_sub'
    metric_score_all_df.loc[metric_score_all_df['mdl_name'].isin(nonAlign_sota_sub_mdl_names),'mdl_type'] = 'nonAlign_sub'
    params = {'legend.fontsize': 10,
              'figure.figsize': (16, 5),
              'axes.labelsize': 14,
              'axes.titlesize': 14,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              }
    pylab.rcParams.update(params)
    fig, ax = plt.subplots()
    #sns.set_style("whitegrid")
    sns.pointplot(data=metric_score_all_df.loc[metric_score_all_df['mdl_type']=='align_sub',:], x="protein_name", y="rho", hue="mdl_name", order=ordered_set_names, palette={'DeepSequence (published)':'green','MSA Transformer':'greenyellow'}, hue_order=align_sota_sub_mdl_names, errorbar=None, dodge=False, join=False, markers='+', scale=0.7)
    sns.pointplot(data=metric_score_all_df.loc[metric_score_all_df['mdl_type']=='nonAlign_sub',:], x="protein_name", y="rho", hue="mdl_name", order=ordered_set_names, palette={'ESM-1v (+further training)':'darkorange','Wavenet':'gold'}, hue_order=nonAlign_sota_sub_mdl_names, errorbar=None, dodge=False, join=False, markers='x', scale=0.7)
    gax = sns.pointplot(data=metric_score_all_df.loc[metric_score_all_df['mdl_type']=='our_sub',:], x="protein_name", y="rho", hue="mdl_name", order=ordered_set_names, palette={'Our SI-pLM (seq_all;fit_valid_best)':'red'}, hue_order=my_mdl_top_names, errorbar=None, dodge=False, join=False, markers='*', scale=0.7) #,'Our SI-pLM (seq_all;pre_valid_best)':'lightcoral'
    gax.set_xticklabels(gax.get_xticklabels(), rotation=45, ha='right')
    ax.xaxis.grid(True) # Show the vertical gridlines
    lg = plt.legend(bbox_to_anchor=(1.00, 1.00), loc='upper right', borderaxespad=0)
    lg.get_texts()[0].set_text("DeepSequence")
    lg.get_texts()[1].set_text("MSA Transformer$^\dagger$")
    lg.get_texts()[2].set_text("ESM-1v$^\star$")
    lg.get_texts()[3].set_text("Wavenet")
    lg.get_texts()[4].set_text("Our SI-pLM$^\diamond$ ($S$+$\hat{T}$,$LS$)")
    plt.xlabel('Fitness set names')
    plt.ylabel("Spearman's rank correlation")
    #plt.ylim([0.2,0.82])
    #plt.setp(gax.lines, zorder=100)
    plt.setp(gax.collections, zorder=100, label="")
    #plt.savefig('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/sota_protein_track_best_valid_group_subMdl.png',dpi=600,bbox_inches='tight')
    plt.clf()

    ## order sets by rho; sota model subset
    # sub_sota_mdl_names = my_mdl_names + ['PSSM','EVMutation (published)','DeepSequence (published)','Wavenet','ESM-1v (+further training)','MSA Transformer','ESM-1b']
    # metric_score_all_subMdl_df = metric_score_all_df.loc[metric_score_all_df['mdl_name'].isin(sub_sota_mdl_names)]
    # fig, ax = plt.subplots(figsize=(14,8))
    # #sns.set_style("whitegrid")
    # gax = sns.pointplot(data=metric_score_all_subMdl_df, x="protein_name", y="rho", hue="mdl_name", order=ordered_set_names,join=False, errorbar=None, dodge=False, palette='tab20',hue_order=sub_sota_mdl_names)
    # gax.set_xticklabels(gax.get_xticklabels(), rotation=45, ha='right')
    # ax.xaxis.grid(True) # Show the vertical gridlines
    # plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0, fontsize=8)
    # plt.savefig('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/sota_protein_track_best_valid_subSOTA.png',dpi=600,bbox_inches='tight')
    # plt.clf()
  if 'sub_sota_protein_track' in fig_name_list:
    sub_sota_mdl_names = ['MSA Transformer','PSSM','EVMutation (published)','DeepSequence (published)','ESM-1v (zero shot)','ESM-1v (+further training)','ProtBERT-BFD','ESM-1b']
    epoch_filter_df = metric_score_multitask_df[['protein_name','lambda','epoch','rho']].loc[metric_score_multitask_df['epoch'] > 30]
    #'protein_name','lambda','epoch','rho','rho_P'
    target_df = epoch_filter_df.groupby(['protein_name','lambda'],as_index=False)['rho'].mean().groupby(['protein_name'],as_index=False)['rho'].max()
    target_df['mdl_name'] = 'MyMdl (joint finetune)'
    target_pre_seq_df = metric_score_seq_df[['protein_name','epoch','rho']].loc[metric_score_seq_df['epoch'] == 0,:]
    target_pre_seq_df['mdl_name'] = 'MyMdl (pretrain)'
    target_ft_seq_df = metric_score_seq_df[['protein_name','epoch','rho']].loc[metric_score_seq_df['epoch'] > 30,:].groupby(['protein_name'],as_index=False)['rho'].mean()
    target_ft_seq_df['mdl_name'] = 'MyMdl (seq finetune)'
    metric_score_sota_df_sub = metric_score_sota_df[['protein_name','rho','mdl_name']].loc[metric_score_sota_df['mdl_name'].isin(sub_sota_mdl_names)]
    metric_score_all_df = pd.concat([metric_score_sota_df_sub,target_df,target_pre_seq_df,target_ft_seq_df])
    fig, ax = plt.subplots(figsize=(14,8))
    #sns.set_style("whitegrid")
    gax = sns.pointplot(data=metric_score_all_df, x="protein_name", y="rho", hue="mdl_name", join=False, errorbar=None, dodge=False, palette='tab20')
    gax.set_xticklabels(gax.get_xticklabels(), rotation=45, ha='right')
    ax.xaxis.grid(True) # Show the vertical gridlines
    plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0, fontsize=8)
    plt.savefig('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/sub_sota_protein_track.png',dpi=600,bbox_inches='tight')
    plt.clf()

  return

def mt_protein_lambda_track(load_processed_df: bool):
  """function named by figure name
  """
  path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  ## load sota predictions
  #header: protein_name,mutant,gt,MSA Transformer,PSSM,EVMutation (published),DeepSequence (published) - single,DeepSequence (published),ESM-1v (zero shot),ESM-1v (+further training),DeepSequence (replicated),EVMutation (replicated),ProtBERT-BFD,TAPE,UniRep,ESM-1b
  sota_pred_df = pd.read_csv(f'{path}/data_process/mutagenesis/esm_predictions/raw_with_wavenet_df.csv',delimiter=',',header=0)
  fitness_fam_info_df = pd.read_csv(f'{path}/data_process/mutagenesis/DeepSequenceMutaSet_reference_file.csv',delimiter=',',header=0)
  #'AMIE_PSEAE_Whitehead','B3VI55_LIPSTSTABLE','B3VI55_LIPST_Whitehead2015','BG_STRSQ_hmmerbit','BLAT_ECOLX_Ostermeier2014','BLAT_ECOLX_Palzkill2012','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Tenaillon2013','BRCA1_HUMAN_BRCT','BRCA1_HUMAN_RING','CALM1_HUMAN_Roth2017','DLG4_RAT_Ranganathan2012','GAL4_YEAST_Shendure2015','HIS7_YEAST_Kondrashov2017','HSP82_YEAST_Bolon2016','IF1_ECOLI_Kishony','KKA2_KLEPN_Mikkelsen2014','MK01_HUMAN_Johannessen','MTH3_HAEAESTABILIZED_Tawfik2015','P84126_THETH_b0','PABP_YEAST_Fields2013-doubles','PABP_YEAST_Fields2013-singles','PTEN_HUMAN_Fowler2018','RASH_HUMAN_Kuriyan','RL401_YEAST_Bolon2013','RL401_YEAST_Bolon2014','RL401_YEAST_Fraser2016','SUMO1_HUMAN_Roth2017','TIM_SULSO_b0','TIM_THEMA_b0','TPK1_HUMAN_Roth2017','TPMT_HUMAN_Fowler2018','UBC9_HUMAN_Roth2017','UBE4B_MOUSE_Klevit2013-singles','YAP1_HUMAN_Fields2012-singles'
  #'BF520_env_Bloom2018','BG505_env_Bloom2018','HG_FLU_Bloom2016','PA_FLU_Sun2015','POLG_HCVJF_Sun2014','POL_HV1N5-CA_Ndungu2014'
  #parEparD_Laub2015_all
  set_names = ['AMIE_PSEAE_Whitehead','RASH_HUMAN_Kuriyan','KKA2_KLEPN_Mikkelsen2014','PTEN_HUMAN_Fowler2018','HIS7_YEAST_Kondrashov2017','MTH3_HAEAESTABILIZED_Tawfik2015','BRCA1_HUMAN_BRCT','BRCA1_HUMAN_RING','DLG4_RAT_Ranganathan2012','PABP_YEAST_Fields2013-singles','PABP_YEAST_Fields2013-doubles','YAP1_HUMAN_Fields2012-singles']
  sign_reverse_set_names = ['BLAT_ECOLX_Palzkill2012','MK01_HUMAN_Johannessen']
  epoch_joint_ft_list = [0,3,7,11,15,19,23,27,31,35,39] #0,5,11,17,23,29,35,41,47,53,59
  epoch_seq_ft_list = ['best',0,3,7,11,15,19,23,27,31,35,39]
  lambda_list = [0.0,0.01,0.1,0.33,0.5,1.0,2.0,5.0,10.0,20.0] #0.0,0.01,0.1,0.33,0.5,1.0,2.0,5.0,10.0,20.0
  if load_processed_df:
    metric_score_multitask_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_multitaskFT_sub12sets.csv',sep=',',header=0)
    metric_score_seq_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_seqFT_sub12sets.csv',sep=',',header=0)
  else:
    ## seq+structure finetuned models
    metric_score_list = []
    for set_nm in set_names:
      prot_nm = fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,:].iloc[0]['Shin2021_set']
      bina_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM']==set_nm,:].iloc[0]['DMS_binarization_cutoff']
      sign_reverse = set_nm in sign_reverse_set_names
      ## load validation mutations
      valid_mut_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/valid_mutation.csv',header=0)
      fit_gt_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/mut_fit_gt.csv',header=0)
      valid_fit_gt_df = fit_gt_df.loc[fit_gt_df['mut_name'].isin(valid_mut_df['mut_name']) & (fit_gt_df['fitness'].notnull())]
      for lmd in lambda_list:
        for epoch in epoch_joint_ft_list:
          print(f'seq+structFT,{set_nm},{lmd},{epoch}')
          ## get model name
          model_path = os.popen(f"grep -E -o 'seq_structure_multi_task.*[0-9]{{6}}' {path}/job_logs/archive_multitask_models_eval/*ss{lmd}_*epoch_{epoch}.*{set_nm}*multitask_fitness_UNsupervise_mutagenesis.lm_multitask_symm_fast.allData.o* | uniq").read().strip('\n')
          ## load my predicted fitness scores
          my_pred_df = pd.read_csv(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/predictions/{model_path}/{set_nm}_{epoch}_wtstruct_rawScores.csv',names=['mutant','myMdl']).drop_duplicates().groupby(['mutant'],as_index=False)['myMdl'].mean()
          my_pred_df['epoch'] = epoch
          my_pred_df['lmd'] = lmd
          my_pred_df['protein_name'] = set_nm
          ## metrics calculation on validation mutations
          valid_merge_df = valid_fit_gt_df.merge(my_pred_df, left_on=['mut_name'], right_on=['mutant'], how='inner')
          rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(valid_merge_df['fitness'].to_numpy().reshape(-1),valid_merge_df['myMdl'].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
          one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,'valid']
          metric_score_list.append(one_score_case)
          ## test mutations, join on common variants
          set_sota_pred_df = sota_pred_df.loc[(sota_pred_df['protein_name'] == set_nm) & (sota_pred_df['gt'].notnull()) & (~sota_pred_df['mutant'].isin(valid_mut_df['mut_name']))]
          sota_myMdl_joint_df = set_sota_pred_df.merge(my_pred_df, on=['mutant','protein_name'], how='inner')
          assert len(set_sota_pred_df)-len(sota_myMdl_joint_df) == 0
          rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_joint_df['gt'].to_numpy().reshape(-1),sota_myMdl_joint_df['myMdl'].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
          one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,'test']
          metric_score_list.append(one_score_case)
    metric_score_multitask_df = pd.DataFrame(metric_score_list,columns=['protein_name','lambda','epoch','rho','rho_P','auroc','auprc','eval_set'])
    metric_score_multitask_df.to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_multitaskFT_sub12sets.csv',index=False,header=True)
    
    ## seq finetuned models (rho is calcualted over test mutation set)
    metric_score_list = []
    for set_nm in set_names:
      valid_mut_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/valid_mutation.csv',header=0)
      bina_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM']==set_nm,:].iloc[0]['DMS_binarization_cutoff']
      sign_reverse = set_nm in sign_reverse_set_names
      for epoch in epoch_seq_ft_list:
        for seq_mode in ['structSeq', 'allSeq']:
          print(f'seqFT,{set_nm},{epoch},{seq_mode}')
          ## get model name
          model_path_list = os.popen(f"grep -E -o 'masked_language_modeling_transformer.*[0-9]{{6}}' {path}/job_logs/archive_baseline_bert_eval/baseline_bert_mutation_fitness_UNsupervise_mutagenesis_rp15_all_2_729.seq_finetune.reweighted.{seq_mode}.TotalEpoch40.Interval*.{epoch}.*.{set_nm}.seqMaxL512.classWno.out | uniq").read().strip('\n').split('\n')
          assert len(model_path_list) == 1
          model_path = model_path_list[0]
          ## load my preds
          my_pred_df = pd.read_csv(f'{path}/eval_results/mutation_fitness_UNsupervise_mutagenesis/predictions/{model_path}/{set_nm}_{epoch}_rawScores.csv',names=['mutant','myMdl']).drop_duplicates().groupby(['mutant'],as_index=False)['myMdl'].mean()
          my_pred_df['epoch'] = epoch
          my_pred_df['protein_name'] = set_nm
          my_pred_df['seq_mode'] = seq_mode
          ## join on common variants
          set_sota_pred_df = sota_pred_df.loc[(sota_pred_df['protein_name'] == set_nm) & (sota_pred_df['gt'].notnull()) & (~sota_pred_df['mutant'].isin(valid_mut_df['mut_name']))]
          sota_myMdl_joint_df = set_sota_pred_df.merge(my_pred_df, on=['mutant','protein_name'], how='inner')
          assert len(set_sota_pred_df)-len(sota_myMdl_joint_df) == 0
          rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_joint_df['gt'].to_numpy().reshape(-1),sota_myMdl_joint_df['myMdl'].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
          one_score_case = [set_nm,epoch,seq_mode,rho_score,rho_p_value,auroc,auprc]
          metric_score_list.append(one_score_case)
    metric_score_seq_df = pd.DataFrame(metric_score_list,columns=['protein_name','epoch','seq_mode','rho','rho_P','auroc','auprc'])
    metric_score_seq_df.to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_seqFT_sub12sets.csv',index=False,header=True)


  target_multitask_df = metric_score_multitask_df.loc[(metric_score_multitask_df['eval_set'] == 'test') & (metric_score_multitask_df['epoch'] > 30),:] #
  metric_score_seq_df['lambda'] = -1
  metric_score_seq_df.loc[metric_score_seq_df['epoch'] == '0','lambda'] = -2
  ## seq pretrained(epoch:0); finetuned model(epoch >30)
  target_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['seq_mode']=='structSeq') & (metric_score_seq_df['epoch'].str.isnumeric()), :]
  target_seq_df = target_seq_df.loc[(target_seq_df['epoch'].astype(int) > 30) | (target_seq_df['epoch'].astype(int) == 0),:]
  target_df = pd.concat([target_multitask_df,target_seq_df])
  ## order protein sets
  pre_rho_df = target_seq_df.loc[target_seq_df['lambda']==-2,['protein_name','epoch','rho']].groupby(['protein_name'],as_index=False)['rho'].mean()
  fine_rho_df = target_seq_df.loc[target_seq_df['lambda']==-1,['protein_name','epoch','rho']].groupby(['protein_name'],as_index=False)['rho'].mean()
  pre_fine_rho_df = pre_rho_df.merge(fine_rho_df,on=['protein_name'],suffixes=['_pre','_fine'])
  pre_fine_rho_df['rho_up'] = pre_fine_rho_df['rho_fine'] - pre_fine_rho_df['rho_pre']
  protein_order = pre_fine_rho_df.sort_values(by='rho_up',ascending=False)['protein_name'].to_list()
  print(protein_order)

  params = {'legend.fontsize': 8,
            'figure.figsize': (10, 4),
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            }
  pylab.rcParams.update(params)
  fig, ax = plt.subplots() #
  sns.set_style("whitegrid")
  
  gax = sns.pointplot(data=target_df.loc[target_df['lambda']>=0,:], x="protein_name", y="rho", order=protein_order, hue="lambda", join=False, errorbar='sd', dodge=True, palette=sns.color_palette("Paired",10), markers='o', scale=0.9) #('pi',100)
  gax = sns.pointplot(data=target_df.loc[target_df['lambda']<0,:], x="protein_name", y="rho", order=protein_order, hue="lambda", join=False, errorbar='sd', dodge=True, palette={-2:'darkgray',-1:'black'}, markers='s', ax=gax, scale=0.7) #('pi',100)
  gax.set_xticklabels(gax.get_xticklabels(), rotation=45, ha='right')
  ax.xaxis.grid(True) # Show the vertical gridlines
  plt.xlabel('Fitness set names')
  plt.ylabel("Spearman's rank correlation")
  plt.ylim([0.,0.85])
  #plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0, fontsize=12)
  lg = plt.legend(bbox_to_anchor=(0.5, 0.99), loc='upper center', borderaxespad=0, ncol=6)
  lg.get_texts()[-1].set_text('fine-tuned')
  lg.get_texts()[-2].set_text('pre-trained')
  plt.setp(gax.collections, zorder=100, label="")
  plt.savefig('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/mt_protein_lambda_track_errbarSD.png',dpi=600, bbox_inches='tight') #
  plt.clf()

def joint_model_select_best_CK(
      prot_nm: str,
      prot_mut_nm: str,
      lmd: float,
      mdl_path_list: List[str],
      ck_freq: int = 6,):
  """Select best checkpoint of joint-model based on aa_ppl on seq+structure validation set
  """
  scalar_value_list = []
  for mdl_nm in mdl_path_list:
    ea = event_accumulator.EventAccumulator(f'/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/logs_to_keep/rp15_all/rp15_all_2_multiTask_models/family_specific_AFDBv3/{prot_nm}/{mdl_nm}')
    ea.Reload()
    for scalar_nm in ['val/aa_ppl','val/ss_ppl','val/rsa_ppl']: #,'val/ss_ppl','val/rsa_ppl','val/dist_ppl'
      for sca_eve in ea.Scalars(scalar_nm):
        if (int(sca_eve.step)+1) % ck_freq == 0:
          scalar_value_list.append([prot_nm,prot_mut_nm,lmd,sca_eve.step,scalar_nm,sca_eve.value])
  scalar_value_df = pd.DataFrame(scalar_value_list,columns=['prot_nm','mut_nm','lambda','epoch','scalar_nm','valid_score']).groupby(['prot_nm','mut_nm','lambda','epoch'],as_index=False)['valid_score'].sum()

  return scalar_value_df

def mutation_depth_group(load_processed_df: bool, fig_name_list: List):
  """performance comparison grouped by mutation depth
  """
  path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  #header: protein_name,mutant,gt,MSA Transformer,PSSM,EVMutation (published),DeepSequence (published) - single,DeepSequence (published),ESM-1v (zero shot),ESM-1v (+further training),DeepSequence (replicated),EVMutation (replicated),ProtBERT-BFD,TAPE,UniRep,ESM-1b
  sota_pred_df = pd.read_csv(f'{path}/data_process/mutagenesis/esm_predictions/raw_with_wavenet_df.csv',delimiter=',',header=0)
  sota_mdl_names = ['PSSM','EVMutation (published)','DeepSequence (published)','Wavenet','MSA Transformer','ESM-1v (zero shot)','ESM-1v (+further training)','ESM-1b','ProtBERT-BFD','TAPE','UniRep']
  fitness_fam_info_df = pd.read_csv(f'{path}/data_process/mutagenesis/DeepSequenceMutaSet_reference_file.csv',delimiter=',',header=0)
  #'prot_nm','mut_nm','lmd','mdl_path','mdl_path_1'
  model_path_multitaskFT_allMSASeq_df = pd.read_csv(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/model_path_multitaskFT_allMSASeq.csv',delimiter=',',header=0)
  model_path_multitaskFT_df = pd.read_csv(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/model_path_multitaskFT.csv',delimiter=',',header=0)
  set_names = ['HIS7_YEAST_Kondrashov2017']
  mutDepth_dict = {'HIS7_YEAST_Kondrashov2017': [1,2,3,4,5,[6,10],[11,28]],
                   'PABP_YEAST_Fields2013-doubles': [2],
                   'PABP_YEAST_Fields2013-singles': [1],
                   'parEparD_Laub2015_all': [1,2,3,4]}
  sign_reverse_set_names = ['BLAT_ECOLX_Palzkill2012','MK01_HUMAN_Johannessen']
  struct_score_names = ['log_ratio_aa','log_ratio_ss','log_ratio_rsa','log_ratio_ss_env','log_ratio_rsa_env','log_ratio_dm']
  epoch_joint_ft_list = ['best',0,5,11,17,23,29,35,41,47,53,59]
  epoch_seq_ft_list = ['best',0,3,7,11,15,19,23,27,31,35,39]
  lambda_list = [0.0,0.5,2.0,20.0] #0.0,0.01,0.1,0.33,0.5,1.0,2.0,5.0,10.0,20.0
  if load_processed_df:
    metric_structure_score_multitask_allMSASeq_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_structure_scores_multitaskFT_allMSASeq_depth.csv',sep=',',header=0)
    metric_structure_score_multitask_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_structure_scores_multitaskFT_depth.csv',sep=',',header=0)
    metric_score_seq_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_seqFT_depth.csv',sep=',',header=0)
    metric_score_sota_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_sota_depth.csv',sep=',',header=0)
  else:
    ## seq+struct finetune, allMSASeq; structure-based mutation scoring
    metric_score_list = []
    #raw_score_list = []
    for set_nm in set_names:
      ## load validation mutations
      valid_mut_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/valid_mutation.csv',header=0)
      fit_gt_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/mut_fit_gt.csv',header=0)
      valid_fit_gt_df = fit_gt_df.loc[fit_gt_df['mut_name'].isin(valid_mut_df['mut_name']) & (fit_gt_df['fitness'].notnull())]
      prot_nm = fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,:].iloc[0]['Shin2021_set']
      bina_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM']==set_nm,:].iloc[0]['DMS_binarization_cutoff']
      sign_reverse = set_nm in sign_reverse_set_names
      for lmd in lambda_list:
        ## get model name
        model_path = model_path_multitaskFT_allMSASeq_df.loc[(model_path_multitaskFT_allMSASeq_df['mut_nm']==set_nm) & (model_path_multitaskFT_allMSASeq_df['lmd']==lmd),'mdl_path'].iloc[0]
        for epoch in epoch_joint_ft_list:
          print(f'joint FT, allMSASeq, mut-depth: {set_nm},{lmd},{epoch}')
          ## load my predicted fitness scores
          #var_name,fit_true,aa_fit,log_ratio_ss,log_ratio_rsa,log_ratio_ss_env,log_ratio_rsa_env,log_ratio_dm
          # my_pred_df = pd.read_csv(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis_structure/predictions/{model_path}/{set_nm}_{epoch}_structProp_rawScores.csv',header=0,delimiter=',').rename(columns={'var_name': 'mutant','aa_fit': 'log_ratio_aa'}).drop_duplicates().groupby(['mutant'],as_index=False)[['fit_true','log_ratio_aa','log_ratio_ss','log_ratio_rsa','log_ratio_ss_env','log_ratio_rsa_env','log_ratio_dm']].mean()
          my_pred_df = pd.read_csv(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/predictions/{model_path}/{set_nm}_{epoch}_wtstruct_rawScores.csv',names=['mutant','myMdl']).drop_duplicates().groupby(['mutant'],as_index=False)['myMdl'].mean().rename(columns={'myMdl':'log_ratio_aa'})
          my_pred_df['epoch'] = epoch
          my_pred_df['lmd'] = lmd
          my_pred_df['protein_name'] = set_nm
          my_pred_df['mut_depth'] = my_pred_df['mutant'].str.count(':')+1
          #raw_score_list.append(my_pred_df)
    
          ## metrics calculation on validation mutations
          valid_merge_df = valid_fit_gt_df.merge(my_pred_df, left_on=['mut_name'], right_on=['mutant'], how='inner')
          for score_nm in ['log_ratio_aa']:
            rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(valid_merge_df['fitness'].to_numpy().reshape(-1),valid_merge_df[score_nm].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
            one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,score_nm,'valid']
            metric_score_list.append(one_score_case)
            
          ## test mutations, join on common variants
          set_sota_pred_df = sota_pred_df.loc[(sota_pred_df['protein_name'] == set_nm) & (sota_pred_df['gt'].notnull()) & (~sota_pred_df['mutant'].isin(valid_mut_df['mut_name']))]
          sota_myMdl_joint_df = set_sota_pred_df.merge(my_pred_df, on=['mutant','protein_name'], how='inner')
          assert len(set_sota_pred_df)-len(sota_myMdl_joint_df) == 0
          for score_nm in ['log_ratio_aa']:
            rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_joint_df['gt'].to_numpy().reshape(-1),sota_myMdl_joint_df[score_nm].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
            one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,score_nm,'test']
            metric_score_list.append(one_score_case)
            ## loop over mut depth ranges
            for depth_range in mutDepth_dict[set_nm]:
              if isinstance(depth_range,int):
                sota_myMdl_depth_joint_df = set_sota_pred_df.merge(my_pred_df.loc[my_pred_df['mut_depth'] == depth_range,:], on=['mutant','protein_name'], how='inner')
                rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_depth_joint_df['gt'].to_numpy().reshape(-1),sota_myMdl_depth_joint_df[score_nm].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
                one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,score_nm,f'test_{depth_range}']
                metric_score_list.append(one_score_case)
              elif isinstance(depth_range,List):
                sota_myMdl_depth_joint_df = set_sota_pred_df.merge(my_pred_df.loc[(my_pred_df['mut_depth'] >= depth_range[0]) & (my_pred_df['mut_depth'] <= depth_range[1]),:], on=['mutant','protein_name'], how='inner')
                rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_depth_joint_df['gt'].to_numpy().reshape(-1),sota_myMdl_depth_joint_df[score_nm].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
                one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,score_nm,f'test_{depth_range[0]}_{depth_range[1]}']
                metric_score_list.append(one_score_case)
              else:
                pass
    metric_structure_score_multitask_allMSASeq_df = pd.DataFrame(metric_score_list,columns=['protein_name','lambda','epoch','rho','rho_P','auroc','auprc','score_nm','eval_set'])
    metric_structure_score_multitask_allMSASeq_df.to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_structure_scores_multitaskFT_allMSASeq_depth.csv',index=False,header=True)

    ## seq+structure finetuned models (only sequence scoring)
    metric_score_list = []
    for set_nm in set_names:
      ## load validation mutations
      valid_mut_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/valid_mutation.csv',header=0)
      fit_gt_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/mut_fit_gt.csv',header=0)
      valid_fit_gt_df = fit_gt_df.loc[fit_gt_df['mut_name'].isin(valid_mut_df['mut_name']) & (fit_gt_df['fitness'].notnull())]
      prot_nm = fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,:].iloc[0]['Shin2021_set']
      bina_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM']==set_nm,:].iloc[0]['DMS_binarization_cutoff']
      sign_reverse = set_nm in sign_reverse_set_names
      for lmd in lambda_list:
        ## get model name
        model_path = model_path_multitaskFT_df.loc[(model_path_multitaskFT_df['mut_nm']==set_nm) & (model_path_multitaskFT_df['lmd']==lmd),'mdl_path'].iloc[0]
        for epoch in epoch_joint_ft_list:
          print(f'joint FT, seq w/ structure, mut-depth: {set_nm},{lmd},{epoch}')
          ## load my predicted fitness scores
          #var_name,fit_true,aa_fit,log_ratio_ss,log_ratio_rsa,log_ratio_ss_env,log_ratio_rsa_env,log_ratio_dm
          my_pred_df = pd.read_csv(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/predictions/{model_path}/{set_nm}_{epoch}_wtstruct_rawScores.csv',names=['mutant','myMdl']).drop_duplicates().groupby(['mutant'],as_index=False)['myMdl'].mean().rename(columns={'myMdl':'log_ratio_aa'})
          my_pred_df['epoch'] = epoch
          my_pred_df['lmd'] = lmd
          my_pred_df['protein_name'] = set_nm
          my_pred_df['mut_depth'] = my_pred_df['mutant'].str.count(':')+1
    
          ## metrics calculation on validation mutations
          valid_merge_df = valid_fit_gt_df.merge(my_pred_df, left_on=['mut_name'], right_on=['mutant'], how='inner')
          for score_nm in ['log_ratio_aa']:
            rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(valid_merge_df['fitness'].to_numpy().reshape(-1),valid_merge_df[score_nm].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
            one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,score_nm,'valid']
            metric_score_list.append(one_score_case)
            
          ## test mutations, join on common variants
          set_sota_pred_df = sota_pred_df.loc[(sota_pred_df['protein_name'] == set_nm) & (sota_pred_df['gt'].notnull()) & (~sota_pred_df['mutant'].isin(valid_mut_df['mut_name']))]
          sota_myMdl_joint_df = set_sota_pred_df.merge(my_pred_df, on=['mutant','protein_name'], how='inner')
          assert len(set_sota_pred_df)-len(sota_myMdl_joint_df) == 0
          for score_nm in ['log_ratio_aa']:
            rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_joint_df['gt'].to_numpy().reshape(-1),sota_myMdl_joint_df[score_nm].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
            one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,score_nm,'test']
            metric_score_list.append(one_score_case)
            ## loop over mut depth ranges
            for depth_range in mutDepth_dict[set_nm]:
              if isinstance(depth_range,int):
                sota_myMdl_depth_joint_df = set_sota_pred_df.merge(my_pred_df.loc[my_pred_df['mut_depth'] == depth_range,:], on=['mutant','protein_name'], how='inner')
                rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_depth_joint_df['gt'].to_numpy().reshape(-1),sota_myMdl_depth_joint_df[score_nm].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
                one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,score_nm,f'test_{depth_range}']
                metric_score_list.append(one_score_case)
              elif isinstance(depth_range,List):
                sota_myMdl_depth_joint_df = set_sota_pred_df.merge(my_pred_df.loc[(my_pred_df['mut_depth'] >= depth_range[0]) & (my_pred_df['mut_depth'] <= depth_range[1]),:], on=['mutant','protein_name'], how='inner')
                rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_depth_joint_df['gt'].to_numpy().reshape(-1),sota_myMdl_depth_joint_df[score_nm].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
                one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,score_nm,f'test_{depth_range[0]}_{depth_range[1]}']
                metric_score_list.append(one_score_case)
              else:
                pass
    metric_structure_score_multitask_df = pd.DataFrame(metric_score_list,columns=['protein_name','lambda','epoch','rho','rho_P','auroc','auprc','score_nm','eval_set'])
    metric_structure_score_multitask_df.to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_structure_scores_multitaskFT_depth.csv',index=False,header=True)

    ## seq finetuned models (rho is calcualted over test mutation set)
    metric_score_list = []
    for set_nm in set_names:
      valid_mut_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/valid_mutation.csv',header=0)
      bina_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM']==set_nm,:].iloc[0]['DMS_binarization_cutoff']
      sign_reverse = set_nm in sign_reverse_set_names
      for epoch in epoch_seq_ft_list:
        for seq_mode in ['structSeq', 'allSeq']:
          print(f'seqFT,{set_nm},{epoch},{seq_mode}')
          ## get model name
          model_path_list = os.popen(f"grep -E -o 'masked_language_modeling_transformer.*[0-9]{{6}}' {path}/job_logs/archive_baseline_bert_eval/baseline_bert_mutation_fitness_UNsupervise_mutagenesis_rp15_all_2_729.seq_finetune.reweighted.{seq_mode}.TotalEpoch40.Interval*.{epoch}.*.{set_nm}.seqMaxL512.classWno.out | uniq").read().strip('\n').split('\n')
          assert len(model_path_list) == 1
          model_path = model_path_list[0]
          ## load my preds
          my_pred_df = pd.read_csv(f'{path}/eval_results/mutation_fitness_UNsupervise_mutagenesis/predictions/{model_path}/{set_nm}_{epoch}_rawScores.csv',names=['mutant','myMdl']).drop_duplicates().groupby(['mutant'],as_index=False)['myMdl'].mean().rename(columns={'myMdl':'log_ratio_aa'})
          my_pred_df['epoch'] = epoch
          my_pred_df['protein_name'] = set_nm
          my_pred_df['seq_mode'] = seq_mode
          my_pred_df['mut_depth'] = my_pred_df['mutant'].str.count(':')+1
          ## join on common variants
          set_sota_pred_df = sota_pred_df.loc[(sota_pred_df['protein_name'] == set_nm) & (sota_pred_df['gt'].notnull()) & (~sota_pred_df['mutant'].isin(valid_mut_df['mut_name']))]
          sota_myMdl_joint_df = set_sota_pred_df.merge(my_pred_df, on=['mutant','protein_name'], how='inner')
          assert len(set_sota_pred_df)-len(sota_myMdl_joint_df) == 0
          rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_joint_df['gt'].to_numpy().reshape(-1),sota_myMdl_joint_df['log_ratio_aa'].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
          one_score_case = [set_nm,epoch,seq_mode,rho_score,rho_p_value,auroc,auprc,'log_ratio_aa','test']
          metric_score_list.append(one_score_case)
          ## loop over mut depth ranges
          for depth_range in mutDepth_dict[set_nm]:
            if isinstance(depth_range,int):
              sota_myMdl_depth_joint_df = set_sota_pred_df.merge(my_pred_df.loc[my_pred_df['mut_depth'] == depth_range,:], on=['mutant','protein_name'], how='inner')
              rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_depth_joint_df['gt'].to_numpy().reshape(-1),sota_myMdl_depth_joint_df['log_ratio_aa'].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
              one_score_case = [set_nm,epoch,seq_mode,rho_score,rho_p_value,auroc,auprc,'log_ratio_aa',f'test_{depth_range}']
              metric_score_list.append(one_score_case)
            elif isinstance(depth_range,List):
              sota_myMdl_depth_joint_df = set_sota_pred_df.merge(my_pred_df.loc[(my_pred_df['mut_depth'] >= depth_range[0]) & (my_pred_df['mut_depth'] <= depth_range[1]),:], on=['mutant','protein_name'], how='inner')
              rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_depth_joint_df['gt'].to_numpy().reshape(-1),sota_myMdl_depth_joint_df['log_ratio_aa'].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
              one_score_case = [set_nm,epoch,seq_mode,rho_score,rho_p_value,auroc,auprc,'log_ratio_aa',f'test_{depth_range[0]}_{depth_range[1]}']
              metric_score_list.append(one_score_case)
            else:
              pass
    metric_score_seq_df = pd.DataFrame(metric_score_list,columns=['protein_name','epoch','seq_mode','rho','rho_P','auroc','auprc','score_nm','eval_set'])
    metric_score_seq_df.to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_seqFT_depth.csv',index=False,header=True)
    
    ## sota models (rho is calcualted over test mutation set)
    metric_score_list = []
    for set_nm in set_names:
      bina_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM']==set_nm,:].iloc[0]['DMS_binarization_cutoff']
      sign_reverse = set_nm in sign_reverse_set_names
      valid_mut_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/valid_mutation.csv',header=0)
      set_sota_pred_df = sota_pred_df.loc[(sota_pred_df['protein_name'] == set_nm) & (sota_pred_df['gt'].notnull()) & (~sota_pred_df['mutant'].isin(valid_mut_df['mut_name']))]
      set_sota_pred_df['mut_depth'] = set_sota_pred_df.loc[:,'mutant'].str.count(':')+1
      for sota_mdl in sota_mdl_names:
        print(f'sota,{set_nm},{sota_mdl}')
        rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(set_sota_pred_df['gt'].to_numpy().reshape(-1),set_sota_pred_df[sota_mdl].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
        one_score_case = [set_nm,sota_mdl,rho_score,rho_p_value,auroc,auprc,'log_ratio_aa','test']
        metric_score_list.append(one_score_case)
        ## loop over mut depth ranges
        for depth_range in mutDepth_dict[set_nm]:
          if isinstance(depth_range,int):
            sota_depth_joint_df = set_sota_pred_df.loc[set_sota_pred_df['mut_depth'] == depth_range,:]
            rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_depth_joint_df['gt'].to_numpy().reshape(-1),sota_depth_joint_df[sota_mdl].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
            one_score_case = [set_nm,sota_mdl,rho_score,rho_p_value,auroc,auprc,'log_ratio_aa',f'test_{depth_range}']
            metric_score_list.append(one_score_case)
          elif isinstance(depth_range,List):
            sota_depth_joint_df = set_sota_pred_df.loc[(set_sota_pred_df['mut_depth'] >= depth_range[0]) & (set_sota_pred_df['mut_depth'] <= depth_range[1]),:]
            rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_depth_joint_df['gt'].to_numpy().reshape(-1),sota_depth_joint_df[sota_mdl].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
            one_score_case = [set_nm,sota_mdl,rho_score,rho_p_value,auroc,auprc,'log_ratio_aa',f'test_{depth_range[0]}_{depth_range[1]}']
            metric_score_list.append(one_score_case)
          else:
            pass
    metric_score_sota_df = pd.DataFrame(metric_score_list,columns=['protein_name','mdl_name','rho','rho_P','auroc','auprc','score_nm','eval_set'])
    metric_score_sota_df.to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_scores_sota_depth.csv',index=False,header=True)

  if 'sota_protein_track_valid_best' in fig_name_list:
    ## joint-model, allMSASeq, fit-valid
    #'protein_name','lambda','epoch','rho','rho_P','auroc','auprc','score_nm','eval_set'
    test_score_allMSASeq_list = []
    mdl_name = 'Our SI-pLM (seq_all;fit_valid_best)'
    for set_nm in set_names:
      for score_nm in ['log_ratio_aa']:
        test_one_score_allMSASeq_list = []
        for metric_nm in ['rho', 'auroc', 'auprc']:
          valid_scoreNm_df = metric_structure_score_multitask_allMSASeq_df.loc[(metric_structure_score_multitask_allMSASeq_df['protein_name'] == set_nm) & (metric_structure_score_multitask_allMSASeq_df['eval_set'] == 'valid') & (metric_structure_score_multitask_allMSASeq_df['score_nm'] == score_nm)] 
          valid_best_idx = valid_scoreNm_df.groupby(['protein_name'],as_index=False)[metric_nm].transform(max) == valid_scoreNm_df[metric_nm].to_frame()
          valid_best_df = metric_structure_score_multitask_allMSASeq_df.iloc[valid_best_idx.loc[valid_best_idx[metric_nm]==True].index].drop_duplicates(subset=['protein_name'], keep='last')
          
          test_loopSet_allMSASeq_list = []
          ## whole test set
          test_score_allMSASeq_df = valid_best_df[['protein_name','lambda','epoch','score_nm',metric_nm]].merge(metric_structure_score_multitask_allMSASeq_df.loc[(metric_structure_score_multitask_allMSASeq_df['eval_set'] == 'test') & (metric_structure_score_multitask_allMSASeq_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch','score_nm'], how='inner',suffixes=('_valid','_test'))
          ## output best (lambda, epoch) selected by validation set
          #print(f'***allMSASeq, fit_valid_best, {score_nm}, {metric_nm}: best lambda, epoch***')
          #print(test_score_allMSASeq_df[['protein_name','lambda','epoch']].to_string())
          #print(test_score_allMSASeq_df['lambda'].value_counts().to_string())
          #print(test_score_allMSASeq_df['epoch'].value_counts().to_string())
          test_score_allMSASeq_df= test_score_allMSASeq_df[['protein_name',f'{metric_nm}_test','score_nm','eval_set']].rename(columns={f'{metric_nm}_test': metric_nm})
          test_score_allMSASeq_df['mdl_name'] = mdl_name
          test_loopSet_allMSASeq_list.append(test_score_allMSASeq_df)
          
          ## loop sub test sets
          for depth_range in mutDepth_dict[set_nm]:
            if isinstance(depth_range,int):
              test_score_allMSASeq_df = valid_best_df[['protein_name','lambda','epoch','score_nm',metric_nm]].merge(metric_structure_score_multitask_allMSASeq_df.loc[(metric_structure_score_multitask_allMSASeq_df['protein_name'] == set_nm) & (metric_structure_score_multitask_allMSASeq_df['eval_set'] == f'test_{depth_range}') & (metric_structure_score_multitask_allMSASeq_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch','score_nm'], how='inner',suffixes=('_valid','_test'))
              ## output best (lambda, epoch) selected by validation set
              #print(f'***allMSASeq, fit_valid_best, {score_nm}, {metric_nm}, {depth_range}: best lambda, epoch***')
              #print(test_score_allMSASeq_df[['protein_name','lambda','epoch']].to_string())
              #print(test_score_allMSASeq_df['lambda'].value_counts().to_string())
              #print(test_score_allMSASeq_df['epoch'].value_counts().to_string())
              test_score_allMSASeq_df= test_score_allMSASeq_df[['protein_name',f'{metric_nm}_test','score_nm','eval_set']].rename(columns={f'{metric_nm}_test': metric_nm})
              test_score_allMSASeq_df['mdl_name'] = mdl_name
              test_loopSet_allMSASeq_list.append(test_score_allMSASeq_df)
            elif isinstance(depth_range,List):
              test_score_allMSASeq_df = valid_best_df[['protein_name','lambda','epoch','score_nm',metric_nm]].merge(metric_structure_score_multitask_allMSASeq_df.loc[(metric_structure_score_multitask_allMSASeq_df['eval_set'] == f'test_{depth_range[0]}_{depth_range[1]}') & (metric_structure_score_multitask_allMSASeq_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch','score_nm'], how='inner',suffixes=('_valid','_test'))
              ## output best (lambda, epoch) selected by validation set
              #print(f'***allMSASeq, fit_valid_best, {score_nm}, {metric_nm}, {depth_range}: best lambda, epoch***')
              #print(test_score_allMSASeq_df[['protein_name','lambda','epoch']].to_string())
              #print(test_score_allMSASeq_df['lambda'].value_counts().to_string())
              #print(test_score_allMSASeq_df['epoch'].value_counts().to_string())
              test_score_allMSASeq_df= test_score_allMSASeq_df[['protein_name',f'{metric_nm}_test','score_nm','eval_set']].rename(columns={f'{metric_nm}_test': metric_nm})
              test_score_allMSASeq_df['mdl_name'] = mdl_name
              test_loopSet_allMSASeq_list.append(test_score_allMSASeq_df)
            else:
              pass
          test_loopSet_allMSASeq_df = pd.concat(test_loopSet_allMSASeq_list,axis=0)
          test_one_score_allMSASeq_list.append(test_loopSet_allMSASeq_df)
        ## merge dfs of three metrics
        test_all_score_allMSASeq_df = test_one_score_allMSASeq_list[0].merge(test_one_score_allMSASeq_list[1],on=['protein_name','mdl_name','score_nm','eval_set'])
        test_all_score_allMSASeq_df = test_all_score_allMSASeq_df.merge(test_one_score_allMSASeq_list[2],on=['protein_name','mdl_name','score_nm','eval_set'])
        test_score_allMSASeq_list.append(test_all_score_allMSASeq_df)
    ## merge dfs of all score types
    test_all_score_allMSASeq_df = pd.concat(test_score_allMSASeq_list,axis=0)

    ## joint-model, allMSASeq, pre-task-valid
    #* option 2: best checkpoint is used within lambda, fitness validation set is used to select cross lambda
    mdl_name = 'Our SI-pLM (seq_all;pre_task_best)'
    test_score_list = []
    for set_nm in set_names:
      for score_nm in ['log_ratio_aa']:
        test_one_score_list = []
        lmdBest_metric_score_multitask_df = metric_structure_score_multitask_allMSASeq_df.loc[(metric_structure_score_multitask_allMSASeq_df['protein_name'] == set_nm) & (metric_structure_score_multitask_allMSASeq_df['eval_set'] == 'valid') & (metric_structure_score_multitask_allMSASeq_df['epoch'] == 'best') & (metric_structure_score_multitask_allMSASeq_df['score_nm'] == score_nm),:]
        for metric_nm in ['rho', 'auroc', 'auprc']:
          valid_best_idx = lmdBest_metric_score_multitask_df.groupby(['protein_name'],as_index=False)[metric_nm].transform(max) == lmdBest_metric_score_multitask_df[metric_nm].to_frame()
          valid_best_df = metric_structure_score_multitask_df.iloc[valid_best_idx.loc[valid_best_idx[metric_nm]==True].index].drop_duplicates(subset=['protein_name'], keep='last')
          
          test_loopSet_list = []
          ## whole test set
          test_score_df = valid_best_df[['protein_name','lambda','epoch','score_nm',metric_nm]].merge(metric_structure_score_multitask_allMSASeq_df.loc[(metric_structure_score_multitask_allMSASeq_df['eval_set'] == 'test') & (metric_structure_score_multitask_allMSASeq_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch','score_nm'], how='inner',suffixes=('_valid','_test'))
          test_score_df= test_score_df[['protein_name',f'{metric_nm}_test','score_nm','eval_set']].rename(columns={f'{metric_nm}_test': metric_nm})
          test_score_df['mdl_name'] = mdl_name
          test_loopSet_list.append(test_score_df)
          
          ## loop sub test sets
          for depth_range in mutDepth_dict[set_nm]:
            if isinstance(depth_range,int):
              test_score_df = valid_best_df[['protein_name','lambda','epoch','score_nm',metric_nm]].merge(metric_structure_score_multitask_allMSASeq_df.loc[(metric_structure_score_multitask_allMSASeq_df['protein_name'] == set_nm) & (metric_structure_score_multitask_allMSASeq_df['eval_set'] == f'test_{depth_range}') & (metric_structure_score_multitask_allMSASeq_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch','score_nm'], how='inner',suffixes=('_valid','_test'))
              test_score_df= test_score_df[['protein_name',f'{metric_nm}_test','score_nm','eval_set']].rename(columns={f'{metric_nm}_test': metric_nm})
              test_score_df['mdl_name'] = mdl_name
              test_loopSet_list.append(test_score_df)
            elif isinstance(depth_range,List):
              test_score_df = valid_best_df[['protein_name','lambda','epoch','score_nm',metric_nm]].merge(metric_structure_score_multitask_allMSASeq_df.loc[(metric_structure_score_multitask_allMSASeq_df['protein_name'] == set_nm) & (metric_structure_score_multitask_allMSASeq_df['eval_set'] == f'test_{depth_range[0]}_{depth_range[1]}') & (metric_structure_score_multitask_allMSASeq_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch','score_nm'], how='inner',suffixes=('_valid','_test'))
              test_score_df= test_score_df[['protein_name',f'{metric_nm}_test','score_nm','eval_set']].rename(columns={f'{metric_nm}_test': metric_nm})
              test_score_df['mdl_name'] = mdl_name
              test_loopSet_list.append(test_score_df)
            else:
              pass
          test_loopSet_df = pd.concat(test_loopSet_list,axis=0)
          test_one_score_list.append(test_loopSet_df)
        ## merge dfs of three metrics
        test_all_score_df = test_one_score_list[0].merge(test_one_score_list[1],on=['protein_name','mdl_name','score_nm','eval_set'])
        test_all_score_df = test_all_score_df.merge(test_one_score_list[2],on=['protein_name','mdl_name','score_nm','eval_set'])
        test_score_list.append(test_all_score_df)
    ## merge dfs of all score types
    test_all_score_allMSASeq_preTask_df = pd.concat(test_score_list,axis=0)


    ## joint-model, seq w/ structure, fit-valid, sequence scoring only
    test_score_list = []
    mdl_name = 'Our SI-pLM (seqWStruct;fit_valid_best)'
    for set_nm in set_names:
      for score_nm in ['log_ratio_aa']:
        test_one_score_list = []
        for metric_nm in ['rho', 'auroc', 'auprc']:
          valid_scoreNm_df = metric_structure_score_multitask_df.loc[(metric_structure_score_multitask_df['protein_name'] == set_nm) & (metric_structure_score_multitask_df['eval_set'] == 'valid') & (metric_structure_score_multitask_df['score_nm'] == score_nm)] 
          valid_best_idx = valid_scoreNm_df.groupby(['protein_name'],as_index=False)[metric_nm].transform(max) == valid_scoreNm_df[metric_nm].to_frame()
          valid_best_df = metric_structure_score_multitask_df.iloc[valid_best_idx.loc[valid_best_idx[metric_nm]==True].index].drop_duplicates(subset=['protein_name'], keep='last')
          
          test_loopSet_list = []
          ## whole test set
          test_score_df = valid_best_df[['protein_name','lambda','epoch','score_nm',metric_nm]].merge(metric_structure_score_multitask_df.loc[(metric_structure_score_multitask_df['eval_set'] == 'test') & (metric_structure_score_multitask_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch','score_nm'], how='inner',suffixes=('_valid','_test'))
          ## output best (lambda, epoch) selected by validation set
          #print(f'***allMSASeq, fit_valid_best, {score_nm}, {metric_nm}: best lambda, epoch***')
          #print(test_score_df[['protein_name','lambda','epoch']].to_string())
          #print(test_score_df['lambda'].value_counts().to_string())
          #print(test_score_df['epoch'].value_counts().to_string())
          test_score_df= test_score_df[['protein_name',f'{metric_nm}_test','score_nm','eval_set']].rename(columns={f'{metric_nm}_test': metric_nm})
          test_score_df['mdl_name'] = mdl_name
          test_loopSet_list.append(test_score_df)
          
          ## loop sub test sets
          for depth_range in mutDepth_dict[set_nm]:
            if isinstance(depth_range,int):
              test_score_df = valid_best_df[['protein_name','lambda','epoch','score_nm',metric_nm]].merge(metric_structure_score_multitask_df.loc[(metric_structure_score_multitask_df['protein_name'] == set_nm) & (metric_structure_score_multitask_df['eval_set'] == f'test_{depth_range}') & (metric_structure_score_multitask_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch','score_nm'], how='inner',suffixes=('_valid','_test'))
              ## output best (lambda, epoch) selected by validation set
              #print(f'***allMSASeq, fit_valid_best, {score_nm}, {metric_nm}, {depth_range}: best lambda, epoch***')
              #print(test_score_df[['protein_name','lambda','epoch']].to_string())
              #print(test_score_df['lambda'].value_counts().to_string())
              #print(test_score_df['epoch'].value_counts().to_string())
              test_score_df= test_score_df[['protein_name',f'{metric_nm}_test','score_nm','eval_set']].rename(columns={f'{metric_nm}_test': metric_nm})
              test_score_df['mdl_name'] = mdl_name
              test_loopSet_list.append(test_score_df)
            elif isinstance(depth_range,List):
              test_score_df = valid_best_df[['protein_name','lambda','epoch','score_nm',metric_nm]].merge(metric_structure_score_multitask_df.loc[(metric_structure_score_multitask_df['eval_set'] == f'test_{depth_range[0]}_{depth_range[1]}') & (metric_structure_score_multitask_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch','score_nm'], how='inner',suffixes=('_valid','_test'))
              ## output best (lambda, epoch) selected by validation set
              #print(f'***allMSASeq, fit_valid_best, {score_nm}, {metric_nm}, {depth_range}: best lambda, epoch***')
              #print(test_score_df[['protein_name','lambda','epoch']].to_string())
              #print(test_score_df['lambda'].value_counts().to_string())
              #print(test_score_df['epoch'].value_counts().to_string())
              test_score_df= test_score_df[['protein_name',f'{metric_nm}_test','score_nm','eval_set']].rename(columns={f'{metric_nm}_test': metric_nm})
              test_score_df['mdl_name'] = mdl_name
              test_loopSet_list.append(test_score_df)
            else:
              pass
          test_loopSet_df = pd.concat(test_loopSet_list,axis=0)
          test_one_score_list.append(test_loopSet_df)
        ## merge dfs of three metrics
        test_all_score_df = test_one_score_list[0].merge(test_one_score_list[1],on=['protein_name','mdl_name','score_nm','eval_set'])
        test_all_score_df = test_all_score_df.merge(test_one_score_list[2],on=['protein_name','mdl_name','score_nm','eval_set'])
        test_score_list.append(test_all_score_df)
    ## merge dfs of all score types
    test_all_score_df = pd.concat(test_score_list,axis=0)

    ## joint-model, seq w/ structure, pre-task-valid
    #* option 2: best checkpoint is used within lambda, fitness validation set is used to select cross lambda
    mdl_name = 'Our SI-pLM (seqWStruct;pre_task_best)'
    test_score_list = []
    for set_nm in set_names:
      for score_nm in ['log_ratio_aa']:
        test_one_score_list = []
        lmdBest_metric_score_multitask_df = metric_structure_score_multitask_df.loc[(metric_structure_score_multitask_df['protein_name'] == set_nm) & (metric_structure_score_multitask_df['eval_set'] == 'valid') & (metric_structure_score_multitask_df['epoch'] == 'best') & (metric_structure_score_multitask_df['score_nm'] == score_nm),:]
        for metric_nm in ['rho', 'auroc', 'auprc']:
          valid_best_idx = lmdBest_metric_score_multitask_df.groupby(['protein_name'],as_index=False)[metric_nm].transform(max) == lmdBest_metric_score_multitask_df[metric_nm].to_frame()
          valid_best_df = metric_structure_score_multitask_df.iloc[valid_best_idx.loc[valid_best_idx[metric_nm]==True].index].drop_duplicates(subset=['protein_name'], keep='last')
          
          test_loopSet_list = []
          ## whole test set
          test_score_df = valid_best_df[['protein_name','lambda','epoch','score_nm',metric_nm]].merge(metric_structure_score_multitask_df.loc[(metric_structure_score_multitask_df['eval_set'] == 'test') & (metric_structure_score_multitask_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch','score_nm'], how='inner',suffixes=('_valid','_test'))
          test_score_df= test_score_df[['protein_name',f'{metric_nm}_test','score_nm','eval_set']].rename(columns={f'{metric_nm}_test': metric_nm})
          test_score_df['mdl_name'] = mdl_name
          test_loopSet_list.append(test_score_df)
          
          ## loop sub test sets
          for depth_range in mutDepth_dict[set_nm]:
            if isinstance(depth_range,int):
              test_score_df = valid_best_df[['protein_name','lambda','epoch','score_nm',metric_nm]].merge(metric_structure_score_multitask_df.loc[(metric_structure_score_multitask_df['protein_name'] == set_nm) & (metric_structure_score_multitask_df['eval_set'] == f'test_{depth_range}') & (metric_structure_score_multitask_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch','score_nm'], how='inner',suffixes=('_valid','_test'))
              test_score_df= test_score_df[['protein_name',f'{metric_nm}_test','score_nm','eval_set']].rename(columns={f'{metric_nm}_test': metric_nm})
              test_score_df['mdl_name'] = mdl_name
              test_loopSet_list.append(test_score_df)
            elif isinstance(depth_range,List):
              test_score_df = valid_best_df[['protein_name','lambda','epoch','score_nm',metric_nm]].merge(metric_structure_score_multitask_df.loc[(metric_structure_score_multitask_df['protein_name'] == set_nm) & (metric_structure_score_multitask_df['eval_set'] == f'test_{depth_range[0]}_{depth_range[1]}') & (metric_structure_score_multitask_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch','score_nm'], how='inner',suffixes=('_valid','_test'))
              test_score_df= test_score_df[['protein_name',f'{metric_nm}_test','score_nm','eval_set']].rename(columns={f'{metric_nm}_test': metric_nm})
              test_score_df['mdl_name'] = mdl_name
              test_loopSet_list.append(test_score_df)
            else:
              pass
          test_loopSet_df = pd.concat(test_loopSet_list,axis=0)
          test_one_score_list.append(test_loopSet_df)
        ## merge dfs of three metrics
        test_all_score_df = test_one_score_list[0].merge(test_one_score_list[1],on=['protein_name','mdl_name','score_nm','eval_set'])
        test_all_score_df = test_all_score_df.merge(test_one_score_list[2],on=['protein_name','mdl_name','score_nm','eval_set'])
        test_score_list.append(test_all_score_df)
    ## merge dfs of all score types
    test_all_score_preTask_df = pd.concat(test_score_list,axis=0)

    ## seq pretrained model
    target_pre_seq_list = []
    for set_nm in set_names:
      for score_nm in ['log_ratio_aa']:
        ## whole test set
        target_pre_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['epoch'] == '0') & (metric_score_seq_df['seq_mode'] == 'allSeq') & (metric_score_seq_df['eval_set'] == 'test') & (metric_score_seq_df['protein_name'] == set_nm) & ((metric_score_seq_df['score_nm'] == score_nm)),:][['protein_name','epoch','rho','auroc','auprc','score_nm','eval_set']].groupby(['protein_name','eval_set','score_nm'],as_index=False)[['rho','auroc','auprc']].mean()
        target_pre_seq_df['mdl_name'] = 'Our pLM (PT)'
        target_pre_seq_list.append(target_pre_seq_df)
        ## loop mutation depth
        for depth_range in mutDepth_dict[set_nm]:
          if isinstance(depth_range,int):
            target_pre_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['epoch'] == '0') & (metric_score_seq_df['seq_mode'] == 'allSeq') & (metric_score_seq_df['eval_set'] == f'test_{depth_range}') & (metric_score_seq_df['protein_name'] == set_nm) & ((metric_score_seq_df['score_nm'] == score_nm)),:][['protein_name','epoch','rho','auroc','auprc','score_nm','eval_set']].groupby(['protein_name','eval_set','score_nm'],as_index=False)[['rho','auroc','auprc']].mean()
            target_pre_seq_df['mdl_name'] = 'Our pLM (PT)'
            target_pre_seq_list.append(target_pre_seq_df)
          elif isinstance(depth_range,List):
            target_pre_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['epoch'] == '0') & (metric_score_seq_df['seq_mode'] == 'allSeq') & (metric_score_seq_df['eval_set'] == f'test_{depth_range[0]}_{depth_range[1]}') & (metric_score_seq_df['protein_name'] == set_nm) & ((metric_score_seq_df['score_nm'] == score_nm)),:][['protein_name','epoch','rho','auroc','auprc','score_nm','eval_set']].groupby(['protein_name','eval_set','score_nm'],as_index=False)[['rho','auroc','auprc']].mean()
            target_pre_seq_df['mdl_name'] = 'Our pLM (PT)'
            target_pre_seq_list.append(target_pre_seq_df)
    target_pre_seq_df = pd.concat(target_pre_seq_list,axis=0)

    ## allSeq finetune
    target_ft_all_seq_list = []
    for set_nm in set_names:
      for score_nm in ['log_ratio_aa']:
        target_ft_all_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['epoch'] == 'best') & (metric_score_seq_df['seq_mode'] == 'allSeq') & (metric_score_seq_df['eval_set'] == 'test') & (metric_score_seq_df['protein_name'] == set_nm) & ((metric_score_seq_df['score_nm'] == score_nm)),:][['protein_name','epoch','rho','auroc','auprc','score_nm','eval_set']].groupby(['protein_name','eval_set','score_nm'],as_index=False)[['rho','auroc','auprc']].mean()
        target_ft_all_seq_df['mdl_name'] = 'Our pLM (FT;seq_all)'
        target_ft_all_seq_list.append(target_ft_all_seq_df)
        ## loop mutation depth
        for depth_range in mutDepth_dict[set_nm]:
          if isinstance(depth_range,int):
            target_ft_all_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['epoch'] == 'best') & (metric_score_seq_df['seq_mode'] == 'allSeq') & (metric_score_seq_df['eval_set'] == f'test_{depth_range}') & (metric_score_seq_df['protein_name'] == set_nm) & ((metric_score_seq_df['score_nm'] == score_nm)),:][['protein_name','epoch','rho','auroc','auprc','score_nm','eval_set']].groupby(['protein_name','eval_set','score_nm'],as_index=False)[['rho','auroc','auprc']].mean()
            target_ft_all_seq_df['mdl_name'] = 'Our pLM (FT;seq_all)'
            target_ft_all_seq_list.append(target_ft_all_seq_df)
          elif isinstance(depth_range,List):
            target_ft_all_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['epoch'] == 'best') & (metric_score_seq_df['seq_mode'] == 'allSeq') & (metric_score_seq_df['eval_set'] == f'test_{depth_range[0]}_{depth_range[1]}') & (metric_score_seq_df['protein_name'] == set_nm) & ((metric_score_seq_df['score_nm'] == score_nm)),:][['protein_name','epoch','rho','auroc','auprc','score_nm','eval_set']].groupby(['protein_name','eval_set','score_nm'],as_index=False)[['rho','auroc','auprc']].mean()
            target_ft_all_seq_df['mdl_name'] = 'Our pLM (FT;seq_all)'
            target_ft_all_seq_list.append(target_ft_all_seq_df)
    target_ft_all_seq_df = pd.concat(target_ft_all_seq_list,axis=0)

    ## structSeq finetune
    target_ft_struct_seq_list = []
    for set_nm in set_names:
      for score_nm in ['log_ratio_aa']:
        target_ft_struct_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['epoch'] == 'best') & (metric_score_seq_df['seq_mode'] == 'structSeq') & (metric_score_seq_df['eval_set'] == 'test') & (metric_score_seq_df['protein_name'] == set_nm) & ((metric_score_seq_df['score_nm'] == score_nm)),:][['protein_name','epoch','rho','auroc','auprc','score_nm','eval_set']].groupby(['protein_name','eval_set','score_nm'],as_index=False)[['rho','auroc','auprc']].mean()
        target_ft_struct_seq_df['mdl_name'] = 'Our pLM (FT;seqWStruct)'
        target_ft_struct_seq_list.append(target_ft_struct_seq_df)
        ## loop mutation depth
        for depth_range in mutDepth_dict[set_nm]:
          if isinstance(depth_range,int):
            target_ft_struct_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['epoch'] == 'best') & (metric_score_seq_df['seq_mode'] == 'structSeq') & (metric_score_seq_df['eval_set'] == f'test_{depth_range}') & (metric_score_seq_df['protein_name'] == set_nm) & ((metric_score_seq_df['score_nm'] == score_nm)),:][['protein_name','epoch','rho','auroc','auprc','score_nm','eval_set']].groupby(['protein_name','eval_set','score_nm'],as_index=False)[['rho','auroc','auprc']].mean()
            target_ft_struct_seq_df['mdl_name'] = 'Our pLM (FT;seqWStruct)'
            target_ft_struct_seq_list.append(target_ft_struct_seq_df)
          elif isinstance(depth_range,List):
            target_ft_struct_seq_df = metric_score_seq_df.loc[(metric_score_seq_df['epoch'] == 'best') & (metric_score_seq_df['seq_mode'] == 'structSeq') & (metric_score_seq_df['eval_set'] == f'test_{depth_range[0]}_{depth_range[1]}') & (metric_score_seq_df['protein_name'] == set_nm) & ((metric_score_seq_df['score_nm'] == score_nm)),:][['protein_name','epoch','rho','auroc','auprc','score_nm','eval_set']].groupby(['protein_name','eval_set','score_nm'],as_index=False)[['rho','auroc','auprc']].mean()
            target_ft_struct_seq_df['mdl_name'] = 'Our pLM (FT;seqWStruct)'
            target_ft_struct_seq_list.append(target_ft_struct_seq_df)
    target_ft_struct_seq_df = pd.concat(target_ft_struct_seq_list,axis=0)

    
    metric_score_all_df = pd.concat([metric_score_sota_df[['protein_name','mdl_name','rho','auroc','auprc','score_nm','eval_set']],test_all_score_allMSASeq_df,test_all_score_allMSASeq_preTask_df,test_all_score_df,test_all_score_preTask_df,target_pre_seq_df,target_ft_all_seq_df,target_ft_struct_seq_df])

    ## ordered outputs
    for set_nm in set_names:
      for depth_range in mutDepth_dict[set_nm]:
        if isinstance(depth_range,int):
          eval_set_str = f'test_{depth_range}'
        elif isinstance(depth_range,List):
          eval_set_str = f'test_{depth_range[0]}_{depth_range[1]}'
        ordered_mdls = metric_score_all_df.loc[(metric_score_all_df['protein_name'] == set_nm) & (metric_score_all_df['eval_set'] == eval_set_str),:].sort_values('rho',ascending=False)[['mdl_name','score_nm','rho']].reset_index().to_string()
        print(f'{set_nm},{eval_set_str},rho,{ordered_mdls}')
      
      ordered_mdls = metric_score_all_df.loc[(metric_score_all_df['protein_name'] == set_nm) & (metric_score_all_df['eval_set'] == 'test'),:].sort_values('rho',ascending=False)[['mdl_name','score_nm','rho']].reset_index().to_string()
      print(f'{set_nm},test,rho,{ordered_mdls}')
      
      ordered_mdls = metric_score_all_df.loc[(metric_score_all_df['protein_name'] == set_nm) & (metric_score_all_df['eval_set'] == 'test'),:].sort_values('auroc',ascending=False)[['mdl_name','score_nm','auroc']].reset_index().to_string()
      print(f'{set_nm},test,auroc,{ordered_mdls}')
      
      ordered_mdls = metric_score_all_df.loc[(metric_score_all_df['protein_name'] == set_nm) & (metric_score_all_df['eval_set'] == 'test'),:].sort_values('auprc',ascending=False)[['mdl_name','score_nm','auprc']].reset_index().to_string()
      print(f'{set_nm},test,auprc,{ordered_mdls}')


    ## latex outputs
    for set_nm in set_names:
      for mdl in metric_score_all_df['mdl_name'].unique().tolist():
        if mdl in ['Our SI-pLM (seq_all;fit_valid_best)']:
          for score_nm in ['log_ratio_aa']:
            out_str = f'{set_nm},{mdl},{score_nm}: '
            for depth_range in mutDepth_dict[set_nm]:
              if isinstance(depth_range,int):
                eval_set_str = f'test_{depth_range}'
              elif isinstance(depth_range,List):
                eval_set_str = f'test_{depth_range[0]}_{depth_range[1]}'
              output_df = metric_score_all_df.loc[(metric_score_all_df['protein_name'] == set_nm) & (metric_score_all_df['mdl_name'] == mdl) & (metric_score_all_df['score_nm'] == score_nm) & (metric_score_all_df['eval_set'] == eval_set_str),:].iloc[0]
              out_str += f"&{output_df['rho']:.3f}"
            output_df = metric_score_all_df.loc[(metric_score_all_df['protein_name'] == set_nm) & (metric_score_all_df['mdl_name'] == mdl) & (metric_score_all_df['score_nm'] == score_nm) & (metric_score_all_df['eval_set'] == 'test'),:].iloc[0]
            out_str += f"&{output_df['rho']:.3f}&{output_df['auroc']:.3f}&{output_df['auprc']:.3f}"
            print(out_str)
        else:
          for score_nm in ['log_ratio_aa']:
            out_str = f'{set_nm},{mdl},{score_nm}: '
            for depth_range in mutDepth_dict[set_nm]:
              if isinstance(depth_range,int):
                eval_set_str = f'test_{depth_range}'
              elif isinstance(depth_range,List):
                eval_set_str = f'test_{depth_range[0]}_{depth_range[1]}'
              output_df = metric_score_all_df.loc[(metric_score_all_df['protein_name'] == set_nm) & (metric_score_all_df['mdl_name'] == mdl) & (metric_score_all_df['score_nm'] == score_nm) & (metric_score_all_df['eval_set'] == eval_set_str),:].iloc[0]
              out_str += f"&{output_df['rho']:.3f}"
            output_df = metric_score_all_df.loc[(metric_score_all_df['protein_name'] == set_nm) & (metric_score_all_df['mdl_name'] == mdl) & (metric_score_all_df['score_nm'] == score_nm) & (metric_score_all_df['eval_set'] == 'test'),:].iloc[0]
            out_str += f"&{output_df['rho']:.3f}&{output_df['auroc']:.3f}&{output_df['auprc']:.3f}"
            print(out_str)

  return

def seq_struct_scoring(load_processed_df: bool = True, fig_name_list: List=None):
  """Comparison between sequence-based and structure-based mutation scoring
  """
  path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  #header: protein_name,mutant,gt,MSA Transformer,PSSM,EVMutation (published),DeepSequence (published) - single,DeepSequence (published),ESM-1v (zero shot),ESM-1v (+further training),DeepSequence (replicated),EVMutation (replicated),ProtBERT-BFD,TAPE,UniRep,ESM-1b
  sota_pred_df = pd.read_csv(f'{path}/data_process/mutagenesis/esm_predictions/raw_with_wavenet_df.csv',delimiter=',',header=0)
  fitness_fam_info_df = pd.read_csv(f'{path}/data_process/mutagenesis/DeepSequenceMutaSet_reference_file.csv',delimiter=',',header=0)
  #'prot_nm','mut_nm','lmd','mdl_path','mdl_path_1'
  model_path_multitaskFT_allMSASeq_df = pd.read_csv(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/model_path_multitaskFT_allMSASeq.csv',delimiter=',',header=0)
  model_path_multitaskFT_df = pd.read_csv(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/model_path_multitaskFT.csv',delimiter=',',header=0)
  #'AMIE_PSEAE_Whitehead','B3VI55_LIPSTSTABLE','B3VI55_LIPST_Whitehead2015','BG_STRSQ_hmmerbit','BLAT_ECOLX_Ostermeier2014','BLAT_ECOLX_Palzkill2012','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Tenaillon2013','BRCA1_HUMAN_BRCT','BRCA1_HUMAN_RING','CALM1_HUMAN_Roth2017','DLG4_RAT_Ranganathan2012','GAL4_YEAST_Shendure2015','HIS7_YEAST_Kondrashov2017','HSP82_YEAST_Bolon2016','IF1_ECOLI_Kishony','KKA2_KLEPN_Mikkelsen2014','MK01_HUMAN_Johannessen','MTH3_HAEAESTABILIZED_Tawfik2015','P84126_THETH_b0','PABP_YEAST_Fields2013-doubles','PABP_YEAST_Fields2013-singles','PTEN_HUMAN_Fowler2018','RASH_HUMAN_Kuriyan','RL401_YEAST_Bolon2013','RL401_YEAST_Bolon2014','RL401_YEAST_Fraser2016','SUMO1_HUMAN_Roth2017','TIM_SULSO_b0','TIM_THEMA_b0','TPK1_HUMAN_Roth2017','TPMT_HUMAN_Fowler2018','UBC9_HUMAN_Roth2017','UBE4B_MOUSE_Klevit2013-singles','YAP1_HUMAN_Fields2012-singles'
  #'BF520_env_Bloom2018','BG505_env_Bloom2018','HG_FLU_Bloom2016','PA_FLU_Sun2015','POLG_HCVJF_Sun2014','POL_HV1N5-CA_Ndungu2014'
  set_names = ['AMIE_PSEAE_Whitehead','B3VI55_LIPSTSTABLE','B3VI55_LIPST_Whitehead2015','BG_STRSQ_hmmerbit','BLAT_ECOLX_Ostermeier2014','BLAT_ECOLX_Palzkill2012','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Tenaillon2013','BRCA1_HUMAN_BRCT','BRCA1_HUMAN_RING','CALM1_HUMAN_Roth2017','DLG4_RAT_Ranganathan2012','GAL4_YEAST_Shendure2015','HSP82_YEAST_Bolon2016','IF1_ECOLI_Kishony','KKA2_KLEPN_Mikkelsen2014','MK01_HUMAN_Johannessen','MTH3_HAEAESTABILIZED_Tawfik2015','P84126_THETH_b0','PABP_YEAST_Fields2013-doubles','PABP_YEAST_Fields2013-singles','PTEN_HUMAN_Fowler2018','RASH_HUMAN_Kuriyan','RL401_YEAST_Bolon2013','RL401_YEAST_Bolon2014','RL401_YEAST_Fraser2016','SUMO1_HUMAN_Roth2017','TIM_SULSO_b0','TIM_THEMA_b0','TPK1_HUMAN_Roth2017','TPMT_HUMAN_Fowler2018','UBC9_HUMAN_Roth2017','UBE4B_MOUSE_Klevit2013-singles','YAP1_HUMAN_Fields2012-singles']

  sign_reverse_set_names = ['BLAT_ECOLX_Palzkill2012','MK01_HUMAN_Johannessen']
  struct_score_names = ['log_ratio_aa','log_ratio_ss','log_ratio_rsa','log_ratio_ss_env','log_ratio_rsa_env','log_ratio_dm','log_ratio_ss_abs','log_ratio_ss_env_abs','log_ratio_rsa_abs','log_ratio_rsa_env_abs','log_ratio_dm_abs']
  struct_score_ensemble_names = ['ss_rsa_cm','ssN_rsaN_cm','aa_ss_rsa_cm','aa_ssN_rsaN_cm','aa_cm','abs(ss_rsa_cm)','abs(ssN_rsaN_cm)','abs(aa_ss_rsa_cm)','abs(aa_ssN_rsaN_cm)','abs(aa_cm)']
  epoch_joint_ft_list = ['best',0,5,11,17,23,29,35,41,47,53,59] #
  epoch_seq_ft_list = ['best',0,3,7,11,15,19,23,27,31,35,39]
  lambda_list = [0.0,0.5,2.0,20.0] #0.0,0.01,0.1,0.33,0.5,1.0,2.0,5.0,10.0,20.0
  if load_processed_df:
    metric_structure_score_multitask_allMSASeq_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_structure_scores_multitaskFT_allMSASeq.csv',sep=',',header=0)
    #metric_structure_score_multitask_df = pd.read_csv(f'eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_structure_scores_multitaskFT.csv',sep=',',header=0)
  else:
    ## seq+struct finetune, allMSASeq; structure-based mutation scoring
    metric_score_list = []
    #raw_score_list = []
    for set_nm in set_names:
      ## load validation mutations
      valid_mut_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/valid_mutation.csv',header=0)
      fit_gt_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/mut_fit_gt.csv',header=0)
      valid_fit_gt_df = fit_gt_df.loc[fit_gt_df['mut_name'].isin(valid_mut_df['mut_name']) & (fit_gt_df['fitness'].notnull())]
      prot_nm = fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,:].iloc[0]['Shin2021_set']
      bina_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM']==set_nm,:].iloc[0]['DMS_binarization_cutoff']
      sign_reverse = set_nm in sign_reverse_set_names
      for lmd in lambda_list:
        ## get model name
        model_path = model_path_multitaskFT_allMSASeq_df.loc[(model_path_multitaskFT_allMSASeq_df['mut_nm']==set_nm) & (model_path_multitaskFT_allMSASeq_df['lmd']==lmd),'mdl_path'].iloc[0]
        for epoch in epoch_joint_ft_list:
          print(f'seq+structFT (allMSASeq;struct-based score),{set_nm},{lmd},{epoch}')
          ## load my predicted fitness scores
          #var_name,fit_true,aa_fit,log_ratio_ss,log_ratio_rsa,log_ratio_ss_env,log_ratio_rsa_env,log_ratio_dm
          my_pred_df = pd.read_csv(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis_structure/predictions/{model_path}/{set_nm}_{epoch}_structProp_rawScores.csv',header=0,delimiter=',').rename(columns={'var_name': 'mutant','aa_fit': 'log_ratio_aa'}).drop_duplicates().groupby(['mutant'],as_index=False)[['fit_true','log_ratio_aa','log_ratio_ss','log_ratio_rsa','log_ratio_ss_env','log_ratio_rsa_env','log_ratio_dm','log_ratio_ss_abs','log_ratio_rsa_abs','log_ratio_ss_env_abs','log_ratio_rsa_env_abs','log_ratio_dm_abs']].mean()
          my_pred_df['epoch'] = epoch
          my_pred_df['lmd'] = lmd
          my_pred_df['protein_name'] = set_nm
          #raw_score_list.append(my_pred_df)
    
          ## metrics calculation on validation mutations
          valid_merge_df = valid_fit_gt_df.merge(my_pred_df, left_on=['mut_name'], right_on=['mutant'], how='inner')
          for score_nm in struct_score_names:
            valid_merge_sub_df = valid_merge_df.loc[valid_merge_df[score_nm].notnull(),:]
            rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(valid_merge_sub_df['fitness'].to_numpy().reshape(-1),valid_merge_sub_df[score_nm].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
            one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,score_nm,'valid']
            metric_score_list.append(one_score_case)
          
          ## label-validation set: ss+rsa+contMap
          valid_merge_df['ss_rsa_cm'] = valid_merge_df['log_ratio_ss'] + valid_merge_df['log_ratio_rsa'] + valid_merge_df['log_ratio_dm']
          valid_merge_df['abs(ss_rsa_cm)'] = valid_merge_df['log_ratio_ss_abs'] + valid_merge_df['log_ratio_rsa_abs'] + valid_merge_df['log_ratio_dm_abs']
          ## label-validation set: ssN+rsaN+contMap
          valid_merge_df['ssN_rsaN_cm'] = valid_merge_df['log_ratio_ss_env'] + valid_merge_df['log_ratio_rsa_env'] + valid_merge_df['log_ratio_dm']
          valid_merge_df['abs(ssN_rsaN_cm)'] = valid_merge_df['log_ratio_ss_env_abs'] + valid_merge_df['log_ratio_rsa_env_abs'] + valid_merge_df['log_ratio_dm_abs']
          ## label-validation set: ss+rsa+contMap+aa
          valid_merge_df['aa_ss_rsa_cm'] = valid_merge_df['log_ratio_aa'] + valid_merge_df['log_ratio_ss'] + valid_merge_df['log_ratio_rsa'] + valid_merge_df['log_ratio_dm']
          valid_merge_df['abs(aa_ss_rsa_cm)'] = valid_merge_df['log_ratio_aa'] + valid_merge_df['log_ratio_ss_abs'] + valid_merge_df['log_ratio_rsa_abs'] + valid_merge_df['log_ratio_dm_abs']
          ## label-validation set: ssN+rsaN+contMap+aa
          valid_merge_df['aa_ssN_rsaN_cm'] = valid_merge_df['log_ratio_aa'] + valid_merge_df['log_ratio_ss_env'] + valid_merge_df['log_ratio_rsa_env'] + valid_merge_df['log_ratio_dm']
          valid_merge_df['abs(aa_ssN_rsaN_cm)'] = valid_merge_df['log_ratio_aa'] + valid_merge_df['log_ratio_ss_env_abs'] + valid_merge_df['log_ratio_rsa_env_abs'] + valid_merge_df['log_ratio_dm_abs']
          ## label-validation set: contMap+aa
          valid_merge_df['aa_cm'] = valid_merge_df['log_ratio_aa'] + valid_merge_df['log_ratio_dm']
          valid_merge_df['abs(aa_cm)'] = valid_merge_df['log_ratio_aa'] + valid_merge_df['log_ratio_dm_abs']
          for score_nm in struct_score_ensemble_names:
            valid_merge_sub_df = valid_merge_df.loc[valid_merge_df[score_nm].notnull(),:]
            rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(valid_merge_sub_df['fitness'].to_numpy().reshape(-1),valid_merge_sub_df[score_nm].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
            one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,score_nm,'valid']
            metric_score_list.append(one_score_case)

          ## test mutations, join on common variants
          set_sota_pred_df = sota_pred_df.loc[(sota_pred_df['protein_name'] == set_nm) & (sota_pred_df['gt'].notnull()) & (~sota_pred_df['mutant'].isin(valid_mut_df['mut_name']))]
          sota_myMdl_joint_df = set_sota_pred_df.merge(my_pred_df, on=['mutant','protein_name'], how='inner')
          assert len(set_sota_pred_df)-len(sota_myMdl_joint_df) == 0
          for score_nm in struct_score_names:
            sota_myMdl_joint_sub_df = sota_myMdl_joint_df.loc[sota_myMdl_joint_df[score_nm].notnull(),:]
            rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_joint_sub_df['gt'].to_numpy().reshape(-1),sota_myMdl_joint_sub_df[score_nm].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
            one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,score_nm,'test']
            metric_score_list.append(one_score_case)
          ## lable-test set: ss+rsa+contMap
          sota_myMdl_joint_df['ss_rsa_cm'] = sota_myMdl_joint_df['log_ratio_ss'] + sota_myMdl_joint_df['log_ratio_rsa'] + sota_myMdl_joint_df['log_ratio_dm']
          sota_myMdl_joint_df['abs(ss_rsa_cm)'] = sota_myMdl_joint_df['log_ratio_ss_abs'] + sota_myMdl_joint_df['log_ratio_rsa_abs'] + sota_myMdl_joint_df['log_ratio_dm_abs']
          ## lable-test set: ssN+rsaN+contMap
          sota_myMdl_joint_df['ssN_rsaN_cm'] = sota_myMdl_joint_df['log_ratio_ss_env'] + sota_myMdl_joint_df['log_ratio_rsa_env'] + sota_myMdl_joint_df['log_ratio_dm']
          sota_myMdl_joint_df['abs(ssN_rsaN_cm)'] = sota_myMdl_joint_df['log_ratio_ss_env_abs'] + sota_myMdl_joint_df['log_ratio_rsa_env_abs'] + sota_myMdl_joint_df['log_ratio_dm_abs']
          ## lable-test set: aa+ss+rsa+contMap
          sota_myMdl_joint_df['aa_ss_rsa_cm'] = sota_myMdl_joint_df['log_ratio_aa'] + sota_myMdl_joint_df['log_ratio_ss'] + sota_myMdl_joint_df['log_ratio_rsa'] + sota_myMdl_joint_df['log_ratio_dm']
          sota_myMdl_joint_df['abs(aa_ss_rsa_cm)'] = sota_myMdl_joint_df['log_ratio_aa'] + sota_myMdl_joint_df['log_ratio_ss_abs'] + sota_myMdl_joint_df['log_ratio_rsa_abs'] + sota_myMdl_joint_df['log_ratio_dm_abs']
          ## lable-test set: aa+ssN+rsaN+contMap
          sota_myMdl_joint_df['aa_ssN_rsaN_cm'] = sota_myMdl_joint_df['log_ratio_aa'] + sota_myMdl_joint_df['log_ratio_ss_env'] + sota_myMdl_joint_df['log_ratio_rsa_env'] + sota_myMdl_joint_df['log_ratio_dm']
          sota_myMdl_joint_df['abs(aa_ssN_rsaN_cm)'] = sota_myMdl_joint_df['log_ratio_aa'] + sota_myMdl_joint_df['log_ratio_ss_env_abs'] + sota_myMdl_joint_df['log_ratio_rsa_env_abs'] + sota_myMdl_joint_df['log_ratio_dm_abs']
          ## lable-test set: aa+contMap
          sota_myMdl_joint_df['aa_cm'] = sota_myMdl_joint_df['log_ratio_aa'] + sota_myMdl_joint_df['log_ratio_dm']
          sota_myMdl_joint_df['abs(aa_cm)'] = sota_myMdl_joint_df['log_ratio_aa'] + sota_myMdl_joint_df['log_ratio_dm_abs']
          for score_nm in struct_score_ensemble_names:
            sota_myMdl_joint_sub_df = sota_myMdl_joint_df.loc[sota_myMdl_joint_df[score_nm].notnull(),:]
            rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(sota_myMdl_joint_sub_df['gt'].to_numpy().reshape(-1),sota_myMdl_joint_sub_df[score_nm].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
            one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,score_nm,'test']
            metric_score_list.append(one_score_case)

    metric_structure_score_multitask_allMSASeq_df = pd.DataFrame(metric_score_list,columns=['protein_name','lambda','epoch','rho','rho_P','auroc','auprc','score_nm','eval_set'])
    metric_structure_score_multitask_allMSASeq_df.to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_structure_scores_multitaskFT_allMSASeq.csv',index=False,header=True)

    ## seq+struct finetune, seq w/ structure; structure-based mutation scoring
    # metric_score_list = []
    # #raw_score_list = []
    # for set_nm in set_names:
    #   ## load validation mutations
    #   valid_mut_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/valid_mutation.csv',header=0)
    #   fit_gt_df = pd.read_csv(f'{path}/data_process/mutagenesis/mutagenesisData/set_data/{set_nm}/mut_fit_gt.csv',header=0)
    #   valid_fit_gt_df = fit_gt_df.loc[fit_gt_df['mut_name'].isin(valid_mut_df['mut_name']) & (fit_gt_df['fitness'].notnull())]
    #   prot_nm = fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,:].iloc[0]['Shin2021_set']
    #   bina_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM']==set_nm,:].iloc[0]['DMS_binarization_cutoff']
    #   sign_reverse = set_nm in sign_reverse_set_names
    #   for lmd in lambda_list:
    #     ## get model name
    #     model_path = model_path_multitaskFT_df.loc[(model_path_multitaskFT_df['mut_nm']==set_nm) & (model_path_multitaskFT_df['lmd']==lmd),'mdl_path'].iloc[0]
    #     for epoch in epoch_joint_ft_list:
    #       print(f'seq+structFT (seq w/ structure;struct-based score),{set_nm},{lmd},{epoch}')
    #       ## load my predicted fitness scores
    #       #var_name,fit_true,aa_fit,log_ratio_ss,log_ratio_rsa,log_ratio_ss_env,log_ratio_rsa_env,log_ratio_dm
    #       my_pred_df = pd.read_csv(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis_structure/predictions/{model_path}/{set_nm}_{epoch}_structProp_rawScores.json',header=0,delimiter=',').rename(columns={'var_name': 'mutant','aa_fit': 'log_ratio_aa'}).drop_duplicates().groupby(['mutant'],as_index=False)[['fit_true','log_ratio_aa','log_ratio_ss','log_ratio_rsa','log_ratio_ss_env','log_ratio_rsa_env','log_ratio_dm']].mean()
    #       my_pred_df['epoch'] = epoch
    #       my_pred_df['lmd'] = lmd
    #       my_pred_df['protein_name'] = set_nm
    #       #raw_score_list.append(my_pred_df)
    
    #       ## metrics calculation on validation mutations
    #       valid_merge_df = valid_fit_gt_df.merge(my_pred_df, left_on=['mut_name'], right_on=['mutant'], how='inner')
    #       for score_nm in struct_score_names:
    #         rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(valid_merge_df['fitness'].to_numpy().reshape(-1),valid_merge_df['score_nm'].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
    #         one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,score_nm,'valid']
    #         metric_score_list.append(one_score_case)
            
    #       ## test mutations, join on common variants
    #       set_sota_pred_df = sota_pred_df.loc[(sota_pred_df['protein_name'] == set_nm) & (sota_pred_df['gt'].notnull()) & (~sota_pred_df['mutant'].isin(valid_mut_df['mut_name']))]
    #       sota_myMdl_joint_df = set_sota_pred_df.merge(my_pred_df, on=['mutant','protein_name'], how='inner')
    #       assert len(set_sota_pred_df)-len(sota_myMdl_joint_df) == 0
    #       for score_nm in struct_score_names:
    #         rho_score,rho_p_value,auroc,auprc = rho_auroc_prc(valid_merge_df['fitness'].to_numpy().reshape(-1),valid_merge_df['score_nm'].to_numpy().reshape(-1),bina_cutoff,sign_reverse)
    #         one_score_case = [set_nm,lmd,epoch,rho_score,rho_p_value,auroc,auprc,score_nm,'valid']
    #         metric_score_list.append(one_score_case)
    # metric_structure_score_multitask_df = pd.DataFrame(metric_score_list,columns=['protein_name','lambda','epoch','rho','rho_P','auroc','auprc','score_nm','eval_set'])
    # metric_structure_score_multitask_df.to_csv('eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/metric_structure_scores_multitaskFT.csv',index=False,header=True)
    
  if 'sota_protein_track_valid_best' in fig_name_list:
    ## joint-model, allMSASeq, fit-valid
    test_score_allMSASeq_list = []
    for score_nm in struct_score_names+struct_score_ensemble_names:
      test_one_score_allMSASeq_list = []
      for metric_nm in ['rho', 'auroc', 'auprc']:
        valid_best_idx = metric_structure_score_multitask_allMSASeq_df.loc[(metric_structure_score_multitask_allMSASeq_df['eval_set'] == 'valid') & (metric_structure_score_multitask_allMSASeq_df['score_nm'] == score_nm)].groupby(['protein_name'],as_index=False)[metric_nm].transform(max) == metric_structure_score_multitask_allMSASeq_df.loc[(metric_structure_score_multitask_allMSASeq_df['eval_set'] == 'valid') & (metric_structure_score_multitask_allMSASeq_df['score_nm'] == score_nm)][metric_nm].to_frame()
        valid_best_df = metric_structure_score_multitask_allMSASeq_df.iloc[valid_best_idx.loc[valid_best_idx[metric_nm]==True].index].drop_duplicates(subset=['protein_name'], keep='last')
        test_score_allMSASeq_df = valid_best_df.merge(metric_structure_score_multitask_allMSASeq_df.loc[(metric_structure_score_multitask_allMSASeq_df['eval_set'] == 'test') & (metric_structure_score_multitask_allMSASeq_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch','score_nm'], how='inner',suffixes=('_valid','_test'))

        test_score_allMSASeq_df= test_score_allMSASeq_df[['protein_name',f'{metric_nm}_test','score_nm']].rename(columns={f'{metric_nm}_test': metric_nm})
        test_score_allMSASeq_df['mdl_name'] = 'Our SI-pLM (seq_all;fit_valid_best)'
        test_one_score_allMSASeq_list.append(test_score_allMSASeq_df)
      ## merge dfs of three metrics
      test_all_score_allMSASeq_df = test_one_score_allMSASeq_list[0].merge(test_one_score_allMSASeq_list[1],on=['protein_name','mdl_name','score_nm'])
      test_all_score_allMSASeq_df = test_all_score_allMSASeq_df.merge(test_one_score_allMSASeq_list[2],on=['protein_name','mdl_name','score_nm'])
      test_score_allMSASeq_list.append(test_all_score_allMSASeq_df)
    ## merge dfs of all score types
    test_all_score_allMSASeq_df = pd.concat(test_score_allMSASeq_list,axis=0)
      

    ## joint-model, seq w/ structure; fit-valid
    # test_score_list = []
    # for score_nm in struct_score_names:
    #   for metric_nm in ['rho', 'auroc', 'auprc']:
    #     valid_best_idx = metric_structure_score_multitask_df.loc[(metric_structure_score_multitask_df['eval_set'] == 'valid') & (metric_structure_score_multitask_df['score_nm'] == score_nm)].groupby(['protein_name'],as_index=False)[metric_nm].transform(max) == metric_structure_score_multitask_df.loc[(metric_structure_score_multitask_df['eval_set'] == 'valid') & (metric_structure_score_multitask_df['score_nm'] == score_nm)][metric_nm].to_frame()
    #     valid_best_df = metric_structure_score_multitask_df.iloc[valid_best_idx.loc[valid_best_idx[metric_nm]==True].index].drop_duplicates(subset=['protein_name'], keep='last')
    #     test_score_df = valid_best_df.merge(metric_structure_score_multitask_df.loc[(metric_structure_score_multitask_df['eval_set'] == 'test') & (metric_structure_score_multitask_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch'], how='inner',suffixes=('_valid','_test'))
    #     ## output best (lambda, epoch) selected by validation set
    #     print(f'***seq w/ structure, fit_valid_best, {metric_nm}: best lambda, epoch***')
    #     #print(test_score_df[['protein_name','lambda','epoch']].to_string())
    #     print(test_score_df['lambda'].value_counts().to_string())
    #     #print(test_score_df['epoch'].value_counts().to_string())

    #     test_score_df= test_score_df[['protein_name',f'{metric_nm}_test','score_nm']].rename(columns={f'{metric_nm}_test': metric_nm})
    #     test_score_df['mdl_name'] = 'Our SI-pLM (seqWStruct;fit_valid_best)'
    #     test_score_list.append(test_score_df)
    # ## merge dfs of three metrics
    # test_all_score_df = test_score_list[0].merge(test_score_list[1],on=['protein_name','mdl_name','score_nm'])
    # test_all_score_df = test_all_score_df.merge(test_score_list[2],on=['protein_name','mdl_name','score_nm'])
    
    metric_score_all_df = pd.concat([test_all_score_allMSASeq_df])

    ## average Rho, AUC
    ave_score_list = []
    for metric_nm in ['rho','auroc','auprc']:
      ave_sota_df = metric_score_all_df.groupby(['mdl_name','score_nm'],as_index=False)[metric_nm].mean().sort_values(by=[metric_nm],ascending=False)
      print(f'***average {metric_nm}***')
      print(ave_sota_df.to_string())
      ave_score_list.append(ave_sota_df)
    ave_score_df = ave_score_list[0].merge(ave_score_list[1],on=['mdl_name','score_nm'])
    ave_score_df = ave_score_df.merge(ave_score_list[2],on=['mdl_name','score_nm'])
    ## latex friendly prints
    mdl_name = 'Our SI-pLM (seq_all;fit_valid_best)'
    for score_nm in struct_score_names+struct_score_ensemble_names:
      ave_one_score_df = ave_score_df.loc[(ave_score_df['mdl_name']==mdl_name) & (ave_score_df['score_nm']==score_nm),:].iloc[0]
      print(f"{mdl_name}, {score_nm}: &{ave_one_score_df['rho']:.3f}&{ave_one_score_df['auroc']:.3f}&{ave_one_score_df['auprc']:.3f}")
  elif 'sota_protein_track_valid_best_2' in fig_name_list:
    ## always use log_ratio_aa to select best model
    ## joint-model, allMSASeq, fit-valid
    test_score_allMSASeq_list = []
    for score_nm in struct_score_names+struct_score_ensemble_names:
      test_one_score_allMSASeq_list = []
      for metric_nm in ['rho', 'auroc', 'auprc']:
        valid_best_idx = metric_structure_score_multitask_allMSASeq_df.loc[(metric_structure_score_multitask_allMSASeq_df['eval_set'] == 'valid') & (metric_structure_score_multitask_allMSASeq_df['score_nm'] == 'log_ratio_aa')].groupby(['protein_name'],as_index=False)[metric_nm].transform(max) == metric_structure_score_multitask_allMSASeq_df.loc[(metric_structure_score_multitask_allMSASeq_df['eval_set'] == 'valid') & (metric_structure_score_multitask_allMSASeq_df['score_nm'] == 'log_ratio_aa')][metric_nm].to_frame()
        valid_best_df = metric_structure_score_multitask_allMSASeq_df.iloc[valid_best_idx.loc[valid_best_idx[metric_nm]==True].index].drop_duplicates(subset=['protein_name'], keep='last')[['protein_name','lambda','epoch']]
        test_score_allMSASeq_df = valid_best_df.merge(metric_structure_score_multitask_allMSASeq_df.loc[(metric_structure_score_multitask_allMSASeq_df['eval_set'] == 'test') & (metric_structure_score_multitask_allMSASeq_df['score_nm'] == score_nm)], on=['protein_name','lambda','epoch'], how='inner')

        test_score_allMSASeq_df= test_score_allMSASeq_df[['protein_name',f'{metric_nm}','score_nm']]
        test_score_allMSASeq_df['mdl_name'] = 'Our SI-pLM (seq_all;fit_valid_best)'
        test_one_score_allMSASeq_list.append(test_score_allMSASeq_df)
      ## merge dfs of three metrics
      test_all_score_allMSASeq_df = test_one_score_allMSASeq_list[0].merge(test_one_score_allMSASeq_list[1],on=['protein_name','mdl_name','score_nm'])
      test_all_score_allMSASeq_df = test_all_score_allMSASeq_df.merge(test_one_score_allMSASeq_list[2],on=['protein_name','mdl_name','score_nm'])
      test_score_allMSASeq_list.append(test_all_score_allMSASeq_df)
    ## merge dfs of all score types
    test_all_score_allMSASeq_df = pd.concat(test_score_allMSASeq_list,axis=0)
    metric_score_all_df = pd.concat([test_all_score_allMSASeq_df])

    ## average Rho, AUC
    ave_score_list = []
    for metric_nm in ['rho','auroc','auprc']:
      ave_sota_df = metric_score_all_df.groupby(['mdl_name','score_nm'],as_index=False)[metric_nm].mean().sort_values(by=[metric_nm],ascending=False)
      print(f'***average {metric_nm}***')
      print(ave_sota_df.to_string())
      ave_score_list.append(ave_sota_df)
    ave_score_df = ave_score_list[0].merge(ave_score_list[1],on=['mdl_name','score_nm'])
    ave_score_df = ave_score_df.merge(ave_score_list[2],on=['mdl_name','score_nm'])
    ## latex friendly prints
    mdl_name = 'Our SI-pLM (seq_all;fit_valid_best)'
    for score_nm in struct_score_names+struct_score_ensemble_names:
      ave_one_score_df = ave_score_df.loc[(ave_score_df['mdl_name']==mdl_name) & (ave_score_df['score_nm']==score_nm),:].iloc[0]
      print(f"{mdl_name}, {score_nm}: &{ave_one_score_df['rho']:.3f}&{ave_one_score_df['auroc']:.3f}&{ave_one_score_df['auprc']:.3f}")

def mutation_embedding_umap(draw_fig: bool=True, load_umap_model: bool=False):
  """generate umaps for mutation embeddings
  """
  path = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  model_path_multitaskFT_allMSASeq_df = pd.read_csv(f'{path}/eval_results/multitask_fitness_UNsupervise_mutagenesis/figures/model_path_multitaskFT_allMSASeq.csv',delimiter=',',header=0)
  
  #setNM,SI_FT_all_fitValid_lambda,SI_FT_all_fitValid_epoch,SI_FT_all_preValid_lambda,SI_FT_all_preValid_epoch,SI_FT_sub_fitValid_lambda,SI_FT_sub_fitValid_epoch,SI_FT_sub_preValid_lambda,SI_FT_sub_preValid_epoch
  best_lambda_epoch_multitaskFT_df = pd.read_csv(f'{path}/eval_results/embedding_analysis/setups/mutaSet_best_hyper.csv',delimiter=',',header=0)
  
  fitness_fam_info_df = pd.read_csv(f'{path}/data_process/mutagenesis/DeepSequenceMutaSet_reference_file_new.csv',delimiter=',',header=0)

  metric_nm_list = ['cosine'] #'euclidean','manhattan','chebyshev','minkowski','canberra','braycurtis','cosine','correlation'
  n_neighbor_list = [5] #2,5,10,15,25,100,8,12,15
  min_dist_list = [1.99]    #0,0.1,0.5,1.5,
  
  score_collect_list = [] # save cohesion and seperation scores

  for set_nm in ['AMIE_PSEAE_Whitehead','B3VI55_LIPSTSTABLE','B3VI55_LIPST_Whitehead2015','BG_STRSQ_hmmerbit','BLAT_ECOLX_Ostermeier2014','BLAT_ECOLX_Palzkill2012','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Tenaillon2013','BRCA1_HUMAN_BRCT','BRCA1_HUMAN_RING','CALM1_HUMAN_Roth2017','DLG4_RAT_Ranganathan2012','GAL4_YEAST_Shendure2015','HSP82_YEAST_Bolon2016','IF1_ECOLI_Kishony','KKA2_KLEPN_Mikkelsen2014','MK01_HUMAN_Johannessen','MTH3_HAEAESTABILIZED_Tawfik2015','P84126_THETH_b0','PABP_YEAST_Fields2013-doubles','PABP_YEAST_Fields2013-singles','PTEN_HUMAN_Fowler2018','RASH_HUMAN_Kuriyan','RL401_YEAST_Bolon2013','RL401_YEAST_Bolon2014','RL401_YEAST_Fraser2016','SUMO1_HUMAN_Roth2017','TIM_SULSO_b0','TIM_THEMA_b0','TPK1_HUMAN_Roth2017','TPMT_HUMAN_Fowler2018','UBC9_HUMAN_Roth2017','UBE4B_MOUSE_Klevit2013-singles','YAP1_HUMAN_Fields2012-singles']:
    # 'MTH3_HAEAESTABILIZED_Tawfik2015','BLAT_ECOLX_Ostermeier2014','B3VI55_LIPSTSTABLE','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Tenaillon2013','RL401_YEAST_Fraser2016','UBE4B_MOUSE_Klevit2013-singles'
    #,'HIS7_YEAST_Kondrashov2017'
    #'AMIE_PSEAE_Whitehead','B3VI55_LIPSTSTABLE','B3VI55_LIPST_Whitehead2015','BG_STRSQ_hmmerbit','BLAT_ECOLX_Ostermeier2014','BLAT_ECOLX_Palzkill2012','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Tenaillon2013','BRCA1_HUMAN_BRCT','BRCA1_HUMAN_RING','CALM1_HUMAN_Roth2017','DLG4_RAT_Ranganathan2012','GAL4_YEAST_Shendure2015','HIS7_YEAST_Kondrashov2017','HSP82_YEAST_Bolon2016','IF1_ECOLI_Kishony','KKA2_KLEPN_Mikkelsen2014','MK01_HUMAN_Johannessen','MTH3_HAEAESTABILIZED_Tawfik2015','P84126_THETH_b0','PABP_YEAST_Fields2013-doubles','PABP_YEAST_Fields2013-singles','PTEN_HUMAN_Fowler2018','RASH_HUMAN_Kuriyan','RL401_YEAST_Bolon2013','RL401_YEAST_Bolon2014','RL401_YEAST_Fraser2016','SUMO1_HUMAN_Roth2017','TIM_SULSO_b0','TIM_THEMA_b0','TPK1_HUMAN_Roth2017','TPMT_HUMAN_Fowler2018','UBC9_HUMAN_Roth2017','UBE4B_MOUSE_Klevit2013-singles','YAP1_HUMAN_Fields2012-singles'
    print(f'>>>{set_nm}<<<',flush=True)
    ## seq-FT model path
    model_path_list = os.popen(f"grep -E -o 'masked_language_modeling_transformer.*[0-9]{{6}}' {path}/job_logs/archive_baseline_bert_eval/baseline_bert_mutation_fitness_UNsupervise_mutagenesis_rp15_all_2_729.seq_finetune.reweighted.allSeq.TotalEpoch40.Interval*.0.*.{set_nm}.seqMaxL512.classWno.mutation_embedding_umap.out | uniq").read().strip('\n').split('\n')
    assert len(model_path_list) == 1
    model_path_seq_pt = model_path_list[0]
    print(f'**pt: {model_path_seq_pt}**')

    model_path_list = os.popen(f"grep -E -o 'masked_language_modeling_transformer.*[0-9]{{6}}' {path}/job_logs/archive_baseline_bert_eval/baseline_bert_mutation_fitness_UNsupervise_mutagenesis_rp15_all_2_729.seq_finetune.reweighted.allSeq.TotalEpoch40.Interval*.best.*.{set_nm}.seqMaxL512.classWno.mutation_embedding_umap.out | uniq").read().strip('\n').split('\n')
    assert len(model_path_list) == 1
    model_path_seq_ft = model_path_list[0]
    print(f'**ft: {model_path_seq_ft}**')

    best_lambda, best_epoch = best_lambda_epoch_multitaskFT_df.loc[best_lambda_epoch_multitaskFT_df['setNM']==set_nm,['SI_FT_all_fitValid_lambda','SI_FT_all_fitValid_epoch']].iloc[0].to_list()
    model_path_si_ft = model_path_multitaskFT_allMSASeq_df.loc[(model_path_multitaskFT_allMSASeq_df['mut_nm'] == set_nm) & (model_path_multitaskFT_allMSASeq_df['lmd'] == best_lambda),'mdl_path'].iloc[0]
    print(f'**si-ft: {model_path_si_ft}**')

    bina_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,'DMS_binarization_cutoff'].iloc[0]
    first_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,'first_cutoff'].iloc[0]
    second_cutoff = fitness_fam_info_df.loc[fitness_fam_info_df['setNM'] == set_nm,'second_cutoff'].iloc[0]

    for embed_mode in ['all_pos_ave']: #'all_pos_ave','mut_pos_ave', 'all_pos_ave_AA_head','mut_pos_ave_AA_head'
      value_dict = {}

      ## load embeddings
      embedding_list, gt_fit_list = [], []
      pt_embeddings = np.load(f'eval_results/embedding_analysis/embedding_save/{model_path_seq_pt}/{set_nm}_0/{embed_mode}.npy')
      embedding_list.append(pt_embeddings)
      fit_gt_arr = np.load(f'eval_results/embedding_analysis/embedding_save/{model_path_seq_pt}/{set_nm}_0/fit_gt.npy')
      #gt_fit_list.append(np.where(fit_gt_arr<bina_cutoff,0,1))
      gt_fit_list.append(fit_gt_arr)
      print(f'PT: {embedding_list[-1].shape}')
      
      ft_embeddings = np.load(f'eval_results/embedding_analysis/embedding_save/{model_path_seq_ft}/{set_nm}_best/{embed_mode}.npy')
      embedding_list.append(ft_embeddings)
      fit_gt_arr = np.load(f'eval_results/embedding_analysis/embedding_save/{model_path_seq_ft}/{set_nm}_best/fit_gt.npy')
      #gt_fit_list.append(np.where(fit_gt_arr<bina_cutoff,0,1))
      gt_fit_list.append(fit_gt_arr)
      print(f'FT: {embedding_list[-1].shape}')

      sift_embeddings = np.load(f'eval_results/embedding_analysis/embedding_save/{model_path_si_ft}/{set_nm}_{best_epoch}/{embed_mode}.npy')
      embedding_list.append(sift_embeddings)
      fit_gt_arr = np.load(f'eval_results/embedding_analysis/embedding_save/{model_path_si_ft}/{set_nm}_{best_epoch}/fit_gt.npy')
      #gt_fit_list.append(np.where(fit_gt_arr<bina_cutoff,0,1))
      gt_fit_list.append(fit_gt_arr)
      print(f'SI-FT: {embedding_list[-1].shape}')

      value_dict[embed_mode] = np.concatenate(embedding_list, axis=0) 
      value_dict['fit_gt'] = np.concatenate(gt_fit_list, axis=0)
      
      ## draw figures
      embedding_fig_path = f'eval_results/embedding_analysis/embedding_figure/{set_nm}'

      ## separate umap
      # for metric_nm in metric_nm_list:
      #   for n_neighbor in n_neighbor_list:
      #     for min_dist in min_dist_list:
      #       umap_corr_obj = umap.UMAP(n_components=2, n_neighbors=n_neighbor, metric=metric_nm, min_dist=min_dist, spread=2.0, random_state=420, verbose=True)
      #       pt_umap = umap_corr_obj.fit_transform(pt_embeddings)
      #       ft_umap = umap_corr_obj.fit_transform(ft_embeddings)
      #       sift_umap = umap_corr_obj.fit_transform(sift_embeddings)

      #       params = {'legend.fontsize': 10,
      #               'figure.figsize': (12, 12),
      #               'axes.labelsize': 14,
      #               'axes.titlesize': 14,
      #               'xtick.labelsize': 10,
      #               'ytick.labelsize': 10,}
      #       #pylab.rcParams.update(params)
      #       fig, ax = plt.subplots()
      #       normalize = colors.Normalize()
      #       #plt.scatter(pt_umap[:, 0], pt_umap[:, 1], c=fit_gt_arr, cmap="Blues", s=10, marker='.',label='PT')
      #       #plt.scatter(ft_umap[:, 0], ft_umap[:, 1], c=fit_gt_arr, cmap="Greens", s=10, marker='.',label='FT')
      #       plt.scatter(sift_umap[:, 0], sift_umap[:, 1], c=fit_gt_arr, cmap="Reds", s=10, marker='.',label='SIFT')
      #       #plt.colorbar()
      #       plt.legend()
      #       plt.savefig(f'{embedding_fig_path}/{metric_nm}_{n_neighbor}_{min_dist}_{embed_mode}_umap_indiv.png',dpi=600, bbox_inches='tight')
      #       plt.clf()

      def sse_ssb_l2(fit_gt,embed,mdl_name,embed_space):
        embed_size = embed.shape[1]
        cluster_num = 2
        nega_idx = np.argwhere(fit_gt <= first_cutoff).reshape(-1)
        posi_idx = np.argwhere(fit_gt > second_cutoff).reshape(-1)
        nega_cluster = embed[nega_idx]
        posi_cluster = embed[posi_idx]
        whole_cluster = np.concatenate((nega_cluster,posi_cluster),axis=0)
        nega_sse = np.sum(np.power(nega_cluster - np.mean(nega_cluster,axis=0),2))/nega_cluster.shape[0]
        posi_sse = np.sum(np.power(posi_cluster - np.mean(posi_cluster,axis=0),2))/posi_cluster.shape[0]
        ssb = (len(nega_cluster)*np.sum((np.mean(nega_cluster,axis=0) - np.mean(whole_cluster,axis=0))**2) + len(posi_cluster)*np.sum((np.mean(posi_cluster,axis=0) - np.mean(whole_cluster,axis=0))**2))/whole_cluster.shape[0]
        ## ch_score(larger)
        ch_score = (ssb*cluster_num)/((nega_sse+posi_sse)*(cluster_num-1))
        ## xb_score (smaller)
        xb_score = (nega_sse + posi_sse) / np.sum((np.mean(nega_cluster,axis=0) - np.mean(posi_cluster,axis=0))**2)
        ## bh_score (smaller)
        bh_score = (nega_sse + posi_sse) / cluster_num
        ## H_score (larger)
        h_score = np.log(ssb/(nega_sse+posi_sse))
        ## Xu_score (smaller)
        xu_score = embed_size*np.log2(np.sqrt((nega_sse+posi_sse)/(embed_size*(len(whole_cluster)**2)))) + np.log(cluster_num)
        ## silhouette (close to 1)
        nega_N = len(nega_cluster)
        if nega_N > 1e5:
          nega_cluster = nega_cluster.astype(np.int8)
          posi_cluster = posi_cluster.astype(np.int8)
        else:
          pass
        nega_repeat_a = np.repeat(nega_cluster.reshape(1,nega_N,embed_size),nega_N,axis=0) # [n_c,n_c,e_s]
        nega_repeat_b = np.repeat(nega_cluster,nega_N,axis=0).reshape(nega_N,nega_N,embed_size)
        nega_intra_dist = np.sum(np.sqrt(np.sum((nega_repeat_a - nega_repeat_b)**2,axis=2)),axis=1)/nega_N
        posi_N = len(posi_cluster)
        posi_repeat_a = np.repeat(posi_cluster.reshape(1,posi_N,embed_size),posi_N,axis=0) # [n_c,n_c,e_s]
        posi_repeat_b = np.repeat(posi_cluster,posi_N,axis=0).reshape(posi_N,posi_N,embed_size)
        posi_intra_dist = np.sum(np.sqrt(np.sum((posi_repeat_a - posi_repeat_b)**2,axis=2)),axis=1)/posi_N

        nega_repeat_posi_a = np.repeat(posi_cluster.reshape(1,posi_N,embed_size),nega_N,axis=0) # [N_nega,N_posi,2]
        nega_repeat_posi_b = np.repeat(nega_cluster,posi_N,axis=0).reshape(nega_N,posi_N,embed_size)
        nega_inter_dist = np.sum(np.sqrt(np.sum((nega_repeat_posi_a - nega_repeat_posi_b)**2,axis=2)),axis=1)/posi_N

        posi_repeat_nega_a = np.repeat(nega_cluster.reshape(1,nega_N,embed_size),posi_N,axis=0) # [N_posi,N_nega,2]
        posi_repeat_nega_b = np.repeat(posi_cluster,nega_N,axis=0).reshape(posi_N,nega_N,embed_size)
        posi_inter_dist = np.sum(np.sqrt(np.sum((posi_repeat_nega_a - posi_repeat_nega_b)**2,axis=2)),axis=1)/nega_N

        nega_conca = np.vstack((nega_intra_dist,nega_inter_dist))
        nega_final_s = (nega_inter_dist - nega_intra_dist) / np.max(nega_conca,axis=0)

        posi_conca = np.vstack((posi_intra_dist,posi_inter_dist))
        posi_final_s = (posi_inter_dist - posi_intra_dist) / np.max(posi_conca,axis=0)

        s_score = np.mean(np.concatenate((nega_final_s,posi_final_s)))

        return_list = [set_nm,embed_mode,metric_nm,n_neighbor,min_dist,mdl_name,'l2',embed_space,nega_sse,posi_sse,ssb,ch_score,xb_score,bh_score,h_score,xu_score,s_score]
        return return_list

      def sse_ssb_cos(fit_gt,embed,mdl_name,embed_space):
        embed_size = embed.shape[1]
        cluster_size = 2
        nega_idx = np.argwhere(fit_gt <= first_cutoff).reshape(-1)
        posi_idx = np.argwhere(fit_gt > second_cutoff).reshape(-1)
        nega_cluster = embed[nega_idx]
        posi_cluster = embed[posi_idx]
        whole_cluster = np.concatenate((nega_cluster,posi_cluster))
        nega_sse = np.mean(cosine_similarity(nega_cluster, np.mean(nega_cluster,axis=0).reshape(-1,embed_size)))
        posi_sse = np.mean(cosine_similarity(posi_cluster, np.mean(posi_cluster,axis=0).reshape(-1,embed_size)))
        ssb = (len(nega_cluster)*cosine_similarity(np.mean(whole_cluster,axis=0).reshape(-1,embed_size), np.mean(nega_cluster,axis=0).reshape(-1,embed_size)).item() + len(posi_cluster)*cosine_similarity(np.mean(whole_cluster,axis=0).reshape(-1,embed_size), np.mean(posi_cluster,axis=0).reshape(-1,embed_size)).item()) / whole_cluster.shape[0]
        ## ch_score(larger)
        ch_score = (ssb*cluster_size)/((nega_sse+posi_sse)*(cluster_size-1))
        ## xb_score (smaller)
        xb_score = (nega_sse + posi_sse) / cosine_similarity(np.mean(posi_cluster,axis=0).reshape(-1,embed_size), np.mean(nega_cluster,axis=0).reshape(-1,embed_size)).item()
        ## bh_score (smaller)
        bh_score = (nega_sse + posi_sse) / cluster_size
        ## H_score (larger)
        h_score = np.log(ssb/(nega_sse+posi_sse))
        ## Xu_score (smaller)
        xu_score = embed_size*np.log2(np.sqrt((nega_sse+posi_sse)/(embed_size*(len(whole_cluster)**2)))) + np.log(cluster_size)
        ## silhouette (close to 1)
        nega_N = len(nega_cluster)
        if nega_N > 1e5:
          nega_cluster = nega_cluster.astype(np.int8)
          posi_cluster = posi_cluster.astype(np.int8)
        else:
          pass
        cos_simi_pairwise = cosine_similarity(nega_cluster) # [N_nega,N_nega]
        np.fill_diagonal(cos_simi_pairwise, 0)
        nega_intra_dist = np.mean(cos_simi_pairwise,axis=1)

        cos_simi_pairwise = cosine_similarity(posi_cluster) # [N_nega,N_nega]
        np.fill_diagonal(cos_simi_pairwise, 0)
        posi_intra_dist = np.mean(cos_simi_pairwise,axis=1)

        cos_simi_pairwise = cosine_similarity(nega_cluster,posi_cluster) # [N_nega,N_posi]
        nega_inter_dist = np.mean(cos_simi_pairwise,axis=1)
        posi_inter_dist = np.mean(cos_simi_pairwise,axis=0)

        nega_conca = np.vstack((nega_intra_dist,nega_inter_dist))
        nega_final_s = (nega_inter_dist - nega_intra_dist) / np.max(nega_conca,axis=0)

        posi_conca = np.vstack((posi_intra_dist,posi_inter_dist))
        posi_final_s = (posi_inter_dist - posi_intra_dist) / np.max(posi_conca,axis=0)

        s_score = np.mean(np.concatenate((nega_final_s,posi_final_s)))

        return_list = [set_nm,embed_mode,metric_nm,n_neighbor,min_dist,mdl_name,'cos',embed_space,nega_sse,posi_sse,ssb,ch_score,xb_score,bh_score,h_score,xu_score,s_score]
        return return_list

      ## all in one umap
      for metric_nm in metric_nm_list:
        for n_neighbor in n_neighbor_list:
          for min_dist in min_dist_list:
            umap_mdl_path = f'eval_results/embedding_analysis/embedding_figure/{set_nm}/umap_mdl_{embed_mode}_{metric_nm}_{n_neighbor}_{min_dist}.sav'
            if load_umap_model:
              print('**load UMAP model**')
              umap_corr_obj = pkl.load((open(umap_mdl_path, 'rb')))
              embedding = umap_corr_obj.transform(value_dict[embed_mode])
            else:
              print('**train UMAP model**')
              umap_corr_obj = umap.UMAP(n_components=2, n_neighbors=n_neighbor, metric=metric_nm, min_dist=min_dist, spread=2.0, random_state=420, verbose=False)
              embedding = umap_corr_obj.fit_transform(value_dict[embed_mode])
              ## save UMAP model
              pkl.dump(umap_corr_obj, open(umap_mdl_path, 'wb'))
            
            num_var = embedding.shape[0] // 3
            ## calculate cohesion and seperation
            for embed_space in ['umap']: #,'origin'
              if embed_space == 'umap':
                embed_arr = embedding
              elif embed_space == 'origin':
                embed_arr = value_dict[embed_mode]
              print(f'**{embed_space},l2**')
              ## euclidean l2 norm
              PT_scores_l2 = sse_ssb_l2(value_dict['fit_gt'][:num_var],embed_arr[:num_var,:],'PT',embed_space)
              FT_scores_l2 = sse_ssb_l2(value_dict['fit_gt'][num_var:2*num_var],embed_arr[num_var:2*num_var,:],'FT',embed_space)
              SIFT_scores_l2 = sse_ssb_l2(value_dict['fit_gt'][2*num_var:],embed_arr[2*num_var:,:],'SIFT',embed_space)
              print(f'**{embed_space},cosine**')
              ## cosine similarity
              PT_scores_cos = sse_ssb_cos(value_dict['fit_gt'][:num_var],embed_arr[:num_var,:],'PT',embed_space)
              FT_scores_cos = sse_ssb_cos(value_dict['fit_gt'][num_var:2*num_var],embed_arr[num_var:2*num_var,:],'FT',embed_space)
              SIFT_scores_cos = sse_ssb_cos(value_dict['fit_gt'][2*num_var:],embed_arr[2*num_var:,:],'SIFT',embed_space)
            
              score_collect_list.append(PT_scores_l2)
              score_collect_list.append(FT_scores_l2)
              score_collect_list.append(SIFT_scores_l2)
              score_collect_list.append(PT_scores_cos)
              score_collect_list.append(FT_scores_cos)
              score_collect_list.append(SIFT_scores_cos)

            if draw_fig:
              params = {'legend.fontsize': 10,
                      'figure.figsize': (12, 12),
                      'axes.labelsize': 14,
                      'axes.titlesize': 14,
                      'xtick.labelsize': 10,
                      'ytick.labelsize': 10,}
              #pylab.rcParams.update(params)
              ## combined figure
              fig, ax = plt.subplots()
              normalize = colors.Normalize()
              plt.scatter(embedding[:num_var, 0], embedding[:num_var, 1], c=value_dict['fit_gt'][:num_var],cmap="Blues", s=10, marker='.',label='PT',norm=normalize)
              plt.scatter(embedding[num_var:2*num_var, 0], embedding[num_var:2*num_var, 1], c=value_dict['fit_gt'][num_var:2*num_var],cmap="Greens", s=10, marker='.',label='FT',norm=normalize)
              plt.scatter(embedding[2*num_var:, 0], embedding[2*num_var:, 1], c=value_dict['fit_gt'][2*num_var:], cmap="Reds", s=10, marker='.',label='SI-FT',norm=normalize)
              #plt.colorbar()
              leg = plt.legend(loc='upper right')
              leg.legend_handles[0].set_color('blue')
              leg.legend_handles[1].set_color('green')
              leg.legend_handles[2].set_color('red')
              plt.savefig(f'{embedding_fig_path}/{metric_nm}_{n_neighbor}_{min_dist}_{embed_mode}_umap_three.png',dpi=300, bbox_inches='tight')
              plt.clf()
              ## PT model
              fig, ax = plt.subplots()
              normalize = colors.Normalize()
              nega_idx = np.argwhere(value_dict['fit_gt'][:num_var] <= first_cutoff).reshape(-1)
              posi_idx = np.argwhere(value_dict['fit_gt'][:num_var] >= second_cutoff).reshape(-1)
              #color_list = np.where(value_dict['fit_gt'][:num_var] < bina_cutoff, 'cyan', 'blue')
              plt.scatter(embedding[:num_var, 0][nega_idx], embedding[:num_var, 1][nega_idx], c='cyan', s=10, marker='.',label='< WT')
              plt.scatter(embedding[:num_var, 0][posi_idx], embedding[:num_var, 1][posi_idx], c='dodgerblue', s=10, marker='.',label='> WT')
              #plt.colorbar()
              leg = plt.legend(loc='upper right')
              plt.savefig(f'{embedding_fig_path}/{metric_nm}_{n_neighbor}_{min_dist}_{embed_mode}_umap_PT.png',dpi=300, bbox_inches='tight')
              plt.clf()
              ## FT model
              fig, ax = plt.subplots()
              normalize = colors.Normalize()
              #color_list = np.where(value_dict['fit_gt'][num_var:2*num_var] < bina_cutoff, 'lime', 'green')
              nega_idx = np.argwhere(value_dict['fit_gt'][num_var:2*num_var] <= first_cutoff).reshape(-1)
              posi_idx = np.argwhere(value_dict['fit_gt'][num_var:2*num_var] >= second_cutoff).reshape(-1)
              plt.scatter(embedding[num_var:2*num_var, 0][nega_idx], embedding[num_var:2*num_var, 1][nega_idx], c='lime', s=10, marker='.',label='< WT')
              plt.scatter(embedding[num_var:2*num_var, 0][posi_idx], embedding[num_var:2*num_var, 1][posi_idx], c='green', s=10, marker='.',label='> WT')
              #plt.colorbar()
              leg = plt.legend(loc='upper right')
              plt.savefig(f'{embedding_fig_path}/{metric_nm}_{n_neighbor}_{min_dist}_{embed_mode}_umap_FT.png',dpi=300, bbox_inches='tight')
              plt.clf()
              ## SI-FT model
              fig, ax = plt.subplots()
              normalize = colors.Normalize()
              #color_list = np.where(value_dict['fit_gt'][2*num_var:] < bina_cutoff, 'lightcoral', 'red')
              nega_idx = np.argwhere(value_dict['fit_gt'][2*num_var:] <= first_cutoff).reshape(-1)
              posi_idx = np.argwhere(value_dict['fit_gt'][2*num_var:] >= second_cutoff).reshape(-1)
              plt.scatter(embedding[2*num_var:, 0][nega_idx], embedding[2*num_var:, 1][nega_idx], c='pink', s=10, marker='.',label='< WT')
              plt.scatter(embedding[2*num_var:, 0][posi_idx], embedding[2*num_var:, 1][posi_idx], c='red', s=10, marker='.',label='> WT')
              #plt.colorbar()
              leg = plt.legend(loc='upper right')
              plt.savefig(f'{embedding_fig_path}/{metric_nm}_{n_neighbor}_{min_dist}_{embed_mode}_umap_SIFT.png',dpi=300, bbox_inches='tight')
              plt.clf()
  
  score_collect_df = pd.DataFrame(score_collect_list,columns=['set_nm','embed_mode','metric_nm','n_neighbor','min_dist','mdl_name','distance_type','embed_space','nega_sse','posi_sse','ssb','ch_score','xb_score','bh_score','h_score','xu_score','s_score'])
  score_collect_df.to_csv(f'eval_results/embedding_analysis/embedding_figure/cohesion_seperation_noMiddle_{embed_mode}.csv')
  for distance_type in ['l2','cos']:
    for embed_space in ['umap']: #,'origin'
      print(f'**{distance_type},{embed_space}**')
      pt_mean = score_collect_df.loc[(score_collect_df['embed_mode']==embed_mode) & (score_collect_df['mdl_name'] == 'PT') & (score_collect_df['distance_type'] == distance_type) & (score_collect_df['embed_space'] == embed_space),['nega_sse','posi_sse','ssb','ch_score','xb_score','bh_score','h_score','xu_score','s_score']].mean()
      ft_mean = score_collect_df.loc[(score_collect_df['embed_mode']==embed_mode) & (score_collect_df['mdl_name'] == 'FT') & (score_collect_df['distance_type'] == distance_type) & (score_collect_df['embed_space'] == embed_space),['nega_sse','posi_sse','ssb','ch_score','xb_score','bh_score','h_score','xu_score','s_score']].mean()
      sift_mean = score_collect_df.loc[(score_collect_df['embed_mode']==embed_mode) & (score_collect_df['mdl_name'] == 'SIFT') & (score_collect_df['distance_type'] == distance_type) & (score_collect_df['embed_space'] == embed_space),['nega_sse','posi_sse','ssb','ch_score','xb_score','bh_score','h_score','xu_score','s_score']].mean()
      print('nega_sse(-),posi_sse(-),ssb(+),ch_score(+),xb_score(-),bh_score(-),h_score(+),xu_score(-),s_score(+)')
      print(f'PT: {pt_mean.to_string()}')
      print(f'FT: {ft_mean.to_string()}')
      print(f'SIFT: {sift_mean.to_string()}')
  return

def embed_quan_analysis():
  #'AMIE_PSEAE_Whitehead','B3VI55_LIPSTSTABLE','B3VI55_LIPST_Whitehead2015','BG_STRSQ_hmmerbit','BLAT_ECOLX_Ostermeier2014','BLAT_ECOLX_Palzkill2012','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Tenaillon2013','BRCA1_HUMAN_BRCT','BRCA1_HUMAN_RING','CALM1_HUMAN_Roth2017','DLG4_RAT_Ranganathan2012','GAL4_YEAST_Shendure2015','HSP82_YEAST_Bolon2016','IF1_ECOLI_Kishony','KKA2_KLEPN_Mikkelsen2014','MK01_HUMAN_Johannessen','MTH3_HAEAESTABILIZED_Tawfik2015','P84126_THETH_b0','PABP_YEAST_Fields2013-doubles','PABP_YEAST_Fields2013-singles','PTEN_HUMAN_Fowler2018','RASH_HUMAN_Kuriyan','RL401_YEAST_Bolon2013','RL401_YEAST_Bolon2014','RL401_YEAST_Fraser2016','SUMO1_HUMAN_Roth2017','TIM_SULSO_b0','TIM_THEMA_b0','TPK1_HUMAN_Roth2017','TPMT_HUMAN_Fowler2018','UBC9_HUMAN_Roth2017','UBE4B_MOUSE_Klevit2013-singles','YAP1_HUMAN_Fields2012-singles'
  target_set_dict = {
    'rho': ['RL401_YEAST_Bolon2013','SUMO1_HUMAN_Roth2017','RL401_YEAST_Fraser2016','RL401_YEAST_Bolon2014','PABP_YEAST_Fields2013-singles','BLAT_ECOLX_Ostermeier2014','PABP_YEAST_Fields2013-doubles','HSP82_YEAST_Bolon2016','TIM_SULSO_b0','UBE4B_MOUSE_Klevit2013-singles'],
    'auroc': ['SUMO1_HUMAN_Roth2017','RL401_YEAST_Fraser2016','RL401_YEAST_Bolon2014','PABP_YEAST_Fields2013-doubles','PABP_YEAST_Fields2013-singles','BLAT_ECOLX_Ostermeier2014','DLG4_RAT_Ranganathan2012','HSP82_YEAST_Bolon2016','TIM_SULSO_b0','RL401_YEAST_Bolon2013'],
    'auroc2': ['MTH3_HAEAESTABILIZED_Tawfik2015','KKA2_KLEPN_Mikkelsen2014','BRCA1_HUMAN_BRCT','AMIE_PSEAE_Whitehead','BLAT_ECOLX_Ostermeier2014','PTEN_HUMAN_Fowler2018','GAL4_YEAST_Shendure2015','B3VI55_LIPSTSTABLE','BLAT_ECOLX_Ranganathan2015','TPMT_HUMAN_Fowler2018'],
    'BLAT_ECOLX_Ostermeier2014': ['BLAT_ECOLX_Ostermeier2014'],
    'auroc3':['MTH3_HAEAESTABILIZED_Tawfik2015','KKA2_KLEPN_Mikkelsen2014','BRCA1_HUMAN_BRCT'],
    'both': ['MTH3_HAEAESTABILIZED_Tawfik2015','BLAT_ECOLX_Ostermeier2014','B3VI55_LIPSTSTABLE','BLAT_ECOLX_Ranganathan2015','BLAT_ECOLX_Tenaillon2013','RL401_YEAST_Fraser2016','UBE4B_MOUSE_Klevit2013-singles']
  }
  score_collect_df = pd.read_csv(f'eval_results/embedding_analysis/embedding_figure/cohesion_seperation_noMiddle.csv',header=0,index_col=0)
  for target_set_nm in ['both']:
    target_set = target_set_dict[target_set_nm]
    for distance_type in ['l2','cos']:
      for embed_space in ['umap']: #,'origin'
        print(f'**{target_set_nm},{distance_type},{embed_space}**')
        pt_mean = score_collect_df.loc[(score_collect_df['embed_mode']=='mut_pos_ave') & (score_collect_df['mdl_name'] == 'PT') & (score_collect_df['distance_type'] == distance_type) & (score_collect_df['embed_space'] == embed_space) & (score_collect_df['set_nm'].isin(target_set)),['nega_sse','posi_sse','ssb','ch_score','xb_score','bh_score','h_score','xu_score','s_score']].mean()
        ft_mean = score_collect_df.loc[(score_collect_df['embed_mode']=='mut_pos_ave') & (score_collect_df['mdl_name'] == 'FT') & (score_collect_df['distance_type'] == distance_type) & (score_collect_df['embed_space'] == embed_space) & (score_collect_df['set_nm'].isin(target_set)),['nega_sse','posi_sse','ssb','ch_score','xb_score','bh_score','h_score','xu_score','s_score']].mean()
        sift_mean = score_collect_df.loc[(score_collect_df['embed_mode']=='mut_pos_ave') & (score_collect_df['mdl_name'] == 'SIFT') & (score_collect_df['distance_type'] == distance_type) & (score_collect_df['embed_space'] == embed_space) & (score_collect_df['set_nm'].isin(target_set)),['nega_sse','posi_sse','ssb','ch_score','xb_score','bh_score','h_score','xu_score','s_score']].mean()
        if distance_type == 'l2':
          print('nega_sse(-),posi_sse(-),ssb(+),ch_score(+),xb_score(-),bh_score(-),h_score(+),xu_score(-),s_score(+)')
        else:
          print('nega_sse(+),posi_sse(+),ssb(-),ch_score(-),xb_score(+),bh_score(+),h_score(-),xu_score(+),s_score(-)')
        print('>>PT')
        print(f'{pt_mean.to_string()}')
        print('>>FT')
        print(f'{ft_mean.to_string()}')
        print('>>SIFT')
        print(f'{sift_mean.to_string()}')
  return

def add_wavenet_to_sota_df():
  """Add Wavenet predictions to sota summarized score file
  """
  path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  ## load sota predictions
  #header: protein_name,mutant,gt,MSA Transformer,PSSM,EVMutation (published),DeepSequence (published) - single,DeepSequence (published),ESM-1v (zero shot),ESM-1v (+further training),DeepSequence (replicated),EVMutation (replicated),ProtBERT-BFD,TAPE,UniRep,ESM-1b
  sota_pred_df = pd.read_csv(f'{path}/data_process/mutagenesis/esm_predictions/raw_df.csv',delimiter=',',header=0)
  # mutation_effect_prediction_all_mean_autoregressive
  wavenet_file_list = pd.read_csv(f'/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/calc_logprobs/output_file_list',delimiter=',',names=['file_name'])
  protein_name_list = sota_pred_df['protein_name'].unique().tolist()
  wavenet_pred_collects = []
  for pro_nm in protein_name_list:
    print(f'>>{pro_nm}')
    target_file_name = wavenet_file_list.loc[wavenet_file_list['file_name'].str.contains(pro_nm),'file_name']
    if len(target_file_name) > 0:
      assert len(target_file_name) == 1
      target_file_name = target_file_name.iloc[0]
      wavenet_pred_df = pd.read_csv(f'/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/calc_logprobs/output_processed/{target_file_name}',delimiter=',',header=0)
      wavenet_pred_df['protein_name'] = pro_nm
      wavenet_pred_df_sele = wavenet_pred_df[['protein_name','mutant','mutation_effect_prediction_all_mean_autoregressive']].drop_duplicates()
      wavenet_pred_collects.append(wavenet_pred_df_sele)
    else:
      continue
  wavenet_pred_collects_df = pd.concat(wavenet_pred_collects, axis=0, ignore_index=True).reset_index(drop=True).rename(columns={'mutation_effect_prediction_all_mean_autoregressive': 'Wavenet'})
  sota_with_wavenet_df = sota_pred_df.merge(wavenet_pred_collects_df,on=['protein_name','mutant'],how='left')
  sota_with_wavenet_df.to_csv(f'{path}/data_process/mutagenesis/esm_predictions/raw_with_wavenet_df.csv',sep=',',header=True,index=False)
  return 

if __name__ == '__main__':
  working_dir   = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  task2run = sys.argv[1]
  
  #proc_ind(working_dir)
  #label_statistic(working_dir)
  if task2run == 'prepare_DeepSeq_mutations':
    prepare_DeepSeq_mutations(working_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/mutagenesis',
                              onlyKeep_label_existing=False,
                              save_json=False)
  elif task2run == 'fireprot_family_query':
    fireprot_family_query()
  elif task2run == 'find_family_across_datasets':
    find_family_across_datasets()
  elif task2run == 'fireprot_fam_label_data':
    fireprot_fam_label_data(fam_unp_list=['P11436','B3VI55','P62593','P38398','P12497','Q56319','P46937'])
  elif task2run == 'thermostability_dataset':
    thermostability_dataset(root_path = working_dir)
  elif task2run == 'cdna_display_dataset':
    cdna_display_dataset(task='hmmscan')
  elif task2run == 'proteingym_domain_seq':
    proteingym_domain_seq()
  elif task2run == 'multitask_fitness_analysis':
    multitask_fitness_analysis(
      family_list=['DLG4_RAT','AMIE_PSEAE','PABP_YEAST','RASH_HUMAN','KKA2_KLEPN','PTEN_HUMAN','MTH3_HAEAESTABILIZED','HIS7_YEAST'],
      figure_list=['3_balance_epoch_graphDef_violin']) #3_balance_epoch_graphDef_violin,3_balance_epoch_graphDef_bar
  elif task2run == 'analysis_region':
    analysis_region(
      working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark',
      shin_muta_path = '/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/calc_logprobs',
      exp_label_set = 'deepsequence',
      sota_mdl_compare = ['mutation_effect_prediction_all_mean_autoregressive','mutation_effect_prediction_forward_mean','mutation_effect_prediction_reverse_mean','mutation_effect_prediction_forward_mean_channels-48','mutation_effect_prediction_reverse_mean_channels-48','mutation_effect_prediction_forward_mean_channels-24','mutation_effect_prediction_reverse_mean_channels-24','mutation_effect_prediction_pairwise','mutation_effect_prediction_independent','mutation_effect_prediction_vae_ensemble','hmm_effect_prediction_mean'],
      my_mdl_compare = ['pretrain_seqMdl_mean','rp75_all_1_mean',	'rp15_all_1_mean', 'rp15_all_2_mean',	'rp15_all_3_mean', 'rp15_all_4_mean','pretrain_seqMdl_mean_s100_reweight','pretrain_seqMdl_mean_s100_nonReweight'],
      mdls_draw = ['hmm_effect_prediction_mean','mutation_effect_prediction_independent','mutation_effect_prediction_pairwise','mutation_effect_prediction_vae_ensemble','mutation_effect_prediction_all_mean_autoregressive','pretrain_seqMdl_mean','pretrain_seqMdl_mean_s100_reweight','pretrain_seqMdl_mean_s100_nonReweight'],
      #'pre_only_mean','pretrain_seqMdl_mean','rp15_all_4_mean','rp15_all_3_mean','rp15_all_2_mean','rp15_all_1_mean','rp75_all_1_mean'
      #'mutation_effect_prediction_all_mean_autoregressive','mutation_effect_prediction_pairwise','mutation_effect_prediction_independent','mutation_effect_prediction_vae_ensemble','hmm_effect_prediction_mean','pretrain_seqMdl_mean','pretrain_seqMdl_mean_s100_reweight','pretrain_seqMdl_mean_s100_nonReweight'
      regions_draw = ['middle'], # also for numerical analysis #'not_middle','all','start','end'
      fig_subNm = 'stacked_bar', #'clean','crowd','three_region'
      save_score = False,
      draw_figure = True,
      domain_group = False,
      nume_analysis = False,
      nume_mdl_pairs = [['pretrain_seqMdl_mean', 'mutation_effect_prediction_all_mean_autoregressive'],
                        ['pretrain_seqMdl_mean', 'hmm_effect_prediction_mean'],
                        ['mutation_effect_prediction_all_mean_autoregressive', 'hmm_effect_prediction_mean'],
                        ['pretrain_seqMdl_mean', 'mutation_effect_prediction_forward_mean'],
                        ['pretrain_seqMdl_mean', 'mutation_effect_prediction_reverse_mean'],
                        ['mutation_effect_prediction_forward_mean','mutation_effect_prediction_reverse_mean']],
      minNum_for_rank = 10)
  elif task2run == 'analysis_fine_region':
    analysis_fine_region(
      working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark',
      shin_muta_path = '/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/calc_logprobs',
      exp_label_set = 'deepsequence',
      sota_mdl_compare = ['mutation_effect_prediction_all_mean_autoregressive','mutation_effect_prediction_forward_mean','mutation_effect_prediction_reverse_mean','mutation_effect_prediction_forward_mean_channels-48','mutation_effect_prediction_reverse_mean_channels-48','mutation_effect_prediction_forward_mean_channels-24','mutation_effect_prediction_reverse_mean_channels-24','mutation_effect_prediction_pairwise','mutation_effect_prediction_independent','mutation_effect_prediction_vae_ensemble','hmm_effect_prediction_mean'],
      my_mdl_compare = ['pretrain_seqMdl_mean','rp75_all_1_mean',	'rp15_all_1_mean', 'rp15_all_2_mean',	'rp15_all_3_mean', 'rp15_all_4_mean'],
      mdls_draw = ['Shin2021_fw','Shin2021_rv','HMM','OurSeqMdl'],
      mdls_diff_draw = ['OurSeqMdl-Shin2021_fw','OurSeqMdl-Shin2021_rv','OurSeqMdl-HMM','Shin2021_fw-Shin2021_rv'],
      fig_subNm = 'fine_interval',
      save_score = True,
      draw_figure = True,
      nume_analysis = True,
      nume_mdl_pairs = [['pretrain_seqMdl_mean', 'mutation_effect_prediction_all_mean_autoregressive'],
                        ['pretrain_seqMdl_mean', 'hmm_effect_prediction_mean'],
                        ['mutation_effect_prediction_all_mean_autoregressive', 'hmm_effect_prediction_mean'],
                        ['pretrain_seqMdl_mean', 'mutation_effect_prediction_forward_mean'],
                        ['pretrain_seqMdl_mean', 'mutation_effect_prediction_reverse_mean'],
                        ['mutation_effect_prediction_forward_mean','mutation_effect_prediction_reverse_mean']],
      minNum_for_rank = 10,
      num_pos_bin = 6)
  elif task2run == 'petase_mutations':
    petase_mutations()
  elif task2run == 'deepSeqSet_wt_seq_file':
    deepSeqSet_wt_seq_file()
  elif task2run == 'change_variant_data_setNm':
    change_variant_data_setNm(set_list=['HIS7_YEAST_Kondrashov2017-singles'])
  elif task2run == 'sota_compare_use_esm_file':
    #'mt_lambda_track','mt_protein_lambda_track','sota_protein_track','sub_sota_protein_track','sota_protein_track_valid_best'
    sota_compare_use_esm_file(load_processed_df=True,fig_name_list=['hist'])
  elif task2run == 'seq_struct_scoring':
    seq_struct_scoring(load_processed_df=False,fig_name_list=['sota_protein_track_valid_best'])
  elif task2run == 'mutation_depth_group':
    mutation_depth_group(load_processed_df=True,fig_name_list=['sota_protein_track_valid_best'])
  elif task2run == 'mt_protein_lambda_track':
    mt_protein_lambda_track(load_processed_df=True)
  elif task2run == 'mutation_embedding_umap':
    mutation_embedding_umap(draw_fig=False, load_umap_model=False)
  elif task2run == 'embed_quan_analysis':
    embed_quan_analysis()
  elif task2run == 'add_wavenet_to_sota_df':
    add_wavenet_to_sota_df()
  elif task2run == 'split_files':
    split_files(
      file_prefix='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/mutagenesis/mutagenesisData/set_data/HIS7_YEAST_Kondrashov2017/HIS7_YEAST_Kondrashov2017_mut_all',
      format='lmdb',
      split_num=3)
  else:
    Exception(f'invalid task {task2run}')

  #mutant_count(working_dir)
  #mutant_label_distribution(working_dir)
  #split_dataset_label_supervise(working_dir)
  #generate_contact_map(working_dir)
  
  
  #mut_fig()
  #mut_precision_fitnessSV_fig()
  #mut_precision_fitnessUNSV_fig()
  
  '''
  mut_epoch_fitnessUNSV_fig(working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark',
                            rpSet_list = ['rp15_all_2'],
                            pretrainMdlNm_list = ['masked_language_modeling_transformer_21-08-23-02-59-06_850428'],
                            mutaSetList_file = None,
                            epoch_list = 224,
                            report_metric = 'spR')
  '''
  '''
  mut_fitnessUNSV_ensemble(working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark',
                          rpSet_list = ['rp75_all_1', 'rp15_all_1', 'rp15_all_2', 'rp15_all_3', 'rp15_all_4'],
                          pretrainMdlNm_list = None,
                          mutaSetList_file = None,
                          epoch_list = None,
                          report_metric = 'spR',
                          pretrain_topk = 3,
                          finetune_topk = 3,
                          ci_alpha = 0.95) 
  '''
  '''
  collect_fitness_ratioScores(working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark',
                              rpSet_list = ['rp75_all_1', 'rp15_all_1', 'rp15_all_2', 'rp15_all_3', 'rp15_all_4'],
                              pretrainMdlNm_list = None,
                              mutaSetList_file = None,
                              epoch_list = None,
                              pretrain_topk = 3,
                              finetune_topk = 3,
                              mutaSetIdx_list = sys.argv[1:])
  '''
  '''
  add_new_prediction(working_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark',
                     rpSet_list=['rp75_all_1', 'rp15_all_1', 'rp15_all_2', 'rp15_all_3', 'rp15_all_4'],
                     epoch_list=[729,224],
                     setNm_list=[])
  '''
  '''
  add_new_mean(working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark',
               shin_muta_path = '/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/calc_logprobs',
               col_list = ['mutation_effect_prediction_reverse_mean_channels-24','mutation_effect_prediction_reverse_mean_channels-48'],
               new_col_name = 'mutation_effect_prediction_reverse_mean')
  '''
  #'mutation_effect_prediction_all_mean_autoregressive','mutation_effect_prediction_pairwise','mutation_effect_prediction_independent','mutation_effect_prediction_vae_ensemble','hmm_effect_prediction_mean','pretrain_seqMdl_mean'
  
  
  
  # ['pretrain_seqMdl_mean', 'mutation_effect_prediction_all_mean_autoregressive'],
  # ['pretrain_seqMdl_mean', 'mutation_effect_prediction_vae_ensemble'],
  # ['pretrain_seqMdl_mean', 'mutation_effect_prediction_pairwise'],
  # ['pretrain_seqMdl_mean', 'mutation_effect_prediction_independent'],
  # ['pretrain_seqMdl_mean', 'hmm_effect_prediction_mean'],
  # ['mutation_effect_prediction_all_mean_autoregressive', 'mutation_effect_prediction_vae_ensemble'],['mutation_effect_prediction_all_mean_autoregressive', 'mutation_effect_prediction_pairwise'],['mutation_effect_prediction_all_mean_autoregressive', 'mutation_effect_prediction_independent'],['mutation_effect_prediction_all_mean_autoregressive', 'hmm_effect_prediction_mean']
  '''
  analysis_mutOrder(
      working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark',
      shin_muta_path = '/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/calc_logprobs',
      exp_label_set = 'deepsequence',
      sota_mdl_compare = ['mutation_effect_prediction_all_mean_autoregressive','mutation_effect_prediction_forward_mean','mutation_effect_prediction_reverse_mean','mutation_effect_prediction_pairwise','mutation_effect_prediction_independent','mutation_effect_prediction_vae_ensemble','hmm_effect_prediction_mean'],
      my_mdl_compare = ['pretrain_seqMdl_mean'],
      mdls_draw = ['mutation_effect_prediction_all_mean_autoregressive','mutation_effect_prediction_pairwise','mutation_effect_prediction_independent','mutation_effect_prediction_vae_ensemble','hmm_effect_prediction_mean','pretrain_seqMdl_mean'],
      region_name = 'all',
      sets_draw = ['HIS7_YEAST_Kondrashov2017'],
      fig_subNm = 'site_analysis',
      count_sites = False,
      draw_figure = True,
      minNum_for_rank = 10)
  '''
  
  '''
  mut_set_spearmanR_fig(working_dir = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark',
                        curve_list = ['ENSM_RW_FT_PfamUniKB','ENSM_RW_FT_ShinData','HMM','Independent','Evmutation','DeepSequence','Shin_2021','Yue_2021'],
                        plot_title = 'unSV_fit_mean_errorbar_sota_withShinData')
  '''
  # unSV_fit_PreBest3_FTBest3_abalation
  # 'All_RW_FT','HMM','Independent','Evmutation','DeepSequence','Shin_2021','Yue_2021'
  # 'B1_rp15_RW_FT', 'B2_rp15_RW_FT', 'B3_rp15_RW_FT', 'B4_rp15_RW_FT', 'All_RW_FT'
  
  '''
  new_wtSeq(wtSeq_fasta=f'{working_dir}/pfam_35.0/Sars_CoV2_RBD/wtSeq_P0DTC2.fasta',
            mut2add=['A67V','T95I','G142D','N211I','G339D','S371L','S373P','S375F','K417N','N440K','G446S','S477N','T478K','E484A','Q493R','G496S','Q498R','N501Y','Y505H','T547K','D614G','H655Y','N679K','P681H','N764K','D796Y','N856K','Q954H','N969K','L981F'],
            save_fasta=f'{working_dir}/pfam_35.0/spike_prot/P0DTC2_BA1.fasta')
  '''
  '''
  spikeMuts = [
    #alpha 'DEL69','DEL70','DEL144','DEL145'
    'N501Y','A570D','D614G','P681H','T716I','S982A','D1118H',
    #beta 'DEL241','DEL242','DEL243'
    'D80A','D215G','K417N','E484K','N501Y','D614G','A701V',
    #gamma
    'L18F','T20N','P26S','D138Y','R190S','K417T','E484K','N501Y','D614G','H655Y','T1027I','V1176F',
    #omicron 'DEL69','DEL70','DEL143','DEL144','DEL145','DEL212'
    'A67V','T95I','G142D','N211I','G339D','S371L','S373P','S375F','K417N','N440K','G446S','S477N','T478K','E484A','Q493R','G496S','Q498R','N501Y','Y505H','T547K','D614G','H655Y','N679K','P681H','N764K','D796Y','N856K','Q954H','N969K','L981F',
    #delta 'DEL157','DEL158',
    'T19R','G142D','E156G','L452R','T478K','D614G','P681R','D950N
    #BA.2 'DEL25', 'DEL26', 'DEL27'
    'T19I','L24S','G142D','V213G','G339D','S371F','S373P','S375F','T376A','D405N','R408S','K417N','N440K','S477N','T478K','E484A','Q493R','Q498R','N501Y','Y505H','D614G','H655Y','N679K','P681H','N764K','D796Y','Q954H','N969K'
  ]
  '''
  '''
  name_range = {'S1N':[1,347],'RBD':[348,526],'S1C':[527,709],'S2':[710,1236]}
  for voc in ['BA1']:  #'alpha','beta','delta','gamma','omicron'
    for name, rangeIdx in name_range.items():
      print(f'{voc},{name},{rangeIdx}')
      prepare_fitnessScan_inputs(wtSeq_fasta=f'{working_dir}/pfam_35.0/spike_prot/P0DTC2_{voc}.fasta',
                            seq_range=rangeIdx, #[348,526],
                            scan_range=rangeIdx, #[348,526],
                            scan_pos=None,
                            save_json=False,
                            save_file=f'{working_dir}/pfam_35.0/spike_prot/{voc}{name}_mutScan.lmdb')
  '''
  # S1N:1,347; RBD:348,526; S1C:527,709; S2:710,1236
  '''
  fitScan_figure(data_path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/pfam_35.0/Sars_CoV2_RBD',
                 mdl_sets=['rp75_all_1','rp15_all_1','rp15_all_2','rp15_all_3','rp15_all_4'],
                 figure_name='mut_wise',
                 topPos_num=30)
  '''
  '''
  pos_select_figure(data_path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/pfam_35.0/spike_prot',
                    mdl_sets=['rp75_all_1','rp15_all_1','rp15_all_2','rp15_all_3','rp15_all_4'],
                    figure_name='mut_wise',
                    var_set='BA2')
  '''
  '''
  fitScan_topKacc(data_path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/pfam_35.0/spike_prot',
                  mdl_sets=['rp75_all_1','rp15_all_1','rp15_all_2','rp15_all_3','rp15_all_4'],
                  parent='wt')
  
  #'rp75_all_1','rp15_all_1','rp15_all_2','rp15_all_3','rp15_all_4'
  '''
  
  '''
  fitScan_topKacc_lineage(data_path='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/pfam_35.0/spike_prot',
                  mdl_sets=['rp75_all_1','rp15_all_1','rp15_all_2','rp15_all_3','rp15_all_4'])
  '''
