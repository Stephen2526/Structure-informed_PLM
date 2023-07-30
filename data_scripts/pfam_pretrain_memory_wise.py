'''
Processing Pfam fasta data, having functions of:
* count statistics of sequence length, family and clan
* create json data set with keys: primary,protein_length,clan,family,id
* create masked LM examples and save in tfrecord format
memory efficient version
process seqs in one json file at a time
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from vocabs import PFAM_VOCAB
from absl import app
from absl import flags
import random, os, re, sys
import json
import numpy as np
import matplotlib.pyplot as plt
import collections

## define global variables
FLAGS = flags.FLAGS

#/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/
flags.DEFINE_string('PROJ_DIR', '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/', 
  "Project root dir")

flags.DEFINE_string('DATA_PATH', 'data_process/pfam/Pfam-A.fasta', 
  "Path for raw input fasta files (or folder of input files)")

flags.DEFINE_string('JSON_PATH', 'data_process/pfam/seq_jsons', 
  "Path for raw input fasta files (or folder of input files)")

flags.DEFINE_string('STAT_PATH', 'data_process/pfam/stat',
  "Path for statistic files")

flags.DEFINE_string('OUTPUT_PATH', 'data_process/pfam/seq_masked_tfrecords_sig_domain',
  "Folder path to save output files)")

flags.DEFINE_string('FAM_CLAN_FILE', 'data_process/pfam/Pfam-A.clans.tsv',
  "family, clan and domain description tsv file")

flags.DEFINE_integer('RANDOM_SEED', 25, 'random seed')

flags.DEFINE_boolean('SPLIT_FASTA', True, 
  'If True, split seqs in every fasta file into subsets')

flags.DEFINE_integer('FASTA_SPLIT_NUM', 60,
  'number of subsets to split seqs in one fasta into')

# 'max_seq_length' takes the value of 'sgl_domain_len_up' + 2 (start and end tokens)
flags.DEFINE_list('MASK_PARAMS', [202, 0.15, 30],
  'params for mask: max_seq_length, mask_prob, max_mask_per_seq')

flags.DEFINE_boolean('SIG_DOM', True,
  'flag for only saving single domain sequences')

flags.DEFINE_integer('sgl_domain_len_low', 18,
  'the lower bound length of single domain')

flags.DEFINE_integer('sgl_domain_len_up', 200,
  'the upper bound length of single domain')

def raw_fasta2dict(input_path, STAT_PATH, fam_clan_tsv):
  '''
  input: 
    input_path: one raw fasta file or a directory containing multiple fasta files,
    STAT_PATH: directory to save statistic file
    fam_clan_tsv: path for fam_clan_tsv file

  output: 
    a list of lists, each list corresponds one fasta file and  
    contains multiple dictionaries. 
    Each dictionary corresponds to one protein sequence/segment.
    keys of dictionary: {sequence,seq_length,uni_iden,family,clan}

  Func:
    statistics of Pfam sequence files:
    -number of sequences
    -length distribution
    -family counts
    -clan counts
  '''

  file_list = []
  if os.path.isdir(input_path):
    file_list = os.listdir(input_path)

  else:
    file_list.append(input_path)
  
  ## create fam-clan dict
  fam_clan = np.loadtxt(fam_clan_tsv, dtype='str', delimiter='\t')
  fam_clan_dict = {fam_clan[i,0]:fam_clan[i,1] for i in range(len(fam_clan))}
  
  seq_total_num = 0

  length_list = [] 
  family_list = []
  clan_list = []

  pfam_list_all = []
  raw_seq = ''

  for fl in file_list:
    print('loading proteins from %s' % (fl), flush=True)
    pfam_list = []
    with open(fl, 'r') as fasta:
      for line in fasta:
        ## parse comment line
        if line[0] == '>':
          ## process last sequence
          if len(raw_seq) > 0:
            seq_len = len(raw_seq) 
            # updates statistics
            #length_fam_clan_list.append([uni_iden, seq_len, pfam_iden, clan_iden])
            length_list.append(seq_len)
            family_list.append(pfam_iden)
            clan_list.append(clan_iden)

            # one protein dict
            one_protein = {
                'sequence': raw_seq,
                'seq_length': seq_len,
                'uni_iden': uni_iden,
                'family': pfam_iden,
                'clan': clan_iden
                }
            pfam_list.append(one_protein)

          seq_total_num += 1
          comment_list = re.split('>| ', line)
          uni_iden = comment_list[1]
          pfam_iden = re.split('\.', comment_list[3])[0]
          try: 
            clan = fam_clan_dict[pfam_iden]
            clan_iden = clan if len(clan)>0 else 'none'
          except:
            print('Pfam %s not in the family list' % (pfam_iden), flush=True)
          raw_seq = ''
        ## parse sequence lines (mulitple lines)
        else:
          raw_seq += line[:-1]
    pfam_list_all.append(pfam_list)

  print('Done', flush=True)
  ## print statistics
  print('In total %d sequences' % (seq_total_num), flush=True)

  #print('length_fam_clan_list size in mem(GB): ', sys.getsizeof(length_list)/(1024**3))
  length_arr = np.array(length_list)
  #print('length_fam_clan_arr size in mem(GB): ', length_arr.nbytes/(1024**3))

  # hist fig of seq length
  fig, axs = plt.subplots(1,1)
  (n,bins,_) = axs.hist(length_arr, bins=50)
  fig.savefig(STAT_PATH+'/seq_len_hist.png')
  np.savetxt(STAT_PATH+'/seq_counts', np.hstack((bins[:,np.newaxis], np.hstack((0,n))[:,np.newaxis])), fmt='%u')
  # distri of pfam_iden, clan_iden
  [fam_uniq_arr, fam_uniq_counts] = np.unique(family_list, return_counts=True)
  [clan_uniq_arr, clan_uniq_counts] = np.unique(clan_list, return_counts=True)
  np.savetxt(STAT_PATH+'/family_counts', np.hstack((fam_uniq_arr[:,np.newaxis], fam_uniq_counts[:,np.newaxis])), fmt='%s')
  np.savetxt(STAT_PATH+'/clan_counts', np.hstack((clan_uniq_arr[:,np.newaxis], clan_uniq_counts[:,np.newaxis])), fmt='%s')
  # barplot of family and clan
  fig1,axs1 = plt.subplots(1,1,figsize=(30,25))
  clan_idx = range(len(clan_uniq_counts))
  axs1.bar(clan_idx, clan_uniq_counts, align='center')
  fig1.savefig(STAT_PATH+'/clan_bar.png')

  fig2,axs2 = plt.subplots(1,1, figsize=(30,25))
  fam_idx = range(len(fam_uniq_counts))
  axs2.bar(fam_idx, fam_uniq_counts, align='center')
  fig2.savefig(STAT_PATH+'/fam_bar.png')

  return pfam_list_all


class TrainingInstance(object):
  """
  class for one training example (one sequence segment)
  token_seq: list, tokens(residues) of a protein seq
  family_id, clan_id: integer, index of family and clan
  masked_lm_positions: list, position index of masked positions
  masked_lm_labels: list, true token(residue) of masked positions 
  """
  def __init__(self, token_seq, family_id, clan_id, masked_lm_positions, masked_lm_labels):
    self.token_seq = token_seq
    self.family_id = family_id
    self.clan_id = clan_id
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __repr__(self):
    instance_dict = {
      'token_seq': token_seq,
      'family_id': family_id,
      'clan_id': clan_id,
      'masked_lm_positions': masked_lm_positions,
      'masked_lm_labels': masked_lm_labels
      }
    return instance_dict

def convert_fam_clan_2_int(fam, clan):
  '''
  convert fam clan identifier to int index
  '''
  fam_id = int(fam[2:])
  if clan == 'none':
    clan_id = 0
  else:
    clan_id = int(clan[2:])

  return ([fam_id], [clan_id])

def create_training_instances(
  fasta_lists, 
  max_seq_length, 
  mask_prob, 
  max_mask_per_seq, 
  rng,
  single_domain = True):
  '''
  create training instance from raw sequence
  
  inputs:
    fasta_lists: 
      list, each element is a list of dicts, each dict is one seq.
      dict keys: {sequence,seq_length,uni_iden,family,clan}   
  output:
    instance_list:
      list, each element is list of TrainingInstances

  '''
  instances_list = []
  total_count = 0
  for fasta_one_list in fasta_lists:
    instances = [] # list holding instances of one seq set

    for seq_one_dict in fasta_one_list:
      raw_seq = seq_one_dict['sequence']
      seq_length = seq_one_dict['seq_length']
      if single_domain and (seq_length < FLAGS.sgl_domain_len_low or seq_length > FLAGS.sgl_domain_len_up): ## only process sequence of single domain
        continue
      else:
        (fam_id, clan_id) = convert_fam_clan_2_int(seq_one_dict['family'], seq_one_dict['clan'])

        '''
        Tokenize protein sequence
          apply a trivial rule:
            each residue tokenize to a single token
        '''
        seq_tokens = list(raw_seq)

        '''
        Padding and splitting
          (check stat/seq_count and stat/seq_len_hist.png for seq len stats)
          -based on the stats, max_seq_length = 256,mask_p = 0.15,max_masks_per_seq=38
          -seqs less than max append with token [PAD] with index 0
          -seqs more than max split into segments from left to right
          -seqs/segments less then 38 are disregarded to reduce computation waste
        '''
        seq_token_list = []
        seq_len = len(seq_tokens)
        start_token = '<CLS>'
        end_token = '<SEP>'
        ## disregard case
        if seq_len < max_mask_per_seq:
          continue
        ## padding case
        elif seq_len <= max_seq_length-2:
          seq_tokens.insert(0, start_token)
          seq_tokens.append(end_token)
          while len(seq_tokens) < max_seq_length:
            seq_tokens.append('<PAD>')
          seq_token_list.append(seq_tokens)
        ## split case (not happen when only consider single domain sequences)
        elif seq_len > max_seq_length-2:
          i = 0
          while i < seq_len:
            seq_tokens_tmp = seq_tokens[i:i+max_seq_length-2]
            if len(seq_tokens_tmp) >= max_mask_per_seq:
              seq_tokens_tmp.insert(0, start_token)
              seq_tokens_tmp.append(end_token)
              seq_token_list.append(seq_tokens_tmp)
            i += max_seq_length-2
        ## create instance
        for tokens in seq_token_list:
          (masked_tokens, masked_lm_positions, masked_lm_labels) = create_masked_predictions(
              tokens, 
              mask_prob, 
              max_mask_per_seq, 
              rng,
              mode = 'single')
          instance = TrainingInstance(
            token_seq = masked_tokens,
            family_id = fam_id,
            clan_id = clan_id,
            masked_lm_positions = masked_lm_positions,
            masked_lm_labels = masked_lm_labels
            )
          instances.append(instance)
    instances_list.append(instances)
    count_n = len(instances)
    total_count += count_n
    print('>>> %d seqs have been processed' % (count_n), flush=True)
  print('>>> In total, %d seqs have been processed' % (total_count), flush=True)
  return instances_list, total_count
    
def create_masked_predictions(
    tokens, 
    mask_prob,
    max_mask_per_seq,
    rng,
    mode = 'single'
    ):
  '''
  create masked input examples
  inputs:
    tokens: list, a seq of tokens for in example
    mode:
      single: uniformly pick single residue to mask
      motif: mask k continuous residues related to motifs
  outputs:

  '''
  if mode == 'single':
    cand_indexes = [] # indexes of residues only
    for (i, token) in enumerate(tokens):
      if token == '<CLS>' or token == '<SEP>' or token == '<PAD>':
        continue
      else:
        cand_indexes.append(i)
    rng.shuffle(cand_indexes)

    num_to_mask = min(max_mask_per_seq, int(round(len(cand_indexes)*mask_prob)))

    ## cand_indexes already shuffled, mask along the list until reach num_to_mask
    mask_list = [] # each elem [idx, true label]
    masked_tokens = tokens
    vocab_tokens = list(PFAM_VOCAB.keys())[4:]
    for idx in cand_indexes:
      if len(mask_list) >= num_to_mask:
        break
      
      mask_list.append([idx, tokens[idx]]) # add this pos to list
      # 80% of time, replace with [MASK]
      if rng.random() < 0.8:
        mask_tk = '<MASK>'
      else:
        # 10% of time, keep original
        if rng.random() < 0.5:
          mask_tk = tokens[idx]
        # 10% of time, replace with other random residue
        else:
          trim_vocabs = list(PFAM_VOCAB.keys())[4:]
          trim_vocabs.remove(tokens[idx])
          mask_tk = trim_vocabs[rng.randint(0, len(trim_vocabs)-1)]

      masked_tokens[idx] = mask_tk

    assert len(mask_list) <= num_to_mask
    ## order masked positions with indexed ascending 
    mask_list = sorted(mask_list, key=lambda x: x[0])
    masked_lm_positions, masked_lm_labels = [], []
    for one_mask in mask_list:
      masked_lm_positions.append(one_mask[0])
      masked_lm_labels.append(one_mask[1])

    ## pad masked_ln positions and labels to max_mask_per_seq
    while len(masked_lm_positions) < max_mask_per_seq:
      masked_lm_positions.append(0)
      masked_lm_labels.append('<PAD>')
      
  return (masked_tokens, masked_lm_positions, masked_lm_labels)
      
def convert_tokens_to_ids(tokens):
  '''
  convert a list of tokens to a list of integers
  '''
  converted_ids = []
  for tk in tokens:
    id_n = PFAM_VOCAB[tk]
    converted_ids.append(id_n)
  return converted_ids

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def write_instance_to_tfrecord_files(instances_list, output_path, writers, num_written, writer_idx, ta_val_test_idx):
  '''
  save instances to tfRecord files
  NOTE: TFRecordWriter don't have append function (use write instand)
  inputs:
    instances_list: each elem is a list of instances
    output_path: a folder path to save instances
    writers: list holding TFRecordWriter objects
    num_written: [train_written, val_written, test_written], number of seqs written to each dateset
    writer_idx: an iterative index for current writer
    ta_val_test_idx: [train_writer, val_writer, test_writer], writer index for each dateset
  '''
  assert os.path.isdir(output_path)
  # copy writer index
  train_wter = ta_val_test_idx[0]
  val_wter = ta_val_test_idx[1]
  test_wter = ta_val_test_idx[2]
  # copy num of written sequences
  train_num = num_written[0]
  val_num = num_written[1]
  test_num = num_written[2]

  for instances in instances_list:
    for instance in instances:
      token_ids = convert_tokens_to_ids(instance.token_seq)
      fam_id = instance.family_id
      clan_id = instance.clan_id
      masked_lm_positions = instance.masked_lm_positions
      masked_lm_ids = convert_tokens_to_ids(instance.masked_lm_labels)

      ## create feature dict
      features = collections.OrderedDict()
      features['token_ids'] = create_int_feature(token_ids)
      features['fam_id'] = create_int_feature(fam_id)
      features['clan_id'] = create_int_feature(clan_id)
      features['masked_lm_positions'] = create_int_feature(masked_lm_positions)
      features['masked_lm_ids'] = create_int_feature(masked_lm_ids)

      tf_example = tf.train.Example(features=tf.train.Features(feature=features))
      writers[writer_idx].write(tf_example.SerializeToString())
      # update num_written
      if writer_idx in train_wter:
        train_num += 1
      elif writer_idx in val_wter:
        val_num += 1
      else:
        test_num += 1
      
      writer_idx = (writer_idx+1) % len(writers) ## iterative writing

  return ([train_num, val_num, test_num], writer_idx)

def main(argv):
  ## copy global params
  PROJ_DIR = FLAGS.PROJ_DIR
  DATA_PATH = PROJ_DIR+FLAGS.DATA_PATH
  JSON_PATH = PROJ_DIR+FLAGS.JSON_PATH
  STAT_PATH = PROJ_DIR+FLAGS.STAT_PATH
  OUTPUT_PATH = PROJ_DIR+FLAGS.OUTPUT_PATH
  FAM_CLAN_FILE = PROJ_DIR+FLAGS.FAM_CLAN_FILE
  RANDOM_SEED = FLAGS.RANDOM_SEED
  SPLIT_FASTA = FLAGS.SPLIT_FASTA
  FASTA_SPLIT_NUM = FLAGS.FASTA_SPLIT_NUM
  [MAX_SEQ_LENGTH, MASK_PROB, MAX_MASK_PER_SEQ] = FLAGS.MASK_PARAMS
  SIG_DOM = FLAGS.SIG_DOM

  print('*** data_path: {}'.format(DATA_PATH), flush=True)
  print('*** json_path: {}'.format(JSON_PATH), flush=True)
  print('*** stat_path: {}'.format(STAT_PATH), flush=True)
  print('*** output_path: {}'.format(OUTPUT_PATH), flush=True)

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  rng = random.Random(RANDOM_SEED)

  readRaw = False
  if readRaw:
    file_list = []
    if os.path.isdir(DATA_PATH):
      file_list = os.listdir(DATA_PATH)
    else:
      file_list.append(DATA_PATH)

    tf.compat.v1.logging.info("*** Reading from fasta files ***")
    for input_file in file_list:
      tf.compat.v1.logging.info(" %s", input_file)

    ## transform raw fasta sequence to list of distionarys
    fasta_all_list = raw_fasta2dict(DATA_PATH, STAT_PATH, FAM_CLAN_FILE)

    ## whether to split sequences in one fasta into N sets
    fasta_all_list_new = [] # new list holding splited lists
    if SPLIT_FASTA:
      for fasta_one_list in fasta_all_list:
        rng.shuffle(fasta_one_list)
        n = FASTA_SPLIT_NUM
        N = len(fasta_one_list)
        step = N // n + 1
        fasta_one_split_list = [fasta_one_list[i:i+step] \
          for i in range(0, N, step)]
        fasta_all_list_new.append(fasta_one_split_list)

  save2json = False
  if save2json:
    ## save seq subsets to json file
    tf.compat.v1.logging.info('*** Saving seqs to jsons ***')
    for idx_fasta in range(len(fasta_all_list_new)):
      ## seqs in one fasta
      out_fl = JSON_PATH+'/fasta_'+str(idx_fasta)+'_seq_set_'
      fasta_one_list = fasta_all_list_new[idx_fasta]
      for idx_seq in range(len(fasta_one_list)):
        with open(out_fl+str(idx_seq), 'w') as fout:
          json.dump(fasta_one_list[idx_seq], fout)
    tf.compat.v1.logging.info('*** Finish saving seqs to jsons ***')

  load_jsons = False
  if load_jsons:
    ##### load sequence dicts from json files #####
    tf.compat.v1.logging.info('*** Loading seqs from json ***')
    json_dir = JSON_PATH

    ##### initialize writer objects #####
    writers = []
    train_wter = []
    val_wter = []
    test_wter = []
    for i in range(FASTA_SPLIT_NUM):
      if i < 54:
        writers.append(tf.io.TFRecordWriter(OUTPUT_PATH+'/train_seqs_masked_TFRecord_'+str(i)))
        train_wter.append(i)
      elif i>=54 and i<58:
        writers.append(tf.io.TFRecordWriter(OUTPUT_PATH+'/validation_seqs_masked_TFRecord_'+str(i)))
        val_wter.append(i)
      elif i>=58:
        writers.append(tf.io.TFRecordWriter(OUTPUT_PATH+'/test_seqs_masked_TFRecord_'+str(i)))
        test_wter.append(i)
    ## counters for file writting
    writer_idx = 0 
    train_written = 0
    val_written = 0
    test_written = 0
    num_written = [train_written, val_written, test_written]
    ## counter for creating instances
    count_set = 1
    total_seq_count = 0
    for fl_str in os.listdir(json_dir):
      seq_dict_all = []
      tf.compat.v1.logging.info('%s', json_dir+'/'+fl_str)
      with open(json_dir+'/'+fl_str, 'r') as fl:
        seq_list = json.load(fl)
      seq_dict_all.append(seq_list)
    
      ##### create sequence instances #####
      tf.compat.v1.logging.info('*** Create training instances ***')
      print('>>> Processing seq set %d' % (count_set), flush=True)
      count_set += 1
      instances, total_count = create_training_instances(seq_dict_all, MAX_SEQ_LENGTH, MASK_PROB, MAX_MASK_PER_SEQ, rng, single_domain=SIG_DOM)     
      total_seq_count += total_count
      ##### write seqs to TFRecord files #####
      tf.compat.v1.logging.info("*** Writing seqs to TFRecord files***")
      (num_written, writer_idx) = write_instance_to_tfrecord_files(instances, OUTPUT_PATH, writers, 
                                                                   num_written, writer_idx, [train_wter, val_wter, test_wter])
  
    # close tfrecord files
    for writer in writers:
      writer.close()

    tf.compat.v1.logging.info('In total, process {} sequences'.format(total_seq_count))
    tf.compat.v1.logging.info('In total, write {} instances with train {}, val {}, test {}'.format(sum(num_written), num_written[0], num_written[1], num_written[2]))

if __name__ == "__main__":
  app.run(main)
