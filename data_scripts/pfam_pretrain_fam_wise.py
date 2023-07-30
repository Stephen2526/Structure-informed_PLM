'''
*Create masked sequence dataset with true label
*Statistical counting

input files: family-wise protein MSA in stockholm/fasta format
output files: 
  masked sequences in tfRecord format
  train/val set: within distribution split with rate 90/10
  test set: hold out specific families and clans
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List

#from torch.utils import data
#import tensorflow as tf
#print('load tf done')
from vocabs import PFAM_VOCAB
from absl import app
from absl import flags
import random, os, re, sys, time, json, lmdb
import pickle as pkl
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import collections
from prody import MSAFile
from Bio import pairwise2, SeqIO
from Bio.SubsMat import MatrixInfo as matlist



## define global variables
FLAGS = flags.FLAGS

#/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/
flags.DEFINE_string('PROJ_DIR', '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/', 
  "Project root dir")

flags.DEFINE_string('DATA_PATH', 'data_process/pfam_32.0', 
  "Path for data files")

flags.DEFINE_string('STAT_PATH', 'data_process/pfam_32.0/stat_rp15',
  "Path for statistic files")

flags.DEFINE_string('OUTPUT_PATH', 'data_process/pfam_32.0/seq_json_rp15',
  "Folder name to save output files")

flags.DEFINE_string('FAM_CLAN_FILE', 'data_process/pfam_32.0/Pfam-A.clans.tsv',
  "family, clan and domain description tsv file")

flags.DEFINE_string('FLNM_PFAM_FILE', 'data_process/pfam_32.0/pfam_rp15_seqs_files_famAcc', 
  'data file name and pfam number, clan number corresponds')

flags.DEFINE_integer('RANDOM_SEED', 25, 'random seed')

flags.DEFINE_boolean('SPLIT', True, 
  'If True, split seqs in every fasta file into subsets')

flags.DEFINE_integer('NUM_SPLIT', 50,
  'number of subsets to split seqs into tfReocrd files')

flags.DEFINE_integer('VOCAB_SIZE', 29,
  'number of vocabularies/tokens')

# 'max_seq_length' takes the value of 'sgl_domain_len_up' + 2 (start and end tokens)
flags.DEFINE_list('MASK_PARAMS', [202, 0.15, 30],
  'params for mask: max_seq_length, mask_prob, max_mask_per_seq')

flags.DEFINE_boolean('SIG_DOM', True,
  'flag for only saving single domain sequences')

flags.DEFINE_integer('sgl_domain_len_low', 18,
  'the lower bound length of single domain')

flags.DEFINE_integer('sgl_domain_len_up', 200,
  'the upper bound length of single domain')

flags.DEFINE_string('IN_FORMAT', 'stockholm', 'format for input seq files')

flags.DEFINE_boolean('MASK_FLAG', True, 'Whether to mask seqs')


blosum_matrix = matlist.blosum62

def seq_identity(x,y):
    '''
    identity for aligned two sequences
    '''
    X = x.upper()
    Y = y.upper()
    same = 0
    full_len = len(X)
    assert len(X) == len(Y)
    for i in range(len(X)):
        if X[i] == Y[i] and X[i] != '-' and X[i] != '.':
            same += 1
        elif X[i] == Y[i] and (X[i] == '.' or X[i] == '-'):
            full_len -= 1
    iden = float(same)/float(full_len)
    return iden

def seq_align_identity(x,y,matrix = blosum_matrix):
    X = x.upper()
    Y = y.upper()
    alignments = pairwise2.align.globaldd(X,Y, matrix,-11,-1,-11,-1)   # Consistent with Blast P grobal alignment
    max_iden = 0
    for i in alignments:
        same = 0
        for j in range(i[-1]):
            if i[0][j] == i[1][j] and i[0][j] != '-':
                same += 1
        iden = float(same)/float(i[-1])
        if iden > max_iden:
            max_iden = iden
    return max_iden

def check_ambigu_letters(ori_seq):
  '''
  replace ambiguous residues: 
   * B to D and N
   * Z to E and Q
   * discard sequences containing X
  Input:
   * original raw sequence
  Output:
   * list of new sequences with replaced residues
  '''
  uniq_lters = set(ori_seq)
  #if 'B' in uniq_lters and 'Z' in uniq_lters:
    
def raw_msa2dict(data_path: str = None,
                 input_format: str = None,
                 fileNm: str = None,
                 famAcc: str = None,
                 clanAcc: str = None,
                 aligned: bool = True,
                 weight_in_header: bool = True,
                 keep_X: bool = True):
  '''
  Description:
    * Load seqs from aligned/unaligned file and convert to dictionary
    * Return family weight (sum of seq weight) in case weights are contained in seq file
  
  Params: 
    * data_path: data dir
    * input_format: msa file format
    * fileNm: path of file
    * famAcc: identifier of family
    * clanAcc: identifier of clan

  Outputs: 
    a list of dictionaries with each dictionary corresponds to one protein sequence/segment.
    
    keys of dictionary:
      - primary: raw protein sequence in aa symbols(upper case)
      - protein_length: length of raw sequence
      - unpIden: uniprot_id
      - range: start_idx-end_idx
      - family: family name
      - clan: clan name
      - seq_reweight: redundancy weight
    
  '''
  seq_dict_list = []
  print('loading sequences from %s' % (fileNm), flush=True)
  # convert pfam,clan to id(int)
  famId = int(famAcc[2:7])
  clanId = int(clanAcc[2:]) if clanAcc is not None else -1
  # initilize msa object
  msa = MSAFile('{}/{}'.format(data_path, fileNm), format=input_format, aligned=aligned)
  family_weight = 0.
  for seq in msa:
    # get unaligned sequence
    raw_seq = ''.join(re.findall('[A-Za-z]', str(seq))).upper()
    if 'X' in raw_seq: 
      if keep_X:
        pass ## keep seqs containing 'X'
      else:
        continue ## jump over seqs containing 'X'
      
    if not weight_in_header:
      res_idx = seq.getResnums()
      start_idx = res_idx[0]
      end_idx = res_idx[-1]
      label = seq.getLabel()
      one_protein = {
        'primary': raw_seq,
        'msa_seq': str(seq),
        'protein_length': len(raw_seq),
        'family': famId,
        'clan': clanId,
        'unpIden': label,
        'range': '{}-{}'.format(start_idx,end_idx)
        }   
    else: ## weight in header: patten follow Yue's definition
      whole_id = seq.getLabel()
      label = whole_id.split(';')[1].split('/')[0]
      start_idx = whole_id.split(';')[1].split('/')[1].split(':')[0].split('-')[0]
      end_idx = whole_id.split(';')[1].split('/')[1].split(':')[0].split('-')[1]
      weight_value = whole_id.split(';')[1].split('/')[1].split(':')[1]
      family_weight += float(weight_value)
      one_protein = {
        'primary': raw_seq,
        'msa_seq': str(seq),
        'protein_length': len(raw_seq),
        'family': famId,
        'clan': clanId,
        'unpIden': label,
        'range': '{}-{}'.format(start_idx,end_idx),
        'seq_reweight': float(weight_value)
        }
    '''
    # old protein dict
    one_protein = {
        'sequence': raw_seq,
        'aligned_seq': str(seq),
        'seq_length': len(raw_seq),
        'uni_iden': '{}/{}-{}'.format(label,start_idx,end_idx),
        'family': famAcc,
        'clan': clanAcc
        }
    ''' 
    seq_dict_list.append(one_protein)

  return seq_dict_list, family_weight

class TrainingInstance(object):
  """
  class for one training example (one sequence segment)
  token_seq: list, tokens(residues) of a protein seq
  family_id, clan_id: integer, index of family and clan
  masked_lm_positions: list, position index of masked positions
  masked_lm_labels: list, true token(residue) of masked positions 
  aligned_binary: binary vector with 1 indicating aligned positions(substu)
  unaligned_binary: binary vector with 1 indicating unaligned positions(insertion)
  gr_1Dprofile_oneHot: size [max_seq_length, num_tokens(29)], ground truth
    1-D distribution/one_hot vector indicating true token
  """
  def __init__(self, token_seq, family_id, clan_id, 
               masked_lm_positions, masked_lm_labels,
               aligned_binary, unaligned_binary, gt_1Dprofile_oneHot):
    self.token_seq = token_seq
    self.family_id = family_id
    self.clan_id = clan_id
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels
    self.aligned_binary = aligned_binary
    self.unaligned_binary = unaligned_binary
    self.gt_1Dprofile_oneHot = gt_1Dprofile_oneHot

  def __repr__(self):
    instance_dict = {
      'token_seq': token_seq,
      'family_id': family_id,
      'clan_id': clan_id,
      'masked_lm_positions': masked_lm_positions,
      'masked_lm_labels': masked_lm_labels,
      'aligned_binary': aligned_binary,
      'unaligned_binary': unaligned_binary,
      'gt_1Dprofile_oneHot': gt_1Dprofile_oneHot
      }
    return instance_dict

def convert_fam_clan_2_int(fam, clan):
  '''
  convert fam clan identifier to int index
  '''
  fam_id = int(fam[2:])
  if clan == 'none' or clan == '':
    clan_id = 0
  else:
    clan_id = int(clan[2:])

  return ([fam_id], [clan_id])

def proc_aligned_seq(aligned_seq, raw_seq):
  '''
  Args:
   - aligned_seq: one seq from msa
   - raw_seq: raw protein seq
  Returns:
   - aligned_bi: binary vec with 1 for aligned positions, size[max_seq_length]
   - unaligned_bi: binary vec with 1 for unaligned positions, size[max_seq_length]
   - pos_type_vec: vector with size [max_seq_length]
          non-negative{0,1,2...} value for aligned residues from left to right
          {-1} for insertion residues
          {-2} for special tokens: start, end, padding
  '''
  max_seq_length = FLAGS.MASK_PARAMS[0]
  num_tokens = FLAGS.VOCAB_SIZE
  raw_seq_len = len(raw_seq)

  type_vec = []
  aligned_idx = 0
  # loop through aligned seq to record types
  for i in range(len(aligned_seq)):
    curr_char = aligned_seq[i]
    if curr_char == '.':
      continue
    elif curr_char == '-':
      aligned_idx += 1
    elif curr_char.isalpha() and curr_char.isupper():
      type_vec.append(aligned_idx)
      aligned_idx += 1
    elif curr_char.isalpha() and curr_char.islower():
      type_vec.append(-1)

  # lengh check    
  assert len(type_vec) == raw_seq_len

  # append special tokens
  type_vec.insert(0,-2)
  while len(type_vec) < max_seq_length:
    type_vec.append(-2)

  type_vec = np.array(type_vec)
  aligned_bi = (type_vec>=0).astype(float)
  unaligned_bi = (type_vec==-1).astype(float)

  return aligned_bi, unaligned_bi, type_vec

def profile1D_oneHot(seq_token, type_vec, aa_1d_prof):
  '''
  Args:
    -seq_token: padded seq of tokens, size[max_seq_length]
    -type_vec: vector showing type of each pos, size[max_seq_length]
    -aa_1d_prof: 1d profile for aligned positions in MSA

  Returns:
    -label_mat: ground truth 1d profile(aligned) and one_hot vec(unaligned)
  '''
  AA_idx = [4,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,24,25,27]
  num_tokens = FLAGS.VOCAB_SIZE
  seq_token_ids = convert_tokens_to_ids(seq_token)
  
  assert len(seq_token_ids) == len(type_vec)

  label_mat = []
  for i in range(len(seq_token_ids)):
    if type_vec[i] < 0:
      one_hot = np.zeros(num_tokens)
      one_hot[seq_token_ids[i]] = 1
      label_mat.append(one_hot)
    else:
      prof1D = np.zeros(num_tokens)
      aa_prof = 1 / np.exp(aa_1d_prof[type_vec[i]]) # convert to prob
      np.put(prof1D, AA_idx, aa_prof)
      label_mat.append(prof1D)

  return np.array(label_mat)

def create_training_instances(
  seq_dict_list, 
  max_seq_length, 
  mask_prob, 
  max_mask_per_seq,
  aa_profile1D,
  rng,
  single_domain = True):
  '''
  create training instance from raw sequence
  
  inputs:
  -seq_dict_list: 
    list of dicts, each is one seq.
    dict keys: {sequence,aligned_seq,seq_length,uni_iden,family,clan}   
  output:
    instance_list:
      list, each element is a TrainingInstance
  '''
  instances_list = [] # list holding training instances
  (fam_id, clan_id) = convert_fam_clan_2_int(seq_dict_list[0]['family'],
      seq_dict_list[0]['clan']) # all seqs in this batch are from same family

  for seq_dict in seq_dict_list:
    raw_seq = seq_dict['sequence']
    aligned_seq = seq_dict['aligned_seq']
    seq_length = seq_dict['seq_length']
    ## only process sequence of single domain
    if single_domain and (seq_length < FLAGS.sgl_domain_len_low or seq_length > FLAGS.sgl_domain_len_up): 
      continue
    else:
      '''
      Tokenize protein sequence
      -apply a trivial rule: each residue tokenize to a single token
      '''
      seq_tokens = list(raw_seq)
      '''
      Padding and splitting
        (check stat/seq_count and stat/seq_len_hist.png for seq len stats)
        -based on the stats, max_seq_length, mask_p, max_masks_per_seq=max_seq_length*mask_p
        -seqs less than max append with token [PAD] with index 0
        -two special tokens: start - <CLS>, end - <SEP>
        -discard sequences with length less than max_mask_per_seq
      '''
      seq_token_list = [] # list to store each token sequence
      aligned_pos_list, unaligned_pos_list = [], [] # store binary vectors recording aligned and unaligned positions
      pos_type_list = []

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
        # gather info from aligned sequence
        aligned_bi, unaligned_bi, pos_type_vec = proc_aligned_seq(aligned_seq, raw_seq)
        aligned_pos_list.append(aligned_bi)
        unaligned_pos_list.append(unaligned_bi)
        pos_type_list.append(pos_type_vec)

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
    ## loop over all seqs in list to create instance
    for seq_i in range(len(seq_token_list)):
      # create masked sequence
      (masked_tokens, masked_lm_positions, masked_lm_labels) = create_masked_predictions(
          seq_token_list[seq_i], 
          mask_prob, 
          max_mask_per_seq, 
          rng,
          mode = 'single')
      # create 1D profile and one_hot label mat, size[max_seq_length,num_tokens]
      label_mat = profile1D_oneHot(seq_token_list[seq_i], pos_type_list[seq_i], aa_profile1D)
  
      
      instance = TrainingInstance(
        token_seq = masked_tokens,
        family_id = fam_id,
        clan_id = clan_id,
        masked_lm_positions = masked_lm_positions,
        masked_lm_labels = masked_lm_labels,
        aligned_binary = aligned_pos_list[seq_i],
        unaligned_binary = unaligned_pos_list[seq_i],
        gt_1Dprofile_oneHot = label_mat
        )
      instances_list.append(instance)
  return instances_list
    
def create_masked_predictions(
    tokens, 
    mask_prob,
    max_mask_per_seq,
    rng,
    mode = 'single'
    ):
  '''
  create masked input examples

  Args:
  -tokens: list, a seq of tokens for in example
  -mode:
    -single: uniformly pick single residue to mask
    -motif: mask k continuous residues related to motifs
  Returns:
  -masked_tokens: masked protein sequence
  -masked_lm_positions: index of masked positions
  -masked_lm_labels: true aa of masked positions
  -
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

def create_bytes_feature(values):
  """Returns a bytes_list from a string / byte."""
  #  convert 2d array binary string, BUT shape is lost
  values_str = values.tostring()
  feature  = tf.train.Feature(bytes_list=tf.train.BytesList(value=[values_str]))
  return feature

def write_instance_to_tfrecord_files(instances_list, output_path,
    test_pfam_list, test_clan_list, writers, test_writers, num_written, writer_idx, ta_val_idx):
  '''
  save instances to tfRecord files
  NOTE: TFRecordWriter don't have append function (use write instand)
  inputs:
    instances_list: each elem is a list of instances
    output_path: a folder path to save instances
    test_pfam, test_clan: pfam and clan held out as test set
    writers: list holding TFRecordWriter objects
    num_written: [train_written, val_written, test_written], number of seqs written to each dateset
    writer_idx: an iterative index for current writer
    ta_val_test_idx: [train_writer, val_writer, test_writer], writer index for each dateset
  '''
  assert os.path.isdir(output_path)
  # copy writer index
  train_wter = ta_val_idx[0]
  val_wter = ta_val_idx[1]
  
  # copy num of written sequences
  train_num = num_written[0]
  val_num = num_written[1]
  test_num = num_written[2]

  for instance in instances_list:
    token_ids = convert_tokens_to_ids(instance.token_seq)
    fam_id = instance.family_id
    clan_id = instance.clan_id
    masked_lm_positions = instance.masked_lm_positions
    masked_lm_ids = convert_tokens_to_ids(instance.masked_lm_labels)
    aligned_binary = instance.aligned_binary
    unaligned_binary = instance.unaligned_binary
    gt_1Dprofile_oneHot = instance.gt_1Dprofile_oneHot


    # convert list to array and set dtype
    if isinstance(token_ids, list):
      token_ids = np.array(token_ids).astype(np.int32)
    else:
      token_ids = token_ids.astype(np.int32)

    if isinstance(fam_id, list):
      fam_id = np.array(fam_id).astype(np.int32)
    else:
      fam_id = fam_id.astype(np.int32)
      
    if isinstance(clan_id, list):
      clan_id = np.array(clan_id).astype(np.int32)
    else:
      clan_id = clan_id.astype(np.int32)

    if isinstance(masked_lm_positions, list):
      masked_lm_positions = np.array(masked_lm_positions).astype(np.int32)
    else:
      masked_lm_positions = masked_lm_positions.astype(np.int32)

    if isinstance(masked_lm_ids, list):
      masked_lm_ids = np.array(masked_lm_ids).astype(np.int32)
    else:
      masked_lm_ids = masked_lm_ids.astype(np.int32)

    if isinstance(aligned_binary, list):
      aligned_binary = np.array(aligned_binary).astype(np.float32)
    else:
      aligned_binary = aligned_binary.astype(np.float32)
  
    if isinstance(unaligned_binary, list):
      unaligned_binary = np.array(unaligned_binary).astype(np.float32)
    else:
      unaligned_binary = unaligned_binary.astype(np.float32)

    if isinstance(gt_1Dprofile_oneHot, list):
      gt_1Dprofile_oneHot = np.array(gt_1Dprofile_oneHot).astype(np.float32)
    else:
      gt_1Dprofile_oneHot = gt_1Dprofile_oneHot.astype(np.float32)
    '''
    print('token_ids: {},\
           fam_id: {},\
           clan_id: {},\
           masked_lm_positions: {},\
           masked_lm_ids: {},\
           aligned_binary: {},\
           unaligned_binary: {},\
           gt_1Dprofile_oneHot: {}'.format(token_ids.dtype, fam_id.dtype
          , clan_id.dtype, masked_lm_positions.dtype, masked_lm_ids.dtype,
          aligned_binary.dtype, unaligned_binary.dtype, gt_1Dprofile_oneHot.dtype))
    input()
    '''
    ## create feature dict
    features = collections.OrderedDict()
    features['token_ids'] = create_int_feature(token_ids)
    features['fam_id'] = create_int_feature(fam_id)
    features['clan_id'] = create_int_feature(clan_id)
    features['masked_lm_positions'] = create_int_feature(masked_lm_positions)
    features['masked_lm_ids'] = create_int_feature(masked_lm_ids)
    features['aligned_binary'] = create_float_feature(aligned_binary)
    features['unaligned_binary'] = create_float_feature(unaligned_binary)
    features['gt_1Dprofile_oneHot'] = create_bytes_feature(gt_1Dprofile_oneHot)
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    
    fam_acc = "PF{:05d}".format(fam_id[0])
    clan_acc = "CL{:04d}".format(clan_id[0])
    if fam_acc in test_pfam_list:
      test_writers[fam_acc].write(tf_example.SerializeToString())
      test_num += 1
    elif clan_acc in test_clan_list:
      test_writers[clan_acc].write(tf_example.SerializeToString())
      test_num += 1
    else:
      writers[writer_idx].write(tf_example.SerializeToString())
    # update num_written
    if writer_idx in train_wter:
      train_num += 1
    elif writer_idx in val_wter:
      val_num += 1
      
    writer_idx = (writer_idx+1) % len(writers) ## iterative writing

  return ([train_num, val_num, test_num], writer_idx)

def create_tfRecord_dataSet(argv):
  ## copy common global params
  PROJ_DIR = FLAGS.PROJ_DIR
  DATA_PATH = PROJ_DIR+FLAGS.DATA_PATH
  STAT_PATH = PROJ_DIR+FLAGS.STAT_PATH
  OUTPUT_PATH = PROJ_DIR+FLAGS.OUTPUT_PATH
  FAM_CLAN_FILE = PROJ_DIR+FLAGS.FAM_CLAN_FILE
  FLNM_PFAM_FILE = PROJ_DIR+FLAGS.FLNM_PFAM_FILE
  RANDOM_SEED = FLAGS.RANDOM_SEED
  SPLIT = FLAGS.SPLIT
  NUM_SPLIT = FLAGS.NUM_SPLIT
  [MAX_SEQ_LENGTH, MASK_PROB, MAX_MASK_PER_SEQ] = FLAGS.MASK_PARAMS
  SIG_DOM = FLAGS.SIG_DOM
  IN_FORMAT = FLAGS.IN_FORMAT
  MASK_FLAG = FLAGS.MASK_FLAG

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  rng = random.Random(RANDOM_SEED)
  '''
  # holdout set following TAPE
  holdOut_pfam = ['PF01112', 'PF03417', 'PF03577', 'PF14604', 'PF18346', 'PF18697'] # similar in seq but diff evolutionarily
  holdOut_clan = ['CL0635', 'CL0624', 'CL0355', 'CL0100', 'CL0417', 'CL0630'] # novel func
  '''
  # holdout set: mutation set
  holdOut_pfamDir = '{}data_process/pfam_32.0/holdOut_sets/muta_pfam_small_set.txt'.format(PROJ_DIR)
  holdOut_clanDir = '{}data_process/pfam_32.0/holdOut_sets/muta_clan_small_set.txt'.format(PROJ_DIR)
  holdOut_pfams = np.loadtxt(holdOut_pfamDir, dtype='str')
  holdOut_clans = np.loadtxt(holdOut_clanDir, dtype='str')

  # load fileNm-PfamAcc pairs for msa files
  flNm_Pfam = np.loadtxt(FLNM_PFAM_FILE, dtype='str', delimiter=' ')
  
  # initialize writer objects
  writers = []
  train_wter = []
  val_wter = []
  
  # train - val rate: 9:1, 
  # families in test set are stored individually(each file has one family or clan)
  train_idx = np.floor(NUM_SPLIT * .9)
  for i in range(NUM_SPLIT):
    if i < train_idx:
      writers.append(tf.io.TFRecordWriter(OUTPUT_PATH+'/train_seqs_masked_TFRecord_'+str(i)))
      train_wter.append(i)
    elif i >= train_idx:
      writers.append(tf.io.TFRecordWriter(OUTPUT_PATH+'/validation_seqs_masked_TFRecord_'+str(i)))
      val_wter.append(i)
  # test set writers
  test_pfam_num = len(holdOut_pfam)
  test_clan_num = len(holdOut_clan)
  test_wter = {}
  for test_pfam  in holdOut_pfam:
    test_wter[test_pfam] = tf.io.TFRecordWriter(OUTPUT_PATH+'/test_seqs_masked_TFRecord_'+test_pfam)
  for test_clan  in holdOut_clan:
    test_wter[test_clan] = tf.io.TFRecordWriter(OUTPUT_PATH+'/test_seqs_masked_TFRecord_'+test_clan)
  

  # counters for file writting
  writer_idx = 0 
  num_written = [0, 0, 0] #seq counting [train_written, val_written, test_written]


  # loop through file names for each family
  for l in range(flNm_Pfam.shape[0]):
  #for l in range(1):
    [fileNm, famAcc, clanAcc] = flNm_Pfam[l]
    print('>>> Processing file %s from family %s' % (fileNm, famAcc), flush=True)

    # load single position emission probability
    print("*** Load single position match emission probability ***", flush=True)
    aa_1d_prob = np.loadtxt('{}data_process/pfam/Pfam-A.hmm_1d_profile/{}'.format(PROJ_DIR,famAcc), dtype='float', delimiter=' ')

    # transform raw fasta sequence to list of distionarys
    print("*** Reading seqs from msa (fasta/stockholm) files ***", flush=True)
    famAcc_no_version = re.split(r'\.', famAcc)[0] 
    seq_dict_list = raw_msa2dict(DATA_PATH, IN_FORMAT, fileNm, famAcc_no_version, clanAcc)
     
    ##### create sequence instances #####
    print('*** Create training instances ***', flush=True)
    instances = create_training_instances(seq_dict_list, MAX_SEQ_LENGTH, MASK_PROB, MAX_MASK_PER_SEQ, 
                                          aa_1d_prob, rng, single_domain=SIG_DOM)     
    ##### write seqs to TFRecord files #####
    print("*** Writing seqs to TFRecord files***", flush=True)
    (num_written, writer_idx) = write_instance_to_tfrecord_files(instances, OUTPUT_PATH, holdOut_pfam, holdOut_clan,  writers, 
                                                                 test_wter, num_written, writer_idx, [train_wter, val_wter])

  for writer in writers:
    writer.close()
  for key, value in test_wter.items():
    value.close()

  print('In total, write {} instances with train {}, val {}, test {}'.format(sum(num_written), num_written[0], num_written[1],
        num_written[2]), flush=True)
  
  os.system("echo 'In total, write {} instances with train {}, val {}, test {}' > {}/1dProfile_num".format(sum(num_written), num_written[0], num_written[1],num_written[2], STAT_PATH))

def get_clanAcc(famAcc: str = None,famClanMap: dict = None):
    '''
    return clan accession name for given fam name
    None if no clan name
    '''
    if re.search(r'\.\d+',famAcc) is not None:
        famAcc = re.split(r'\.', famAcc)[0]
    if famAcc[:2] != 'PF':
        famAcc = 'PF{}'.format(famAcc)

    if len(famClanMap[famAcc]) == 0:
        return None
    elif len(famClanMap[famAcc]) > 0:
        return famClanMap[famAcc]
    else:
      Exception('No such family {}'.format(famAcc))

def create_finetune_dataSet(
    working_path: str = None,
    famClanMap: dict = None,
    seq_dir: str = None,
    seq_format: str = 'stockholm',
    family_list: List = None,
    familyList_file: str = None,
    output_dir: str = None,
    output_format: str = 'lmdb',
    iden_cutoff: float = 0.8,
    align_pairwise: bool = False,
    reweight_bool: bool = True,
    use_diamond: bool = False,
    diamond_path: str = None,
    use_mmseqs2: bool = True,
    mmseqs2_path: str = None,
    len_hist: bool = False,
    weight_in_header: bool = False):

  """
  Prepare finetune dataset (lmdb format) from Pfam MSA files
  
  Descriptions
    * extract unaligned seq
    * calculate weighting for each seq 
      * reciprocal of the number of neighbors for each sequence at a minimum identity of 80%
    * calculate weighting for each family
      * reciprocal of sum of seq weights in this family 
  
  Parameters
    * align_pairwise: bool, True if aligned seqs are provided

  Outputs
    * processed data saved in lmdb format
  """
  if family_list is not None:
    family_list = np.asarray(family_list,dtype='str').reshape((-1,))
  ## load family list to process
  if familyList_file is not None:
    family_list = np.loadtxt('{}/{}'.format(working_path,familyList_file), dtype='str', delimiter=',')

  # loop through family list
  for l in range(family_list.shape[0]):
  #for l in range(1):
    famAcc = family_list[l]
    print('>>> Processing %s' % (famAcc), flush=True)

    ## get clan 
    #clanAcc = get_clanAcc(famAcc,famClanMap)

    # transform raw fasta sequence to list of distionarys
    print("*** Reading seqs from msa (fasta/stockholm) files ***", flush=True)
    ## remove family version number
    #if re.search(r'\.\d+',famAcc) is not None:
    #  famAcc = re.split(r'\.', famAcc)[0]
   
    seq_dict_list,family_reweight = raw_msa2dict(
                                      data_path=f'{working_path}/{seq_dir}',
                                      input_format=seq_format,
                                      fileNm=f'{famAcc}.{seq_format}',  #famAcc,
                                      famAcc=famAcc,
                                      clanAcc=None, #clanAcc,
                                      aligned=True,
                                      weight_in_header=weight_in_header,
                                      keep_X=True)
    
    print('>>> Num of seqs: {}'.format(len(seq_dict_list)))
    if reweight_bool:
      if use_diamond:
        print('>>Using Diamond')
        seqReweight_dict = {}
        ## stockholm to fasta
        os.system('esl-reformat -u -o {}/{}/{}.fa fasta {}/{}/{}'.format(diamond_path,famAcc,famAcc,working_path,seq_dir,famAcc))
        ## build diamond database
        os.system('diamond makedb --in {}/{}/{}.fa -d {}/{}/{}'.format(diamond_path,famAcc,famAcc,diamond_path,famAcc,famAcc))
        ## diamond search
        # get split fasta names
        fasta_splitList = os.popen("ls -v {}/{}/splits".format(diamond_path,famAcc)).readlines() #contain '\n'
        # loop fasta splits
        for fa_split in fasta_splitList:
          fa_split = fa_split.replace('\n','')
          # search
          #dmSearch_out = os.popen(f'diamond blastp -q {diamond_path}/splits/{fa_split} -d {diamond_path}/{famAcc}/{famAcc} -o {diamond_path}/{famAcc}/{famAcc}.out.tsv -v -k0 --compress 1 -f 6 qseqid sseqid pident length mismatch').readlines()
          os.system(f'diamond blastp -q {diamond_path}/splits/{fa_split} -d {diamond_path}/{famAcc}/{famAcc} -o {diamond_path}/{famAcc}/{famAcc}.out.tsv -v -k0 --compress 1 -f 6 qseqid sseqid pident length mismatchi')
          # loop each seq
          with open(f'{diamond_path}/splits/{fa_split}') as handle:
            for record in SeqIO.parse(handle, "fasta"):
              seqId = record.id
              # extract lines for this seq
              iden_tar = os.popen(f"grep '^{seqId}' {diamond_path}/{famAcc}/{famAcc}.out.tsv | cut -d$'\t' -f3").readlines()
              iden_tar = np.array([float(iden.strip('\n')) for iden in iden_tar])
              num_neighbors = np.sum(iden_tar >= iden_cutoff) - 1 
              seqReweight_dict[seqId] == 1. / num_neighbors
              ##TODO unfinished
      elif use_mmseqs2: ## use mmSeqs2 to cluster seqs, then use reweight = 1/cluster_size for seqs in one cluster
        print('>>Using MMseqs2')
        ## stockholm to fasta
        if not os.path.isdir(f'{mmseqs2_path}/{famAcc}'):
          os.mkdir(f'{mmseqs2_path}/{famAcc}')
          os.mkdir(f'{mmseqs2_path}/{famAcc}/tmpDir')
        os.system(f'esl-reformat -u -o {mmseqs2_path}/{famAcc}/{famAcc}.fa fasta {working_path}/{seq_dir}/{famAcc}.{seq_format}')
        ## run mmseqs2 (BFD -c 0.9 --cov-mode 1)
        os.system(f'mmseqs easy-linclust {mmseqs2_path}/{famAcc}/{famAcc}.fa {mmseqs2_path}/{famAcc}/{famAcc}_mmseqs2 {mmseqs2_path}/{famAcc}/tmpDir -c 0.9 --cov-mode 1 --cluster-mode 2 --alignment-mode 3 --min-seq-id {iden_cutoff}')
        ## loop representative seqs
        seqReweight_dict = {}
        family_reweight = 0.
        with open(f'{mmseqs2_path}/{famAcc}/{famAcc}_mmseqs2_rep_seq.fasta') as handle:
          for record in SeqIO.parse(handle,"fasta"):
            seqId = record.id
            ## get neighbor seqIds and nums
            num_neighbors = int(os.popen(f"grep -c '{seqId}' {mmseqs2_path}/{famAcc}/{famAcc}_mmseqs2_cluster.tsv").read().strip('\n'))-1# remove count of itself
            assert num_neighbors >= 0
            if num_neighbors == 0:
              seqReweight_dict[seqId] = 1.0
              family_reweight += 1.0
            else:
              seq_neighbors = os.popen(f"grep '{seqId}' {mmseqs2_path}/{famAcc}/{famAcc}_mmseqs2_cluster.tsv").read().strip('\n').split('\n')
              for seq_pair in seq_neighbors:
                seqNei_id = seq_pair.split('\t')[-1]
                seqReweight_dict[seqNei_id] = 1./num_neighbors
                family_reweight += 1./num_neighbors
        ## check num of seqs in seqReweight_dict
        ## CAUTION: seqs containing 'X'
        #assert len(seq_dict_list) == len(seqReweight_dict)
        ## assign seq reweight score
        for seq_dict in seq_dict_list:
          seqId = '{}/{}'.format(seq_dict['unpIden'],seq_dict['range'])
          seq_dict['seq_reweight'] = seqReweight_dict[seqId]
      else: ## calculate pairwise %identity mamually
        print('>>manual pairwise iden')
        start_time = time.time()
        print("*** Calculating sequence reweighting scores ***", flush=True)
        ## calculate reweighting score for each sequence and whole family
        family_reweight = 0.
        for seq_dict_query in seq_dict_list:
          idenScore_list = []
          for seq_dict in seq_dict_list:
            if align_pairwise:
              iden_score = seq_align_identity(seq_dict_query['primary'],seq_dict['primary'],matrix=blosum_matrix)
            else:
              iden_score = seq_identity(seq_dict_query['msa_seq'],seq_dict['msa_seq'])
            idenScore_list.append(iden_score)
          idenScore_list = np.array(idenScore_list).astype(float)
          ## exclude compare to itself(-1), avoid devided by 0
          num_similar_neighbors = np.sum(idenScore_list >= iden_cutoff) - 1.
          seq_reweight = min(1., 1. / (num_similar_neighbors + 1e-6))
          seq_dict_query['seq_reweight'] = seq_reweight
          family_reweight += seq_reweight
        end_time = time.time()
        print('>>> Takes {}s'.format(end_time - start_time))
    
    ## save family reweighting score, seq_length, uniq_char
    seqLen_list = []
    uniq_chars = []
    if weight_in_header:
      assert family_reweight > 0.
    for seq_dict in seq_dict_list:
      #rand_num = rng.random()
      if reweight_bool or weight_in_header:
        seq_dict['family_reweight'] = family_reweight
      seqLen_list.append(seq_dict['protein_length'])
      seq = seq_dict['primary']
      uniq_chars = list(set(uniq_chars + list(seq)))
    print("*** Save data and draw figures ***", flush=True)
    if len_hist:
      ## seq length histogram figure
      fig = plt.figure()
      plt.hist(seqLen_list, density=False, bins=50)  # density=False would make counts
      plt.ylabel('Count')
      plt.xlabel('Length')
      plt.savefig('{}/{}/seqLenDist_{}.png'.format(working_path,output_dir,famAcc))
      plt.close()
      ## save
      np.savetxt('{}/{}/seqLenList_{}'.format(working_path,output_dir,famAcc),seqLen_list,fmt='%s',delimiter=',')
      np.savetxt('{}/{}/uniqCharList_{}'.format(working_path,output_dir,famAcc),uniq_chars,fmt='%s',delimiter=',')
    
    if output_format == 'json':
      with open('{}/{}/{}.json'.format(working_path,output_dir,famAcc),'w') as fl2wt:
        json.dump(seq_dict_list,fl2wt)
    elif output_format == 'lmdb':
      wrtEnv = lmdb.open('{}/{}/{}.lmdb'.format(working_path,output_dir,famAcc), map_size=(1024 * 20)*(2 ** 20))
      with wrtEnv.begin(write=True) as txn:
        for i, entry in enumerate(seq_dict_list):
          txn.put(str(i).encode(), pkl.dumps(entry))
        txn.put(b'num_examples', pkl.dumps(i + 1))
      wrtEnv.close()
    else:
      Exception('invalid output format: {}'.format(output_format))
    print('*** In total, write {} instances ***'.format(len(seq_dict_list)), flush=True)
  return None

def create_json_dataSet(argv):
  '''
  Output:
  *train, validation, test set: list, each element a dist for a seq
  '''
  ## copy common global params
  PROJ_DIR = FLAGS.PROJ_DIR
  DATA_PATH = PROJ_DIR+FLAGS.DATA_PATH
  STAT_PATH = PROJ_DIR+FLAGS.STAT_PATH
  OUTPUT_PATH = PROJ_DIR+FLAGS.OUTPUT_PATH
  FAM_CLAN_FILE = PROJ_DIR+FLAGS.FAM_CLAN_FILE
  FLNM_PFAM_FILE = PROJ_DIR+FLAGS.FLNM_PFAM_FILE
  RANDOM_SEED = FLAGS.RANDOM_SEED
  SPLIT = FLAGS.SPLIT
  NUM_SPLIT = FLAGS.NUM_SPLIT
  [MAX_SEQ_LENGTH, MASK_PROB, MAX_MASK_PER_SEQ] = FLAGS.MASK_PARAMS
  SIG_DOM = FLAGS.SIG_DOM
  IN_FORMAT = FLAGS.IN_FORMAT
  MASK_FLAG = FLAGS.MASK_FLAG

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  rng = random.Random(RANDOM_SEED)
  
  # holdout set: mutation set
  holdOut_pfamDir = '{}data_process/pfam_32.0/holdOut_sets/muta_pfam_small_set.txt'.format(PROJ_DIR)
  holdOut_clanDir = '{}data_process/pfam_32.0/holdOut_sets/muta_clan_small_set.txt'.format(PROJ_DIR)
  holdOut_pfams = np.loadtxt(holdOut_pfamDir, dtype='str')
  holdOut_clans = np.loadtxt(holdOut_clanDir, dtype='str')

  # load fileNm-PfamAcc pairs for msa files
  flNm_Pfam = np.loadtxt(FLNM_PFAM_FILE, dtype='str', delimiter=' ')
  
  # train - val rate: 9:1, 
  train_set = []
  val_set = []
  holdOut_set = []

  # counters for file writting
  writer_idx = 0 
  num_written = [0, 0, 0] #seq counting [train_written, val_written, test_written]


  # loop through file names for each family
  for l in range(flNm_Pfam.shape[0]):
  #for l in range(1):
    [fileNm, famAcc, clanAcc] = flNm_Pfam[l]
    print('>>> Processing file %s from family %s' % (fileNm, famAcc), flush=True)

    # transform raw fasta sequence to list of distionarys
    print("*** Reading seqs from msa (fasta/stockholm) files ***", flush=True)
    famAcc_no_version = re.split(r'\.', famAcc)[0] 
    seq_dict_list = raw_msa2dict(DATA_PATH, IN_FORMAT, fileNm, famAcc_no_version, clanAcc)
    if famAcc in holdOut_pfams or clanAcc in holdOut_clans:
      # add 'id' pair
      tmp_id = 0
      for seq_dict in seq_dict_list:
        seq_dict["id"] = str(tmp_id)
        tmp_id += 1
      # save a individual file
      with open('{}/holdout_indiSet/pfam_holdout_{}.json'.format(OUTPUT_PATH,famAcc_no_version),'w') as fl2wt:
        json.dump(seq_dict_list,fl2wt)
      
      # append to whole holdout set
      for seq_dict in seq_dict_list:
        seq_dict['id'] = str(num_written[2])
        num_written[2] += 1
      holdOut_set.extend(seq_dict_list)
    else:
      for seq_dict in seq_dict_list:
        rand_num =  rng.random()
        if rand_num < 0.1:
          seq_dict["id"] = str(num_written[1])
          num_written[1] += 1
          val_set.append(seq_dict)
        else:
          seq_dict["id"] = str(num_written[0])
          num_written[0] += 1
          train_set.append(seq_dict)
  # save three sets
  with open('{}/pfam_train.json'.format(OUTPUT_PATH),'w') as fl2wt:
    json.dump(train_set,fl2wt)
  with open('{}/pfam_valid.json'.format(OUTPUT_PATH),'w') as fl2wt:
    json.dump(val_set,fl2wt)
  with open('{}/pfam_holdout.json'.format(OUTPUT_PATH),'w') as fl2wt:
    json.dump(holdOut_set,fl2wt)


  print('In total, write {} instances with train {}, val {}, test {}'.format(sum(num_written), num_written[0], num_written[1], num_written[2]), flush=True)

  os.system("echo 'In total, write {} instances with train {}, val {}, test {}' > {}/seq_json_num".format(sum(num_written), num_written[0], num_written[1],num_written[2], STAT_PATH))

def randomSplit_train_val(data_path: str = None,
                          flNm_list: List = None,
                          val_ratio: float = 0.1,
                          data_format: str = 'lmdb',
                          map_size: int = (1024 * 20)*(2 ** 20)):
  """ randomly split all data into train and validation set with a ratio
  """
  random.seed(25)
  if data_format == 'lmdb':
    for fl in flNm_list:
      print(f'>> {fl}')
      ## load all data
      all_data = []
      input_env = lmdb.open(f'{data_path}/{fl}.{data_format}', readonly=True, lock=False, readahead=False, meminit=False)
      with input_env.begin(write=False) as txn:
        num_examples = pkl.loads(txn.get(b'num_examples'))
        for idx in range(num_examples):
          item = pkl.loads(txn.get(str(idx).encode()))
          all_data.append(item)
      ## split data
      train_list = []
      val_list = []
      for one_dt in all_data:
        #if 'X' in one_dt['primary'].upper(): ## exclude seq containing 'X'
        #  pass
        prob = random.random()
        if prob < val_ratio:
          val_list.append(one_dt)
        else:
          train_list.append(one_dt)
      ## save data
      wrtEnv = lmdb.open(f'{data_path}/train_{fl}.lmdb',map_size=map_size)
      with wrtEnv.begin(write=True) as txn:
        for i, entry in enumerate(train_list):
          txn.put(str(i).encode(), pkl.dumps(entry))
        txn.put(b'num_examples', pkl.dumps(i+1))
      wrtEnv.close()

      wrtEnv = lmdb.open(f'{data_path}/val_{fl}.lmdb',map_size=map_size)
      with wrtEnv.begin(write=True) as txn:
        for i, entry in enumerate(val_list):
          txn.put(str(i).encode(), pkl.dumps(entry))
        txn.put(b'num_examples', pkl.dumps(i+1))
      wrtEnv.close()

      ## report
      print(f'>> split results: train: {len(train_list)}; validation: {len(val_list)}; train-val ratio: {len(train_list)/len(val_list):0.3f}:1')
  elif data_format in ['stockholm','fasta']:
    for fl in flNm_list:
      print(f'>> {fl}')
      ## split data
      train_list = []
      val_list = []
      with open(f'{data_path}/{fl}.{data_format}') as handle:
        for record in SeqIO.parse(handle,data_format):
          #seqId = record.id
          #seqRaw = record.seq
          prob = random.random()
          if prob < val_ratio:
            val_list.append(record)
          else:
            train_list.append(record)
      with open(f'{data_path}/train_{fl}.{data_format}', "w") as output_handle:
        SeqIO.write(train_list, output_handle, data_format)
      with open(f'{data_path}/val_{fl}.{data_format}', "w") as output_handle:
        SeqIO.write(val_list, output_handle, data_format)
  return None

def combineDataset(data_path: str = None,
                  flNm_list: List = None,
                  combinedFileNameList: List = None,
                  data_format: str = 'lmdb',
                  map_size: int = (1024 * 20)*(2 ** 20)):
  """ Combine multiple files into single one

  Arguments:
  * flNm_list: e.g. [[file1_1, file1_2], [file2_1, file2_2, file2_3]]
  """
  if data_format == 'lmdb':
    for i in range(len(flNm_list)):
      flSet = flNm_list[i]
      combinedFileName = combinedFileNameList[i]
      print(f'>> {flSet}')
      all_data = []
      ## load all example into all_data
      for flSig in flSet:
        input_env = lmdb.open(f'{data_path}/{flSig}.lmdb', readonly=True, lock=False, readahead=False, meminit=False)
        with input_env.begin(write=False) as txn:
          num_examples = pkl.loads(txn.get(b'num_examples'))
          print(f'>>{flSig},{num_examples}')
          for idx in range(num_examples):
            item = pkl.loads(txn.get(str(idx).encode()))
            all_data.append(item)
      ## save data
      wrtEnv = lmdb.open(f'{data_path}/{combinedFileName}.lmdb',map_size=map_size)
      with wrtEnv.begin(write=True) as txn:
        for i, entry in enumerate(all_data):
          txn.put(str(i).encode(), pkl.dumps(entry))
        txn.put(b'num_examples', pkl.dumps(i+1))
        print(f'>>{combinedFileName},{i+1}')
      wrtEnv.close()

def checkDataNum(data_path: str = None,
                 flNm_list: List = None,
                 data_format: str = 'lmdb',
                 output_file: str = None):
  output_list = []
  if data_format == 'lmdb':
    for fl in flNm_list:
      input_env = lmdb.open(f'{data_path}/{fl}.lmdb', readonly=True, lock=False, readahead=False, meminit=False)
      with input_env.begin(write=False) as txn:
        num_examples = pkl.loads(txn.get(b'num_examples'))
      print(f'>> {fl}: {num_examples}')
      output_list.append([fl, num_examples])
  # save output
  np.savetxt(f'{data_path}/{output_file}.csv',output_list,fmt='%s',delimiter=',')

def filterByLen(argv):
  """
  filter out sequences with length <= 500
  """
  PROJ_DIR = FLAGS.PROJ_DIR 
  OUTPUT_PATH = PROJ_DIR+FLAGS.OUTPUT_PATH
  STAT_PATH = PROJ_DIR+FLAGS.STAT_PATH

  # load json data
  with open('{}/pfam_train.json'.format(OUTPUT_PATH),'r') as fl:
    train_json = json.load(fl)
  with open('{}/pfam_valid.json'.format(OUTPUT_PATH),'r') as fl:
    val_json = json.load(fl)
  with open('{}/pfam_holdout.json'.format(OUTPUT_PATH),'r') as fl:
    test_json = json.load(fl)

  len_cutoff = 500
  # loop through json and filter seq by length
  train_lenCut = []
  val_lenCut = []
  test_lenCut = []

  for train_one in train_json:
    if int(train_one['protein_length']) <= len_cutoff:
      train_lenCut.append(train_one)

  for val_one in val_json:
    if int(val_one['protein_length']) <= len_cutoff:
      val_lenCut.append(val_one)
  
  for test_one in test_json:
    if int(test_one['protein_length']) <= len_cutoff:
      test_lenCut.append(test_one)

  # save to json files
  with open('{}/pfam_train_lenCut.json'.format(OUTPUT_PATH),'w') as fl2wt:
    json.dump(train_lenCut,fl2wt)
  with open('{}/pfam_valid_lenCut.json'.format(OUTPUT_PATH),'w') as fl2wt:
    json.dump(val_lenCut,fl2wt)
  with open('{}/pfam_holdout_lenCut.json'.format(OUTPUT_PATH),'w') as fl2wt:
    json.dump(test_lenCut,fl2wt)
  
  num_written = [len(train_lenCut), len(val_lenCut), len(test_lenCut)]
  print('In total, write {} instances with train {}, val {}, test {}'.format(sum(num_written), num_written[0], num_written[1], num_written[2]), flush=True)

  os.system("echo 'In total, write {} instances with train {}, val {}, test {}' > {}/seq_json_num_lenCut".format(sum(num_written), num_written[0], num_written[1],num_written[2], STAT_PATH))

def checkSeqLen(argv):
  """
  check sequence length distribution
  """
  PROJ_DIR = FLAGS.PROJ_DIR 
  OUTPUT_PATH = PROJ_DIR+FLAGS.OUTPUT_PATH
  STAT_PATH = PROJ_DIR+FLAGS.STAT_PATH

  # load json data
  with open('{}/pfam_train.json'.format(OUTPUT_PATH),'r') as fl:
    train_json = json.load(fl)
  with open('{}/pfam_valid.json'.format(OUTPUT_PATH),'r') as fl:
    val_json = json.load(fl)
  with open('{}/pfam_holdout.json'.format(OUTPUT_PATH),'r') as fl:
    test_json = json.load(fl)

  # loop through json and filter seq by length
  train_lenDis = []
  val_lenDis = []
  test_lenDis = []
  train_collect = []
  for train_one in train_json:
    assert len(train_one['primary']) == int(train_one['protein_length'])
    train_lenDis.append(train_one['protein_length'])
    if int(train_one['protein_length']) > 1000:
      train_collect.append(train_one['id'])

  for val_one in val_json:
    assert len(val_one['primary']) == int(val_one['protein_length'])
    val_lenDis.append(val_one['protein_length'])
  
  for test_one in test_json:
    assert len(test_one['primary']) == int(test_one['protein_length'])
    test_lenDis.append(test_one['protein_length'])
  
  all_lenDis = train_lenDis + val_lenDis + test_lenDis

  # draw histgram of sequence length 
  fig = plt.figure(0)
  plt.hist(all_lenDis, 50)
  fig.savefig('{}/json_all_lenDis.png'.format(STAT_PATH))
  os.system("echo 'train min_len:{},max_len:{}; val min_len:{},max_len:{}; test min_len:{},max_len:{}' > {}/seq_json_lenDis".format(min(train_lenDis),max(train_lenDis),min(val_lenDis),max(val_lenDis),min(test_lenDis),max(test_lenDis),STAT_PATH))
  print(train_collect)

def selectModel(work_path: str = None,
                fam_list: List = None,                
                rpSet_list: List = ['rp75_all_1'],
                epoch_list: List = None,
                topK: int = 3):
  """select top K epochs for each family as starting point for finetune
     based on language modeling perplexity of family sequences

     Paramaters:

     Outputs:

  """
  ## load muta set name list
  #muta_pfam_clan_list = np.loadtxt('/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/mutagenesis/DeepSequenceMutaSet_pfam_clan', dtype='str',delimiter=',')
  
  ## dict holding mdl_subIdx for each epock
  # rp75_all_1
  mdl_subsubIdx_dict = {
  'rp75_all_1': {
    '1': [104],
    '2': [119,134,149],
    '3': [164,179],
    '4': [194,209,224]
    },
  'rp15_all_1': {
    '2': [459],
    '3': [489,519],
    '4': [549,579,609,639,669,699],
    '5': [729]
    },
  'rp15_all_2': {
    '2': [459,489,519,549,579,609,639,669],
    '3': [699,729]
    },
  'rp15_all_3': {
    '1': [459],
    '2': [489,519,549,579,609,639,669,699],
    '3': [729]
    },
  'rp15_all_4': {
    '1': [459,489],
    '2': [519,549,579,609,639,669,699,729]
    }
  }

  ## loop over each family
  for famId in fam_list:
    metric_values = []
    model_ids = []
    ## loop over rpSet
    for rpSet in rpSet_list:
      rp_split = re.split(r'_',rpSet)
      rpNm = f"{rp_split[0]}_{rp_split[1]}"
      mdl_subIdx = rp_split[2]
      ## loop over epochs within each rpSet
      for epoch in epoch_list:
        ## query model path from log file and load metric values from json
        log_file = f'{work_path}/job_logs/archive_baseline_bert_eval/baseline_bert_{rpSet}_torch_eval.{famId}_lm.0.{epoch}.out'
        #mdl_dir = os.popen(f"grep 'loading weights file' {log_file} | tail -n1 | cut -d'/' -f12").read().strip('\n')
        #eval_epoch = os.popen(f"grep 'loading weights file' {log_file} | tail -n1 | cut -d'/' -f13 | cut -d'_' -f3 | cut -d'.' -f1").read().strip('\n')
        with open(log_file) as handle:
          mdl_dir = re.findall(r'loading weights file.+(masked_language_modeling_transformer.+)\/pytorch_model.+',handle.read())[0]
        ## query mdl_subsubIdx
        for subI,epoL in mdl_subsubIdx_dict[rpSet].items():
          if int(epoch) in epoL:
            mdl_subsubIdx = subI
        with open(f'{work_path}/results_to_keep/{rpNm}/{rpNm}_pretrain_{mdl_subIdx}_models/{mdl_dir}/results_metrics_finetune_datasets_{famId}_{epoch}.json') as handle:
          metric_json = json.load(handle)
        #print(f"{rpNm}_{mdl_subIdx};{famId};{mutaNm};{metric_json['lm_ece']},{metric_json['accuracy']}")
        metric_values.append(metric_json['lm_ece'])
        model_ids.append(f'{rpSet}_{mdl_subsubIdx}_{epoch}')
    ## report mean/std ppl over all epochs and topK models
    topK_idx = np.argsort(metric_values)[:topK]
    #print(f'{famId},{np.mean(metric_values)},{np.std(metric_values)},{[model_ids[i] for i in topK_idx]},{[metric_values[i] for i in topK_idx]}')
    print(f"{famId},{' '.join([model_ids[i] for i in topK_idx])}")

def process_shin2021Data(seq_dir: str=None,
                        setNm_file: str=None,
                        save_dir: str=None,
                        use_mmseqs2: bool=False,
                        mmseqs2_path: str=None,
                        update_data: bool=False,
                        save_data: bool=False,
                        report_weights: bool=False):
  """ 
  Process Shin2021's sequence data for finetune
  """
  ## load set names
  setNm_list = np.loadtxt(f'{seq_dir}/{setNm_file}',dtype='str',delimiter='\t')
  if report_weights:
    weight_report_wrt = open(f'{seq_dir}/report_weights.tsv','w')
    weight_report_wrt.write(f'file_name\tset_name\tseqL\tMeff\tMeff_mmseqs2\tMtotal\n')

  ## loop over sets
  for i in range(setNm_list.shape[0]):
    seqFile_name = setNm_list[i,0]
    save_name = setNm_list[i,1]
    iden_cutoff = setNm_list[i,2]
    range_list = setNm_list[i,4].split('-')
    print(f'>>{seqFile_name}')
    
    ## use mmseqs2
    if use_mmseqs2: ## use mmSeqs2 to cluster seqs, then use reweight = 1/cluster_size for seqs in one cluster
      print('>>Doing MMseqs2')
      ## run mmseqs2
      os.system(f'mmseqs easy-linclust {seq_dir}/sequences/{seqFile_name}.fa {seq_dir}/sequences_mmseqs2/{save_name}_mmseqs2 {mmseqs2_path}/tmpDir -c {iden_cutoff} --cov-mode 1 --cluster-mode 2 --alignment-mode 3 --min-seq-id {iden_cutoff}')
      ## loop representative seqs
      seqReweight_dict = {}
      family_reweight_mmseqs2 = 0.
      with open(f'{seq_dir}/sequences_mmseqs2/{save_name}_mmseqs2_rep_seq.fasta') as handle:
        for record in SeqIO.parse(handle,"fasta"):
          seqId_split = record.id.split(':')
          seqId = seqId_split[0] # AMIE_PSEAE/1-346
          ## get neighbor seqIds and nums
          num_neighbors = int(os.popen(f"grep -c '{seqId}' {seq_dir}/sequences_mmseqs2/{save_name}_mmseqs2_cluster.tsv").read().strip('\n'))-1# remove count of itself
          assert num_neighbors >= 0
          if num_neighbors == 0:
            seqReweight_dict[seqId] = 1.0
            family_reweight_mmseqs2 += 1.0
          else:
            seq_neighbors = os.popen(f"grep '{seqId}' {seq_dir}/sequences_mmseqs2/{save_name}_mmseqs2_cluster.tsv").read().strip('\n').split('\n')
            for seq_pair in seq_neighbors:
              seqNei_id = seq_pair.split('\t')[-1].split(':')[0]
              seqReweight_dict[seqNei_id] = 1./num_neighbors
              family_reweight_mmseqs2 += 1./num_neighbors
    
    ## update mmseqs2 reweight in train/val splits
    if update_data:
      print('>>Updating train/val')
      for spl_set in ['train_','val_','']:
        wrtEnv = lmdb.open(f'{save_dir}/{spl_set}{save_name}.lmdb', map_size=(1024 * 20)*(2 ** 20))
        with wrtEnv.begin(write=True) as txn:
          num_examples = pkl.loads(txn.get(b'num_examples'))
          for i in range(num_examples):
            item = pkl.loads(txn.get(str(i).encode()))
            seqId = item['unp_range']
            seqPri = item['primary']
            # item.update({'primary': str(seqPri),
            #              'seq_reweight_mmseqs2': seqReweight_dict[seqId],
            #              'family_reweight_mmseqs2': family_reweight_mmseqs2,})
            item.update({'primary': str(seqPri)})
            txn.replace(str(i).encode(), pkl.dumps(item))
        wrtEnv.close()
    if save_data:
      dt2save = []
      family_reweight = 0.
      ## load Shin2021's fasta seq
      print('>>process Shin fasta')
      with open(f'{seq_dir}/sequences/{seqFile_name}.fa') as handle:
        for record in SeqIO.parse(handle, "fasta"):
          seqId_split = record.id.split(':')
          seqId = seqId_split[0] # AMIE_PSEAE/1-346
          reweight_score = float(seqId_split[1])
          family_reweight += reweight_score
          ## get mmseqs2 reweight
          reweight_mmseqs2 = seqReweight_dict[seqId]
          dt2save.append({'unp_range': seqId,
                          'primary': str(record.seq),
                          'seq_reweight_mmseqs2': reweight_mmseqs2,
                          'family_reweight_mmseqs2': family_reweight_mmseqs2,
                          'seq_reweight': reweight_score})
      ## append family_reweight
      for dt in dt2save:
        dt['family_reweight'] = family_reweight
      ## save to lmdb
      if not os.path.isdir(f'{save_dir}'):
        os.makedirs(f'{save_dir}')
      wrtEnv = lmdb.open(f'{save_dir}/{save_name}.lmdb', map_size=(1024 * 20)*(2 ** 20))
      with wrtEnv.begin(write=True) as txn:
        for i, entry in enumerate(dt2save):
          txn.put(str(i).encode(), pkl.dumps(entry))
        txn.put(b'num_examples', pkl.dumps(i + 1))
      wrtEnv.close()
    if report_weights: ## report Meff and L
      ## load Meff and L from saved lmdb data (just load from the first example)
      wrtEnv = lmdb.open(f'{save_dir}/{save_name}.lmdb', map_size=(1024 * 20)*(2 ** 20))
      with wrtEnv.begin(write=True) as txn:
        num_examples = pkl.loads(txn.get(b'num_examples'))
        item = pkl.loads(txn.get(str(0).encode()))
      seqId = item['unp_range'].split('/')[0]
      family_reweight_mmseqs2 = item['family_reweight_mmseqs2']
      family_reweight = item['family_reweight']
      seqL = int(range_list[1]) - int(range_list[0]) + 1
      weight_report_wrt.write(f'{seqFile_name}\t{save_name}\t{seqL}\t{family_reweight}\t{family_reweight_mmseqs2}\t{num_examples}\n')
  if report_weights:
    weight_report_wrt.close()

if __name__ == "__main__":
  work_path = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark'
  
  ## params for create_finetuen_dataSet
  diamond_path = '/scratch/user/sunyuanfei/Projects/diamond_run'
  mmseqs2_path = '/scratch/user/sunyuanfei/Projects/mmseqs2_run'
  famClanMap_path = 'data_process/pfam_34.0/Pfam-A.clans.tsv'
  #fam_clan_map = np.loadtxt('{}/{}'.format(work_path,famClanMap_path),dtype='str',delimiter='\t')
  fam_clan_dict = {}
  # for i in range(fam_clan_map.shape[0]):
  #   famAcc = fam_clan_map[i][0]
  #   clanAcc = fam_clan_map[i][1]
  #   fam_clan_dict[famAcc] = clanAcc
  
  ## cmd line argv: iden_cutoff, family_1, family_2, family_... 
  #base_dir = '{}/data_process/pfam_34.0'.format(work_path)
  seq_dir = 'Pfam-A.full.uniprot'
  #proj_name = sys.argv[1] # cagi6, mutagenesis
  #iden_cutoff = sys.argv[2]
  #family_list = []
  #for i in range(3,len(sys.argv)):
  #  family_list.append(sys.argv[i])
  task2run = sys.argv[1]
  if task2run == 'create_finetune_dataSet':
    create_finetune_dataSet(
      working_path = f'{work_path}/data_process/stability/cdna-display',
      famClanMap = fam_clan_dict,
      seq_dir = 'selected_domains',
      seq_format = 'stockholm', #'fasta'
      family_list = ['PF00018','PF01378','PF00313','PF02209','PF00046','PF02216','PF00226','PF00240','PF13499','PF00397','PF08239'],
      familyList_file = None,
      output_dir = 'selected_domains', # 'finetune_datasets','finetune_datasets_noReweight',
      output_format = 'lmdb',
      iden_cutoff = 0.8,
      align_pairwise = False,
      reweight_bool = True,
      use_diamond = False,
      diamond_path = None,
      use_mmseqs2 = True,
      mmseqs2_path = mmseqs2_path,
      len_hist = True,
      weight_in_header = False)
  elif task2run == 'process_shin2021Data':
    process_shin2021Data(
      seq_dir='/scratch/user/sunyuanfei/Projects/seqdesign-pytorch/examples/datasets',
      setNm_file='sequence_list',
      save_dir='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/mutagenesis/shin2021_data/finetune_seqData_withX',
      use_mmseqs2 = False,
      mmseqs2_path = '/scratch/user/sunyuanfei/Projects/mmseqs2_run',
      update_data = False,
      save_data=False,
      report_weights=True)
    #muta:'PF00071','PF00076','PF00097','PF00145','PF00172','PF00179','PF00218','PF00232','PF00240','PF00397','PF00475','PF00509','PF00516_PF00517','PF00533','PF00595','PF00603','PF00607','PF00782_PF10409','PF00795','PF01176','PF01636','PF03693','PF03702','PF04263_PF04265','PF04564','PF05724','PF08300','PF11976','PF13354','PF13499'
    #cagi6: 'PF01379_PF03900','PF13499','PF00069'
    #Shin2021: 'AMIE_PSEAE','B3VI55_LIPST','BF520_env','BG505_env','BG_STRSQ','BLAT_ECOLX','BRCA1_HUMAN_BRCT','BRCA1_HUMAN_RING','CALM1_HUMAN','DLG4_RAT','GAL4_YEAST','HG_FLU','HIS7_YEAST','HSP82_YEAST','IF1_ECOLI','KKA2_KLEPN','MK01_HUMAN','MTH3_HAEAESTABILIZED','TIM_THETH','PABP_YEAST','PA_FLU','POLG_HCVJF','PTEN_HUMAN','RASH_HUMAN','RL401_YEAST','SUMO1_HUMAN','TPK1_HUMAN','TPMT_HUMAN','TIM_SULSO','TIM_THEMA','UBC9_HUMAN','UBE4B_MOUSE','YAP1_HUMAN'
    #['AMIE_PSEAE','B3VI55_LIPST','BF520_env','BG505_env','BG_STRSQ','BLAT_ECOLX','BRCA1_HUMAN_BRCT','BRCA1_HUMAN_RING','CALM1_HUMAN','DLG4_RAT','GAL4_YEAST','HG_FLU','HIS7_YEAST','HSP82_YEAST','IF1_ECOLI','KKA2_KLEPN','MK01_HUMAN','MTH3_HAEAESTABILIZED','TIM_THETH','PABP_YEAST','PA_FLU','POLG_HCVJF','PTEN_HUMAN','RASH_HUMAN','RL401_YEAST','SUMO1_HUMAN','TPK1_HUMAN','TPMT_HUMAN','TIM_SULSO','TIM_THEMA','UBC9_HUMAN','UBE4B_MOUSE','YAP1_HUMAN']
  elif task2run == 'randomSplit_train_val':
    randomSplit_train_val(
      data_path = '/scratch/user/sunyuanfei/Projects/CAGI6/ARSA',
      #'/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/mutagenesis/shin2021_data/finetune_seqData'
      flNm_list = ['PF00884.alignment.uniprot','PF14707.alignment.uniprot'],
      val_ratio = 0.1,
      data_format ='lmdb', #'stockholm'
      map_size = (1024 * 20)*(2 ** 20))
  elif task2run == 'combineDataset':
    combineDataset(
      data_path = '/scratch/user/sunyuanfei/Projects/CAGI6/ARSA',
      flNm_list = [['train_PF00884.alignment.uniprot','train_PF14707.alignment.uniprot'],['val_PF00884.alignment.uniprot','val_PF14707.alignment.uniprot']],
      combinedFileNameList = ['train_ARSA','val_ARSA'],
      data_format = 'lmdb',
      map_size = (1024 * 20)*(2 ** 20))
  elif task2run == 'selectModel':
    selectModel(
      work_path = '/scratch/user/yutingg/YSun_Projs/ProteinMutEffBenchmark',
      fam_list = ['PF01379_PF03900','PF13499','PF00071','PF00076','PF00097','PF00145','PF00172','PF00179','PF00218','PF00232','PF00240','PF00397','PF00475','PF00509','PF00516_PF00517','PF00533','PF00595','PF00603','PF00607','PF00782_PF10409','PF00795','PF01176','PF01636','PF02518','PF03693','PF03702','PF04263_PF04265','PF04564','PF05724','PF08300','PF11976','PF13354'],                
      rpSet_list = ['rp15_all_4'],
      epoch_list = [459,489,519,549,579,609,639,669,699,729],
      topK = 3)
  else:
    print(f'invalid task {task2run}')
