'''
*Create masked sequence dataset with true label
*Process seq with ambiguous letters(X,B,Z)
*Counting

input files: 
  family-wise protein MSA in stockholm/fasta format
mask policy:
  mask top-N positions with low entropy/uncertainty(more conserved positions):
  this is why called 'entropy_mask'
output files: 
  masked sequences and labels in tfRecord format
  train/val set: within distribution split with rate 90/10
  test set: hold out specific families and clans
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
from prody import MSAFile

## define global variables
FLAGS = flags.FLAGS

#/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/
flags.DEFINE_string('PROJ_DIR', '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark', 
  "Project root dir")

flags.DEFINE_string('DATA_PATH', 'data_process/pfam_32.0', 
  "Path for data files")

flags.DEFINE_string('STAT_PATH', 'stat_rp15',
  "Path for statistic files")

flags.DEFINE_string('OUTPUT_PATH',
'seq_masked_tfrecords_maxLen500_1dProfile_entropy_rp15',
  "Folder path to save output files)")

flags.DEFINE_string('UNMATCH_PATH', '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/dataProcess_scripts/unmatch_rp35/unmatch_fams',
  "Path for unmatch_fams")


flags.DEFINE_string('FAM_CLAN_FILE', 'Pfam-A.clans.tsv',
  "family, clan and domain description tsv file")

flags.DEFINE_string('FLNM_PFAM_FILE', 'pfam_rp15_seqs_files_famAcc', 
  'data file name and pfam number, clan number corresponds')

flags.DEFINE_integer('RANDOM_SEED', 25, 'random seed')

flags.DEFINE_boolean('SPLIT', True, 
  'If True, split seqs in every fasta file into subsets')

flags.DEFINE_integer('NUM_SPLIT', 50,
  'number of subsets to split seqs into tfReocrd files')

flags.DEFINE_integer('VOCAB_SIZE', 29,
  'number of vocabularies/tokens')

# 'max_seq_length' takes the value of 'sgl_domain_len_up' + 2 (start and end tokens)
flags.DEFINE_list('MASK_PARAMS', [502, 0.15, 75],
  'params for mask: max_seq_length, mask_prob, max_mask_per_seq')

flags.DEFINE_boolean('SIG_DOM', True,
  'flag for only saving single domain sequences(in fact, just a length filter)')

flags.DEFINE_integer('sgl_domain_len_low', 1,
  'the lower bound length of single domain')

flags.DEFINE_integer('sgl_domain_len_up', 500,
  'the upper bound length of single domain')

flags.DEFINE_string('IN_FORMAT', 'stockholm', 'format for input seq files')

flags.DEFINE_boolean('MASK_FLAG', True, 'Whether to mask seqs')



def raw_fasta2dict(data_path, input_format, fileNm, famAcc_ver, famAcc_no_ver, clanAcc):
  '''
  input: 
    -data_path: data dir
    -input_format: msa file format
    -fileNm: path of file
    -famAcc: identifier of family with version / no version
    -clanAcc: identifier of clan

  output: 
    -a list of dictionaries with each dictionary corresponds to one protein sequence/segment.
    keys of dictionary:
    -sequence: raw protein sequence in aa symbols(upper case)
    -aligned_seq: aligned sequence in msa file
    -seq_length: length of raw sequence
    -uni_iden: uniprot_id/start_idx-end_idx
    -family: family id
    -clan: clan id

  Count statistics in create_instance function
  '''
  special_letters = ['B','X','Z']
  seq_dict_list = []
  print('loading sequences from %s' % (fileNm), flush=True)

  # initilize msa object
  msa = MSAFile('{}/{}'.format(data_path, fileNm), format=input_format, aligned=True)
  for seq in msa:
    res_idx = seq.getResnums()
    start_idx = res_idx[0]
    end_idx = res_idx[-1]
    label = seq.getLabel()
    # get unaligned sequence
    raw_seq = ''.join(re.findall('[A-Za-z]', str(seq))).upper()
    # handle special letters
    uniq_seqChars = list(set(raw_seq))
    if 'X' in uniq_seqChars:
      pass
    elif 'B' in uniq_seqChars and 'Z' not in uniq_seqChars:
      # B -> D
      new_rawSeq = raw_seq.replace('B','D')
      new_seq = str(seq).replace('B','D').replace('b','d')
      # one protein dict
      one_protein_1 = {
        'sequence': new_rawSeq,
        'aligned_seq': new_seq,
        'seq_length': len(new_rawSeq),
        'uni_iden': '{}/{}-{}'.format(label,start_idx,end_idx),
        'family_ver': famAcc_ver,
        'family': famAcc_no_ver,
        'clan': clanAcc
      }
      # B -> N
      new_rawSeq = raw_seq.replace('B','N')
      new_seq = str(seq).replace('B','N').replace('b','n')
      # one protein dict
      one_protein_2 = {
        'sequence': new_rawSeq,
        'aligned_seq': new_seq,
        'seq_length': len(new_rawSeq),
        'uni_iden': '{}/{}-{}'.format(label,start_idx,end_idx),
        'family_ver': famAcc_ver,
        'family': famAcc_no_ver,
        'clan': clanAcc
        }
      seq_dict_list.append(one_protein_1)
      seq_dict_list.append(one_protein_2)

    elif 'Z' in uniq_seqChars and 'B' not in uniq_seqChars:
      # Z -> E
      new_rawSeq = raw_seq.replace('Z','E')
      new_seq = str(seq).replace('Z','E').replace('z','e')
      # one protein dict
      one_protein_1 = {
        'sequence': new_rawSeq,
        'aligned_seq': new_seq,
        'seq_length': len(new_rawSeq),
        'uni_iden': '{}/{}-{}'.format(label,start_idx,end_idx),
        'family_ver': famAcc_ver,
        'family': famAcc_no_ver,
        'clan': clanAcc
      }
      # Z -> Q
      new_rawSeq = raw_seq.replace('Z','Q')
      new_seq = str(seq).replace('Z','Q').replace('z','q')
      # one protein dict
      one_protein_2 = {
        'sequence': new_rawSeq,
        'aligned_seq': new_seq,
        'seq_length': len(new_rawSeq),
        'uni_iden': '{}/{}-{}'.format(label,start_idx,end_idx),
        'family_ver': famAcc_ver,
        'family': famAcc_no_ver,
        'clan': clanAcc
        }
      seq_dict_list.append(one_protein_1)
      seq_dict_list.append(one_protein_2)
    elif 'B' in uniq_seqChars and 'Z' in uniq_seqChars:
      # Z -> E, B -> D
      new_rawSeq = raw_seq.replace('Z','E').replace('B','D')
      new_seq = str(seq).replace('Z','E').replace('z','e').replace('B','D').replace('b','d')
      # one protein dict
      one_protein_1 = {
        'sequence': new_rawSeq,
        'aligned_seq': new_seq,
        'seq_length': len(new_rawSeq),
        'uni_iden': '{}/{}-{}'.format(label,start_idx,end_idx),
        'family_ver': famAcc_ver,
        'family': famAcc_no_ver,
        'clan': clanAcc
      }
      # Z -> E, B -> N
      new_rawSeq = raw_seq.replace('Z','E').replace('B','N')
      new_seq = str(seq).replace('Z','E').replace('z','e').replace('B','N').replace('b','n')
      # one protein dict
      one_protein_2 = {
        'sequence': new_rawSeq,
        'aligned_seq': new_seq,
        'seq_length': len(new_rawSeq),
        'uni_iden': '{}/{}-{}'.format(label,start_idx,end_idx),
        'family_ver': famAcc_ver,
        'family': famAcc_no_ver,
        'clan': clanAcc
        }
      # Z -> Q, B -> D
      new_rawSeq = raw_seq.replace('Z','Q').replace('B','D')
      new_seq = str(seq).replace('Z','Q').replace('z','q').replace('B','D').replace('b','d')
      # one protein dict
      one_protein_3 = {
        'sequence': new_rawSeq,
        'aligned_seq': new_seq,
        'seq_length': len(new_rawSeq),
        'uni_iden': '{}/{}-{}'.format(label,start_idx,end_idx),
        'family_ver': famAcc_ver,
        'family': famAcc_no_ver,
        'clan': clanAcc
        }
      # Z -> Q, B -> N
      new_rawSeq = raw_seq.replace('Z','Q').replace('B','N')
      new_seq = str(seq).replace('Z','Q').replace('z','q').replace('B','N').replace('b','n')
      # one protein dict
      one_protein_4 = {
        'sequence': new_rawSeq,
        'aligned_seq': new_seq,
        'seq_length': len(new_rawSeq),
        'uni_iden': '{}/{}-{}'.format(label,start_idx,end_idx),
        'family_ver': famAcc_ver,
        'family': famAcc_no_ver,
        'clan': clanAcc
        }

      seq_dict_list.append(one_protein_1)
      seq_dict_list.append(one_protein_2)
      seq_dict_list.append(one_protein_3)
      seq_dict_list.append(one_protein_4)
    else:
      # one protein dict
      one_protein = {
        'sequence': raw_seq,
        'aligned_seq': str(seq),
        'seq_length': len(raw_seq),
        'uni_iden': '{}/{}-{}'.format(label,start_idx,end_idx),
        'family_ver': famAcc_ver,
        'family': famAcc_no_ver,
        'clan': clanAcc
        }
      seq_dict_list.append(one_protein)

  return seq_dict_list

class TrainingInstance(object):
  """
  class for one training example (one sequence segment)
  token_seq: list, tokens(residues) of a protein seq
  family_id, clan_id: integer, index of family and clan
  masked_lm_positions: list, position index of masked positions
  masked_lm_labels: list, true token(residue) of masked positions 
  aligned_binary: binary vector with 1 indicating aligned positions(substitution)
  unaligned_binary: binary vector with 1 indicating unaligned positions(insertion)
  gr_1Dprofile_oneHot: size [max_seq_length, num_tokens], ground truth
    1-D distribution(for aligned positions) or one_hot vector indicating true
    token(for unaligned positions)
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
          non-negative{0,1,2...} value as indices for residues from left to right among all aligned postions
          {-1} for insertion residues
          {-2} for special tokens: start, end, padding
  E.g.:
   - aligned_seq : -A-Bcd..eFG--H..iJk-
   - raw_seq:       AB C D EFGH IJ K
   - pos_type_vec:-213-1-1-1458-19-1-2-2
   - aligned_bi  : 011 0 0 0111 01 0 0 0
   - unaligned_bi: 000 1 1 1000 10 1 0 0
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
  
  total_aligned_pos_n = aligned_idx
  # lengh check    
  assert len(type_vec) == raw_seq_len

  # append special tokens
  type_vec.insert(0,-2)
  while len(type_vec) < max_seq_length:
    type_vec.append(-2)

  type_vec = np.array(type_vec)
  aligned_bi = (type_vec>=0).astype(float)
  unaligned_bi = (type_vec==-1).astype(float)

  return aligned_bi, unaligned_bi, type_vec, total_aligned_pos_n

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
  aligned_pos_entropy, # entropy arr of aligned pos
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
  #famAcc_ver = seq_dict_list[0]['family_ver'] # family number with version, e.g. PF130002.23

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
      pos_type_list = [] # list to store position type vector
      total_aligned_pos_N_list = [] # list to store num of total aligned positions

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
        aligned_bi, unaligned_bi, pos_type_vec, total_aligned_pos_n= proc_aligned_seq(aligned_seq, raw_seq)
        
        # aligned length match check
        if total_aligned_pos_n != len(aligned_pos_entropy):
          print('!!!UNMATCH!!!')
          os.system('echo {}, {} vs {} >> {}'.format(seq_dict_list[0]['family'], total_aligned_pos_n, len(aligned_pos_entropy), FLAGS.UNMATCH_PATH))
          break

        
        aligned_pos_list.append(aligned_bi)
        unaligned_pos_list.append(unaligned_bi)
        pos_type_list.append(pos_type_vec)
        total_aligned_pos_N_list.append(total_aligned_pos_n)

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
  ''' 
  ## loop over all seqs in list to create instance
  for seq_i in range(len(seq_token_list)):
   
    # create masked sequence
    (masked_tokens, masked_lm_positions, masked_lm_labels) = create_masked_predictions(
        seq_token_list[seq_i], 
        mask_prob, 
        max_mask_per_seq,
        aligned_pos_entropy,
        pos_type_list[seq_i],
        total_aligned_pos_N_list[seq_i],
        seq_dict_list[0]['family'],
        rng,
        mode = 'single'
        )
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
    '''
  return instances_list
    
def create_masked_predictions(
    tokens, 
    mask_prob,
    max_mask_per_seq,
    aligned_pos_entropy,
    pos_type_vec,
    total_aligned_pos_n, #to verify total num of aligned positions
    fam_id,
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
    '''
    # collect indices of 20AAs
    cand_indexes = [] # indexes of residues only
    for (i, token) in enumerate(tokens):
      if token == '<CLS>' or token == '<SEP>' or token == '<PAD>':
        continue
      else:
        cand_indexes.append(i)
    rng.shuffle(cand_indexes)
    '''
    # collect indices of aligned 20AAs
    cand_idx = []
    aligned_idx = []
    for (i, type_val) in enumerate(pos_type_vec):
      if type_val == -2 or type_val == -1:
        continue
      else:
        cand_idx.append(i)
        aligned_idx.append(type_val)

    assert len(cand_idx) <= total_aligned_pos_n

    # only residue amoung aligned positions(gaps are also aligned position)
    aligned_resi_entropy = aligned_pos_entropy[aligned_idx]
    idx_ordered_list = np.argsort(aligned_resi_entropy)
    
    num_to_mask = min(max_mask_per_seq, int(round(len(cand_idx)*mask_prob)))

    ## cand_indexes already shuffled, mask along the list until reach num_to_mask
    mask_list = [] # each elem [idx, true label]
    masked_tokens = tokens
    vocab_tokens = list(PFAM_VOCAB.keys())[4:]

    # loop from position with lowest H to highest
    for idx in idx_ordered_list:
      glo_idx = cand_idx[idx] # get the idx of this aligned AA in processed seq
      if len(mask_list) >= num_to_mask: 
        break
      mask_list.append([glo_idx, tokens[glo_idx]]) # add this pos to list
      # 80% of time, replace with [MASK]
      if rng.random() < 0.8:
        mask_tk = '<MASK>'
      else:
        # 10% of time, keep original
        if rng.random() < 0.5:
          mask_tk = tokens[glo_idx]
        # 10% of time, replace with other random residue
        else:
          trim_vocabs = list(PFAM_VOCAB.keys())[4:]
          trim_vocabs.remove(tokens[glo_idx])
          mask_tk = trim_vocabs[rng.randint(0, len(trim_vocabs)-1)]

      masked_tokens[glo_idx] = mask_tk

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

def order_posit_entropy(aa_1d_prob, famAcc_ver):
  '''
  return an array storing entropy of each aligned position
  '''
  aligned_posit_n = aa_1d_prob.shape[0]
  entropy_arr = np.sum(-1.*aa_1d_prob*np.log(aa_1d_prob), axis=1)
  # save entropy_arr and ordered_idx
  np.savetxt('{}data_process/pfam/Pfam-A.hmm_1d_profile_entropy/{}'.format(FLAGS.PROJ_DIR,famAcc_ver),
      entropy_arr, fmt='%.5e')
  return entropy_arr

def main(argv):
  ## copy common global params
  PROJ_DIR = FLAGS.PROJ_DIR
  DATA_PATH = PROJ_DIR+'/'+FLAGS.DATA_PATH
  STAT_PATH = DATA_PATH+'/'+FLAGS.STAT_PATH
  OUTPUT_PATH = DATA_PATH+'/'+FLAGS.OUTPUT_PATH
  FAM_CLAN_FILE = DATA_PATH+'/'+FLAGS.FAM_CLAN_FILE
  FLNM_PFAM_FILE = DATA_PATH+'/'+FLAGS.FLNM_PFAM_FILE
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
  val_pfam = [] # to be determined
  '''

  # mutagenesis hold-out set
  holdOut_pfam = np.loadtxt('{}/holdOut_sets/muta_pfam_large_set.txt'.format(DATA_PATH),dtype='str')
  holdOut_clan = np.loadtxt('{}/holdOut_sets/muta_clan_large_set.txt'.format(DATA_PATH),dtype='str')

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
    #print("*** Load single position match emission probability ***", flush=True)
    aa_1d_prob = np.loadtxt('{}/Pfam-A.hmm_1d_profile/{}'.format(DATA_PATH,famAcc), dtype='float', delimiter=' ')
  
    # generate index list from lowest entropy to highest
    aligned_pos_entropy = order_posit_entropy(aa_1d_prob,famAcc)
    
    # transform raw fasta sequence to list of distionarys
    #print("*** Reading seqs from msa (fasta/stockholm) files ***", flush=True)
    famAcc_no_version = re.split(r'\.', famAcc)[0] 
    seq_dict_list = raw_fasta2dict(DATA_PATH, IN_FORMAT, fileNm, famAcc, famAcc_no_version, clanAcc)

    ##### create sequence instances #####
    print('*** Create training instances ***', flush=True)
    instances = create_training_instances(seq_dict_list, MAX_SEQ_LENGTH, MASK_PROB, MAX_MASK_PER_SEQ, 
                                          aa_1d_prob, aligned_pos_entropy, rng, single_domain=SIG_DOM)     
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
  
  os.system("echo 'In total, write {} instances with train {}, val {}, test {}' > {}/{}".format(sum(num_written), num_written[0],
        num_written[1],num_written[2], STAT_PATH,OUTPUT_PATH))


if __name__ == "__main__":
  app.run(main)
