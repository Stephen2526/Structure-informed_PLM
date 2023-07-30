import tensorflow as tf
import numpy as np

max_seq_length = 202
num_tokens = 29
max_mask_per_seq = 30

name_to_features = {
  "token_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
  "fam_id": tf.io.FixedLenFeature([], tf.int64),
  "clan_id": tf.io.FixedLenFeature([], tf.int64),
  "masked_lm_positions": tf.io.FixedLenFeature([max_mask_per_seq], tf.int64),
  "masked_lm_ids": tf.io.FixedLenFeature([max_mask_per_seq], tf.int64),
  "aligned_binary": tf.io.FixedLenFeature([max_seq_length], tf.float32),
  "unaligned_binary": tf.io.FixedLenFeature([max_seq_length], tf.float32),
  "gt_1Dprofile_oneHot": tf.io.FixedLenFeature([], tf.string)
}
def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  parsed_examples = tf.io.parse_single_example(example_proto, name_to_features)
  
  token_ids_ts = parsed_examples['token_ids']
  fam_id_ts = parsed_examples['fam_id']
  clan_id_ts = parsed_examples['clan_id']
  masked_pos_ts = parsed_examples['masked_lm_positions'] 
  masked_ids_ts = parsed_examples['masked_lm_ids']
  aligned_binary = parsed_examples['aligned_binary']
  unaligned_binary = parsed_examples['unaligned_binary']
  gt_1Dprofile_oneHot = parsed_examples['gt_1Dprofile_oneHot']
  
  def numpy_proc(label_str):
    label_1d = np.frombuffer(label_str, dtype=np.float32)
    label_mat = label_1d.reshape((max_seq_length, num_tokens))
    #print(label_mat[:10])
    return label_mat

  label_mat_ts = tf.numpy_function(numpy_proc, [gt_1Dprofile_oneHot], tf.float32)
  
  parsed_examples['gt_1Dprofile_oneHot'] = label_mat_ts
  return parsed_examples

filename = '/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/pfam/seq_masked_tfrecords_sig_domain_test/train_seqs_masked_TFRecord_0'
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
parsed_dataset = raw_dataset.map(_parse_function)
for parsed_record in parsed_dataset.take(1):
  print(repr(parsed_record))
'''
for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)
'''
