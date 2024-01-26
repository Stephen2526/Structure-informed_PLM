from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from copy import copy
from pathlib import Path
import pickle as pkl
import logging
import random
from itertools import chain

import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform

from tokenizers import BaseTokenizer, ab_H_subclass, ab_L_subclass, ab_HL_subclass
from mapping import registry

from torch_geometric.data import Data as pyg_Data
from torch_geometric.data import Batch as pyg_Batch

logger = logging.getLogger(__name__)


def dataset_factory(data_file: Union[str, Path], *args, **kwargs) -> Dataset:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    if data_file.suffix == '.lmdb':
        return LMDBDataset(data_file, *args, **kwargs)
    elif data_file.suffix in {'.fasta', '.fna', '.ffn', '.faa', '.frn'}:
        return FastaDataset(data_file, *args, **kwargs)
    elif data_file.suffix == '.json':
        return JSONDataset(data_file, *args, **kwargs)
    elif data_file.is_dir():
        return NPZDataset(data_file, *args, **kwargs)
    else:
        raise ValueError(f"Unrecognized datafile type {data_file.suffix}")

def kth_diag_indices(arr: np.ndarray , k_list: List):
  """
  return indices of k-th diagnals
  Input:
  * arr: array of size (N,N)
  * k: list of k-th diagonals, e.g. [-1,0,1]
  """
  rowIdxs,colIdxs = np.diag_indices_from(arr)
  out_row_idxs = []
  out_col_idxs = []
  for k in k_list:
    if k < 0:
      out_row_idxs.extend(rowIdxs[-k:])
      out_col_idxs.extend(colIdxs[:k])
    elif k > 0:
      out_row_idxs.extend(rowIdxs[:-k])
      out_col_idxs.extend(colIdxs[k:])
    else:
      out_row_idxs.extend(rowIdxs)
      out_col_idxs.extend(colIdxs)
  return out_row_idxs,out_col_idxs

def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.amax([seq.shape for seq in sequences], axis=0).tolist()
    #shape = [batch_size] + [475]*len(sequences[0].shape)


    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)
    else:
        Exception(f'invalid element type {dtype}')

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

def pad_distogram(array2d: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(array2d)
    shape = [batch_size] + np.amax([cm.shape for cm in array2d], axis=0).tolist()
    #shape = [batch_size] + [475]*len(contactMaps[0].shape)

    if dtype is None:
        dtype = array2d[0].dtype

    if isinstance(array2d[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(array2d[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)
    else:
        Exception(f'invalid element type {dtype}')

    for arr, cm in zip(array, array2d):
        arrslice = tuple(slice(dim) for dim in cm.shape)
        arr[arrslice] = cm

    return array
    
def add_special_dimens(contact_map: np.ndarray, value: Union[float,int]) -> np.ndarray:
    """Add one row/col dimension for start and end token with value
    """ 
    #contact_mat = np.asarray(contact_map)
    contact_pad = np.pad(contact_map, ((1,1),(1,1)), 'constant', constant_values=value)
    return contact_pad

def check_seq_neighbors(contact_map: Sequence[Sequence[Union[float,int]]], value: Union[float,int] ,diag_idx: List=[-5,-4,-3,-2,-1,0,1,2,3,4,5]) -> np.ndarray:
    """Set values for residues within 6 seq neighbors |i-j| < 6
    """
    if not isinstance(contact_map, np.ndarray):
        contact_map = np.asarray(contact_map)
    diagIdxs = kth_diag_indices(contact_map, diag_idx)
    contact_map[diagIdxs] = value
    return contact_map
    
def min_max_normalize_array(array: np.ndarray) -> np.ndarray:
    """Normalize provided array to have values be in range [0, 1]."""
    min_value = min(array)
    max_value = max(array)
    array = np.array([(value - min_value) / (max_value - min_value) for value in array])
    return array

def generate_graph(distance_map: np.ndarray, neighbor_strategy: str, knn_value: int, dist_cutoff: float)->np.ndarray:
    """generate graph edges based on neighbor_strategy (NO self loop).
    messages from nodes in edge_index[0] are sent to nodes in edge_index[1] (flow="source_to_target")

    Args:
        distance_map: distance matrix of size L * L, non-existing pair has np.nan value, dtype='float64'
        neighbor_strategy: strategy to define neighboring nodes
            options are: 
                'sequential': sequential neighbors from seq
                'random': random neighbors
                'knn': k nearest neighbors
                'distCut': neighbors within a distance radius
                'full': fully connected graph
                'knnDistCut': neighbors within a distance radius, but max to be k
        knn_value: value defining k nearest neighbors
        dist_cutoff: distance cutoff for 'distCut'
    
    Return: 
        edge_index: [2,num_edges]
    """
    seq_len = distance_map.shape[0]
    if neighbor_strategy == 'sequential':
        first_idxs = np.arange(seq_len-1)   
        second_idxs = np.arange(1,seq_len)
        edge_index = np.vstack((np.hstack((first_idxs,second_idxs)),np.hstack((second_idxs,first_idxs))))
    elif neighbor_strategy == 'random':
        edge_index_source,edge_index_target = [],[]
        for n in range(seq_len):
            canda_nodes = list(range(n)) + list(range(n+1,seq_len))
            random_nodes = np.random.choice(canda_nodes,min(len(canda_nodes),knn_value),replace=False)
            edge_index_source.extend(random_nodes)
            edge_index_target.extend([n]*len(random_nodes))
        edge_index = np.vstack((edge_index_source,edge_index_target))
    elif neighbor_strategy == 'knn':
        edge_index_source,edge_index_target = [],[]
        for n in range(seq_len):
            valid_idxs = np.argwhere(~np.isnan(distance_map[n,:])).reshape(-1)
            knn_nodes = np.argsort(distance_map[n,:][valid_idxs])[1:knn_value+1]
            valid_nodes = [valid_idxs[i] for i in knn_nodes]
            edge_index_source.extend(valid_nodes)
            edge_index_target.extend([n]*len(valid_nodes))
        edge_index = np.vstack((edge_index_source,edge_index_target))
    elif neighbor_strategy == 'distCut':
        edge_index_source,edge_index_target = [],[]
        for n in range(seq_len):
            radius_idxs = np.argwhere(distance_map[n,:] <= dist_cutoff).reshape(-1)
            knn_nodes = np.argsort(distance_map[n,:][radius_idxs])[1:]
            valid_nodes = [radius_idxs[i] for i in knn_nodes]
            edge_index_source.extend(valid_nodes)
            edge_index_target.extend([n]*len(valid_nodes))
        edge_index = np.vstack((edge_index_source,edge_index_target))    
    elif neighbor_strategy == 'full':
        edge_index_source,edge_index_target = [],[]
        for n in range(seq_len):
            canda_nodes = list(range(n)) + list(range(n+1,seq_len))
            edge_index_source.extend(canda_nodes)
            edge_index_target.extend([n]*len(canda_nodes))
        edge_index = np.vstack((edge_index_source,edge_index_target))
    elif neighbor_strategy == 'noGS':
        edge_index_source,edge_index_target = [],[]
        for n in range(seq_len):
            edge_index_source.append(n)
            edge_index_target.append(n)
        edge_index = np.vstack((edge_index_source,edge_index_target))
    elif neighbor_strategy == 'knnDistCut':
        edge_index_source,edge_index_target = [],[]
        for n in range(seq_len):
            radius_idxs = np.argwhere(distance_map[n,:] <= dist_cutoff).reshape(-1)
            knn_nodes = np.argsort(distance_map[n,:][radius_idxs])[1:knn_value+1]
            valid_nodes = [radius_idxs[i] for i in knn_nodes]
            edge_index_source.extend(valid_nodes)
            edge_index_target.extend([n]*len(valid_nodes))
        edge_index = np.vstack((edge_index_source,edge_index_target))
    else:
        raise Exception(f"requested neighbor strategy {neighbor_strategy} not acceptable")
    # handle no edge cases
    if edge_index.shape[1] == 0:
        edge_index = np.vstack((list(range(seq_len)),list(range(seq_len))))
    return edge_index

def discretize_distogram(distance_map: np.ndarray,
                         first_cutoff: float,
                         last_cutoff: float,
                         num_bins: int,
                         ignore_index: int=-1):
    """discretize distance value into bins. 
    """
    def bin_a_value(v):
        if v == np.nan:
            return ignore_index
        bin_cutoffs = np.linspace(first_cutoff,last_cutoff,num=num_bins-1)
        assign_bin = np.sum(v > bin_cutoffs, axis=-1)
        return assign_bin
    
    return np.vectorize(bin_a_value)(distance_map)

def discretize_distogram_fast(distance_map: np.ndarray,
                            first_cutoff: float,
                            last_cutoff: float,
                            num_bins: int,
                            ignore_index: int=-1,
                            dtype: np.dtype = np.int8):
    """discretize distance value into bins. 
    """
    nan_indices = np.nonzero(np.isnan(distance_map))
    bin_cutoffs = np.linspace(first_cutoff,last_cutoff,num=num_bins-1)
    assign_bin = np.sum([distance_map > cutof for cutof in bin_cutoffs],axis=0,dtype=dtype)
    assign_bin[nan_indices] = ignore_index
    return assign_bin



class FastaDataset(Dataset):
    """Creates a dataset from a fasta file.
    Args:
        data_file (Union[str, Path]): Path to fasta file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        from Bio import SeqIO
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        # if in_memory:
        cache = list(SeqIO.parse(str(data_file), 'fasta'))
        num_examples = len(cache)
        self._cache = cache
        # else:
            # records = SeqIO.index(str(data_file), 'fasta')
            # num_examples = len(records)
#
            # if num_examples < 10000:
                # logger.info("Reading full fasta file into memory because number of examples "
                            # "is very low. This loads data approximately 20x faster.")
                # in_memory = True
                # cache = list(records.values())
                # self._cache = cache
            # else:
                # self._records = records
                # self._keys = list(records.keys())

        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        # if self._in_memory and self._cache[index] is not None:
        record = self._cache[index]
        # else:
            # key = self._keys[index]
            # record = self._records[key]
            # if self._in_memory:
                # self._cache[index] = record

        item = {'id': record.id,
                'primary': str(record.seq),
                'protein_length': len(record.seq)}
        return item


class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=126, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))
        
        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = None
        self._in_memory = in_memory
        self._num_examples = num_examples
        self.data_file = str(data_file)

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        
        if self._env is None:
            self._env = lmdb.open(self.data_file, max_readers=126, readonly=True,
                        lock=False, readahead=False, meminit=False)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        self._env = None
        return item


class JSONDataset(Dataset):
    """Creates a dataset from a json file. Assumes that data is
       a JSON serialized list of record, where each record is
       a dictionary.
    Args:
        data_file (Union[str, Path]): Path to json file.
        in_memory (bool): Dummy variable to match API of other datasets
    """

    def __init__(self, data_file: Union[str, Path], in_memory: bool = True):
        import json
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        records = json.loads(data_file.read_text())
        # *insufficient shared memory*: python list has 'copy-on-write' which
        # will gradually use up memory. Check this post:
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-445446603
        # Convert python distionary list to pandas dataFrame
        if not isinstance(records, list):
            raise TypeError(f"TAPE JSONDataset requires a json serialized list, "
                            f"received {type(records)}")

        records = pd.DataFrame(records)
        self._records = records
        self._num_examples = len(records)

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        item = dict(self._records.loc[index,:])
        if not isinstance(item, dict):
            raise TypeError(f"Expected dataset to contain a list of dictionary "
                            f"records, received record of type {type(item)}")
        if 'id' not in item:
            item['id'] = str(index)
        return item


class NPZDataset(Dataset):
    """Creates a dataset from a directory of npz files.
    Args:
        data_file (Union[str, Path]): Path to directory of npz files
        in_memory (bool): Dummy variable to match API of other datasets
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = True,
                 split_files: Optional[Collection[str]] = None):
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not data_file.is_dir():
            raise NotADirectoryError(data_file)
        file_glob = data_file.glob('*.npz')
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.name in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory")

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_file}")

        self._file_list = file_list

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)

        item = dict(np.load(self._file_list[index]))
        if not isinstance(item, dict):
            raise TypeError(f"Expected dataset to contain a list of dictionary "
                            f"records, received record of type {type(item)}")
        if 'id' not in item:
            item['id'] = self._file_list[index].stem
        return item


@registry.register_task('embed')
class EmbedDataset(Dataset):

    def __init__(self,
                 data_file: Union[str, Path],
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = False,
                 convert_tokens_to_ids: bool = True):
        super().__init__()

        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self.data = dataset_factory(data_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return item['id'], token_ids, input_mask

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        ids, tokens, input_mask = zip(*batch)
        ids = list(ids)
        tokens = torch.from_numpy(pad_sequences(tokens))
        input_mask = torch.from_numpy(pad_sequences(input_mask))
        return {'ids': ids, 'input_ids': tokens, 'input_mask': input_mask}  # type: ignore


@registry.register_task('embed_seq')
class EmbedSeqDataset(Dataset):
    """ DataLoader module for sequence embedding
    
    input lmdb data should contain:
        'seq_id': str, identifier of sequence, can be index, uniport_id or others
        'seq_primary': str, protein amino acid sequence, check tokenization for allowed characters

    """
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = False,
                 file_format: str = 'lmdb',
                 **kwargs):
        super().__init__()
        
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f"{split}.{file_format}"
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['seq_primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, item['seq_id']

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        tokens, input_mask, seq_id = zip(*batch)
        tokens = torch.from_numpy(pad_sequences(tokens,0))
        input_mask = torch.from_numpy(pad_sequences(input_mask,0))
        return {'input_ids': tokens, 'input_mask': input_mask, 'seq_id': seq_id}  # type: ignore

@registry.register_task('masked_language_modeling')
class MaskedLanguageModelingDataset(Dataset):
    """Creates the Masked Language Modeling Pfam Dataset
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
        file_format (str): format of data file (Default: 'json')
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        super().__init__()
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        
        data_path = Path(data_path)
        data_file = f"{split}.{file_format}"
        self.data = dataset_factory(data_path / data_file, in_memory)

        self.model_config = kwargs.get('model_config')
        self.multi_copy_num = self.model_config.multi_copy_num if hasattr(self.model_config, 'multi_copy_num') else 1
        self.max_len = int(self.model_config.seq_max_length) - 2 # remove two special positions: start and end
        self.eval_phase = self.model_config.eval_phase if hasattr(self.model_config, 'eval_phase') else False

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # for i-th example, add start/end token, mask seq, convert to ids, make input_mask
        item = self.data[index]
        # seq str
        if 'primary' in item:
            seq_str = item['primary']
        elif 'seq_primary' in item:
            seq_str = item['seq_primary']
        else:
            pass
        # seq windom-crop
        seq_len = len(seq_str)
        if seq_len > self.max_len and not self.eval_phase: # cut while training
            seg_start = np.random.choice(seq_len-self.max_len+1, 1)[0]
            seg_end = seg_start + self.max_len
        else:
            seg_start = 0
            seg_end = seq_len
        
        tokens = self.tokenizer.tokenize(seq_str[seg_start:seg_end])
        tokens = self.tokenizer.add_special_tokens(tokens)
        if self.multi_copy_num > 1:
            masked_tokens, labels = self._apply_bert_mask_copy(tokens,self.multi_copy_num)
            masked_token_ids = np.array([self.tokenizer.convert_tokens_to_ids(ele) for ele in masked_tokens], np.int64)
            seq_ids = [item['seq_id']]*self.multi_copy_num if 'seq_id' in item else None
        else:
            masked_tokens, labels = self._apply_bert_mask(tokens)
            masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
            seq_ids = item['seq_id'] if 'seq_id' in item else None
        input_mask = np.ones_like(masked_token_ids)
        
        
        ## no family/clan for Shin2021 data
        #clanId = int(item['clan']) if len(str(item['clan'])) > 0 else -1
        #famId = int(item['family'])
        
        if 'seq_reweight' in item:
            if self.multi_copy_num > 1:
                reweight_nor = [float(item['seq_reweight'])]*self.multi_copy_num
            else:
                reweight_nor = float(item['seq_reweight'])
        else:
            if self.multi_copy_num > 1:
                reweight_nor = [1.0]*self.multi_copy_num
            else:
                reweight_nor = 1.0

        return masked_token_ids, input_mask, labels, seq_ids, reweight_nor

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, lm_label_ids, seq_ids, reweight_nor = tuple(zip(*batch))
        
        ## if multiple copies, dimension conversion
        if self.multi_copy_num > 1:
            input_ids = list(chain(*input_ids))
            input_mask = list(chain(*input_mask))
            lm_label_ids = list(chain(*lm_label_ids))
            #reweight_nor = list(chain(*reweight_nor))
            seq_ids = list(chain(*seq_ids))
        else:
            pass

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))
        
        #clan = torch.LongTensor(clan)  # type: ignore
        #family = torch.LongTensor(family)  # type: ignore

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': lm_label_ids,
                'set_nm': seq_ids}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                continue

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)

                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token(not special tokens)
                    token = self.tokenizer.convert_id_to_token(
                        random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                else:
                    # 10% chance to keep current token
                    pass

                masked_tokens[i] = token

        return masked_tokens, labels

    def _apply_bert_mask_copy(self, tokens: List[str], copy_num: int) -> Tuple[List[List], List[List]]:
        """
        return multiple copies of masked seqs for the single input seq
        """
        masked_tokens_aug = []
        labels_aug = []
        for cpy in range(copy_num):
            masked_tokens = copy(tokens)
            labels = np.zeros([len(tokens)], np.int64) - 1
            for i, token in enumerate(tokens):
                # Tokens begin and end with start_token and stop_token, ignore these
                if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                    continue
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    labels[i] = self.tokenizer.convert_token_to_id(token)
                    if prob < 0.8:
                        # 80% random change to mask token
                        token = self.tokenizer.mask_token
                    elif prob < 0.9:
                        # 10% chance to change to random token(not special tokens)
                        token = self.tokenizer.convert_id_to_token(
                            random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                    else:
                        # 10% chance to keep current token
                        pass
                    masked_tokens[i] = token
            masked_tokens_aug.append(masked_tokens)
            labels_aug.append(labels)

        return masked_tokens_aug, labels_aug

@registry.register_task('language_modeling')
class LanguageModelingDataset(Dataset):
    """Creates the Language Modeling Pfam Dataset
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()
        '''
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")
        '''
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'pfam/pfam_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)

        return token_ids, input_mask, item['clan'], item['family']

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, clan, family = tuple(zip(*batch))

        torch_inputs = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        # ignore_index is -1
        torch_labels = torch.from_numpy(pad_sequences(input_ids, -1))
        clan = torch.LongTensor(clan)  # type: ignore
        family = torch.LongTensor(family)  # type: ignore

        return {'input_ids': torch_inputs,
                'input_mask': input_mask,
                'targets': torch_labels}


@registry.register_task('fluorescence')
class FluorescenceDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'iupac',
                 in_memory: bool = False):
        '''
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")
        '''
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'fluorescence/fluorescence_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, float(item['log_fluorescence'][0])

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, fluorescence_true_value = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        fluorescence_true_value = torch.FloatTensor(fluorescence_true_value)  # type: ignore
        fluorescence_true_value = fluorescence_true_value.unsqueeze(1)

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': fluorescence_true_value}


@registry.register_task('stability')
class StabilityDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'iupac',
                 in_memory: bool = False):
        '''
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")
        '''
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'stability/stability_{split}.lmdb'

        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, float(item['stability_score'][0])

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, stability_true_value = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        stability_true_value = torch.FloatTensor(stability_true_value)  # type: ignore
        stability_true_value = stability_true_value.unsqueeze(1)

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': stability_true_value}


@registry.register_task('remote_homology', num_labels=1195)
class RemoteHomologyDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'iupac',
                 in_memory: bool = False):
        '''
        if split not in ('train', 'valid', 'test_fold_holdout',
                         'test_family_holdout', 'test_superfamily_holdout'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test_fold_holdout', "
                             f"'test_family_holdout', 'test_superfamily_holdout']")
        '''
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'remote_homology/remote_homology_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, item['fold_label']

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, fold_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        fold_label = torch.LongTensor(fold_label)  # type: ignore

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': fold_label}


@registry.register_task('contact_prediction')
class ProteinnetDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'iupac',
                 in_memory: bool = False):
        '''
        if split not in ('train', 'train_unfiltered', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'train_unfiltered', 'valid', 'test']")
        '''
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'proteinnet/proteinnet_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        protein_length = len(item['primary'])
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)

        valid_mask = item['valid_mask']
        contact_map = np.less(squareform(pdist(item['tertiary'])), 8.0).astype(np.int64)

        yind, xind = np.indices(contact_map.shape)
        invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
        invalid_mask |= np.abs(yind - xind) < 6
        contact_map[invalid_mask] = -1

        return token_ids, input_mask, contact_map, protein_length

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, contact_labels, protein_length = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        contact_labels = torch.from_numpy(pad_sequences(contact_labels, -1))
        protein_length = torch.LongTensor(protein_length)  # type: ignore

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': contact_labels,
                'protein_length': protein_length}


@registry.register_task('secondary_structure', num_labels=3)
class SecondaryStructureDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'iupac',
                 in_memory: bool = False):
        '''
        if split not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'casp12', "
                             f"'ts115', 'cb513']")
        '''
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss3'], np.int64)
        labels = np.pad(labels, (1, 1), 'constant', constant_values=-1)

        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(pad_sequences(ss_label, -1))

        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': ss_label}

        return output


@registry.register_task('trrosetta')
class TRRosettaDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'iupac',
                 in_memory: bool = False,
                 max_seqlen: int = 300):
        '''
        if split not in ('train', 'valid'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid']")
        '''
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_path = data_path / 'trrosetta'
        split_files = (data_path / f'{split}_files.txt').read_text().split()
        self.data = NPZDataset(data_path / 'npz', in_memory, split_files=split_files)

        self._dist_bins = np.arange(2, 20.1, 0.5)
        self._dihedral_bins = (15 + np.arange(-180, 180, 15)) / 180 * np.pi
        self._planar_bins = (15 + np.arange(0, 180, 15)) / 180 * np.pi
        self._split = split
        self.max_seqlen = max_seqlen
        self.msa_cutoff = 0.8
        self.penalty_coeff = 4.5

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        msa = item['msa']
        dist = item['dist6d']
        omega = item['omega6d']
        theta = item['theta6d']
        phi = item['phi6d']

        if self._split == 'train':
            msa = self._subsample_msa(msa)
        elif self._split == 'valid':
            msa = msa[:20000]  # runs out of memory if msa is way too big
        msa, dist, omega, theta, phi = self._slice_long_sequences(
            msa, dist, omega, theta, phi)

        mask = dist == 0

        dist_bins = np.digitize(dist, self._dist_bins)
        omega_bins = np.digitize(omega, self._dihedral_bins) + 1
        theta_bins = np.digitize(theta, self._dihedral_bins) + 1
        phi_bins = np.digitize(phi, self._planar_bins) + 1

        dist_bins[mask] = 0
        omega_bins[mask] = 0
        theta_bins[mask] = 0
        phi_bins[mask] = 0

        dist_bins[np.diag_indices_from(dist_bins)] = -1

        # input_mask = np.ones_like(msa[0])

        return msa, dist_bins, omega_bins, theta_bins, phi_bins

    def _slice_long_sequences(self, msa, dist, omega, theta, phi):
        seqlen = msa.shape[1]
        if self.max_seqlen > 0 and seqlen > self.max_seqlen:
            start = np.random.randint(seqlen - self.max_seqlen + 1)
            end = start + self.max_seqlen

            msa = msa[:, start:end]
            dist = dist[start:end, start:end]
            omega = omega[start:end, start:end]
            theta = theta[start:end, start:end]
            phi = phi[start:end, start:end]

        return msa, dist, omega, theta, phi

    def _subsample_msa(self, msa):
        num_alignments, seqlen = msa.shape

        if num_alignments < 10:
            return msa

        num_sample = int(10 ** np.random.uniform(np.log10(num_alignments)) - 10)

        if num_sample <= 0:
            return msa[0][None, :]
        elif num_sample > 20000:
            num_sample = 20000

        indices = np.random.choice(
            msa.shape[0] - 1, size=num_sample, replace=False) + 1
        indices = np.pad(indices, [1, 0], 'constant')  # add the sequence back in
        return msa[indices]

    def collate_fn(self, batch):
        msa, dist_bins, omega_bins, theta_bins, phi_bins = tuple(zip(*batch))
        # features = pad_sequences([self.featurize(msa_) for msa_ in msa], 0)
        msa1hot = pad_sequences(
            [F.one_hot(torch.LongTensor(msa_), 21) for msa_ in msa], 0, torch.float)
        # input_mask = torch.FloatTensor(pad_sequences(input_mask, 0))
        dist_bins = torch.LongTensor(pad_sequences(dist_bins, -1))
        omega_bins = torch.LongTensor(pad_sequences(omega_bins, 0))
        theta_bins = torch.LongTensor(pad_sequences(theta_bins, 0))
        phi_bins = torch.LongTensor(pad_sequences(phi_bins, 0))

        return {'msa1hot': msa1hot,
                # 'input_mask': input_mask,
                'dist': dist_bins,
                'omega': omega_bins,
                'theta': theta_bins,
                'phi': phi_bins}

    def featurize(self, msa):
        msa = torch.LongTensor(msa)
        msa1hot = F.one_hot(msa, 21).float()

        seqlen = msa1hot.size(1)

        weights = self.reweight(msa1hot)
        features_1d = self.extract_features_1d(msa1hot, weights)
        features_2d = self.extract_features_2d(msa1hot, weights)

        features = torch.cat((
            features_1d.unsqueeze(1).repeat(1, seqlen, 1),
            features_1d.unsqueeze(0).repeat(seqlen, 1, 1),
            features_2d), -1)

        features = features.permute(2, 0, 1)

        return features

    def reweight(self, msa1hot):
        # Reweight
        seqlen = msa1hot.size(1)
        id_min = seqlen * self.msa_cutoff
        id_mtx = torch.tensordot(msa1hot, msa1hot, [[1, 2], [1, 2]])
        id_mask = id_mtx > id_min
        weights = 1.0 / id_mask.float().sum(-1)
        return weights

    def extract_features_1d(self, msa1hot, weights):
        # 1D Features
        seqlen = msa1hot.size(1)
        f1d_seq = msa1hot[0, :, :20]

        # msa2pssm
        beff = weights.sum()
        f_i = (weights[:, None, None] * msa1hot).sum(0) / beff + 1e-9
        h_i = (-f_i * f_i.log()).sum(1, keepdims=True)
        f1d_pssm = torch.cat((f_i, h_i), dim=1)

        f1d = torch.cat((f1d_seq, f1d_pssm), dim=1)
        f1d = f1d.view(seqlen, 42)
        return f1d

    def extract_features_2d(self, msa1hot, weights):
        # 2D Features
        num_alignments = msa1hot.size(0)
        seqlen = msa1hot.size(1)
        num_symbols = 21
        if num_alignments == 1:
            # No alignments, predict from sequence alone
            f2d_dca = torch.zeros(seqlen, seqlen, 442, dtype=torch.float)
        else:
            # fast_dca

            # covariance
            x = msa1hot.view(num_alignments, seqlen * num_symbols)
            num_points = weights.sum() - weights.mean().sqrt()
            mean = (x * weights[:, None]).sum(0, keepdims=True) / num_points
            x = (x - mean) * weights[:, None].sqrt()
            cov = torch.matmul(x.transpose(-1, -2), x) / num_points

            # inverse covariance
            reg = torch.eye(seqlen * num_symbols) * self.penalty_coeff / weights.sum().sqrt()
            cov_reg = cov + reg
            inv_cov = torch.inverse(cov_reg)

            x1 = inv_cov.view(seqlen, num_symbols, seqlen, num_symbols)
            x2 = x1.permute(0, 2, 1, 3)
            features = x2.reshape(seqlen, seqlen, num_symbols * num_symbols)

            x3 = (x1[:, :-1, :, :-1] ** 2).sum((1, 3)).sqrt() * (1 - torch.eye(seqlen))
            apc = x3.sum(0, keepdims=True) * x3.sum(1, keepdims=True) / x3.sum()
            contacts = (x3 - apc) * (1 - torch.eye(seqlen))

            f2d_dca = torch.cat([features, contacts[:, :, None]], axis=2)

        return f2d_dca

@registry.register_task('penalize_nonContact_attention')
class Penalize_nonContact(Dataset):
  def __init__(self,
               data_path: Union[str, Path],
               split: str,
               tokenizer: Union[str,BaseTokenizer] = 'pfam',
               in_memory: bool = False,
               file_format: str = 'lmdb',
               **kwargs):
      super().__init__()
      '''
      if split not in ('train', 'valid', 'holdout', 'tr-val'):
          raise ValueError(
            f"Unrecognized split: {split}. "
            f"Must be one of ['train', 'valid', 'holdout']")
      '''
      if isinstance(tokenizer, str):
          tokenizer = BaseTokenizer(vocab=tokenizer)

      self.tokenizer = tokenizer

      data_path = Path(data_path)
      data_file = f"allData_lenCut_l8h500_{split}.{file_format}"
      self.data = dataset_factory(data_path / data_file, in_memory)

  def __len__(self) -> int:
      return len(self.data)

  def __getitem__(self, index):
      # for i-th example's
      # sequence: add start/end token, mask seq, convert to ids, make input_mask
      # contact-map: 
      #   * add row/col(set 1) for start/end token;
      item = self.data[index]
      tokens = self.tokenizer.tokenize(item['target_seq'])
      seq_length = len(tokens)
      tokens = self.tokenizer.add_special_tokens(tokens)
      masked_tokens, labels = self._apply_bert_mask(tokens)
      masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
      input_mask = np.ones_like(masked_token_ids)
      valid_mask = np.array([False] + item['valid_mask'] + [False])
      valid_mask_nc = np.array([True] + item['valid_mask'] + [True]) # True for special tokens

      famId = item['pfamAcc'] #e.g. PF00001
      unpId = item['unpAcc'] #e.g. O43613    

      # Residues |i-j| < 6 use value 1 (not penalize)
      binary_contact = check_seq_neighbors(item['contact-map'], value=1)
      # special tokens in contact-map use value 0 (penalize)
      binary_contact = add_special_dimens(binary_contact, value=0)
      
      # for sparse logistic model
      if 'type_flag' in item.keys():
        type_flag = item['type_flag']
      else:
        type_flag = -1

      # for wt seq precision
      set_nm = None
      if 'set_nm' in item.keys():
        set_nm = item['set_nm']

      return masked_token_ids, input_mask, labels, binary_contact, valid_mask, valid_mask_nc, famId, unpId, seq_length, type_flag, set_nm
  
  def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
      input_ids, input_mask, lm_label_ids, bi_contact, valid_mask, valid_mask_nc, famId, unpId, seq_length, type_flag, set_nm = tuple(zip(*batch))

      input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
      input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
      # ignore_index is -1
      lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))
      valid_mask = torch.from_numpy(pad_sequences(valid_mask, False))
      valid_mask_nc = torch.from_numpy(pad_sequences(valid_mask_nc, True)) # True of padding positions
      # use 0 to penalize for paddings
      bi_contact = torch.from_numpy(pad_distogram(bi_contact, 0))
      seq_length = torch.ShortTensor(seq_length)
      type_flag = torch.CharTensor(type_flag)
      # pytorch has no string tensor, for famId, unpId, need convert to id(int)

      dict_to_return = {'input_ids': input_ids,
                        'input_mask': input_mask,
                        'targets': lm_label_ids,
                        'targets_contact': bi_contact,
                        'valid_mask': valid_mask,
                        'valid_mask_nc': valid_mask_nc,
                        'seq_length': seq_length,
                        'type_flag': type_flag}
      if set_nm[0] is None:
        return dict_to_return
      else:
        dict_to_return['set_nm'] = np.array(set_nm)
        return dict_to_return

  def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
      masked_tokens = copy(tokens)
      labels = np.zeros([len(tokens)], np.int64) - 1

      for i, token in enumerate(tokens):
          # Tokens begin and end with start_token and stop_token, ignore these
          if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
              continue

          prob = random.random()
          if prob < 0.15:
              prob /= 0.15
              labels[i] = self.tokenizer.convert_token_to_id(token)

              if prob < 0.8:
                  # 80% random change to mask token
                  token = self.tokenizer.mask_token
              elif prob < 0.9:
                  # 10% chance to change to random token(not special tokens)
                  token = self.tokenizer.convert_id_to_token(
                      random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
              else:
                  # 10% chance to keep current token
                  pass

              masked_tokens[i] = token

      return masked_tokens, labels



@registry.register_task('promote_contact_attention')
class Promote_contact(Dataset):
  def __init__(self,
               data_path: Union[str, Path],
               split: str,
               tokenizer: Union[str,BaseTokenizer] = 'pfam',
               in_memory: bool = False,
               file_format: str = 'lmdb',
               **kwargs):
      super().__init__()
      '''
      if split not in ('train', 'valid', 'holdout','tr-val'):
          raise ValueError(
            f"Unrecognized split: {split}. "
            f"Must be one of ['train', 'valid', 'holdout']")
      '''
      if isinstance(tokenizer, str):
          tokenizer = BaseTokenizer(vocab=tokenizer)

      self.tokenizer = tokenizer

      data_path = Path(data_path)
      data_file = f"allData_lenCut_l8h500_{split}.{file_format}"
      self.data = dataset_factory(data_path / data_file, in_memory)

  def __len__(self) -> int:
      return len(self.data)

  def __getitem__(self, index):
      # for i-th example's
      # sequence: add start/end token, mask seq, convert to ids, make input_mask
      # contact-map: 
      #   * add row/col(set 1) for start/end token;
      item = self.data[index]
      tokens = self.tokenizer.tokenize(item['target_seq'])
      seq_length = len(tokens)
      tokens = self.tokenizer.add_special_tokens(tokens)
      masked_tokens, labels = self._apply_bert_mask(tokens)
      masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
      input_mask = np.ones_like(masked_token_ids)
      valid_mask = np.array([False] + item['valid_mask'] + [False])

      famId = item['pfamAcc'] #e.g. PF00001
      unpId = item['unpAcc'] #e.g. O43613    

      # Residues |i-j| < 6 use value 0 (not promote)
      binary_contact = check_seq_neighbors(item['contact-map'], value=0)

      # process contact-map: add dimen for start/end token use value 0 (not promote)
      binary_contact = add_special_dimens(binary_contact, value=0)
     
      # for logistic model training
      if 'type_flag' in item.keys():
        type_flag = item['type_flag']
      else:
        type_flag = -1
      
      # for wt seq precision
      set_nm = None
      if 'set_nm' in item.keys():
        set_nm = item['set_nm']

      return masked_token_ids, input_mask, labels, binary_contact, valid_mask, famId, unpId, seq_length, type_flag, set_nm
  
  def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
      input_ids, input_mask, lm_label_ids, bi_contact, valid_mask, famId, unpId, seq_length, type_flag, set_nm = tuple(zip(*batch))

      input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
      input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
      # ignore_index is -1
      lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))

      # use 0 to penalize for paddings
      bi_contact = torch.from_numpy(pad_distogram(bi_contact, 0))
      valid_mask = torch.from_numpy(pad_sequences(valid_mask, False))
      # pytorch has no string tensor, for famId, unpId, need convert to id(int)
      seq_length = torch.ShortTensor(seq_length)
      type_flag = torch.CharTensor(type_flag)
      
      dict_to_return = {'input_ids': input_ids,
                        'input_mask': input_mask,
                        'targets': lm_label_ids,
                        'targets_contact': bi_contact,
                        'valid_mask': valid_mask,
                        'seq_length': seq_length,
                        'type_flag': type_flag}

      if set_nm[0] is None:
        return dict_to_return
      else:
        dict_to_return['set_nm'] = np.array(set_nm)
        return dict_to_return

  def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
      masked_tokens = copy(tokens)
      labels = np.zeros([len(tokens)], np.int64) - 1

      for i, token in enumerate(tokens):
          # Tokens begin and end with start_token and stop_token, ignore these
          if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
              continue

          prob = random.random()
          if prob < 0.15:
              prob /= 0.15
              labels[i] = self.tokenizer.convert_token_to_id(token)

              if prob < 0.8:
                  # 80% random change to mask token
                  token = self.tokenizer.mask_token
              elif prob < 0.9:
                  # 10% chance to change to random token(not special tokens)
                  token = self.tokenizer.convert_id_to_token(
                      random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
              else:
                  # 10% chance to keep current token
                  pass

              masked_tokens[i] = token

      return masked_tokens, labels
   
@registry.register_task('contact_ce_attention_weightnor')  
@registry.register_task('contact_ce_attention')
class Promote_contact(Dataset):
  def __init__(self,
               data_path: Union[str, Path],
               split: str,
               tokenizer: Union[str,BaseTokenizer] = 'pfam',
               in_memory: bool = False,
               file_format: str = 'lmdb',
               **kwargs):
      super().__init__()
      '''
      if split not in ('train', 'valid', 'holdout','tr-val'):
          raise ValueError(
            f"Unrecognized split: {split}. "
            f"Must be one of ['train', 'valid', 'holdout']")
      '''
      if isinstance(tokenizer, str):
          tokenizer = BaseTokenizer(vocab=tokenizer)

      self.tokenizer = tokenizer

      data_path = Path(data_path)
      data_file = f"allData_lenCut_l8h500_{split}.{file_format}"
      self.data = dataset_factory(data_path / data_file, in_memory)

  def __len__(self) -> int:
      return len(self.data)

  def __getitem__(self, index):
      # for i-th example's
      # sequence: add start/end token, mask seq, convert to ids, make input_mask
      # contact-map: 
      #   * add row/col(set 1) for start/end token;
      item = self.data[index]
      tokens = self.tokenizer.tokenize(item['target_seq'])
      seq_length = len(tokens)
      tokens = self.tokenizer.add_special_tokens(tokens)
      masked_tokens, labels = self._apply_bert_mask(tokens)
      masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
      input_mask = np.ones_like(masked_token_ids)
      valid_mask = np.array([False] + item['valid_mask'] + [False])

      famId = item['pfamAcc'] #e.g. PF00001
      unpId = item['unpAcc'] #e.g. O43613    

      # Residues |i-j| < 6 use value 0 (not promote)
      binary_contact = check_seq_neighbors(item['contact-map'], value=0)

      # process contact-map: add dimen for start/end token use value 0 (not promote)
      binary_contact = add_special_dimens(binary_contact, value=0)
     
      # for sparse logistic model
      if 'type_flag' in item.keys():
        type_flag = item['type_flag']
      else:
        type_flag = -1
      
      # for wt seq precision
      set_nm = None
      if 'set_nm' in item.keys():
        set_nm = item['set_nm']

      return masked_token_ids, input_mask, labels, binary_contact, valid_mask, famId, unpId, seq_length, type_flag, set_nm
  
  def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
      input_ids, input_mask, lm_label_ids, bi_contact, valid_mask, famId, unpId, seq_length, type_flag, set_nm = tuple(zip(*batch))

      input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
      input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
      # ignore_index is -1
      lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))

      # use 0 to penalize for paddings
      bi_contact = torch.from_numpy(pad_distogram(bi_contact, 0))
      valid_mask = torch.from_numpy(pad_sequences(valid_mask, False))
      # pytorch has no string tensor, for famId, unpId, need convert to id(int)
      seq_length = torch.ShortTensor(seq_length)
      type_flag = torch.CharTensor(type_flag)
   
      dict_to_return = {'input_ids': input_ids,
                        'input_mask': input_mask,
                        'targets': lm_label_ids,
                        'targets_contact': bi_contact,
                        'valid_mask': valid_mask,
                        'seq_length': seq_length,
                        'type_flag': type_flag}
      if set_nm[0] is None:
        return dict_to_return
      else:
        dict_to_return['set_nm'] = np.array(set_nm)
        return dict_to_return


  def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
      masked_tokens = copy(tokens)
      labels = np.zeros([len(tokens)], np.int64) - 1

      for i, token in enumerate(tokens):
          # Tokens begin and end with start_token and stop_token, ignore these
          if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
              continue

          prob = random.random()
          if prob < 0.15:
              prob /= 0.15
              labels[i] = self.tokenizer.convert_token_to_id(token)

              if prob < 0.8:
                  # 80% random change to mask token
                  token = self.tokenizer.mask_token
              elif prob < 0.9:
                  # 10% chance to change to random token(not special tokens)
                  token = self.tokenizer.convert_id_to_token(
                      random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
              else:
                  # 10% chance to keep current token
                  pass

              masked_tokens[i] = token

      return masked_tokens, labels

@registry.register_task('esm_eval')
class EsmModel_eval(Dataset):
  def __init__(self,
               data_path: Union[str, Path],
               split: str,
               tokenizer: Union[str,BaseTokenizer] = 'pfam',
               in_memory: bool = False,
               file_format: str = 'lmdb',
               **kwarg):
      super().__init__()
      
      data_path = Path(data_path)
      data_file = f"allData_lenCut_l8h500_{split}.{file_format}"
      self.data = dataset_factory(data_path / data_file, in_memory)
      if "alphabet_obj" in kwarg.keys():
          self.alphabet = kwarg["alphabet_obj"]
  
  def __len__(self) -> int:
      return len(self.data)

  def __getitem__(self, index):
      # for i-th example's
      item = self.data[index]
      tar_seq = item['target_seq'] # [seq_l,]
      seq_length = len(tar_seq)
      valid_mask = np.array(item['valid_mask']) # [seq_l,]

      binary_contact = np.array(item['contact-map']) # [seq_l,seq_l]
      
      if 'type_flag' in item.keys():
        type_flag = item['type_flag']
      else:
        type_flag = -1

      return tar_seq, binary_contact, valid_mask, seq_length, type_flag
  
  def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
      # esm only add start token at beginning and pad to batch max length for seq
      # valid_mask: [bs,l_max-1] no start token
      # contact-map: [bs,l_max-1,l_max-1] no start token
      # batch_tokens: [bs,l_max] with start token and padding
      tar_seq, bi_contact, valid_mask, seq_length, type_flag = tuple(zip(*batch))
      esm_data = []
      seq_count  = 0
      for seq_p in tar_seq:
          esm_data.append(('pro_{}'.format(seq_count), seq_p))
          seq_count += 1
      #print(esm_data)
      batch_converter = self.alphabet.get_batch_converter()
      batch_labels, batch_strs, batch_tokens = batch_converter(esm_data)
      
      # use 0 to penalize for paddings
      bi_contact = torch.from_numpy(pad_distogram(bi_contact, 0))
      valid_mask = torch.from_numpy(pad_sequences(valid_mask, False))
      # pytorch has no string tensor, for famId, unpId, need convert to id(int)
      seq_length = torch.ShortTensor(seq_length)
      type_flag = torch.CharTensor(type_flag)
 
      return {'batch_tokens': batch_tokens,
              'batch_strs': batch_strs,
              'targets_contacts': bi_contact,
              'valid_masks': valid_mask,
              'seq_lengths': seq_length,
              'type_flag': type_flag}

@registry.register_task('mutation_fitness_supervise_mutagenesis')
@registry.register_task('mutation_fitness_supervise_CAGI')
class MutagenesisDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        
        self.model_config = kwargs.get('model_config',None)

        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        if self.model_config.set_nm is not None:
          data_path = Path(data_path)
          data_file = f'{self.model_config.set_nm}/{split}.lmdb'
          self.data = dataset_factory(data_path / data_file, in_memory)
        else:
          data_path = Path(data_path)
          data_file = f'mut_{split}.lmdb'
          self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['mut_seq'])
        input_mask = np.ones_like(token_ids)
        if self.model_config.label_apply_ln:
            fit_score = np.log(float(item['fitness']))
        else:
            fit_score = float(item['fitness'])
        return token_ids, input_mask, fit_score, item['set_nm'], item['mutants']

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, fitness_true_value, set_nm, mutants = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        fitness_true_value = torch.FloatTensor(fitness_true_value)  # type: ignore
        fitness_true_value = fitness_true_value.unsqueeze(1)
        set_nm = np.array(set_nm)

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': fitness_true_value,
                'set_nm': set_nm,
                'mutants': list(mutants)}

@registry.register_task('mutation_fitness_UNsupervise_mutagenesis')
class MutagenesisUnSVDataset(Dataset):
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        set_nm = kwargs.get('mutgsis_set',None)
        self.model_config = kwargs.get('model_config')
        self.max_len = int(self.model_config.seq_max_length) - 2 # remove two special positions: start and end
        
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        if set_nm is not None:
          data_path = Path(data_path)
          data_file = f'{set_nm}/{set_nm}_mut_all.{file_format}'
          self.data = dataset_factory(data_path / data_file, in_memory)
        else:
          data_path = Path(data_path)
          data_file = f'mut_all.{file_format}'
          self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        seq_len = int(item['seq_len'])
        mut_relative_idxs = item['mut_relative_idxs']
        if seq_len > self.max_len:
            seg_start,seg_end,mut_relative_idxs = self._trim_seq_center(seq_len,mut_relative_idxs)
        else:
            seg_start = 0
            seg_end = seq_len
        tokens_wt = self.tokenizer.tokenize(item['wt_seq'][seg_start:seg_end])
        tokens_mut = self.tokenizer.tokenize(item['mut_seq'][seg_start:seg_end])
        masked_tokens, labels_wt, labels_mut = self._apply_mut_mask(tokens_wt,tokens_mut,mut_relative_idxs)
        masked_tokens = self.tokenizer.add_special_tokens(masked_tokens)
        labels_wt = np.concatenate(([-1],labels_wt,[-1]))
        labels_mut = np.concatenate(([-1],labels_mut,[-1]))
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64) 
        input_mask = np.ones_like(masked_token_ids)
        return masked_token_ids, input_mask, labels_wt, labels_mut, float(item['fitness']), item['set_nm'], item['mutants'], mut_relative_idxs

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, lm_label_wt_ids, lm_label_mut_ids, fitness_true_value, set_nm, mutants, mut_relative_idxs = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        lm_label_wt_ids = torch.from_numpy(pad_sequences(lm_label_wt_ids, -1))
        lm_label_mut_ids = torch.from_numpy(pad_sequences(lm_label_mut_ids, -1))
        fitness_true_value = torch.FloatTensor(fitness_true_value)  # type: ignore
        set_nm = np.array(set_nm)
        # mutant: tuple
        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': lm_label_wt_ids,
                'targets_mut': lm_label_mut_ids,
                'fitness_gt': fitness_true_value,
                'set_nm': set_nm,
                'mutants': mutants,
                'mut_relative_idxs': mut_relative_idxs,}

    def _apply_mut_mask(self, 
                        tokens_wt: List[str], 
                        tokens_mut: List[str],
                        rela_pos_list: List[int]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens_wt)
        assert len(tokens_wt) == len(tokens_mut)
        labels_wt = np.zeros([len(tokens_wt)], np.int64) - 1
        labels_mut = np.zeros([len(tokens_mut)], np.int64) - 1
        ## pos_list, wtSeq_startIdx start from 1
        for i in rela_pos_list:
            labels_wt[i] = self.tokenizer.convert_token_to_id(tokens_wt[i])
            labels_mut[i] = self.tokenizer.convert_token_to_id(tokens_mut[i])
            masked_tokens[i] = self.tokenizer.mask_token     
        return masked_tokens, labels_wt, labels_mut

    def _trim_seq_center(self,seq_len,mut_relative_idxs):
        """trim seq which is longer than max_len, and keep variant position at center if possible
        """
        assert seq_len > self.max_len
        mut_idx_min, mut_idx_max = min(mut_relative_idxs), max(mut_relative_idxs)
        if mut_idx_min < mut_idx_max: # multi-site mut
            middle_idx = (mut_idx_min + mut_idx_max) // 2
            half_size = (self.max_len - 1) // 2
            tmp_seg_start = middle_idx - half_size
            tmp_seg_end = middle_idx + (self.max_len - half_size)
            if tmp_seg_start < 0:
                seg_start = 0
                seg_end = self.max_len
            elif tmp_seg_end > seq_len:
                seg_start = seq_len - self.max_len
                seg_end = seq_len
            else:
                seg_start = tmp_seg_start
                seg_end = tmp_seg_end
            mut_relative_idxs = [idx-seg_start for idx in mut_relative_idxs]
        elif mut_idx_min == mut_idx_max: #single-site mut
            half_size = (self.max_len - 1) // 2
            tmp_seg_start = mut_idx_min - half_size
            tmp_seg_end = mut_idx_min + (self.max_len - half_size)
            if tmp_seg_start < 0:
                seg_start = 0
                seg_end = self.max_len
            elif tmp_seg_end > seq_len:
                seg_start = seq_len - self.max_len
                seg_end = seq_len
            else:
                seg_start = tmp_seg_start
                seg_end = tmp_seg_end
            mut_relative_idxs = [idx-seg_start for idx in mut_relative_idxs]
        else:
            Exception('Invalid mutation indices')
        return seg_start, seg_end, mut_relative_idxs

@registry.register_task('multitask_fitness_UNsupervise_mutagenesis')
class MultiTaskMutagenesisUnSVDataset(Dataset):
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = False,
                 file_format: str = 'lmdb',
                 **kwargs):
        set_nm = kwargs.get('mutgsis_set',None)
        self.neighbor_strategy = kwargs.get('neighbor_strategy')
        self.knn_value = kwargs.get('knn_value')
        self.dist_cutoff = kwargs.get('dist_cutoff')
        self.model_config = kwargs.get('model_config')
        self.max_len = int(self.model_config.seq_max_length) - 2 # remove two special positions: start and end
        self.eval_label_name = self.model_config.eval_label_name
        self.split = split
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        if self.eval_label_name == 'fitness':
            if set_nm is not None:
                data_path = Path(data_path)
                data_file = f'{set_nm}/{set_nm}_mut_all.{file_format}'
                self.data = dataset_factory(data_path / data_file, in_memory)
            else:
                data_path = Path(data_path)
                data_file = f'mut_all.{file_format}'
                self.data = dataset_factory(data_path / data_file, in_memory)
        elif self.eval_label_name == 'stability':
            data_path = Path(data_path)
            data_file = f'{set_nm}.{file_format}'
            self.data = dataset_factory(data_path / data_file, in_memory)
        else:
            pass
            Exception(f'invalid label name for evaluation: {self.eval_label_name}')

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        seq_len = int(item['seq_len'])
        mut_relative_idxs = item['mut_relative_idxs']
        if seq_len > self.max_len:
            seg_start,seg_end,mut_relative_idxs = self._trim_seq_center(seq_len,mut_relative_idxs)
        else:
            seg_start = 0
            seg_end = seq_len
        tokens_wt = self.tokenizer.tokenize(item['wt_seq'][seg_start:seg_end])
        tokens_mut = self.tokenizer.tokenize(item['mut_seq'][seg_start:seg_end])
        masked_tokens, labels_wt, labels_mut = self._apply_mut_mask(tokens_wt,tokens_mut,mut_relative_idxs)
        masked_tokens = self.tokenizer.add_special_tokens(masked_tokens)
        labels_wt = np.concatenate(([-1],labels_wt,[-1]))
        labels_mut = np.concatenate(([-1],labels_mut,[-1]))
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64) 
        input_mask = np.ones_like(masked_token_ids)
        aa_seq_mask = np.ones_like(masked_token_ids)
        aa_seq_mask[0] = aa_seq_mask[-1] = 0
        
        ## fake distance_map
        #distance_map = np.ones((len(tokens_wt),len(tokens_wt)),dtype=np.float64)
        ## define neighbors (tmp neighbor_strtegy w/o structures)
        # if self.neighbor_strategy in ['knn','distCut','knnDistCut']:
        #     neighbor_strategy = 'full'
        # else:
        #     neighbor_strategy = self.neighbor_strategy

        #env_structure = lmdb.open(f'/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/mutagenesis/wt_seq_structure/#set_structure_data/{self.split}_AFDB.lmdb', max_readers=126, readonly=True, lock=False, readahead=False, meminit=False)
        #with env_structure.begin(write=False) as txn:
        #  item_structure = pkl.loads(txn.get(str(0).encode()))
        #distance_map = np.frombuffer(item_structure['distance_map'],dtype=np.float64).reshape(seq_len,seq_len)[seg_start:seg_end,seg_start:seg_end]
        #env_structure.close()
        #neighbor_strategy = self.neighbor_strategy

        #edge_index_arr = generate_graph(distance_map, neighbor_strategy, self.knn_value, self.dist_cutoff)

        # Positional encoding for each node (used for GNNs)
        #node_pos_encode_arr = min_max_normalize_array(np.arange(len(tokens_wt))).reshape(-1, 1)  # [num_node, 1]
        # Positional encoding for each edge (used for sequentially-ordered inputs like proteins)
        #edge_pos_encode_arr = np.sin((edge_index_arr[0] - edge_index_arr[1]).astype(np.float64)).reshape(-1, 1)  # [num_edges, 1]
        #graph_data = pyg_Data(edge_index=torch.tensor(edge_index_arr,dtype=torch.long),pos=torch.tensor(node_pos_encode_arr,dtype=torch.float),edge_pos=torch.tensor(edge_pos_encode_arr,dtype=torch.float))

        return masked_token_ids, input_mask, aa_seq_mask, labels_wt, labels_mut, float(item['fitness']), item['set_nm'], item['mutants'], mut_relative_idxs

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, aa_seq_mask, lm_label_wt_ids, lm_label_mut_ids, fitness_true_value, set_nm, mutant, mut_relative_idxs = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        aa_seq_mask = torch.from_numpy(pad_sequences(aa_seq_mask, 0))
        lm_label_wt_ids = torch.from_numpy(pad_sequences(lm_label_wt_ids, -1))
        lm_label_mut_ids = torch.from_numpy(pad_sequences(lm_label_mut_ids, -1))
        fitness_true_value = torch.FloatTensor(fitness_true_value)  # type: ignore
        set_nm = np.array(set_nm)
        #graph_batch = pyg_Batch.from_data_list(list(graph_data_tuple))
        #graph_batch_idxs = torch.arange(graph_batch.num_graphs,dtype=torch.int16)
        # mutant: tuple
        return {'input_seq_ids': input_ids,
                'input_seq_mask': input_mask,
                'aa_seq_mask': aa_seq_mask,
                #'graph_batch': graph_batch,
                #'graph_batch_idxs': graph_batch_idxs,
                'targets_seq': lm_label_wt_ids,
                'targets_mut': lm_label_mut_ids,
                'fitness_gt': fitness_true_value,
                'set_nm': set_nm,
                'mutants': mutant,
                'mut_relative_idxs': mut_relative_idxs,}

    def _apply_mut_mask(self, 
                        tokens_wt: List[str], 
                        tokens_mut: List[str],
                        rela_pos_list: List[int]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens_wt)
        assert len(tokens_wt) == len(tokens_mut)
        labels_wt = np.zeros([len(tokens_wt)], np.int64) - 1
        labels_mut = np.zeros([len(tokens_mut)], np.int64) - 1
        ## pos_list, wtSeq_startIdx start from 1
        for i in rela_pos_list:
            labels_wt[i] = self.tokenizer.convert_token_to_id(tokens_wt[i])
            labels_mut[i] = self.tokenizer.convert_token_to_id(tokens_mut[i])
            masked_tokens[i] = self.tokenizer.mask_token     
        return masked_tokens, labels_wt, labels_mut

    def _trim_seq_center(self,seq_len,mut_relative_idxs):
        """trim seq which is longer than max_len, and keep variant position at center if possible
        """
        assert seq_len > self.max_len
        mut_idx_min, mut_idx_max = min(mut_relative_idxs), max(mut_relative_idxs)
        if mut_idx_min < mut_idx_max: # multi-site mut
            middle_idx = (mut_idx_min + mut_idx_max) // 2
            half_size = (self.max_len - 1) // 2
            tmp_seg_start = middle_idx - half_size
            tmp_seg_end = middle_idx + (self.max_len - half_size)
            if tmp_seg_start < 0:
                seg_start = 0
                seg_end = self.max_len
            elif tmp_seg_end > seq_len:
                seg_start = seq_len - self.max_len
                seg_end = seq_len
            else:
                seg_start = tmp_seg_start
                seg_end = tmp_seg_end
            mut_relative_idxs = [idx-seg_start for idx in mut_relative_idxs]
        elif mut_idx_min == mut_idx_max: #single-site mut
            half_size = (self.max_len - 1) // 2
            tmp_seg_start = mut_idx_min - half_size
            tmp_seg_end = mut_idx_min + (self.max_len - half_size)
            if tmp_seg_start < 0:
                seg_start = 0
                seg_end = self.max_len
            elif tmp_seg_end > seq_len:
                seg_start = seq_len - self.max_len
                seg_end = seq_len
            else:
                seg_start = tmp_seg_start
                seg_end = tmp_seg_end
            mut_relative_idxs = [idx-seg_start for idx in mut_relative_idxs]
        else:
            Exception('Invalid mutation indices')
        return seg_start, seg_end, mut_relative_idxs

@registry.register_task('multitask_fitness_UNsupervise_mutagenesis_structure')
class MultiTaskMutagenesisUnSVStructureDataset(Dataset):
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        set_nm = kwargs.get('mutgsis_set',None)
        self.neighbor_strategy = kwargs.get('neighbor_strategy')
        self.knn_value = kwargs.get('knn_value')
        self.dist_cutoff = kwargs.get('dist_cutoff')
        self.model_config = kwargs.get('model_config')
        self.max_len = int(self.model_config.seq_max_length) - 2 # remove two special positions: start and end
        self.eval_label_name = getattr(self.model_config, 'eval_label_name', 'fitness')
        self.split = split
        self.dist_num_bins = getattr(self.model_config, 'num_dist_classes', 32)
        self.dist_first_cutoff = getattr(self.model_config, 'dist_first_cutoff', 2.3125)
        self.dist_last_cutoff = getattr(self.model_config, 'dist_last_cutoff', 21.6875)
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        if self.eval_label_name == 'fitness':
            if set_nm is not None:
                data_path = Path(data_path)
                data_file = f'{set_nm}/{set_nm}_mut_all.{file_format}'
                self.data = dataset_factory(data_path / data_file, in_memory)
            else:
                data_path = Path(data_path)
                data_file = f'mut_all.{file_format}'
                self.data = dataset_factory(data_path / data_file, in_memory)
        elif self.eval_label_name == 'stability':
            data_path = Path(data_path)
            data_file = f'{set_nm}.{file_format}'
            self.data = dataset_factory(data_path / data_file, in_memory)
        else:
            pass
            Exception(f'invalid label name for evaluation: {self.eval_label_name}')

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        seq_len = int(item['seq_len'])
        mut_relative_idxs = item['mut_relative_idxs']
        if seq_len > self.max_len:
            seg_start,seg_end,mut_relative_idxs = self._trim_seq_center(seq_len,mut_relative_idxs)
        else:
            seg_start = 0
            seg_end = seq_len
        tokens_wt = self.tokenizer.tokenize(item['wt_seq'][seg_start:seg_end])
        tokens_mut = self.tokenizer.tokenize(item['mut_seq'][seg_start:seg_end])
        input_tokens_wt = self.tokenizer.add_special_tokens(tokens_wt)
        input_tokens_mut = self.tokenizer.add_special_tokens(tokens_mut)
        input_token_ids_wt = np.array(self.tokenizer.convert_tokens_to_ids(input_tokens_wt), dtype=np.int32)
        input_token_ids_mut = np.array(self.tokenizer.convert_tokens_to_ids(input_tokens_mut), dtype=np.int32)
        input_mask = np.ones_like(input_token_ids_wt)
        aa_seq_mask = np.ones_like(input_token_ids_wt)
        aa_seq_mask[0] = aa_seq_mask[-1] = 0

        env_structure = lmdb.open(f'/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/mutagenesis/wt_seq_structure/set_structure_data/{self.split}_AFDB.lmdb', max_readers=126, readonly=True, lock=False, readahead=False, meminit=False)
        with env_structure.begin(write=False) as txn:
          item_structure = pkl.loads(txn.get(str(0).encode()))
        distance_map = np.frombuffer(item_structure['distance_map']).astype(np.float16).reshape(seq_len,seq_len)[seg_start:seg_end,seg_start:seg_end]
        env_structure.close()
        neighbor_strategy = self.neighbor_strategy

        ## graph data
        #edge_index_arr = generate_graph(distance_map, neighbor_strategy, self.knn_value, self.dist_cutoff)
        ## Positional encoding for each node (used for GNNs)
        #node_pos_encode_arr = min_max_normalize_array(np.arange(len(tokens_wt))).reshape(-1, 1)  # [num_node, 1]
        ## Positional encoding for each edge (used for sequentially-ordered inputs like proteins)
        #edge_pos_encode_arr = np.sin((edge_index_arr[0] - edge_index_arr[1]).astype(np.float64)).reshape(-1, 1)  # [num_edges, 1]
        #graph_data = pyg_Data(edge_index=torch.tensor(edge_index_arr,dtype=torch.long),pos=torch.tensor(node_pos_encode_arr,dtype=torch.float),#edge_pos=torch.tensor(edge_pos_encode_arr,dtype=torch.float))

        ## SS,RSA,DistMap labels
        ss3_str = item_structure['ss3'][seg_start:seg_end]
        ss8_str = item_structure['ss8'][seg_start:seg_end]
        rsa2_str = item_structure['rsa_class'][seg_start:seg_end]
        rsa2_value = np.frombuffer(item_structure['rsa_value']).astype(np.float16)[seg_start:seg_end]
        # discretize distance map
        distMap_label_ids = discretize_distogram_fast(distance_map,self.dist_first_cutoff,self.dist_last_cutoff,self.dist_num_bins,ignore_index=-1,dtype=np.int8)
        distMap_label_ids = check_seq_neighbors(distMap_label_ids,value=-1,diag_idx=[0])

        # pass structure label of whole seq
        ss3_label_ids = np.array(self.tokenizer.struct_convert_tokens_to_ids(list(ss3_str),'ss3'),dtype=np.int8) # [L,]
        rsa2_label_ids = np.array(self.tokenizer.struct_convert_tokens_to_ids(list(rsa2_str),'rsa2'),dtype=np.int8) # [L,]

        ## mutation names, e.g. ['A20','S35']
        wt_names, mut_names = [':'.join(item['mutants'])], [':'.join(item['mutants'])]
        for mut_name in item['mutants']:
            wt_names.append(f'{mut_name[0]}{mut_name[1:-1]}')
            mut_names.append(f'{mut_name[-1]}{mut_name[1:-1]}')

        return [input_token_ids_wt,input_token_ids_mut], [input_mask]*2, [aa_seq_mask]*2, [ss3_label_ids]*2, [rsa2_label_ids]*2, [distMap_label_ids]*2, [item['set_nm']]*2, [wt_names,mut_names], [mut_relative_idxs]*2, [float(item['fitness'])]*2

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, aa_seq_mask, ss3_label_ids, rsa2_label_ids, distMap_label_ids, set_nms, mutant, mut_relative_idxs, fitness_score = tuple(zip(*batch))
        # flatten sub-batch
        input_ids = list(chain(*input_ids))
        input_mask = list(chain(*input_mask))
        aa_seq_mask = list(chain(*aa_seq_mask))
        ss3_label_ids = list(chain(*ss3_label_ids))
        rsa2_label_ids = list(chain(*rsa2_label_ids))
        distMap_label_ids = list(chain(*distMap_label_ids))
        set_nms = list(chain(*set_nms))
        mutant = list(chain(*mutant))
        mut_relative_idxs = list(chain(*mut_relative_idxs))
        #graph_data_tuple = list(chain(*graph_data_tuple))
        fitness_score = torch.FloatTensor(list(chain(*fitness_score)))

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        aa_seq_mask = torch.from_numpy(pad_sequences(aa_seq_mask, 0))
        #graph_batch = pyg_Batch.from_data_list(graph_data_tuple)
        #graph_batch_idxs = torch.arange(graph_batch.num_graphs,dtype=torch.int16)
        # mutant: tuple
        return {'input_seq_ids': input_ids,
                'input_seq_mask': input_mask,
                'aa_seq_mask': aa_seq_mask,
                'ss3_label_ids':ss3_label_ids,
                'rsa2_label_ids':rsa2_label_ids,
                'distMap_label_ids':distMap_label_ids,
                #'graph_batch': graph_batch,
                #'graph_batch_idxs': graph_batch_idxs,
                'fitness_score': fitness_score,
                'set_nms': set_nms,
                'mutants': mutant,
                'mut_relative_idxs': mut_relative_idxs}

    def _apply_mut_mask(self, 
                        tokens_wt: List[str], 
                        tokens_mut: List[str],
                        rela_pos_list: List[int]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens_wt)
        assert len(tokens_wt) == len(tokens_mut)
        labels_wt = np.zeros([len(tokens_wt)], np.int64) - 1
        labels_mut = np.zeros([len(tokens_mut)], np.int64) - 1
        ## pos_list, wtSeq_startIdx start from 1
        for i in rela_pos_list:
            labels_wt[i] = self.tokenizer.convert_token_to_id(tokens_wt[i])
            labels_mut[i] = self.tokenizer.convert_token_to_id(tokens_mut[i])
            masked_tokens[i] = self.tokenizer.mask_token     
        return masked_tokens, labels_wt, labels_mut

    def _trim_seq_center(self,seq_len,mut_relative_idxs):
        """trim seq which is longer than max_len, and keep variant position at center if possible
        """
        assert seq_len > self.max_len
        mut_idx_min, mut_idx_max = min(mut_relative_idxs), max(mut_relative_idxs)
        if mut_idx_min < mut_idx_max: # multi-site mut
            middle_idx = (mut_idx_min + mut_idx_max) // 2
            half_size = (self.max_len - 1) // 2
            tmp_seg_start = middle_idx - half_size
            tmp_seg_end = middle_idx + (self.max_len - half_size)
            if tmp_seg_start < 0:
                seg_start = 0
                seg_end = self.max_len
            elif tmp_seg_end > seq_len:
                seg_start = seq_len - self.max_len
                seg_end = seq_len
            else:
                seg_start = tmp_seg_start
                seg_end = tmp_seg_end
            mut_relative_idxs = [idx-seg_start for idx in mut_relative_idxs]
        elif mut_idx_min == mut_idx_max: #single-site mut
            half_size = (self.max_len - 1) // 2
            tmp_seg_start = mut_idx_min - half_size
            tmp_seg_end = mut_idx_min + (self.max_len - half_size)
            if tmp_seg_start < 0:
                seg_start = 0
                seg_end = self.max_len
            elif tmp_seg_end > seq_len:
                seg_start = seq_len - self.max_len
                seg_end = seq_len
            else:
                seg_start = tmp_seg_start
                seg_end = tmp_seg_end
            mut_relative_idxs = [idx-seg_start for idx in mut_relative_idxs]
        else:
            Exception('Invalid mutation indices')
        return seg_start, seg_end, mut_relative_idxs

@registry.register_task('mutation_fitness_UNsupervise_scanning')
class MutagenesisUnSVScanDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        '''
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")
        '''
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'{split}.{file_format}'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        tokens_wt = self.tokenizer.tokenize(item['wt_seq'])
        masked_tokens, labels_wt = self._apply_mut_mask(tokens_wt,item['mut_relative_idxs'])
        masked_tokens = self.tokenizer.add_special_tokens(masked_tokens)
        labels_wt = np.concatenate(([-1],labels_wt,[-1]))
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64) 
        input_mask = np.ones_like(masked_token_ids)
        return masked_token_ids, input_mask, labels_wt, item['mut_abso_idxs']

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, lm_label_wt_ids, mut_abso_idxs = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        lm_label_wt_ids = torch.from_numpy(pad_sequences(lm_label_wt_ids, -1))
        # mutant: tuple
        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': lm_label_wt_ids,
                'mut_abso_idxs': mut_abso_idxs}

    def _apply_mut_mask(self, 
                        tokens_wt: List[str], 
                        rela_pos_list: List[int]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens_wt)
        labels_wt = np.zeros([len(tokens_wt)], np.int64) - 1
        ## pos_list, wtSeq_startIdx start from 1
        for i in rela_pos_list:
            labels_wt[i] = self.tokenizer.convert_token_to_id(tokens_wt[i])
            masked_tokens[i] = self.tokenizer.mask_token     
        return masked_tokens, labels_wt

@registry.register_task('mutation_fitness_UNsupervise_CAGI')
class MutationUnSVDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = False,
                 file_format: str = 'lmdb',
                 **kwargs):
        '''
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")
        '''
        
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        tokens = self.tokenizer.tokenize(item['wt_seq'])
        masked_tokens, labels = self._apply_mut_mask(tokens,item['mut_pos'],item['wt_aa'],item['wtSeq_startIdx'])
        masked_tokens = self.tokenizer.add_special_tokens(masked_tokens)
        labels = np.concatenate(([-1],labels,[-1]))
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)
        return masked_token_ids, input_mask, labels, item['mut_pos']

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, lm_label_ids, mut_pos = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': lm_label_ids,
                'mut_pos': mut_pos}

    def _apply_mut_mask(self, 
                        tokens: List[str], 
                        pos_list: List[int],
                        wt_aa: List[str], 
                        wtSeq_startIdx: int) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1
        ## pos_list, wtSeq_startIdx start from 1
        for i in range(len(pos_list)):
            mut_pos = pos_list[i]
            aaRef = wt_aa[i]
            relIdx = mut_pos - wtSeq_startIdx
            try:
                assert relIdx >= 0
                assert aaRef == masked_tokens[relIdx]
            except:
                Exception('mut_pos,wtSeq_startIdx;aaRef,masked_tokens[relIdx]'.format(mut_pos,wtSeq_startIdx,aaRef,masked_tokens[relIdx]))
            labels[relIdx] = self.tokenizer.convert_token_to_id(masked_tokens[relIdx])
            masked_tokens[relIdx] = self.tokenizer.mask_token
        return masked_tokens, labels


@registry.register_task('antibody_mlm_seqConcate')
class ABSeqConcateMaskedLanguageModelingDataset(Dataset):
    """Creates the Masked Language Modeling for antibody Dataset
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout', ...], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory. Default: False.
        file_format (str): format of data file (Default: 'lmdb')
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        super().__init__()
        '''
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")
        '''
        self.mask_stragy = kwargs.get('mlm_mask_stragy') # if not exist, return None
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        data_path = Path(data_path)
        data_file = f"HL_pair_{split}.{file_format}"
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # for i-th example, add start/end token, mask seq, convert to ids, make input_mask
        item = self.data[index]
        tokens_VH = self.tokenizer.tokenize(item['seqVH'])
        tokens_VL = self.tokenizer.tokenize(item['seqVL'])
        tokens_pair = tokens_VH + ['<sep>'] + tokens_VL
        tokens = self.tokenizer.add_special_tokens(tokens_pair)
        if self.mask_stragy == 'vanilla':
            masked_tokens, labels = self._apply_bert_mask(tokens)
            masked_token_ids = np.array(self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        elif self.mask_stragy == 'cdr_vanilla':
            masked_tokens, labels = self._apply_bert_mask_cdrVani(tokens,len(tokens_VH),len(tokens_VL),
                                        [item['cdr1HIdx'],item['cdr2HIdx'],item['cdr3HIdx']],[item['cdr1LIdx'],item['cdr2LIdx'],item['cdr3LIdx']])
            masked_token_ids = np.array(self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        elif self.mask_stragy == 'cdr_margin':
            ## size change: [L,] -> [6,L] (each seq mask one whole cdr region, 3+3)
            masked_tokens, labels = self._apply_bert_mask_cdr_margin(tokens,len(tokens_VH),len(tokens_VL),
                                        [item['cdr1HIdx'],item['cdr2HIdx'],item['cdr3HIdx']],[item['cdr1LIdx'],item['cdr2LIdx'],item['cdr3LIdx']])
            masked_token_ids = np.array([self.tokenizer.convert_tokens_to_ids(masked_tokens_i) for masked_tokens_i in masked_tokens], np.int64)
        elif self.mask_stragy == 'cdr_pair':
            ## size change: [L,] -> [9,L] (each seq mask one whole cdr region, 3*3)
            masked_tokens, labels = self._apply_bert_mask_cdr_pair(tokens,len(tokens_VH),len(tokens_VL),
                                        [item['cdr1HIdx'],item['cdr2HIdx'],item['cdr3HIdx']],[item['cdr1LIdx'],item['cdr2LIdx'],item['cdr3LIdx']])
            masked_token_ids = np.array([self.tokenizer.convert_tokens_to_ids(masked_tokens_i) for masked_tokens_i in masked_tokens], np.int64)
        else:
            Exception('Unrecognized MLM mask strategy: {}'.format(self.mask_stragy))
        input_mask = np.ones_like(masked_token_ids)
        ## token_type_ids(segment_id)
        ## <cls> VH <sep> VL <sep> <pad> ...
        ##   0  {0}   0  {1}   1     1 ...
        aug_size = 1 if len(masked_token_ids.shape) == 1 else masked_token_ids.shape[0]
        token_type_ids = np.array([[0] + [0]*len(tokens_VH) + [0] + [1]*len(tokens_VL) + [1]]*aug_size, np.int64).squeeze()
        subClassH = np.array([ab_H_subclass[item['subclassH']]]*aug_size).squeeze() if item['subclassH'].lower() != 'unknown' else np.array([-1]*aug_size).squeeze()
        subClassL = np.array([ab_L_subclass[item['subclassL']]]*aug_size).squeeze() if item['subclassL'].lower() != 'unknown' else np.array([-1]*aug_size).squeeze()
        if item['subclassH'].lower() != 'unknown' and item['subclassL'].lower() != 'unknown':
            subClassHLPair = np.array([ab_HL_subclass['{}-{}'.format(item['subclassH'],item['subclassL'])]]*aug_size).squeeze()
        else:
            subClassHLPair = np.array([-1]*aug_size).squeeze()
        return masked_token_ids, input_mask, labels, token_type_ids, subClassH, subClassL, subClassHLPair

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, lm_label_ids, token_type_ids, subClassH, subClassL, subClassHLPair = tuple(zip(*batch))
        
        ## reshape mini-batch of cdrOne masked seqs
        ## e.g. input_ids: tuple -> list, [bs,augment_size,...] -> [bs*augment_size,...]
        if self.mask_stragy == 'cdr_margin' or self.mask_stragy == 'cdr_pair':
            input_ids = list(chain(*input_ids))
            input_mask = list(chain(*input_mask))
            lm_label_ids = list(chain(*lm_label_ids))
            token_type_ids = list(chain(*token_type_ids))
            subClassH = list(chain(*subClassH))
            subClassL = list(chain(*subClassL))
            subClassHLPair = list(chain(*subClassHLPair))

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        token_type_ids = torch.from_numpy(pad_sequences(token_type_ids, 1))
        # ignore_index is -1
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))
        subClassH = torch.LongTensor(np.array(subClassH))
        subClassL = torch.LongTensor(np.array(subClassL))
        subClassHLPair = torch.LongTensor(np.array(subClassHLPair))
        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': lm_label_ids,
                'token_type_ids': token_type_ids,
                'subClassH': subClassH,
                'subClassL': subClassL,
                'subClassHLPair': subClassHLPair}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1
        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            # also stop token is used as separation token between H/L
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                continue
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)
                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token(not special tokens)
                    token = self.tokenizer.convert_id_to_token(
                        random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                else:
                    # 10% chance to keep current token
                    pass
                masked_tokens[i] = token    
        return masked_tokens, labels
    def _apply_bert_mask_cdrVani(self, tokens: List[str],
                                 lenVH: int, lenVL: int, 
                                 cdrHIdx: List[List], cdrLIdx: List[List]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1
        ## rescale index of cdr region
        ## <cls>,VH,<sep>,<VL>,<sep>
        cdrIdx_list = list(range(cdrHIdx[0][0]+1,cdrHIdx[0][1]+2)) + \
                      list(range(cdrHIdx[1][0]+1,cdrHIdx[1][1]+2)) + \
                      list(range(cdrHIdx[2][0]+1,cdrHIdx[2][1]+2)) + \
                      list(range(cdrLIdx[0][0]+lenVH+2,cdrLIdx[0][1]+lenVH+3)) + \
                      list(range(cdrLIdx[1][0]+lenVH+2,cdrLIdx[1][1]+lenVH+3)) + \
                      list(range(cdrLIdx[2][0]+lenVH+2,cdrLIdx[2][1]+lenVH+3))
        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            # also stop token is used as separation token between H/L
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                continue
            prob = random.random()
            ## non-cdr region, only change residue
            if i not in cdrIdx_list:
                if prob < 0.15:
                    prob /= 0.15
                    labels[i] = self.tokenizer.convert_token_to_id(token)
                    if prob < 0.8:
                        # 80% random change to mask token
                        #token = self.tokenizer.mask_token
                        pass
                    elif prob < 0.9:
                        # 10% chance to change to random token(not special tokens)
                        token = self.tokenizer.convert_id_to_token(
                            random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                    else:
                        # 10% chance to keep current token
                        pass
                    masked_tokens[i] = token 
            else:
                if prob < 0.15:
                    prob /= 0.15
                    labels[i] = self.tokenizer.convert_token_to_id(token)
                    if prob < 0.8:
                        # 80% random change to mask token
                        token = self.tokenizer.mask_token
                    elif prob < 0.9:
                        # 10% chance to change to random token(not special tokens)
                        token = self.tokenizer.convert_id_to_token(
                            random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                    else:
                        # 10% chance to keep current token
                        pass
                    masked_tokens[i] = token
        return masked_tokens, labels
    
    def _apply_bert_mask_cdr_margin(self, tokens: List[str],
                                    lenVH: int, lenVL: int, 
                                    cdrHIdx: List[List], cdrLIdx: List[List]) -> Tuple[List[List], List[List]]:
        masked_tokens_aug = []
        labels_aug = []
        ## rescale index of cdr region
        ## <cls>,VH,<sep>,<VL>,<sep>
        cdrHIdx_list = [list(range(cdrHIdx[0][0]+1,cdrHIdx[0][1]+2)),list(range(cdrHIdx[1][0]+1,cdrHIdx[1][1]+2)),list(range(cdrHIdx[2][0]+1,cdrHIdx[2][1]+2))]
        cdrLIdx_list = [list(range(cdrLIdx[0][0]+lenVH+2,cdrLIdx[0][1]+lenVH+3)),list(range(cdrLIdx[1][0]+lenVH+2,cdrLIdx[1][1]+lenVH+3)),list(range(cdrLIdx[2][0]+lenVH+2,cdrLIdx[2][1]+lenVH+3))]
        cdrAllIdx_list = cdrHIdx_list + cdrLIdx_list
        for cdr_i in range(len(cdrAllIdx_list)): 
            cdr_range = cdrAllIdx_list[cdr_i]
            masked_tokens = copy(tokens)
            labels = np.zeros([len(tokens)], np.int64) - 1
            for i, token in enumerate(tokens):
                # Tokens begin and end with start_token and stop_token, ignore these
                # also stop token is used as separation token between H/L
                if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                    continue
                ## mask cdr regions in H/L chain
                if i in cdr_range:
                    labels[i] = self.tokenizer.convert_token_to_id(token) 
                    token = self.tokenizer.mask_token
                    masked_tokens[i] = token
                else:
                    prob = random.random()
                    if prob < 0.15:
                        prob /= 0.15
                        labels[i] = self.tokenizer.convert_token_to_id(token)
                        if prob < 0.8:
                            # 80% random change to mask token
                            #token = self.tokenizer.mask_token
                            pass
                        elif prob < 0.9:
                            # 10% chance to change to random token(not special tokens)
                            token = self.tokenizer.convert_id_to_token(
                                random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                        else:
                            # 10% chance to keep current token
                            pass
                        masked_tokens[i] = token
            masked_tokens_aug.append(masked_tokens)
            labels_aug.append(labels)
        return masked_tokens_aug, labels_aug

    def _apply_bert_mask_cdr_pair(self, tokens: List[str],
                                    lenVH: int, lenVL: int, 
                                    cdrHIdx: List[List], cdrLIdx: List[List]) -> Tuple[List[List], List[List]]:
        masked_tokens_aug = []
        labels_aug = []
        ## rescale index of cdr region
        ## <cls>,VH,<sep>,<VL>,<sep>
        cdrHIdx_list = [list(range(cdrHIdx[0][0]+1,cdrHIdx[0][1]+2)),list(range(cdrHIdx[1][0]+1,cdrHIdx[1][1]+2)),list(range(cdrHIdx[2][0]+1,cdrHIdx[2][1]+2))]
        cdrLIdx_list = [list(range(cdrLIdx[0][0]+lenVH+2,cdrLIdx[0][1]+lenVH+3)),list(range(cdrLIdx[1][0]+lenVH+2,cdrLIdx[1][1]+lenVH+3)),list(range(cdrLIdx[2][0]+lenVH+2,cdrLIdx[2][1]+lenVH+3))]
        for cdrh in range(len(cdrHIdx_list)):
            for cdrl in range(len(cdrLIdx_list)): 
                cdrh_range = cdrHIdx_list[cdrh]
                cdrl_range = cdrLIdx_list[cdrl]
                masked_tokens = copy(tokens)
                labels = np.zeros([len(tokens)], np.int64) - 1
                for i, token in enumerate(tokens):
                    # Tokens begin and end with start_token and stop_token, ignore these
                    # also stop token is used as separation token between H/L
                    if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                        continue
                    ## mask cdr regions in H/L chain
                    if i in cdrh_range or i in cdrl_range:
                        labels[i] = self.tokenizer.convert_token_to_id(token) 
                        token = self.tokenizer.mask_token
                        masked_tokens[i] = token
                    else:
                        prob = random.random()
                        if prob < 0.15:
                            prob /= 0.15
                            labels[i] = self.tokenizer.convert_token_to_id(token)
                            if prob < 0.8:
                                # 80% random change to mask token
                                #token = self.tokenizer.mask_token
                                pass
                            elif prob < 0.9:
                                # 10% chance to change to random token(not special tokens)
                                token = self.tokenizer.convert_id_to_token(
                                    random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                            else:
                                # 10% chance to keep current token
                                pass
                            masked_tokens[i] = token
            masked_tokens_aug.append(masked_tokens)
            labels_aug.append(labels)
        return masked_tokens_aug, labels_aug

@registry.register_task('antibody_embed_seqConcate')
class ABSeqConcateEmbedLanguageModelingDataset(Dataset):
    """Creates the Masked Language Modeling for antibody Dataset
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout', ...], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory. Default: False.
        file_format (str): format of data file (Default: 'lmdb')
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        super().__init__()
        '''
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")
        '''
        self.mask_stragy = kwargs.get('mlm_mask_stragy') # if not exist, return None
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        data_path = Path(data_path)
        data_file = f"{split}.{file_format}" # 'HL_pair_{split}'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # for i-th example, add start/end token, mask seq, convert to ids, make input_mask
        item = self.data[index]
        tokens_VH = self.tokenizer.tokenize(item['seqVH'])
        tokens_VL = self.tokenizer.tokenize(item['seqVL'])
        tokens_pair = tokens_VH + ['<sep>'] + tokens_VL
        tokens_pair = self.tokenizer.add_special_tokens(tokens_pair)
        token_ids = np.array(self.tokenizer.convert_tokens_to_ids(tokens_pair), np.int64)
        input_mask = np.ones_like(token_ids)
        ## token_type_ids(segment_id)
        ## <cls> VH <sep> VL <sep> <pad> ...
        ##   0  {0}   0  {1}   1     1 ...
        token_type_ids = np.array([[0] + [0]*len(tokens_VH) + [0] + [1]*len(tokens_VL) + [1]], np.int64).squeeze()
        subClassH_str = item['subclassH'] if 'subclassH' in item.keys() else 'unknown'
        subClassL_str = item['subclassL'] if 'subclassL' in item.keys() else 'unknown'
        subClassH = ab_H_subclass[subClassH_str]
        subClassL = ab_L_subclass[subClassL_str]
        subClassHLPair = ab_HL_subclass['{}-{}'.format(subClassH_str,subClassL_str)]
        return token_ids, input_mask, token_type_ids, subClassH, subClassL, subClassHLPair, item['entityH'], item['entityL']

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, token_type_ids, subClassH, subClassL, subClassHLPair, entityH, entityL = tuple(zip(*batch))
        
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        token_type_ids = torch.from_numpy(pad_sequences(token_type_ids, 1))
        # ignore_index is -1
        subClassH = torch.LongTensor(np.array(subClassH))
        subClassL = torch.LongTensor(np.array(subClassL))
        subClassHLPair = torch.LongTensor(np.array(subClassHLPair))
        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'token_type_ids': token_type_ids,
                'subClassH': subClassH,
                'subClassL': subClassL,
                'subClassHLPair': subClassHLPair,
                'entityH': entityH,
                'entityL': entityL}

@registry.register_task('antibody_mlm_seqIndiv')
class ABSeqIndivMaskedLanguageModelingDataset(Dataset):
    """Creates the Masked Language Modeling for antibody Dataset
       Heavy and light chain seqs are encoded individually
       Self-attention for intra-seq; across-attention for inter-seq
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout', ...], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory. Default: False.
        file_format (str): format of data file (Default: 'lmdb')
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        super().__init__()
        '''
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")
        '''
        self.mask_stragy = kwargs.get('mlm_mask_stragy') # if not exist, return None
        #print('**mask_stragy:{}**'.format(self.mask_stragy))
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        data_path = Path(data_path)
        data_file = f"HL_pair_{split}.{file_format}"
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # for i-th example, add start/end token, mask seq, convert to ids, make input_mask
        item = self.data[index]
        tokens_VH = self.tokenizer.tokenize(item['seqVH'])
        tokens_VL = self.tokenizer.tokenize(item['seqVL'])
        tokensSE_VH = self.tokenizer.add_special_tokens(tokens_VH)
        tokensSE_VL = self.tokenizer.add_special_tokens(tokens_VL)
        masked_token_ids_VH, masked_token_ids_VL = None, None
        if self.mask_stragy == 'vanilla':
            masked_tokens_VH, labels_VH = self._apply_bert_mask(tokensSE_VH)
            masked_tokens_VL, labels_VL = self._apply_bert_mask(tokensSE_VL)
            masked_token_ids_VH = np.array(
                self.tokenizer.convert_tokens_to_ids(masked_tokens_VH), np.int64)
            masked_token_ids_VL = np.array(
                self.tokenizer.convert_tokens_to_ids(masked_tokens_VL), np.int64)
        elif self.mask_stragy == 'cdr_vanilla':
            masked_tokens_VH, labels_VH = self._apply_bert_mask_cdrVani(tokensSE_VH,[item['cdr1HIdx'],item['cdr2HIdx'],item['cdr3HIdx']])
            masked_tokens_VL, labels_VL = self._apply_bert_mask_cdrVani(tokensSE_VL,[item['cdr1LIdx'],item['cdr2LIdx'],item['cdr3LIdx']])
            masked_token_ids_VH = np.array(
                self.tokenizer.convert_tokens_to_ids(masked_tokens_VH), np.int64)
            masked_token_ids_VL = np.array(
                self.tokenizer.convert_tokens_to_ids(masked_tokens_VL), np.int64)
        elif self.mask_stragy == 'cdr_margin':
            ## size change: [L,] -> [6,L] (each seq mask one whole cdr region)
            masked_tokens_VH_cdr, labels_VH_cdr = self._apply_bert_mask_cdrOne(tokensSE_VH,[item['cdr1HIdx'],item['cdr2HIdx'],item['cdr3HIdx']])
            masked_tokens_VH_noise, labels_VH_noise = self._apply_bert_mask_noise(tokensSE_VH)
            masked_tokens_VH = np.concatenate((masked_tokens_VH_cdr,masked_tokens_VH_noise), axis=0)
            labels_VH = np.concatenate((labels_VH_cdr,labels_VH_noise),axis=0)

            masked_tokens_VL_cdr, labels_VL_cdr = self._apply_bert_mask_cdrOne(tokensSE_VL,[item['cdr1LIdx'],item['cdr2LIdx'],item['cdr3LIdx']])
            masked_tokens_VL_noise, labels_VL_noise = self._apply_bert_mask_noise(tokensSE_VL)
            masked_tokens_VL = np.concatenate((masked_tokens_VL_cdr,masked_tokens_VL_noise), axis=0)
            labels_VL = np.concatenate((labels_VL_noise,labels_VL_cdr),axis=0)

            masked_token_ids_VH = np.array([self.tokenizer.convert_tokens_to_ids(masked_tokens_i) for masked_tokens_i in masked_tokens_VH], np.int64)
            masked_token_ids_VL = np.array([self.tokenizer.convert_tokens_to_ids(masked_tokens_i) for masked_tokens_i in masked_tokens_VL], np.int64)
        elif self.mask_stragy == 'cdr_pair':
            ## size change: [L,] -> [9,L] (each seq mask one whole cdr region)
            masked_tokens_VH_unpair, labels_VH_unpair = self._apply_bert_mask_cdrOne(tokensSE_VH,[item['cdr1HIdx'],item['cdr2HIdx'],item['cdr3HIdx']])
            masked_tokens_VL_unpair, labels_VL_unpair = self._apply_bert_mask_cdrOne(tokensSE_VL,[item['cdr1LIdx'],item['cdr2LIdx'],item['cdr3LIdx']])
            masked_token_ids_VH = np.array([self.tokenizer.convert_tokens_to_ids(masked_tokens_i) for masked_tokens_i in masked_tokens_VH_unpair], np.int64)
            masked_token_ids_VL = np.array([self.tokenizer.convert_tokens_to_ids(masked_tokens_i) for masked_tokens_i in masked_tokens_VL_unpair], np.int64)
            masked_token_ids_VH = np.repeat(masked_token_ids_VH,[3,3,3],axis=0) #[a1,a1,a1,b1,b1,b1,c1,c1,c1]
            masked_token_ids_VL = np.tile(masked_token_ids_VL,(3,1)) #[a2,b2,c2,a2,b2,c2,a2,b2,c2]
            labels_VH = np.repeat(labels_VH_unpair,[3,3,3],axis=0)
            labels_VL = np.tile(labels_VL_unpair,(3,1))
        else:
            Exception('Unrecognized MLM mask strategy: {}'.format(self.mask_stragy))
        assert (masked_token_ids_VH is not None) and (masked_token_ids_VL is not None)
        input_mask_VH = np.ones_like(masked_token_ids_VH)
        input_mask_VL = np.ones_like(masked_token_ids_VL)
        aug_size = 1 if len(masked_token_ids_VH.shape) == 1 else masked_token_ids_VH.shape[0]
        subClassH = np.array([ab_H_subclass[item['subclassH']]]*aug_size).squeeze()
        subClassL = np.array([ab_L_subclass[item['subclassL']]]*aug_size).squeeze()
        subClassHLPair = np.array([ab_HL_subclass['{}-{}'.format(item['subclassH'],item['subclassL'])]]*aug_size).squeeze()
        return masked_token_ids_VH, masked_token_ids_VL,\
               input_mask_VH, input_mask_VL,\
               labels_VH, labels_VL,\
               subClassH, subClassL, subClassHLPair, \
               item['entityH'], item['entityL']

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids_VH, input_ids_VL, input_mask_VH, input_mask_VL,\
        lm_label_ids_VH, lm_label_ids_VL, subClassH, subClassL, subClassHLPair, entityH, entityL = tuple(zip(*batch))

        ## reshape mini-batch of cdrOne masked seqs
        ## e.g. input_ids: tuple -> list, [bs,augment_size,...] -> [bs*augment_size,...]
        if self.mask_stragy == 'cdr_margin' or self.mask_stragy == 'cdr_pair':
            input_ids_VH = list(chain(*input_ids_VH))
            input_ids_VL = list(chain(*input_ids_VL))
            input_mask_VH = list(chain(*input_mask_VH))
            input_mask_VL = list(chain(*input_mask_VL))
            lm_label_ids_VH = list(chain(*lm_label_ids_VH))
            lm_label_ids_VL = list(chain(*lm_label_ids_VL))
            subClassH = list(chain(*subClassH))
            subClassL = list(chain(*subClassL))
            subClassHLPair = list(chain(*subClassHLPair))

        input_ids_VH = torch.from_numpy(pad_sequences(input_ids_VH, 0))
        input_ids_VL = torch.from_numpy(pad_sequences(input_ids_VL, 0))
        input_mask_VH = torch.from_numpy(pad_sequences(input_mask_VH, 0))
        input_mask_VL = torch.from_numpy(pad_sequences(input_mask_VL, 0))
        # ignore_index is -1
        lm_label_ids_VH = torch.from_numpy(pad_sequences(lm_label_ids_VH, -1))
        lm_label_ids_VL = torch.from_numpy(pad_sequences(lm_label_ids_VL, -1))
        subClassH = torch.LongTensor(np.array(subClassH))  # type: ignore
        subClassL = torch.LongTensor(np.array(subClassL))  # type: ignore
        subClassHLPair = torch.LongTensor(np.array(subClassHLPair))
        return {'input_ids_VH': input_ids_VH,
                'input_ids_VL': input_ids_VL,
                'input_mask_VH': input_mask_VH,
                'input_mask_VL': input_mask_VL,
                'targets_VH': lm_label_ids_VH,
                'targets_VL': lm_label_ids_VL,
                'subClassH': subClassH,
                'subClassL': subClassL,
                'subClassHLPair': subClassHLPair,
                'entityH': entityH,
                'entityL': entityL}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1
        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            # also stop token is used as separation token between H/L
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                continue
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)
                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token(not special tokens)
                    token = self.tokenizer.convert_id_to_token(
                        random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                else:
                    # 10% chance to keep current token
                    pass
                masked_tokens[i] = token    
        return masked_tokens, labels

    def _apply_bert_mask_noise(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens_aug = []
        labels_aug = []
        for cpy in range(3):
            masked_tokens = copy(tokens)
            labels = np.zeros([len(tokens)], np.int64) - 1
            for i, token in enumerate(tokens):
                # Tokens begin and end with start_token and stop_token, ignore these
                # also stop token is used as separation token between H/L
                if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                    continue
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    labels[i] = self.tokenizer.convert_token_to_id(token)
                    if prob < 0.8:
                        # 80% random change to mask token
                        #token = self.tokenizer.mask_token
                        pass
                    elif prob < 0.9:
                        # 10% chance to change to random token(not special tokens)
                        token = self.tokenizer.convert_id_to_token(
                            random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                    else:
                        # 10% chance to keep current token
                        pass
                    masked_tokens[i] = token    
            masked_tokens_aug.append(masked_tokens)
            labels_aug.append(labels)
        return masked_tokens_aug, labels_aug

    def _apply_bert_mask_cdrVani(self, tokens: List[str],
                                 cdrIdx: List[List]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1
        ## rescale index of cdr region
        ## <cls>,VH,<sep>,<VL>,<sep>
        cdrIdx_list = list(range(cdrIdx[0][0]+1,cdrIdx[0][1]+2)) + \
                      list(range(cdrIdx[1][0]+1,cdrIdx[1][1]+2)) + \
                      list(range(cdrIdx[2][0]+1,cdrIdx[2][1]+2))
        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            # also stop token is used as separation token between H/L
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                continue
            prob = random.random()
            ## non-cdr region, only change residue
            if i not in cdrIdx_list:
                if prob < 0.15:
                    prob /= 0.15
                    labels[i] = self.tokenizer.convert_token_to_id(token)
                    if prob < 0.8:
                        # 80% random change to mask token
                        #token = self.tokenizer.mask_token
                        pass
                    elif prob < 0.9:
                        # 10% chance to change to random token(not special tokens)
                        token = self.tokenizer.convert_id_to_token(
                            random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                    else:
                        # 10% chance to keep current token
                        pass
                    masked_tokens[i] = token 
            else:
                if prob < 0.15:
                    prob /= 0.15
                    labels[i] = self.tokenizer.convert_token_to_id(token)
                    if prob < 0.8:
                        # 80% random change to mask token
                        token = self.tokenizer.mask_token
                    elif prob < 0.9:
                        # 10% chance to change to random token(not special tokens)
                        token = self.tokenizer.convert_id_to_token(
                            random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                    else:
                        # 10% chance to keep current token
                        pass
                    masked_tokens[i] = token
        return masked_tokens, labels
    def _apply_bert_mask_cdrOne(self, tokens: List[str],
                                cdrIdx: List[List]) -> Tuple[List[List], List[List]]:
        masked_tokens_aug = []
        labels_aug = []
        ## rescale index of cdr region
        ## <cls>,VH,<sep>,<VL>,<sep>
        cdrIdx_list = [list(range(cdrIdx[0][0]+1,cdrIdx[0][1]+2)),list(range(cdrIdx[1][0]+1,cdrIdx[1][1]+2)),list(range(cdrIdx[2][0]+1,cdrIdx[2][1]+2))]
        for cdr in range(len(cdrIdx_list)):
            cdr_range = cdrIdx_list[cdr]
            masked_tokens = copy(tokens)
            labels = np.zeros([len(tokens)], np.int64) - 1
            for i, token in enumerate(tokens):
                # Tokens begin and end with start_token and stop_token, ignore these
                # also stop token is used as separation token between H/L
                if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                    continue
                ## mask cdr region
                if i in cdr_range:
                    labels[i] = self.tokenizer.convert_token_to_id(token) 
                    token = self.tokenizer.mask_token
                    masked_tokens[i] = token
                else:
                    prob = random.random()
                    if prob < 0.15:
                        prob /= 0.15
                        labels[i] = self.tokenizer.convert_token_to_id(token)
                        if prob < 0.8:
                            # 80% random change to mask token
                            #token = self.tokenizer.mask_token
                            pass
                        elif prob < 0.9:
                            # 10% chance to change to random token(not special tokens)
                            token = self.tokenizer.convert_id_to_token(
                                random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                        else:
                            # 10% chance to keep current token
                            pass
                        masked_tokens[i] = token
            masked_tokens_aug.append(masked_tokens)
            labels_aug.append(labels)
        return masked_tokens_aug, labels_aug

@registry.register_task('antibody_embed_seqIndiv')
class ABSeqIndivEmbedLanguageModelingDataset(Dataset):
    """Creates the Masked Language Modeling for antibody Dataset
       Heavy and light chain seqs are encoded individually
       Self-attention for intra-seq; across-attention for inter-seq
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout', ...], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory. Default: False.
        file_format (str): format of data file (Default: 'lmdb')
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        super().__init__()
        '''
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")
        '''
        self.mask_stragy = kwargs.get('mlm_mask_stragy') # if not exist, return None
        #print('**mask_stragy:{}**'.format(self.mask_stragy))
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        data_path = Path(data_path)
        data_file = f"{split}.{file_format}" # 'HL_pair_{split}'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # for i-th example, add start/end token, mask seq, convert to ids, make input_mask
        item = self.data[index]
        tokens_VH = self.tokenizer.tokenize(item['seqVH'])
        tokens_VL = self.tokenizer.tokenize(item['seqVL'])
        tokensSE_VH = self.tokenizer.add_special_tokens(tokens_VH)
        tokensSE_VL = self.tokenizer.add_special_tokens(tokens_VL)
        token_ids_VH = np.array(self.tokenizer.convert_tokens_to_ids(tokensSE_VH), np.int64)
        token_ids_VL = np.array(self.tokenizer.convert_tokens_to_ids(tokensSE_VL), np.int64)
        input_mask_VH = np.ones_like(token_ids_VH)
        input_mask_VL = np.ones_like(token_ids_VL)
        subClassH_str = item['subclassH'] if 'subclassH' in item.keys() else 'unknown'
        subClassL_str = item['subclassL'] if 'subclassL' in item.keys() else 'unknown'
        subClassH = ab_H_subclass[subClassH_str]
        subClassL = ab_L_subclass[subClassL_str]
        subClassHLPair = ab_HL_subclass['{}-{}'.format(subClassH_str,subClassL_str)]
        return token_ids_VH, token_ids_VL,\
               input_mask_VH, input_mask_VL,\
               subClassH, subClassL, subClassHLPair,\
               item['entityH'], item['entityL']

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids_VH, input_ids_VL, input_mask_VH, input_mask_VL, subClassH, subClassL, subClassHLPair, entityH, entityL = tuple(zip(*batch))

        input_ids_VH = torch.from_numpy(pad_sequences(input_ids_VH, 0))
        input_ids_VL = torch.from_numpy(pad_sequences(input_ids_VL, 0))
        input_mask_VH = torch.from_numpy(pad_sequences(input_mask_VH, 0))
        input_mask_VL = torch.from_numpy(pad_sequences(input_mask_VL, 0))
        
        subClassH = torch.LongTensor(np.array(subClassH))  # type: ignore
        subClassL = torch.LongTensor(np.array(subClassL))  # type: ignore
        subClassHLPair = torch.LongTensor(np.array(subClassHLPair))
        return {'input_ids_VH': input_ids_VH,
                'input_ids_VL': input_ids_VL,
                'input_mask_VH': input_mask_VH,
                'input_mask_VL': input_mask_VL,
                'subClassH': subClassH,
                'subClassL': subClassL,
                'subClassHLPair': subClassHLPair,
                'entityH': entityH,
                'entityL': entityL}

@registry.register_task('seq_structure_multi_task')
class SeqStructureMultiTaskDataset(Dataset):
    """Creates dataset for seq+structure cross modality multi-task model
    The first and last cutoff and bin number for distogram follows AlphaFold2's definition
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
        file_format (str): format of data file (Default: 'json')
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 dist_first_cutoff: float=2.3125,
                 dist_last_cutoff: float=21.6875,
                 **kwargs):
        super().__init__()
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        data_path = Path(data_path)
        data_file = f"{split}.{file_format}"
        self.data = dataset_factory(data_path / data_file, in_memory)
        
        ## retrieve vars from kwargs
        self.model_config = kwargs.get('model_config')
        self.dist_num_bins = self.model_config.num_dist_classes
        self.neighbor_strategy = kwargs.get('neighbor_strategy')
        self.knn_value = kwargs.get('knn_value')
        self.dist_cutoff = kwargs.get('dist_cutoff')
        self.max_len = int(self.model_config.seq_max_length) - 2 # remove two special positions: start and end
        self.dist_first_cutoff = getattr(self.model_config, 'dist_first_cutoff', 2.3125)
        self.dist_last_cutoff = getattr(self.model_config, 'dist_last_cutoff', 21.6875)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        '''
        'seq_id': f'{family_nm}/{unp_id}/{unp_range_str}/{af_id}',
        'seq_primary': pdb_seq_target,
        'seq_len': seq_len,
        'contact_map': contact_map_target.tobytes(fill_value=np.nan),
        'distance_map': distance_map_target.tobytes(fill_value=np.nan),
        'ss3': ss3_seq_target,
        'ss8': ss8_seq_target,
        'rsa_class': rsa_class_seq_target,
        'rsa_value': rsa_value_seq_target.tobytes(fill_value=np.nan),
        'seq_reweight': weight 
        '''
        # i-th data point
        item = self.data[index]
        seq_len = int(item['seq_len'])
        if seq_len > self.max_len:
            seg_start = np.random.choice(seq_len-self.max_len+1, 1)[0]
            seg_end = seg_start + self.max_len
        else:
            seg_start = 0
            seg_end = seq_len
        seq_primary = item['seq_primary'][seg_start:seg_end]
        distance_map = np.frombuffer(item['distance_map'],dtype=np.float64).reshape(seq_len,seq_len)[seg_start:seg_end,seg_start:seg_end]
        ss3_str = item['ss3'][seg_start:seg_end]
        ss8_str = item['ss8'][seg_start:seg_end]
        rsa2_str = item['rsa_class'][seg_start:seg_end]
        rsa2_value = np.frombuffer(item['rsa_value'],dtype=np.float64)[seg_start:seg_end]
        
        # tokenize amino acid seq
        aa_tokens = self.tokenizer.tokenize(seq_primary)
        aa_tokens = self.tokenizer.add_special_tokens(aa_tokens)
        aa_masked_tokens, aa_labels = self._apply_bert_mask(aa_tokens)
        aa_masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(aa_masked_tokens), np.int64)
        aa_input_mask = aa_seq_mask = np.ones_like(aa_masked_token_ids)
        aa_seq_mask[0] = aa_seq_mask[-1] = 0

        # tokenize structure elements: ss,rsa
        ss3_tokens = self.tokenizer.tokenize(ss3_str)
        rsa2_tokens = self.tokenizer.tokenize(rsa2_str)
        ss3_tokens = self.tokenizer.add_placeholder_tokens(ss3_tokens, '-')
        rsa2_tokens = self.tokenizer.add_placeholder_tokens(rsa2_tokens, '-')
        ss3_ids = np.array(self.tokenizer.struct_convert_tokens_to_ids(ss3_tokens,'ss3'))
        rsa2_ids = np.array(self.tokenizer.struct_convert_tokens_to_ids(rsa2_tokens,'rsa2'))
        
        # discretize distance map
        dist_bin_mat = discretize_distogram_fast(distance_map,self.dist_first_cutoff,self.dist_last_cutoff,self.dist_num_bins)
        dist_bin_mat = add_special_dimens(dist_bin_mat,-1)

        # define neighbors
        edge_index_arr = generate_graph(distance_map, self.neighbor_strategy, self.knn_value, self.dist_cutoff)
        
        # Positional encoding for each node (used for GNNs)
        node_pos_encode_arr = min_max_normalize_array(np.arange(len(seq_primary))).reshape(-1, 1)  # [num_node, 1]
        # Positional encoding for each edge (used for sequentially-ordered inputs like proteins)
        edge_pos_encode_arr = np.sin((edge_index_arr[0] - edge_index_arr[1]).astype(np.float64)).reshape(-1, 1)  # [num_edges, 1]
        graph_data = pyg_Data(edge_index=torch.tensor(edge_index_arr,dtype=torch.long),pos=torch.tensor(node_pos_encode_arr,dtype=torch.float),edge_pos=torch.tensor(edge_pos_encode_arr,dtype=torch.float))

        ## sequence reweighting
        if 'seq_reweight' in item:
            reweight_nor = float(item['seq_reweight'])
        else:
            reweight_nor = 1.0

        return aa_masked_token_ids, aa_input_mask, aa_seq_mask, aa_labels, ss3_ids, rsa2_ids, dist_bin_mat, graph_data, reweight_nor

    def collate_fn(self, batch: List[Any]) -> Dict[str, Union[torch.Tensor,tuple]]:
        aa_masked_token_ids, aa_input_mask, aa_seq_mask, aa_labels_ids, ss3_label_ids, rsa2_label_ids, dist_bin_mat, graph_data_tuple, reweight_nor = tuple(zip(*batch))
        
        aa_masked_token_ids = torch.from_numpy(pad_sequences(aa_masked_token_ids, 0))
        aa_input_mask = torch.from_numpy(pad_sequences(aa_input_mask, 0))
        aa_seq_mask = torch.from_numpy(pad_sequences(aa_seq_mask, 0))
        aa_labels_ids = torch.from_numpy(pad_sequences(aa_labels_ids, -1))
        ss3_label_ids = torch.from_numpy(pad_sequences(ss3_label_ids, -1))
        rsa2_label_ids = torch.from_numpy(pad_sequences(rsa2_label_ids, -1))
        dist_bin_mat = torch.from_numpy(pad_distogram(dist_bin_mat, -1))
        # pyg Data/Batch's to() function is not applicable to apex
        # may need to define a custom pyg Data/Batch class to overwrite original to()
        graph_batch = pyg_Batch.from_data_list(list(graph_data_tuple))

        return {'input_seq_ids': aa_masked_token_ids,
                'input_seq_mask': aa_input_mask,
                'aa_seq_mask': aa_seq_mask,
                'graph_batch': graph_batch,
                'targets_seq':aa_labels_ids,
                'targets_ss':ss3_label_ids,
                'targets_rsa':rsa2_label_ids,
                'targets_dist':dist_bin_mat}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                continue

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)

                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token(not special tokens)
                    token = self.tokenizer.convert_id_to_token(
                        random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                else:
                    # 10% chance to keep current token
                    pass

                masked_tokens[i] = token

        return masked_tokens, labels


@registry.register_task('seq_structure_multi_task_multiCopy')
class SeqStructureMultiTaskDataset_MultiCopy(Dataset):
    """Creates dataset for seq+structure cross modality multi-task model
    The first and last cutoff and bin number for distogram follows AlphaFold2's definition
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
        file_format (str): format of data file (Default: 'json')
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 dist_first_cutoff: float=2.3125,
                 dist_last_cutoff: float=21.6875,
                 **kwargs):
        super().__init__()
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        data_path = Path(data_path)
        data_file = f"{split}.{file_format}"
        self.data = dataset_factory(data_path / data_file, in_memory)
        
        ## retrieve vars from kwargs
        self.model_config = kwargs.get('model_config')
        self.dist_num_bins = self.model_config.num_dist_classes
        self.multi_copy_num = getattr(self.model_config, 'multi_copy_num', 1)
        self.neighbor_strategy = kwargs.get('neighbor_strategy')
        self.knn_value = kwargs.get('knn_value')
        self.dist_cutoff = kwargs.get('dist_cutoff')
        self.max_len = int(self.model_config.seq_max_length) - 2 # remove two special positions: start and end
        self.dist_first_cutoff = getattr(self.model_config, 'dist_first_cutoff', 2.3125)
        self.dist_last_cutoff = getattr(self.model_config, 'dist_last_cutoff', 21.6875)
        self.ignore_seq_neighbors = getattr(self.model_config, 'ignore_seq_neighbors', False)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # i-th data point
        item = self.data[index]
        seq_len = int(item['seq_len'])
        if seq_len > self.max_len:
            max_len = self.max_len
        elif seq_len <= self.max_len and self.multi_copy_num == 1:
            max_len = seq_len
        elif seq_len <= self.max_len and self.multi_copy_num > 1:
            max_len = min(self.max_len, round(seq_len * 0.9))
        else:
            Exception(f'invalid max_len {self.max_len} or multi_copy_num {self.multi_copy_num}')
        aa_masked_token_ids_multi = []
        aa_labels_multi = []
        ss3_ids_multi = []
        rsa2_ids_multi = []
        dist_bin_mat_multi = []
        #graph_data_multi = []
        for copy_i in range(self.multi_copy_num):
            seg_start = np.random.choice(seq_len-max_len+1, 1)[0]
            seg_end = seg_start + max_len
            seq_primary = item['seq_primary'][seg_start:seg_end]

            # tokenize amino acid seq
            aa_tokens = self.tokenizer.tokenize(seq_primary)
            aa_tokens = self.tokenizer.add_special_tokens(aa_tokens)
            aa_masked_tokens, aa_labels = self._apply_bert_mask(aa_tokens)
            aa_masked_token_ids = np.array(
                self.tokenizer.convert_tokens_to_ids(aa_masked_tokens), np.int64)
            aa_masked_token_ids_multi.append(aa_masked_token_ids)
            aa_labels_multi.append(aa_labels)
            aa_input_mask = np.ones_like(aa_masked_token_ids)
            aa_seq_mask = np.ones_like(aa_masked_token_ids)
            aa_seq_mask[0] = aa_seq_mask[-1] = 0
            
            if item['ss3'] is not None:
                distance_map = np.frombuffer(item['distance_map'],dtype=np.float64).astype(np.float32).reshape(seq_len,seq_len)[seg_start:seg_end,seg_start:seg_end]
                ss3_str = item['ss3'][seg_start:seg_end]
                #ss8_str = item['ss8'][seg_start:seg_end]
                rsa2_str = item['rsa_class'][seg_start:seg_end]
                #rsa2_value = np.frombuffer(item['rsa_value'],dtype=np.float64)[seg_start:seg_end]

                # tokenize structure elements: ss,rsa
                ss3_tokens = self.tokenizer.tokenize(ss3_str)
                rsa2_tokens = self.tokenizer.tokenize(rsa2_str)
                ss3_tokens = self.tokenizer.add_placeholder_tokens(ss3_tokens, '-')
                rsa2_tokens = self.tokenizer.add_placeholder_tokens(rsa2_tokens, '-')
                ss3_ids = np.array(self.tokenizer.struct_convert_tokens_to_ids(ss3_tokens,'ss3'))
                rsa2_ids = np.array(self.tokenizer.struct_convert_tokens_to_ids(rsa2_tokens,'rsa2'))
                ss3_ids_multi.append(ss3_ids)
                rsa2_ids_multi.append(rsa2_ids)
                
                # discretize distance map
                dist_bin_mat = discretize_distogram_fast(distance_map,self.dist_first_cutoff,self.dist_last_cutoff,self.dist_num_bins,dtype=np.int8)
                ## ignore sequential neighbors
                if self.ignore_seq_neighbors:
                    dist_bin_mat = check_seq_neighbors(dist_bin_mat,value=-1)
                dist_bin_mat = add_special_dimens(dist_bin_mat,-1)
                dist_bin_mat_multi.append(dist_bin_mat)

                # define neighbors
                edge_index_arr = generate_graph(distance_map, self.neighbor_strategy, self.knn_value, self.dist_cutoff)
                
                # Positional encoding for each node (used for GNNs)
                #node_pos_encode_arr = min_max_normalize_array(np.arange(len(seq_primary))).reshape(-1, 1)  # [num_node, 1]
                # Positional encoding for each edge (used for sequentially-ordered inputs like proteins)
                #edge_pos_encode_arr = np.sin((edge_index_arr[0] - edge_index_arr[1]).astype(np.float64)).reshape(-1, 1)  # [num_edges, 1]
                #graph_data = pyg_Data(edge_index=torch.tensor(edge_index_arr,dtype=torch.long),pos=torch.tensor(node_pos_encode_arr,dtype=torch.float),edge_pos=torch.tensor(edge_pos_encode_arr,dtype=torch.float))
                #graph_data_multi.append(graph_data)
            else:
                ss3_ids_multi.append(-1*np.ones(max_len+2, dtype=np.int8))
                rsa2_ids_multi.append(-1*np.ones(max_len+2, dtype=np.int8))
                dist_bin_mat_multi.append(-1*np.ones((max_len+2,max_len+2), dtype=np.int8))
                #graph_data_multi.append(pyg_Data())
        ## generate aa_input_mask and aa_seq_mask
        aa_masked_token_ids_multi = np.array(aa_masked_token_ids_multi)
        aa_input_mask_multi = np.ones_like(aa_masked_token_ids_multi)
        aa_seq_mask_multi = np.ones_like(aa_masked_token_ids_multi)
        aa_seq_mask_multi[:,0] = 0
        aa_seq_mask_multi[:,-1] = 0

        ## sequence reweighting
        if 'seq_reweight' in item:
            if self.multi_copy_num > 1:
                reweight_nor = [float(item['seq_reweight'])]*self.multi_copy_num
            else:
                reweight_nor = float(item['seq_reweight'])
        else:
            if self.multi_copy_num > 1:
                reweight_nor = [1.0]*self.multi_copy_num
            else:
                reweight_nor = 1.0
        return aa_masked_token_ids_multi, aa_input_mask_multi, aa_seq_mask_multi, aa_labels_multi, ss3_ids_multi, rsa2_ids_multi, dist_bin_mat_multi

    
    def __getitem_weight__(self, index):
        item = self.data[index]
        if 'seq_reweight' in item:
            reweight_nor = float(item['seq_reweight'])
        else:
            reweight_nor = 1.0
        return reweight_nor

    def __getitem_seqLen__(self, index):
        item = self.data[index]
        seq_len = int(item['seq_len'])
        if seq_len > self.max_len:
            max_len = self.max_len
        elif seq_len <= self.max_len and self.multi_copy_num == 1:
            max_len = seq_len
        elif seq_len <= self.max_len and self.multi_copy_num > 1:
            max_len = min(self.max_len, round(seq_len * 0.9))
        else:
            Exception(f'invalid max_len {self.max_len} or multi_copy_num {self.multi_copy_num}')
        return max_len+2 # two special token [CLS], [SEP]  

    def collate_fn(self, batch: List[Any]) -> Dict[str, Union[torch.Tensor,tuple]]:
        aa_masked_token_ids, aa_input_mask, aa_seq_mask, aa_labels_ids, ss3_label_ids, rsa2_label_ids, dist_bin_mat = tuple(zip(*batch))
        
        aa_masked_token_ids = list(chain(*aa_masked_token_ids))
        aa_input_mask = list(chain(*aa_input_mask))
        aa_seq_mask = list(chain(*aa_seq_mask))
        aa_labels_ids = list(chain(*aa_labels_ids))
        ss3_label_ids = list(chain(*ss3_label_ids))
        rsa2_label_ids = list(chain(*rsa2_label_ids))
        dist_bin_mat = list(chain(*dist_bin_mat))
        #graph_data_tuple = list(chain(*graph_data_tuple))

        aa_masked_token_ids = torch.from_numpy(pad_sequences(aa_masked_token_ids, 0))
        aa_input_mask = torch.from_numpy(pad_sequences(aa_input_mask, 0))
        aa_seq_mask = torch.from_numpy(pad_sequences(aa_seq_mask, 0))
        aa_labels_ids = torch.from_numpy(pad_sequences(aa_labels_ids, -1, dtype=np.int64))
        ss3_label_ids = torch.from_numpy(pad_sequences(ss3_label_ids, -1, dtype=np.int64))
        rsa2_label_ids = torch.from_numpy(pad_sequences(rsa2_label_ids, -1, dtype=np.int64))
        dist_bin_mat = torch.from_numpy(pad_distogram(dist_bin_mat, -1, dtype=np.int64))
        # pyg Data/Batch's to() function is not applicable to apex
        # may need to define a custom pyg Data/Batch class to overwrite original to()
        #graph_batch = pyg_Batch.from_data_list(graph_data_tuple)

        return {'input_seq_ids': aa_masked_token_ids,
                'input_seq_mask': aa_input_mask,
                'aa_seq_mask': aa_seq_mask,
                #'graph_batch': graph_batch,
                'targets_seq':aa_labels_ids,
                'targets_ss':ss3_label_ids,
                'targets_rsa':rsa2_label_ids,
                'targets_dist':dist_bin_mat}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                continue

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)

                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token(not special tokens)
                    token = self.tokenizer.convert_id_to_token(
                        random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                else:
                    # 10% chance to keep current token
                    pass

                masked_tokens[i] = token

        return masked_tokens, labels

@registry.register_task('structure_awareness_1d')
@registry.register_task('structure_awareness_2d')
class StructureAwarenessEvalDataset(Dataset):
    """Creates the dataset for structure awareness evaluation tasks
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        super().__init__()

        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        data_path = Path(data_path)
        data_file = f"{split}.{file_format}"
        self.data = dataset_factory(data_path / data_file, in_memory)
        self.model_config = kwargs.get('model_config')
        self.label_type = self.model_config.label_type
        self.eval_phase = self.model_config.eval_phase
        self.multi_copy_num = self.model_config.multi_copy_num

        if self.label_type in ['ss','rsa']:
            self.max_len = 400
        elif self.label_type == 'distMap':
            self.max_len = 256
            self.dist_first_cutoff = 2.3125
            self.dist_last_cutoff = 21.6875
            self.dist_num_bins = self.model_config.class_size
        else:
            Exception(f"label_type is not correct: {self.label_type}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # for i-th example, add start/end token, mask seq, convert to ids, make input_mask
        item = self.data[index]
        if self.multi_copy_num == 1:
            aa_token_ids, input_mask, targets = self._single_copy_process(item)
            seq_ids = item['seq_id']
        elif self.multi_copy_num > 1:
            aa_token_ids, input_mask, targets = self._multi_copy_process(item)
            seq_ids = [item['seq_id']]*self.multi_copy_num
        else:
            raise Exception(f"invalid copy number: {self.multi_copy_num}")

        return aa_token_ids, input_mask, targets, seq_ids

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        #input_ids, input_mask, lm_label_ids, clan, family, reweight_nor= tuple(zip(*batch))
        input_ids, input_mask, label_targets, seq_ids = tuple(zip(*batch))
        
        if self.multi_copy_num > 1:
            input_ids = list(chain(*input_ids))
            input_mask = list(chain(*input_mask))
            label_targets = list(chain(*label_targets))
            seq_ids = list(chain(*seq_ids))
        else:
            pass

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        # ignore_index is -1
        if self.label_type in ['ss','rsa']:
            label_targets = torch.from_numpy(pad_sequences(label_targets, -1))
        elif self.label_type == 'distMap': 
            label_targets = torch.from_numpy(pad_distogram(label_targets, -1))

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': label_targets,
                'set_nm': seq_ids}

    def _single_copy_process(self, item: Dict[str, Any]):
        """process item data, single copy version    
        """
        seq_len = int(item['seq_len'])
        if seq_len > self.max_len and not self.eval_phase:
            seg_start = np.random.choice(seq_len-self.max_len+1, 1)[0]
            seg_end = seg_start + self.max_len
        else:
            seg_start = 0
            seg_end = seq_len

        aa_str = item['seq_primary'][seg_start:seg_end]
        aa_tokens = self.tokenizer.tokenize(aa_str)
        aa_tokens = self.tokenizer.add_special_tokens(aa_tokens)
        aa_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(aa_tokens), np.int64)
        input_mask = np.ones_like(aa_token_ids)
        
        # process structure labels
        if self.label_type == 'ss':
            ss3_str = item['ss3'][seg_start:seg_end]
            ss3_tokens = self.tokenizer.tokenize(ss3_str)
            ss3_tokens = self.tokenizer.add_placeholder_tokens(ss3_tokens, '-')
            targets = np.array(self.tokenizer.struct_convert_tokens_to_ids(ss3_tokens,'ss3'))
        elif self.label_type == 'rsa':
            rsa2_str = item['rsa_class'][seg_start:seg_end]
            rsa2_tokens = self.tokenizer.tokenize(rsa2_str)
            rsa2_tokens = self.tokenizer.add_placeholder_tokens(rsa2_tokens, '-')
            targets = np.array(self.tokenizer.struct_convert_tokens_to_ids(rsa2_tokens,'rsa2'))
        elif self.label_type == 'distMap':
            distance_map = np.frombuffer(item['distance_map'],dtype=np.float64).reshape(seq_len,seq_len)[seg_start:seg_end,seg_start:seg_end]
            dist_bin_mat = discretize_distogram_fast(distance_map,self.dist_first_cutoff,self.dist_last_cutoff,self.dist_num_bins)
            targets = add_special_dimens(dist_bin_mat,-1)
        else:
            raise Exception(f"invalid label type: {self.label_type}")

        return aa_token_ids, input_mask, targets

    def _multi_copy_process(self, item: Dict[str,Any]):
        """process item data, multi copy version. window-cropping dy default  
        """
        seq_len = int(item['seq_len'])
        max_len = min(self.max_len, round(seq_len * 0.9))
        aa_token_ids_multi, targets_multi = [], []
        for copy_i in range(self.multi_copy_num):
            seg_start = np.random.choice(seq_len-max_len+1, 1)[0]
            seg_end = seg_start + max_len
            aa_str = item['seq_primary'][seg_start:seg_end]
            aa_tokens = self.tokenizer.tokenize(aa_str)
            aa_tokens = self.tokenizer.add_special_tokens(aa_tokens)
            aa_token_ids = np.array(
                self.tokenizer.convert_tokens_to_ids(aa_tokens), np.int64)
            aa_token_ids_multi.append(aa_token_ids)

            # process structure labels
            if self.label_type == 'ss':
                ss3_str = item['ss3'][seg_start:seg_end]
                ss3_tokens = self.tokenizer.tokenize(ss3_str)
                ss3_tokens = self.tokenizer.add_placeholder_tokens(ss3_tokens, '-')
                targets = np.array(self.tokenizer.struct_convert_tokens_to_ids(ss3_tokens,'ss3'))
                targets_multi.append(targets)
            elif self.label_type == 'rsa':
                rsa2_str = item['rsa_class'][seg_start:seg_end]
                rsa2_tokens = self.tokenizer.tokenize(rsa2_str)
                rsa2_tokens = self.tokenizer.add_placeholder_tokens(rsa2_tokens, '-')
                targets = np.array(self.tokenizer.struct_convert_tokens_to_ids(rsa2_tokens,'rsa2'))
                targets_multi.append(targets)
            elif self.label_type == 'distMap':
                distance_map = np.frombuffer(item['distance_map'],dtype=np.float64).reshape(seq_len,seq_len)[seg_start:seg_end,seg_start:seg_end]
                dist_bin_mat = discretize_distogram_fast(distance_map,self.dist_first_cutoff,self.dist_last_cutoff,self.dist_num_bins)
                targets = add_special_dimens(dist_bin_mat,-1)
                targets_multi.append(targets)
            else:
                raise Exception(f"invalid label type: {self.label_type}")
        
        aa_token_ids_multi = np.array(aa_token_ids_multi)
        targets_multi = np.array(targets_multi)
        input_mask_multi = np.ones_like(aa_token_ids_multi)

        return aa_token_ids_multi, input_mask_multi, targets_multi
