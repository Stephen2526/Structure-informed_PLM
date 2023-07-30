"""Implementation of a bucketed data sampler from PyTorch-NLP.
Modified by Roshan Rao.

See https://github.com/PetrochukM/PyTorch-NLP/
"""
import typing
import math
import operator
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SubsetRandomSampler


class SortedSampler(Sampler):
    """ Samples elements sequentially, always in the same order.
    Args:
        data (iterable): Iterable data.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.
    Example:
        >>> list(SortedSampler(range(10), sort_key=lambda i: -i))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    """

    def __init__(self,
                 dataset,
                 sort_key: typing.Callable[[int], typing.Any],
                 indices: typing.Optional[typing.Iterable[int]] = None):
        super().__init__(dataset)
        self.dataset = dataset
        self.sort_key = sort_key
        if indices is None:
            new_sort_key = getattr(dataset, "__getitem_seqLen__", None)
            if callable(new_sort_key):
                sort_keys = ((i, new_sort_key(i)) for i in range(len(dataset)))
            else:
                sort_keys = map(sort_key, dataset)
        else:
            new_sort_key = getattr(dataset, "__getitem_seqLen__", None)
            if callable(new_sort_key):
                sort_keys = ((i, new_sort_key(i)) for i in indices)
            else:
                ## dataset[i] will call __getitem__(self, i) in dataset class
                sort_keys = ((i, sort_key(dataset[i])) for i in indices) 
            
        self.sorted_indices = [i for i, _ in sorted(sort_keys, key=operator.itemgetter(1))]

    def __iter__(self):
        return iter(self.sorted_indices)

    def __len__(self):
        return len(self.dataset)

class MyWeightedRandomSampler(Sampler):
    """ customized WeightedRaondomSampler, sample with repalcement considering weight balancing
        sample weights are collected in runtime.
        
    Convention:
        dataset[i][-1] - normalized seq reweight score (seq_weight/family_weight)

    Args:
        dataset (iterable): Iterable data.
        indices (iterable): pool of indices to sample
    """

    def __init__(self,
                 dataset,
                 indices: typing.Optional[typing.Iterable[int]] = None):
        super().__init__(dataset)
        self.dataset = dataset
        self.indices = indices
        ## collect sample weights
        if indices is None:
            get_weight = getattr(dataset, "__getitem_weight__", None)
            if callable(get_weight):
                weight_list = [get_weight(i) for i in range(len(dataset))]
            else:
                weight_list = [dataset[i][-1] for i in range(len(dataset))]
            weight_list = np.asarray(weight_list).reshape(-1).astype('float64')
            weight_list = weight_list / np.sum(weight_list)
            self.weighted_indices = np.random.choice(len(dataset),size=len(dataset),p=weight_list).tolist()
        else:
            get_weight = getattr(dataset, "__getitem_weight__", None)
            if callable(get_weight):
                weight_list = [get_weight(i) for i in indices] # applicable for multi-copy case
            else:
                weight_list = [dataset[i][-1] for i in indices]
            ## normalize weights within bucket
            weight_list = np.asarray(weight_list).reshape(-1).astype('float64')
            weight_list = weight_list / np.sum(weight_list)
            self.weighted_indices = np.random.choice(indices,size=len(indices),p=weight_list).tolist()

    def __iter__(self):
        return iter(self.weighted_indices)

    def __len__(self):
        if self.indices is None:
            return len(self.dataset) 
        else:
            return len(self.indices) 


class BucketBatchSampler(BatchSampler):
    """ `BucketBatchSampler` toggles between `sampler` batches and sorted batches.
    Typically, the `sampler` will be a `RandomSampler` allowing the user to toggle between
    random batches and sorted batches. A larger `bucket_size_multiplier` is more sorted
    and vice versa. Provides ~10-25 percent speedup.

    Background:
        ``BucketBatchSampler`` is similar to a ``BucketIterator`` found in popular
        libraries like ``AllenNLP`` and ``torchtext``. A ``BucketIterator`` pools together
        examples with a similar size length to reduce the padding required for each batch
        while maintaining some noise through bucketing.

    Args:
        sampler (torch.data.utils.sampler.Sampler):
        batch_size (int): Size of mini-batch.
        drop_last (bool): If `True` the sampler will drop the last batch if its size
            would be less than `batch_size`.
        sort_key (callable, optional): Callable to specify a comparison key for sorting.
        bucket_size_multiplier (int, optional): Buckets are of size
            `batch_size * bucket_size_multiplier`.
    Example:
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(10)))
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=False))
        [[6, 7, 8], [0, 1, 2], [3, 4, 5], [9]]
        >>> list(BucketBatchSampler(sampler, batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self,
                 sampler,
                 batch_size,
                 drop_last,
                 sort_key,
                 dataset,
                 balancing: bool,
                 bucket_size_multiplier=100):
        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        self.dataset = dataset
        self.balancing = balancing
        ## each bucket contains (batch_size*bucket_size_multiplier) sequences
        self.bucket_sampler = BatchSampler(
            sampler, min(batch_size * bucket_size_multiplier, len(sampler)), False)

    def __iter__(self):
        ## loop through each bucket (batch_size*bucket_size_multiplier)
        for bucket in self.bucket_sampler:
            if self.balancing: ## weighted sampling with replacement within bucket
              bucket = MyWeightedRandomSampler(self.dataset, indices=bucket)
            ## sort samples in each bucket by seq length (the output of __getitem__(self, index))
            ## that sequences with similar length as likely to be in one batch
            sorted_sampler = SortedSampler(self.dataset, self.sort_key, indices=bucket)
            ## split samples in one bucket into batches, then randomize order of batches in one bucket
            for batch in SubsetRandomSampler(
                    list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))):
                yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)

