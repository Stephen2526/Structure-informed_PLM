"""Utility functions to help setup the model, optimizer, distributed compute, etc.
"""
from email.policy import default
import typing
import logging
from pathlib import Path
import sys, os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from optimization import AdamW

from mapping import registry

from .base_utils import get_effective_batch_size
from ._sampler import BucketBatchSampler

logger = logging.getLogger(__name__)


def setup_logging(is_local_master: bool,
                  is_global_master: bool,
                  save_path: typing.Optional[Path] = None,
                  log_level: typing.Union[str, int] = None) -> None:
    if log_level is None:
        level = logging.INFO
    elif isinstance(log_level, str):
        level = getattr(logging, log_level.upper())
    elif isinstance(log_level, int):
        level = log_level

    if not is_global_master:
        level = max(level, logging.WARN)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)


    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S")

    if not root_logger.hasHandlers():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        if save_path is not None:
            file_handler = logging.FileHandler(save_path / 'log')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)


def setup_optimizer(model,
                    learning_rate: float):
    """Create the AdamW optimizer for the given model with the specified learning rate. Based on
    creation in the pytorch_transformers repository.

    Args:
        model (PreTrainedModel): The model for which to create an optimizer
        learning_rate (float): Default learning rate to use when creating the optimizer

    Returns:
        optimizer (AdamW): An AdamW optimizer

    """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


def setup_dataset(task: str,
                  data_dir: typing.Union[str, Path],
                  split: str,
                  tokenizer: str,
                  data_format: str,
                  in_memory: bool=False,
                  **kwarg) -> Dataset:
    task_spec = registry.get_task_spec(task)
    #mlm_mask_stragy = kwarg.get('mlm_mask_stragy')
    #mutgsis_set = kwarg.get('mutgsis_set')
    #alphabet_obj = kwarg.get('alphabet_obj')
    #in_memory = False if kwarg.get('in_memory') is None else True
    return task_spec.dataset(data_path=data_dir, 
                             split=split,
                             tokenizer=tokenizer,
                             in_memory=in_memory,
                             file_format=data_format,
                             **kwarg)



def setup_loader(dataset: Dataset,
                 batch_size: int,
                 local_rank: int,
                 n_gpu: int,
                 gradient_accumulation_steps: int,
                 num_workers: int,
                 **kwarg) -> DataLoader:
    balancing = kwarg.get('balancing')
    sampler = DistributedSampler(dataset) if local_rank != -1 else RandomSampler(dataset)
    batch_size = get_effective_batch_size(
        batch_size, local_rank, n_gpu, gradient_accumulation_steps) * n_gpu
    # WARNING: this will fail if the primary sequence is not the first thing the dataset returns
    # x[0] is intended to get first returned variable of func __getitem__(self, index) = x
    batch_sampler = BucketBatchSampler(
        sampler, batch_size, False, lambda x: len(x[0]), dataset, balancing)
    
    if 'collate_fn' in kwarg.keys():
      collate_fn_cusm = kwarg['collate_fn']
    else:
      collate_fn_cusm = dataset.collate_fn

    loader = DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=collate_fn_cusm,  # type: ignore
        batch_sampler=batch_sampler)

    return loader


def setup_distributed(local_rank: int,
                      no_cuda: bool) -> typing.Tuple[torch.device, int, bool]:
    if local_rank != -1 and not no_cuda:
        torch.cuda.set_device(local_rank)
        device: torch.device = torch.device("cuda", local_rank)
        n_gpu = 1
        dist.init_process_group(backend="nccl")
    elif not torch.cuda.is_available() or no_cuda:
        device = torch.device("cpu")
        n_gpu = 1
    else:
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()

    is_master = local_rank in (-1, 0)
    is_global_master = os.environ.get('RANK')
    if is_global_master:
        is_global_master = int(os.environ["RANK"]) in (0,)
    else:
        is_global_master = True

    return device, n_gpu, is_master, is_global_master
