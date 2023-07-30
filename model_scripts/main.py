import typing
import os,sys
import logging
import argparse
import warnings
import inspect


try:
    import apex  # noqa: F401
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False

import datasets, metrics, training
from models import modeling_pt_bert, modeling_pt_modality
import utils
from mapping import registry

CallbackList = typing.Sequence[typing.Callable]
OutputDict = typing.Dict[str, typing.List[typing.Any]]


logger = logging.getLogger(__name__)
warnings.filterwarnings(  # Ignore pytorch warning about loss gathering
    'ignore', message='Was asked to gather along dimension 0', module='torch.nn.parallel')


def create_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Parent parser for tape functions',
                                     add_help=False)
    parser.add_argument('run_type', choices=['run_train', 'run_eval','run_embed','run_train_distributed'], 
                        default='run_train', help='which function to run')
    parser.add_argument('model_type', help='Base model class to run')
    parser.add_argument('--model_config_file', default=None, type=utils.check_is_file,
                        help='Config file for model')
    parser.add_argument('--extra_config_file', default=None, type=utils.check_is_file,
                        help='Extrac config file for model besides pretrain config')
    parser.add_argument('--vocab_file', default=None,
                        help='Pretrained tokenizer vocab file')
    parser.add_argument('--output_dir', default='./results', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='CPU-only flag')
    parser.add_argument('--seed', default=42, type=int, help='Random seed to use')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank of process in distributed training. '
                             'Set by launch script.')
    parser.add_argument('--tokenizer', choices=['iupac', 'unirep', 'pfam'],
                        default='pfam', help='Tokenizes to use on the amino acid sequences')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers to use for multi-threaded data loading')
    parser.add_argument('--data_format', choices=['json','lmdb'],default='lmdb', help='file format of data')
    parser.add_argument('--log_level', default=logging.INFO,
                        choices=['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR',
                                 logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR],
                        help="log level for the experiment")
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    return parser


def create_train_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run Training on the TAPE datasets',
                                     parents=[base_parser])
    parser.add_argument('task', choices=list(registry.task_name_mapping.keys()),
                        help='TAPE Task to train/eval on')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Batch size')
    parser.add_argument('--data_dir', default='/scratch/user/sunyuanfei/Projects/ProteinMutEffBenchmark/data_process/pfam_32.0/seq_json_rp75', type=utils.check_is_dir,
                        help='Directory from which to load task data')
    parser.add_argument('--train_split', default='train', type=str,
                        help='split/subset name of training data')
    parser.add_argument('--valid_split', default='valid', type=str,
                        help='split/subset name of validation data')
    parser.add_argument('--num_train_epochs', default=10, type=int,
                        help='Number of training epochs in current cycle')
    parser.add_argument('--num_train_total_epochs', default=100, type=int,
                        help='Number of total training epochs')
    parser.add_argument('--num_log_iter', default=20, type=int,
                        help='Number of training steps per log iteration')
    parser.add_argument('--fp16', action='store_true', help='Whether to use fp16 weights')
    parser.add_argument('--fp16_opt_level', default='O1', type=str,
                        help='optimization level for apex')
    parser.add_argument('--warmup_steps', default=10000, type=int,
                        help='Number of learning rate warmup steps')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='Number of forward passes to make for each backwards pass')
    parser.add_argument('--loss_scale', default=0, type=int,
                        help='Loss scaling. Only used during fp16 training.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Maximum gradient norm')
    parser.add_argument('--exp_name', default=None, type=str,
                        help='Name to give to this experiment')
    parser.add_argument('--from_pretrained', default=None, type=str,
                        help='Directory containing config and pretrained model weights')
    parser.add_argument('--pretrained_epoch', default=None, type=utils.int_or_str,
                        help='the epoch number of pretrained model to load')
    parser.add_argument('--log_dir', default='./logs', type=str)
    parser.add_argument('--pytorch_profiler', action='store_true', help='Whether to record profiler')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help="Frequency of eval pass. A value <= 0 means the eval pass is "
                             "not run")
    parser.add_argument('--save_freq', default=1, type=utils.int_or_str,
                        help="How often to save the model during training. Either an integer "
                             "frequency or the string 'improvement'")
    parser.add_argument('--save_freq_opt_checkpoint', default='improvement', type=utils.int_or_str,
                        help="How often to save the optimization checkpoint during training. Either an integer "
                             "frequency or the string 'improvement'")
    parser.add_argument('--patience', default=-1, type=int,
                        help="How many epochs without improvement to wait before ending "
                             "training")
    parser.add_argument('--resume_from_checkpoint', action='store_true',
                        help="whether to resume training from the checkpoint")
    parser.add_argument('--mlm_mask_stragy', default='vanilla', type=str,
                        help='specify masking strategy for MLM (e.g. antibody: cdr_vanilla; cdr_one)')
    parser.add_argument('--lr_scheduler', default='constant', type=str,
                        help='learning rate scheduler')
    parser.add_argument('--balancing', action='store_true', help='whether use sequence weight balancing when training')
    parser.add_argument('--save_checkpoint', action='store_true', help='whether saving checkpoint when training')
    parser.add_argument('--neighbor_strategy', default='knn', choices=['full','knnDistCut','distCut','knn','random','sequential','noGS'], help='strategy to define neighbors in graph, used for seq_structure_multi_task dataset')
    parser.add_argument('--knn_value', default=20, type=int,
                        help='num of k nearest neighbors in graph, used for seq_structure_multi_task dataset')
    parser.add_argument('--dist_cutoff', default=12.0, type=float,
                        help='max ditance in angstrom to define neighbors in graph, used for seq_structure_multi_task dataset')
    return parser


def create_eval_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run Eval on the TAPE Datasets',
                                     parents=[base_parser])
    parser.add_argument('task', choices=list(registry.task_name_mapping.keys()),
                        help='TAPE Task to train/eval on')
    parser.add_argument('from_pretrained', type=str,
                        help='Directory containing config and pretrained model weights')
    parser.add_argument('--pretrained_epoch', default=None, type=utils.int_or_str,
                        help='the epoch number of pretrained model to load')
    parser.add_argument('--mutgsis_set', default=None, type=str, help='mutagenesis set name to evaluate')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Batch size')
    parser.add_argument('--data_dir', default='./data', type=utils.check_is_dir,
                        help='Directory from which to load task data')
    parser.add_argument('--eval_save_dir', default='./eval_results', type=utils.check_is_dir,
                        help='Directory to save evaluation results')
    parser.add_argument('--metrics', default=[],
                        help=f'Metrics to run on the result. '
                             f'Choices: {list(registry.metric_name_mapping.keys())}',
                        nargs='*')
    parser.add_argument('--split', default='test', type=str,
                        help='Which split to run on')
    parser.add_argument('--model_name', default='esm', type=str,
                        help='name of esm model to load')
    parser.add_argument('--repr_layers', default=[1],
                        help='layer index list to collect hidden representation from esm',                             
                        nargs='*',type=int)
    parser.add_argument('--output_pred', action='store_true', help='Whether to save outputs')
    parser.add_argument('--mlm_mask_stragy', default='vanilla', type=str,
                        help='specify masking strategy for MLM (e.g. antibody: cdr_vanilla; cdr_one)')   
    parser.add_argument('--embed_modelNm', default=None, type=str,
                        help='model name for embedding (bert_1_rp75,bert_[1,2,3,4]_rp15)')
    parser.add_argument('--neighbor_strategy', default='knn', choices=['full','knnDistCut','distCut','knn','random','sequential','noGS'], help='strategy to define neighbors in graph, used for seq_structure_multi_task dataset')
    parser.add_argument('--knn_value', default=20, type=int,
                        help='num of k nearest neighbors in graph, used for seq_structure_multi_task dataset')
    parser.add_argument('--dist_cutoff', default=12.0, type=float,
                        help='max ditance in angstrom to define neighbors in graph, used for seq_structure_multi_task dataset')
    return parser


def create_embed_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Embed a set of proteins wiht a pretrained model',
        parents=[base_parser])
    parser.add_argument('data_file', type=str,
                        help='File containing set of proteins to embed')
    parser.add_argument('out_file', type=str,
                        help='Name of output file')
    parser.add_argument('from_pretrained', type=str,
                        help='Directory containing config and pretrained model weights')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Batch size')
    parser.add_argument('--full_sequence_embed', action='store_true',
                        help='If true, saves an embedding at every amino acid position '
                             'in the sequence. Note that this can take a large amount '
                             'of disk space.')
    parser.set_defaults(task='embed')
    return parser


def create_distributed_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, parents=[base_parser])
    # typing.Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=2,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=47493, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")
    return parser


def run_train(args: typing.Optional[argparse.Namespace] = None, env=None) -> None:
    if env is not None:
        os.environ = env

    if args is None:
        base_parser = create_base_parser()
        train_parser = create_train_parser(base_parser)
        args = train_parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: "
            f"{args.gradient_accumulation_steps}, should be >= 1")

    if (args.fp16 or args.local_rank != -1) and not APEX_FOUND:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex "
            "to use distributed and fp16 training.")

    arg_dict = vars(args)
    arg_names = inspect.getfullargspec(training.run_train).args

    missing = set(arg_names) - set(arg_dict.keys())
    if missing:
        raise RuntimeError(f"Missing arguments: {missing}")
    train_args = {name: arg_dict[name] for name in arg_names}
    #print('_>main/run_train, train_args:{}'.format(train_args))
    training.run_train(**train_args)


def run_eval(args: typing.Optional[argparse.Namespace] = None) -> typing.Dict[str, float]:
    if args is None:
        base_parser = create_base_parser()
        parser = create_eval_parser(base_parser)
        args = parser.parse_args()

    if args.from_pretrained is None:
        raise ValueError("Must specify pretrained model")
    if args.local_rank != -1:
        raise ValueError("Not support distributed validation pass")

    arg_dict = vars(args)
    if args.task == 'esm_eval':
        arg_names = inspect.getfullargspec(training.run_eval_esm).args
    else:
        arg_names = inspect.getfullargspec(training.run_eval).args


    missing = set(arg_names) - set(arg_dict.keys())
    if missing:
        raise RuntimeError(f"Missing arguments: {missing}")
    eval_args = {name: arg_dict[name] for name in arg_names}
    print('_>main/run_eval, eval_args:{}'.format(eval_args))
    
    if args.task == 'esm_eval':
        return training.run_eval_esm(**eval_args)
    else:
        return training.run_eval(**eval_args)


def run_embed(args: typing.Optional[argparse.Namespace] = None) -> None:
    if args is None:
        base_parser = create_base_parser()
        parser = create_embed_parser(base_parser)
        args = parser.parse_args()
    if args.from_pretrained is None:
        raise ValueError("Must specify pretrained model")
    if args.local_rank != -1:
        raise ValueError("TAPE does not support distributed validation pass")

    arg_dict = vars(args)
    arg_names = inspect.getfullargspec(training.run_embed).args

    missing = set(arg_names) - set(arg_dict.keys())
    if missing:
        raise RuntimeError(f"Missing arguments: {missing}")
    embed_args = {name: arg_dict[name] for name in arg_names}
    print('_>main/run_embed, embed_args:{}'.format(embed_args))

    training.run_embed(**embed_args)


def run_train_distributed(args: typing.Optional[argparse.Namespace] = None) -> None:
    """Runs distributed training via multiprocessing.
    """
    if args is None:
        base_parser = create_base_parser()
        distributed_parser = create_distributed_parser(base_parser)
        distributed_train_parser = create_train_parser(distributed_parser)
        args = distributed_train_parser.parse_args()

    # Define the experiment name here, instead of dealing with barriers and communication
    # when getting the experiment name
    exp_name = utils.get_expname(args.exp_name, args.task, args.model_type)
    args.exp_name = exp_name
    print('_>main/run_train_distributed, args:{}'.format(args))
    utils.launch_process_group(
        run_train, args, args.nproc_per_node, args.nnodes,
        args.node_rank, args.master_addr, args.master_port)


if __name__ == '__main__':
    # debug
    #print('task:{}'.format(list(registry.task_name_mapping.keys())))
    #for task_key,task_val in registry.task_name_mapping.items():
      #print('>>task:{} - models:{}'.format(task_key,list(task_val.models.keys())))
    #print('metric:{}'.format(list(registry.metric_name_mapping.keys())))
    run_type = sys.argv[1]
    print('>>>run_type: {}'.format(run_type))
    if run_type == 'run_train_distributed':
      run_train_distributed()
    elif run_type == 'run_train':
      run_train()
    elif run_type == 'run_embed':
      run_embed()
    elif run_type == 'run_eval':
      run_eval()
    else:
      print('wrong cmd to run')
