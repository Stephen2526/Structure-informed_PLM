import typing
import os,sys,re
import logging
from timeit import default_timer as timer
import json
from pathlib import Path
import inspect
import pickle as pkl
import numpy as np
import scipy.stats

from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.optim as optim
from torch.utils.data import DataLoader
from optimization import ConstantLRSchedule, WarmupConstantSchedule, WarmupLinearSchedule, WarmupCosineSchedule, WarmupCosineWithHardRestartsSchedule
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torch.profiler

import utils
import errors
import visualization
from models.modeling_utils import BaseModel, BaseConfig
from mapping import registry
from tokenizers import BaseTokenizer

try:
    from apex import amp
    import amp_C
    import apex_C
    from apex.amp import _amp_state
    from apex.parallel.distributed import flat_dist_call
    from apex.parallel.distributed import DistributedDataParallel as DDP
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False

logger = logging.getLogger(__name__)

MetricsDict = typing.Dict[str, float]
LossAndMetrics = typing.Tuple[float, MetricsDict]
OutputDict = typing.Dict[str, typing.Any]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        else:
            return json.JSONEncoder.default(self, obj)

class ForwardRunner:

    def __init__(self,
                 model: BaseModel,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1,
                 fp16: bool = False,
                 local_rank: int = -1):

        self.model = model
        self.device = device
        self.n_gpu = n_gpu
        self.fp16 = fp16
        self.local_rank = local_rank

        forward_arg_keys = inspect.getfullargspec(model.forward).args
        forward_arg_keys = forward_arg_keys[1:]  # remove self argument
        self._forward_arg_keys = forward_arg_keys
        #assert 'input_ids' in self._forward_arg_keys

    def initialize_distributed_model(self):
        if self.local_rank != -1:
            if not self.fp16:
                self.model = DDP(self.model)
            else:
                flat_dist_call([param.data for param in self.model.parameters()],
                               torch.distributed.broadcast, (0,))
        elif self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

    def forward(self,
                batch: typing.Dict[str, torch.Tensor],
                return_outputs: bool = False,
                no_loss: bool = False):
        # Filter out batch items that aren't used in this model
        # Requires that dataset keys match the forward args of the model
        # Useful if some elements of the data are only used by certain models
        # e.g. PSSMs / MSAs and other evolutionary data
        batch = {name: tensor for name, tensor in batch.items()
                 if name in self._forward_arg_keys}
        if self.device.type == 'cuda':
            # for name, val in batch.items():
            #     if isinstance(val,torch.Tensor):
            #         batch[name] = val.cuda(device=self.device, non_blocking=True)
            #     elif isinstance(val, typing.List):
            #         batch[name] = [tensor.cuda(device=self.device, non_blocking=True) for tensor in val]

            batch = {name: tensor.cuda(device=self.device, non_blocking=True)
                     for name, tensor in batch.items()}
        
        outputs = self.model(**batch)

        if no_loss:
            return outputs

        if isinstance(outputs[0], tuple) and len(outputs[0]) == 2:
            # model also returned metrics
            loss, metrics = outputs[0]
        else:
            # no metrics and loss
            loss = None
            metrics = {}

        if self.n_gpu > 1 and loss is not None :  # pytorch DataDistributed doesn't mean scalars
            loss = loss.mean()
            metrics = {name: metric.mean() for name, metric in metrics.items()}

        if return_outputs:
            return loss, metrics, outputs
        else:
            return loss, metrics

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self


class BackwardRunner(ForwardRunner):

    def __init__(self,
                 model: BaseModel,
                 optimizer: optim.Optimizer,  # type: ignore
                 gradient_accumulation_steps: int = 1,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1,
                 fp16: bool = False,
                 fp16_opt_level: str = 'O1',
                 local_rank: int = -1,
                 max_grad_norm: float = 1.0,
                 warmup_steps: int = 0,
                 num_train_optimization_steps: int = 1000000,
                 lr_scheduler: str = 'constant',
                 num_epoch: int = 50):

        super().__init__(model, device, n_gpu, fp16, local_rank)
        self.fp16_opt_level = fp16_opt_level
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self._global_step = 0
        self._local_rank = local_rank
        self._overflow_buf = torch.cuda.IntTensor([0])  # type: ignore
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._delay_accumulation = fp16 and local_rank != -1
        if lr_scheduler == 'constant':
            self.scheduler = ConstantLRSchedule(self.optimizer)
        elif lr_scheduler == 'warmupConstant':
            self.scheduler = WarmupConstantSchedule(
                self.optimizer, warmup_steps)
        elif lr_scheduler == 'warmupLinear':
            self.scheduler = WarmupLinearSchedule(
                self.optimizer, warmup_steps, num_train_optimization_steps)
        elif lr_scheduler == 'warmupCosine':
            self.scheduler = WarmupCosineSchedule(
                self.optimizer, warmup_steps, num_train_optimization_steps, cycles=.5)
        elif lr_scheduler == 'CosineAnnealingWarmRestarts':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, num_train_optimization_steps // (num_epoch*4), T_mult=2, eta_min=1e-8, verbose=False)

    def initialize_fp16(self):
        if self.fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.fp16_opt_level, loss_scale="dynamic") # master_weights=True
            _amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    def resume_from_checkpoint(self, checkpoint_dir: str, checkpoint_epoch: int) -> int:
        if checkpoint_epoch is None:
            logger.info("loading checkpoint from {}checkpoint.bin".format(checkpoint_dir))
            checkpoint = torch.load(
                os.path.join(checkpoint_dir, 'checkpoint.bin'), map_location=self.device)
        else:
            logger.info("loading checkpoint from {}checkpoint_{}.bin".format(checkpoint_dir,checkpoint_epoch))
            checkpoint = torch.load(
                os.path.join(checkpoint_dir, 'checkpoint_{}.bin'.format(checkpoint_epoch)), map_location=self.device)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.fp16:
            self.optimizer._lazy_init_maybe_master_weights()
            self.optimizer._amp_stash.lazy_init_called = True
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved in zip(
                    amp.master_params(self.optimizer), checkpoint['master params']):
                param.data.copy_(saved.data)
            amp.load_state_dict(checkpoint['amp'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        if 'global_step' in checkpoint.keys():
            start_gloStep = checkpoint['global_step'] + 1
        else:
            start_gloStep = 0
        return start_epoch, start_gloStep

    def save_state(self, 
                   save_directory: typing.Union[str, Path],
                   epoch_id: int,
                   save_freq: typing.Union[str, int],
                   save_freq_opt_checkpoint: typing.Union[str, int],
                   save_checkpoint: bool,
                   num_train_epochs: int,
                   num_evals_no_improvement: int):
        save_directory = Path(save_directory)
        if not save_directory.exists():
            save_directory.mkdir()
        else:
            assert save_directory.is_dir(), "Save path should be a directory"
        model_to_save = getattr(self.model, 'module', self.model)
        model_to_save.save_pretrained(save_directory, epoch_id, save_freq, num_train_epochs, num_evals_no_improvement)
        optimizer_state: typing.Dict[str, typing.Any] = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch_id,
            'global_step': self.global_step}
        if APEX_FOUND:
            optimizer_state['master params'] = list(amp.master_params(self.optimizer))
            try:
                optimizer_state['amp'] = amp.state_dict()
            except AttributeError:
                pass
        if save_checkpoint:
            if isinstance(save_freq_opt_checkpoint, int):
                if (((epoch_id + 1) % save_freq_opt_checkpoint == 0) or ((epoch_id + 1) == num_train_epochs)) and num_evals_no_improvement == 0:
                    torch.save(optimizer_state, save_directory / 'checkpoint_{}.bin'.format(epoch_id))
                    torch.save(optimizer_state, save_directory / 'checkpoint.bin')
                elif ((epoch_id + 1) % save_freq_opt_checkpoint == 0) or ((epoch_id + 1) == num_train_epochs) or (epoch_id == 0):
                    torch.save(optimizer_state, save_directory / 'checkpoint_{}.bin'.format(epoch_id))
                elif num_evals_no_improvement == 0:
                    torch.save(optimizer_state, save_directory / 'checkpoint.bin')
            else:
                torch.save(optimizer_state, save_directory / 'checkpoint.bin')


    def backward(self, loss) -> None:
        if not self._delay_accumulation:
            loss = loss / self.gradient_accumulation_steps
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer,
                                delay_overflow_check=self._delay_accumulation) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def step(self) -> None:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, norm_type=2.0, error_if_nonfinite=False)
        if self._local_rank == -1:
            self._step()
        elif not self.fp16:
            # TODO: Can you do this allreduce after accumulation also?
            self._step()
        else:
            self._step_distributed_fp16()

    def _step(self) -> None:
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()  # type: ignore
        self._global_step += 1

    def _step_distributed_fp16(self) -> None:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        scaler = _amp_state.loss_scalers[0]
        master_grads = [p.grad for p in amp.master_params(self.optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        # allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else \
            # torch.float32
        allreduce_dtype = torch.float16
        flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        self._overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            self._overflow_buf,
            [master_grads, allreduced_views],
            scaler.loss_scale() / (
                torch.distributed.get_world_size() * self.gradient_accumulation_steps))
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        self._overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            self._overflow_buf,
            [allreduced_views, master_grads],
            1. / scaler.loss_scale())
        # 5. update loss scale
        scaler = _amp_state.loss_scalers[0]
        old_overflow_buf = scaler._overflow_buf
        scaler._overflow_buf = self._overflow_buf
        had_overflow = scaler.update_scale()
        scaler._overfloat_buf = old_overflow_buf
        # 6. call optimizer step function
        if had_overflow == 0:
            self._step()
        else:
            # Overflow detected, print message and clear gradients
            logger.info(f"Gradient overflow.  Skipping step, reducing loss scale to "
                        f"{scaler.loss_scale()}")
            if _amp_state.opt_properties.master_weights:
                for param in self.optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in self.model.parameters():
            param.grad = None

    @property
    def global_step(self) -> int:
        return self._global_step
    
    @property
    def set_global_step(self, gloStep) -> None:
        self._global_step = gloStep

def run_train_epoch(epoch_id: int,
                    train_loader: DataLoader,
                    runner: BackwardRunner,
                    viz: typing.Optional[visualization.TAPEVisualizer] = None,
                    num_log_iter: int = 20,
                    gradient_accumulation_steps: int = 1,
                    log_dir: str = None,
                    pytorch_profiler: bool = False,
                    start_epoch: int = None) -> LossAndMetrics:
    if viz is None:
        viz = visualization.DummyVisualizer()
    smoothing = 1 - 1 / num_log_iter
    accumulator = utils.MetricsAccumulator(smoothing)

    torch.set_grad_enabled(True)
    runner.train()

    def make_log_str(step: int, time: float, forward_time: float=0, backward_time: float=0, data_time: float=0) -> str:
        ep_percent = epoch_id + step / len(train_loader)
        if runner.scheduler is not None:
            #curr_lr = runner.scheduler.get_lr()[0]  # type: ignore
            curr_lr = runner.scheduler.get_last_lr()[0] # type: ignore
        else:
            curr_lr = runner.optimizer.param_groups[0]['lr']

        print_str = []
        print_str.append(f"[Ep: {ep_percent:.2f}]")
        print_str.append(f"[Iter: {runner.global_step}]")
        print_str.append(f"[Time: {time:5.2f}s; F/B/D: {forward_time:.1f}/{backward_time:.1f}/{data_time:.1f}]")
        print_str.append(f"[Loss: {accumulator.loss():.5g}]")

        for name, value in accumulator.metrics().items():
            print_str.append(f"[{name.capitalize()}: {value:.5g}]")

        print_str.append(f"[LR: {curr_lr:.5g}]")
    
        ## GPU mem inspect
        curr_device_idx = torch.cuda.current_device()
        mem_divider = 1.049e+6 # byte to MiB
        ma_mib = torch.cuda.memory_allocated(curr_device_idx) // mem_divider
        max_ma_mib = torch.cuda.max_memory_allocated(curr_device_idx) // mem_divider
        mr_mib = torch.cuda.memory_reserved(curr_device_idx) // mem_divider
        max_mr_mib = torch.cuda.max_memory_reserved(curr_device_idx) // mem_divider
        free_mem, total_mem = torch.cuda.mem_get_info(curr_device_idx)
        active_mem, free_mem, total_mem = (total_mem - free_mem) // mem_divider, free_mem // mem_divider, total_mem // mem_divider
        print_str.append(f"[Mem({curr_device_idx}): {int(ma_mib)}(ma),{int(max_ma_mib)}(mma),{int(mr_mib)}(mr),{int(max_mr_mib)}(mmr),{int(active_mem)}/{int(free_mem)}/{int(total_mem)}(a/f/t)]")
    
        return ''.join(print_str)
    
    if pytorch_profiler and epoch_id == start_epoch:
        # setup pytorch profiler
        pfer = torch.profiler.profile(
            #schedule=torch.profiler.schedule(skip_first=0, wait=max(1,int(len(train_loader)*0.1)), warmup=max(1,int(len(train_loader)*0.1)), active=min(3,int(len(train_loader)*0.2)), repeat=2),
            schedule=torch.profiler.schedule(skip_first=30, wait=10, warmup=10, active=3, repeat=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=False,
            profile_memory=True,
            with_stack=True)
        # set timer
        pfer.start()
        start_t = timer()
        for step, batch in enumerate(train_loader):
            loss, metrics = runner.forward(batch)  # type: ignore
            runner.backward(loss)
            accumulator.update(loss, metrics, step=False)
            if (step + 1) % gradient_accumulation_steps == 0:
                runner.step()
                viz.log_metrics(accumulator.step(), "train", runner.global_step)
                viz.log_scalars("lr","train", runner.scheduler.get_last_lr()[0], runner.global_step)
                if runner.global_step % num_log_iter == 0:
                    end_t = timer()
                    logger.info(make_log_str(step, end_t - start_t))
                    start_t = end_t
            pfer.step()
        pfer.stop()
    else:
        forward_t,backward_t,data_t = 0,0,0
        start_t = timer()
        data_start = timer()
        for step, batch in enumerate(train_loader):
            data_t += timer() - data_start
            forward_start = timer()
            loss, metrics = runner.forward(batch)  # type: ignore
            forward_t += timer() - forward_start
            
            backward_start = timer()
            runner.backward(loss)
            accumulator.update(loss, metrics, step=False)
            backward_t += timer() - backward_start
            
            if (step + 1) % gradient_accumulation_steps == 0:
                backward_start = timer()
                runner.step()
                backward_t += timer() - backward_start
                
                viz.log_metrics(accumulator.step(), "train", runner.global_step)
                viz.log_scalars("lr","train", runner.scheduler.get_last_lr()[0], runner.global_step)
                if runner.global_step % num_log_iter == 0:
                    end_t = timer()
                    logger.info(make_log_str(step, end_t - start_t, forward_t, backward_t, data_t))
                    forward_t,backward_t,data_t = 0,0,0
                    start_t = end_t
            
            data_start = timer()

    final_print_str = f"Train: [Loss: {accumulator.final_loss():.5g}]"
    for name, value in accumulator.final_metrics().items():
        final_print_str += f"[{name.capitalize()}: {value:.5g}]"
    logger.info(final_print_str)
    return accumulator.final_loss(), accumulator.final_metrics()


def run_valid_epoch(epoch_id: int,
                    valid_loader: DataLoader,
                    runner: ForwardRunner,
                    viz: typing.Optional[visualization.TAPEVisualizer] = None,
                    is_master: bool = True) -> typing.Tuple[float, typing.Dict[str, float]]:
    num_batches = len(valid_loader)
    accumulator = utils.MetricsAccumulator()

    torch.set_grad_enabled(False)
    runner.eval()
    
    # !!This need to be corrected in utils.MetricsAccumulator():
    # The total perplexity is calculated in the way like: 1/2*(s1/n1 + s2/n2)
    # But should be (s1+s2)/(n1+n2)
    for batch in tqdm(valid_loader, desc='Running Eval', total=num_batches,
                      disable=not is_master, leave=False):
        loss, metrics = runner.forward(batch)  # type: ignore
        accumulator.update(loss, metrics)

    # Reduce loss across all processes if multiprocessing
    eval_loss = utils.reduce_scalar(accumulator.final_loss())
    metrics = {name: utils.reduce_scalar(value)
               for name, value in accumulator.final_metrics().items()}

    print_str = f"Evaluation: [Loss: {eval_loss:.5g}]"
    for name, value in metrics.items():
        print_str += f"[{name.capitalize()}: {value:.5g}]"

    metrics['loss'] = eval_loss
    if viz is not None:
        #viz.log_metrics(metrics, "val", getattr(runner, 'global_step', epoch_id))
        viz.log_metrics(metrics, "val", epoch_id)
        logger.info("** Visualization log saved after epoch {} **".format(epoch_id))


    logger.info(print_str)

    return eval_loss, metrics


def _get_outputs_to_save(batch, outputs):
    targets = batch['targets'].cpu().numpy()
    outputs = outputs.cpu().numpy()
    protein_length = batch['protein_length'].sum(1).cpu().numpy()

    reshaped_output = []
    for target, output, plength in zip(targets, outputs, protein_length):
        output_slices = tuple(slice(1, plength - 1) if dim == protein_length.max() else
                              slice(0, dim) for dim in output.shape)
        output = output[output_slices]
        target = target[output_slices]

        reshaped_output.append((target, output))
    reshaped_output


def run_eval_epoch(eval_loader: DataLoader,
                   runner: ForwardRunner,
                   metric_names: typing.Sequence[str],
                   metric_functions: typing.Sequence[typing.Callable],
                   data_dir: str,
                   task: str,
                   from_pretrained: str,
                   pretrained_epoch: typing.Union[int,str],
                   model_config: BaseConfig,
                   split: str,
                   eval_save_dir: str,
                   output_pred: bool = False,
                   is_master: bool = True,
                   **kwargs) -> typing.Union[typing.Dict,typing.Tuple]:  #typing.List[typing.Dict[str, typing.Any]]:
    torch.set_grad_enabled(False)
    runner.eval()
    
    ## antibody task related
    mlm_mask_stragy = kwargs.get('mlm_mask_stragy')
    mlm_maskStragy_id = f'_{mlm_mask_stragy}' if mlm_mask_stragy is not None else ''
    embed_modelNm = kwargs.get('embed_modelNm')
    
    ## mutagenesis
    mutgsis_set = kwargs.get('mutgsis_set')

    # load some config params
    num_layers = model_config.num_hidden_layers
    num_heads = model_config.num_attention_heads
    head_selector = model_config.head_selector

    save_outputs = []
    metric_values = {} # Dict[str,Any]
    accumulator = utils.MetricsAccumulator() # (moving) average across batches

    ## initialize metric_values
    for name in metric_names:
        if name == 'perplexity':
            metric_values[name] = [0.,0.,0.] # [ece, ppl, makedToken_count]
        elif name == 'perplexity_subClass_AB':
            metric_values[name] = [0.,0.,0.] # [ece, ppl, makedToken_count]
        elif 'precision' in name:
            # 1.correct [step,n_layer,n_head]; 2.total [step,n_layer,n_head] 3.indiv_prec [step,bs,n_layer,n_head]
            metric_values[name] = [[],[],[]]
        elif name == 'test_logisticRegression':
            metric_values[name] = [np.zeros((4,3,2)),[]] # [n_range(all,short,medium,long), k_top(1,2,5), 2(correst,total)]
        elif name == 'test_logisticRegression_layerwise':
            metric_values[name] = [np.zeros((num_layers,4,3,2)),[]] # [n_layer,n_range,k_top,2]
        elif name == 'test_logisticRegression_layersupervise':
            metric_values[name] = [np.zeros((4,3,2)),[]] # [n_range(all,short,medium,long), k_top(1,2,5), 2(correst,total)] 
        elif 'all_pred_distribution' in name:
            # 1st: [n_layer,n_head,3(short,medium,long)]
            metric_values[name] = [np.zeros((num_layers,num_heads,3)),np.zeros((num_layers,num_heads)),np.zeros((num_layers,num_heads))]
        elif name == 'fitness_assess_supervise':
            metric_values[name] = {} # record [target_fit,pred_fit] pair for each set
        elif name == 'fitness_unsupervise_CAGI':
            metric_values[name] = [[],[],[]] # first predicted logits [bs,n_mut_pos,n_token], second target tokens [bs,n_mut_pos], third: mut_pos [bs,n_mut_pos]
        elif name == 'fitness_supervise_CAGI':
            metric_values[name] = {'pred_fit':[], # predicted fitness scores
                                   'mutants':[] # mutation strings
                                  }
        elif name == 'fitness_unsupervise_mutagenesis':
            metric_values[name] = [[],[],[],[],[]] # 1.predicted logits,[bs,n_mut,n_token];2.wt_resi_id,[bs,n_mut];3.mut_resi_id,[bs,n_mut];4.fitness_groundtruth [bs,];5.mutation_name_list [bs,]
        elif name == 'fitness_unsupervise_scanning':
            metric_values[name] = [[],[],[]] # 1.predicted logits,[bs,1,n_token];2.wt_resi_id,[bs,1];3.abso_pos_idxs,[bs,1]
        elif name == 'multitask_unsupervise_mutagenesis':
            metric_values[name] = {
                    'aa_logits': [], # [bs,aa_tokens,l_max]
                    'ss_logits': [], # [bs,ss_tokens,l_max]
                    'rsa_logits': [], # [bs,rsa_tokens,l_max]
                    'dist_logits': [], # [bs,dist_tokens,l_max,l_max]
                    'wt_aa_ids': [], # [bs,l_max]
                    'mut_aa_ids': [], # [bs,l_max]
                    'fitness_gt': [], # [bs,]
                    'mutation_list': [], # [bs,n_mut]
                    'wt_ss_ids': [], # [bs,l]
                    'wt_rsa_ids': [], # [bs,l]
                    'wt_dist_ids': [] # [bs,l]
                    }
        elif name == 'multitask_unsupervise_mutagenesis_structure':
            metric_values[name] = {
                    'aa_seq_mask': [],
                    'ss3_logits': [],
                    'ss3_labels': [],
                    'rsa2_logits': [],
                    'rsa2_labels': [],
                    'distMap_logits': [],
                    'distMap_labels': [],
                    'mutants': [],
                    'mut_relative_idxs': [],
                    'fitness_score': []
                    }
        elif name == 'seqModel_seq_struct_eval':
            metric_values[name] = {
                    'pred_logits': [], # 1d: [bs,l_max,n_token], 2d: [bs,l_max,l_max,n_token]
                    'targets_label': [], # 1d: [bs,l_max], 2d: [bs,l_max,l_max]
                    'ce_value_tensor': [], # 1d: [num_pos,] non-reduced CE loss tensor
                    'batch_ce_values': [] # 1d: [num_batch,] ce loss for each batch
                    }
        elif name == 'multitask_seq_struct_eval':
          metric_values[name] = {
                'aa_logits': [],
                'ss_logits': [],
                'rsa_logits': [],
                'dist_logits': [],
                'targets_aa': [], # [bs,padded_l]
                'targets_ss': [], # [bs,padded_l]
                'targets_rsa': [], # [bs,padded_l]
                'targets_dist': [], # [bs,padded_l,padded_l]
                }
        elif 'contact_background' in name:
            metric_values[name] = [0., 0., []]
        elif name == 'save_embedding':
            metric_values[name] = {} # key-value: seq_id-embedding
        elif name in ['embed_antibody','embed_antibody_internal']:
            metric_values[name] = [] # save data for tSNE
        elif name == 'mutation_embedding_umap':
            metric_values[name] = {'mut_nm': [],'all_pos_ave': [],'mut_pos_ave': [],'fit_gt': [],'all_pos_ave_AA_head': [],'mut_pos_ave_AA_head': []}
        else:
            metric_values[name] = [0., 0.]

    for batch in tqdm(eval_loader, desc='Evaluation', total=len(eval_loader),
                      disable=not is_master):
        loss, metrics, outputs = runner.forward(batch, return_outputs=True) # type: ignore
        
        if task == 'antibody_mlm_seqConcate':
            pred_token_logits = outputs[1].cpu().numpy() #[bs,l_max,n_token]
            pred_subClassHLPair_logits = outputs[2].cpu().numpy() #[bs,n_subClassPair]
            targets_subClassHLPair = batch['subClassHLPair'].cpu().numpy() #[bs,]
            targets = batch['targets'].cpu().numpy() #[bs,l_max]
            predictions = pred_token_logits
        elif task == 'antibody_embed_seqConcate':
            hidden_states_transform_token = outputs[0].cpu().numpy() #[bs,l_max,hidden_d]
            hidden_states_transform_subClassHLPair = outputs[1].cpu().numpy() #[bs,hidden_d]
            hidden_states_encoder_lastLayer = outputs[2][-1].cpu().numpy()  #[bs,l_max,hidden_d]
            targets_subClassHLPair = batch['subClassHLPair'].cpu().numpy() #[bs,]
            entityH = batch['entityH'] #tuple, [bs,]
            entityL = batch['entityL'] #tuple, [bs,]
            input_masks = batch['input_mask'].cpu().numpy() #[bs,l_max]
            token_type_ids = batch['token_type_ids'].cpu().numpy() #[bs,l_max]
        elif task == 'antibody_mlm_seqIndiv':
            pred_token_logits_VH = outputs[1].cpu().numpy() #[bs,l_max_VH,n_token]
            pred_token_logits_VL = outputs[2].cpu().numpy() #[bs,l_max_VL,n_token]
            pred_subClassHLPair_logits = outputs[3].cpu().numpy() #[bs,n_subClassPair]
            targets_VH = batch['targets_VH'].cpu().numpy() #[bs,l_max_VH]
            targets_VL = batch['targets_VL'].cpu().numpy() #[bs,l_max_VL]
            targets_subClassHLPair = batch['subClassHLPair'].cpu().numpy() #[bs,]
            targets = np.concatenate((targets_VH,targets_VL),axis=1) #[bs,l_max_VH+l_max_VL]
            predictions = np.concatenate((pred_token_logits_VH,pred_token_logits_VL),axis=1) #[bs,l_max_VH+l_max_VL,n_token]
        elif task == 'antibody_embed_seqIndiv':
            hidden_states_transform_token_VH = outputs[0].cpu().numpy() #[bs,l_VH,hidden_d], hidden vec in pred_head
            hidden_states_transform_token_VL = outputs[1].cpu().numpy() #[bs,l_VL,hidden_d], hidden vec in pred_head
            hidden_states_transform_subClassHLPair = outputs[2].cpu().numpy() #[bs,hidden_d]
            hidden_states_encoder_lastLayer_VH = outputs[7].cpu().numpy() #[bs,l_VH,hidden_d], after crossAttention
            hidden_states_encoder_lastLayer_VL = outputs[9].cpu().numpy() #[bs,l_VL,hidden_d], after crossAttention
            targets_subClassHLPair = batch['subClassHLPair'].cpu().numpy() #[bs,]
            entityH = batch['entityH'] #tuple, [bs,]
            entityL = batch['entityL'] #tuple, [bs,]
            input_masks_VH = batch['input_mask_VH'].cpu().numpy() #[bs,l_max]
            input_masks_VL = batch['input_mask_VL'].cpu().numpy() #[bs,l_max]
        elif task in ['embed_seq', 'embed']:
            hiddenMats = np.transpose([hidden.cpu().numpy() for hidden in outputs[2]], (1,0,2,3)) # [bs,n_layer,L_max,hidden_d]
        elif task == 'mutation_fitness_UNsupervise_mutagenesis':
            metric_tuple = outputs[0][1]
            predictions = outputs[1].cpu().numpy() # mlm: [bs,l_max,n_token]; fitness:[bs,] 
            targets = batch['targets'].cpu().numpy() # mlm: [bs,l_max]; fitness:[bs,]
            if metric_names[0] == 'mutation_embedding_umap':
                lastHiddenMats = outputs[2][-1].cpu().numpy().astype(np.float32)# [bs,L_max,hidden_d]
                AAHeadHiddenMats = outputs[4].cpu().numpy().astype(np.float32)
        elif task == 'multitask_fitness_UNsupervise_mutagenesis':
            metric_tuple = outputs[0][1]
            predictions = [pred_logits.cpu().numpy() for pred_logits in outputs[1]]
            targets = batch['targets_seq'].cpu().numpy()
            if metric_names[0] == 'mutation_embedding_umap':
                lastHiddenMats = outputs[2].cpu().numpy().astype(np.float32)# [bs,L_max,hidden_d]
                AAHeadHiddenMats = outputs[4].cpu().numpy().astype(np.float32)
        elif task == 'multitask_fitness_UNsupervise_mutagenesis_structure':
            metric_tuple = outputs[0][1]
            pred_logits = [pred_logits.cpu().numpy().astype(np.float32) for pred_logits in outputs[1]]
        elif task == 'seq_structure_multi_task':
            metric_tuple = outputs[0][1]
            predictions = [pred_logits.cpu().numpy() for pred_logits in outputs[1]]
            targets_aa = batch['targets_seq'].cpu().numpy()
            targets_ss = batch['targets_ss'].cpu().numpy()
            targets_rsa = batch['targets_rsa'].cpu().numpy()
            targets_dist = batch['targets_dist'].cpu().numpy()
        else:
            metric_tuple = outputs[0][1]
            predictions = outputs[1].cpu().numpy() # mlm: [bs,l_max,n_token]; fitness:[bs,] 
            targets = batch['targets'].cpu().numpy() # mlm: [bs,l_max]; fitness:[bs,]
            hiddenMats, attentionMats = None, None
            #hiddenMats = np.transpose([hiddenL.cpu().numpy() for hiddenL in outputs[2]], (1,0,2,3)) # [bs,n_layer,L_max,hidden_d]
            #attentionMats = np.transpose([attenL.cpu().numpy() for attenL in outputs[3]], (1,0,2,3,4)) # [bs, n_layer, n_head,L_max,L_max]
        
            # targets_contact, valid_mask, seq_length are co existing
            if "targets_contact" in batch.keys():
                target_contacts = batch['targets_contact'].cpu().numpy() #size: [bs,l_max,l_max]
                valid_masks = batch ['valid_mask'].cpu().numpy() # size: [bs, l_max]
                seq_lengths = batch['seq_length'].cpu().numpy() # size: [bs,]
                loss_mlm = np.log(metric_tuple['perplexity'].cpu().numpy()) # size: [n_gpu,] batch mean on each gpu
                if 'nonCon_att_Fnorm2' in metric_tuple.keys(): 
                  loss_fnorm2 = metric_tuple['nonCon_att_Fnorm2'].cpu().numpy() # size: [n_gpu,]
                  loss_fnorm2_local = metric_tuple['nonCon_att_Fnorm2_local'].cpu().numpy() # size: [n_gpus, ]
                  loss_fnorm2_local_nor = metric_tuple['nonCon_att_Fnorm2_local_nor'].cpu().numpy() # size: [n_gpu, ]
                elif 'con_att_Fnorm2' in metric_tuple.keys(): 
                  loss_fnorm2 = metric_tuple['con_att_Fnorm2'].cpu().numpy() # size: [n_gpu,]
                  loss_fnorm2_local = metric_tuple['con_att_Fnorm2_local'].cpu().numpy() # size: [n_gpu, ]
                  loss_fnorm2_local_nor = metric_tuple['con_att_Fnorm2_local_nor'].cpu().numpy() # size: [n_gpu, ]
                else:
                  pass
            else:
                if 'perplexity' in metric_tuple.keys():
                    loss_mlm = np.log(metric_tuple['perplexity'].cpu().numpy())
        
        ## batch data saved to cpu for different tasks
        if task in ['penalize_nonContact_attention','promote_contact_attention','contact_ce_attention_weightnor','contact_ce_attention','esm_eval']:
            type_flags = batch['type_flag'].cpu().numpy() #size: [bs,]
        elif task in ['mutation_fitness_UNsupervise_mutagenesis']:
            set_nms = batch['set_nm'] # [bs,]
            mutants = batch['mutants'] # tuple (bs,n_mut), mutation name e.g. 'M1W'
            fitness_gt = batch['fitness_gt'].cpu().numpy() # [bs,]
            input_mask = batch['input_mask'].cpu().numpy() #[bs,l_max], for save embedding 
            targets_mut = batch['targets_mut'].cpu().numpy() #[bs,l_max]
            mut_relative_idxs = batch['mut_relative_idxs']
        elif task in ['multitask_fitness_UNsupervise_mutagenesis']:
            set_nms = batch['set_nm'] # [bs,]
            mutants = batch['mutants'] # tuple (bs,n_mut), mutation name e.g. 'M1W'
            fitness_gt = batch['fitness_gt'].cpu().numpy() # [bs,]
            input_mask = batch['input_seq_mask'].cpu().numpy() #[bs,l_max], for save embedding 
            targets_mut = batch['targets_mut'].cpu().numpy() #[bs,l_max]
            mut_relative_idxs = batch['mut_relative_idxs']
        elif task in ['multitask_fitness_UNsupervise_mutagenesis_structure']:
            ss3_labels = batch['ss3_label_ids']
            rsa2_labels = batch['rsa2_label_ids']
            distMap_labels = batch['distMap_label_ids']
            set_nms = batch['set_nms']
            mutants = batch['mutants']
            mut_relative_idxs = batch['mut_relative_idxs']
            fitness_score = batch['fitness_score'].cpu().tolist()
            aa_seq_mask = batch['aa_seq_mask'].cpu().numpy().astype(np.bool_)
        elif task in ['structure_awareness_1d','structure_awareness_2d','masked_language_modeling']:
            set_nms = batch['set_nm']
        elif task in ['mutation_fitness_UNsupervise_CAGI']:
            mut_pos = batch['mut_pos'] # [bs,n_mut_pos]
        elif task in ['mutation_fitness_supervise_CAGI']:
            mutants = batch['mutants'] # mutation str
        elif task in ['save_embedding']:
            seqId_list = batch['seq_id'] # tuple, [bs,], for save embedding 
            input_mask = batch['input_mask'].cpu().numpy() #[bs,l_max]
        
        ## save needed outputs 
        if output_pred:
            if "targets_contact" in batch.keys():
                for pred, target, attentMat, tar_cont, seq_len in zip(predictions, targets, attentionMats, target_contacts, seq_lengths):
                    save_outputs.append({"prediction": pred, "target": target, "attenMat": attentMat, "tar_contMap": tar_cont, "seq_length": seq_len})
            else:
                 for pred, target in zip(predictions, targets):
                    save_outputs.append({"prediction": pred, "target": target})
        
        ## loop over metrics and fill in metric_values
        for name, metric in zip(metric_names, metric_functions):
            if name == 'perplexity':
                metric_values[name] = list(map(sum, zip(metric_values[name],metric(targets, predictions, normalize=False))))
            elif name == 'perplexity_subClass_AB':
                metric_values[name] = list(map(sum, zip(metric_values[name],metric(targets_subClassHLPair, pred_subClassHLPair_logits, normalize=False))))
            elif 'contact_background' in name:
                #name == 'contact_background_prec_all':
                name_split = re.split('_',name)
                cal_range = name_split[-1]                
                #metric_values[name] = list(map(sum, zip(metric_values[name],
                #  metric(target_contacts,normalize=False,valid_mask=valid_masks,seq_length=seq_lengths,cal_range=cal_range))))
                corr_num, total_num, indiv_list = metric(target_contacts,normalize=False,valid_mask=valid_masks,seq_length=seq_lengths,cal_range=cal_range)
                metric_values[name][0] += corr_num
                metric_values[name][1] += total_num
                metric_values[name][2].extend(indiv_list) #[bs,]
            elif 'precision' in name:
                #name == 'max_precision_all_5':
                name_split = re.split('_',name)
                top_cut = int(name_split[-1])
                symm_way = name_split[0]
                cal_range = name_split[2]                
                corr_list, total_list, indiv_prec_list = metric(target_contacts, attentionMats, normalize=False, valid_mask=valid_masks,seq_length=seq_lengths, top_cut=top_cut, symm_way=symm_way, cal_range=cal_range)
                metric_values[name][0].append(corr_list)
                metric_values[name][1].append(total_list)
                if indiv_prec_list is not None:
                  metric_values[name][2].append(indiv_prec_list)
            elif name == 'train_logisticRegression':
                pretrain_model = re.split('/',from_pretrained)[-1]
                metric_values[name] = metric(target_contacts,attentionMats,type_flags,valid_mask=valid_masks,seq_length=seq_lengths, data_dir=data_dir,task=task,pretrain_model=pretrain_model,pretrained_epoch=pretrained_epoch)
            elif name == 'train_logisticRegression_layerwise':
                pretrain_model = re.split('/',from_pretrained)[-1]
                metric_values[name] = metric(target_contacts,attentionMats,type_flags,valid_mask=valid_masks,seq_length=seq_lengths, data_dir=data_dir,task=task,pretrain_model=pretrain_model,pretrained_epoch=pretrained_epoch)
            elif name == 'train_logisticRegression_layersupervise':
                pretrain_model = re.split('/',from_pretrained)[-1]
                metric_values[name] = metric(target_contacts,attentionMats,type_flags,valid_mask=valid_masks,seq_length=seq_lengths, data_dir=data_dir,task=task,pretrain_model=pretrain_model,head_selector=head_selector,pretrained_epoch=pretrained_epoch)
            elif name == 'test_logisticRegression':
                best_mdl_set = []
                pretrain_model = re.split('/',from_pretrained)[-1]
                prec_set, indiv_prec_set = metric(target_contacts,attentionMats,best_mdl_set,data_dir=data_dir,valid_mask=valid_masks, seq_length=seq_lengths,mdl_save_dir='logistic_models',pretrain_model=pretrain_model,pretrained_epoch=pretrained_epoch)
                metric_values[name][0] += prec_set
                metric_values[name][1].append(indiv_prec_set)
            elif name == 'test_logisticRegression_layerwise':
                best_mdl_set = []
                pretrain_model = re.split('/',from_pretrained)[-1]
                prec_set, indiv_prec_set = metric(target_contacts,attentionMats,best_mdl_set,data_dir=data_dir,valid_mask=valid_masks, seq_length=seq_lengths,mdl_save_dir='logistic_models',pretrain_model=pretrain_model,pretrained_epoch=pretrained_epoch) 
                metric_values[name][0] += prec_set
                metric_values[name][1].append(indiv_prec_set)
            elif name == 'test_logisticRegression_layersupervise':
                best_mdl_set = []
                pretrain_model = re.split('/',from_pretrained)[-1]
                prec_set, indiv_prec_set = metric(target_contacts,attentionMats,best_mdl_set, data_dir=data_dir,valid_mask=valid_masks,seq_length=seq_lengths, mdl_save_dir='logistic_models',pretrain_model=pretrain_model, head_selector=head_selector,pretrained_epoch=pretrained_epoch) 
                metric_values[name][0] += prec_set
                metric_values[name][1].append(indiv_prec_set)
            elif 'all_pred_distribution' in name:
                name_split = re.split('_',name)
                top_cut = int(name_split[-1])
                symm_way = name_split[-2]
                pred_dis_list, corr_list, total_list = metric(target_contacts,attentionMats,valid_mask=valid_masks, seq_length=seq_lengths,top_cut=top_cut,symm_way=symm_way)
                metric_values[name][0] += pred_dis_list
                metric_values[name][1] += corr_list
                metric_values[name][2] += total_list
            elif name == 'fitness_assess_supervise': # add target_fit, pred_fit pair to json
                for i_idx in range(len(targets)):
                    i_setNm = set_nms[i_idx]
                    pred_fitness = predictions[i_idx]
                    tar_fitness = targets[i_idx]
                    if i_setNm not in metric_values[name].keys():
                        metric_values[name][i_setNm] = [[tar_fitness,pred_fitness]]
                    else:
                        metric_values[name][i_setNm].append([tar_fitness,pred_fitness])
            elif name == 'fitness_unsupervise_CAGI':
                ## only keep predicted logits of masked positions
                for bs_i in range(len(targets)):
                    pred_logits = predictions[bs_i] #[l_max,n_tokens]
                    tar_labels = targets[bs_i] #[l_max,]
                    mask_pos = tar_labels != -1 
                    mask_pred_logits = pred_logits[mask_pos] #[n_masks,n_tokens]
                    mask_labels = tar_labels[mask_pos] #[n_masks,]
                    metric_values[name][0].append(mask_pred_logits) #[bs,n_masks,n_tokens]
                    metric_values[name][1].append(mask_labels) #[bs,n_masks]
                    metric_values[name][2].append(mut_pos[bs_i]) #[bs,n_masks]
            elif name == 'fitness_supervise_CAGI':
                ## only keep predicted logits of masked positions
                for bs_i in range(len(targets)):
                    pred_fit = predictions[bs_i] # scaler
                    mut_str = mutants[bs_i]
                    metric_values[name]['pred_fit'].append(pred_fit) #[bs,]
                    metric_values[name]['mutants'].append(mut_str) #[bs,]
            elif name == 'fitness_unsupervise_mutagenesis':
                ## only keep predicted logits of masked positions
                for bs_i in range(len(targets)):
                    pred_logits = predictions[bs_i] #[l_max,n_token]
                    tar_labels_wt = targets[bs_i] #[l_max,]
                    tar_labels_mut = targets_mut[bs_i] #[l_max,]
                    mask_pos = tar_labels_wt != -1 
                    mask_pred_logits = pred_logits[mask_pos] #[n_mask,n_token]
                    mask_label_wt = tar_labels_wt[mask_pos] #[n_mask,]
                    mask_label_mut = tar_labels_mut[mask_pos]
                    mutant_list = mutants[bs_i]
                    metric_values[name][0].append(mask_pred_logits) #[bs,n_mask,n_token]
                    metric_values[name][1].append([mask_label_wt]) #[bs,n_mask]
                    metric_values[name][2].append([mask_label_mut]) #[bs,n_mask]
                    metric_values[name][3].append(fitness_gt[bs_i]) #[bs,]
                    metric_values[name][4].append(mutant_list) #[bs,]
            elif name == 'multitask_unsupervise_mutagenesis':
                for bs_i in range(len(targets)):
                    metric_values[name]['aa_logits'].append(predictions[0][bs_i]) # [bs,aa_tokens,l_max]
                    #metric_values[name]['ss_logits'].append(predictions[1][bs_i]) # [bs,ss_tokens,l_max]    
                    #metric_values[name]['rsa_logits'].append(predictions[2][bs_i]) # [bs,rsa_tokens,l_max]
                    #metric_values[name]['dist_logits'].append(predictions[3][bs_i]) # [bs,dist_tokens,l_max,l_max]
                    metric_values[name]['wt_aa_ids'].append(targets[bs_i]) # [bs,l_max]
                    metric_values[name]['mut_aa_ids'].append(targets_mut[bs_i]) # [bs,l_max]
                    metric_values[name]['fitness_gt'].append(fitness_gt[bs_i]) # [bs,]
                    metric_values[name]['mutation_list'].append(mutants[bs_i]) # (bs,n_mut)
                    #metric_values[name]['wt_ss_ids'].append() # [bs,l]
                    #metric_values[name]['wt_rsa_ids'].append() # [bs,l]
                    #metric_values[name]['wt_dist_ids'].append() # [bs,l]
            elif name == 'multitask_unsupervise_mutagenesis_structure':
                for bs_i in range(len(mutants)):
                    metric_values[name]['aa_seq_mask'].append(aa_seq_mask[bs_i])
                    metric_values[name]['ss3_logits'].append(pred_logits[1][bs_i])
                    metric_values[name]['ss3_labels'].append(ss3_labels[bs_i])
                    metric_values[name]['rsa2_logits'].append(pred_logits[2][bs_i])
                    metric_values[name]['rsa2_labels'].append(rsa2_labels[bs_i])
                    metric_values[name]['distMap_logits'].append(pred_logits[3][bs_i])
                    metric_values[name]['distMap_labels'].append(distMap_labels[bs_i])
                    metric_values[name]['mutants'].append(mutants[bs_i])
                    metric_values[name]['mut_relative_idxs'].append(mut_relative_idxs[bs_i])
                    metric_values[name]['fitness_score'].append(fitness_score[bs_i])                   
            elif name == 'fitness_unsupervise_scanning':
                ## only keep predicted logits of masked positions
                for bs_i in range(len(targets)):
                    pred_logits = predictions[bs_i] #[l_max,n_token]
                    tar_labels_wt = targets[bs_i] #[l_max,]
                    ab_pos = batch['mut_abso_idxs'][bs_i] #[1,]
                    mask_pos = tar_labels_wt != -1
                    mask_pred_logits = pred_logits[mask_pos] #[1,n_token]
                    mask_label_wt = tar_labels_wt[mask_pos] #[1,]
                    metric_values[name][0].append(mask_pred_logits) #[bs,1,n_token]
                    metric_values[name][1].append([mask_label_wt]) #[bs,1]
                    metric_values[name][2].append(ab_pos) #[bs,1]
            elif name == 'seqModel_seq_struct_eval':
                if model_config.label_type == 'distMap':
                    value={'pred_logits': predictions, 'targets_label': targets}
                    metric(value,set_nms[0],label_type=model_config.label_type,accumulator=accumulator)
                else:
                    for bs_i in range(len(targets)):
                        metric_values[name]['pred_logits'].append(predictions[bs_i])
                        metric_values[name]['targets_label'].append(targets[bs_i])
            elif name == 'multitask_seq_struct_eval':
                value={'dist_logits': predictions[3], 'targets_dist': targets_dist}
                pretrain_model = re.split('/',from_pretrained)[-1]
                metric(value,split,eval_save_dir,pretrain_model,task,pretrained_epoch=pretrained_epoch,accumulator=accumulator,batch_accumu=True,class_channel_last=model_config.class_channel_last)
                for bs_i in range(len(targets_aa)):
                    metric_values[name]['aa_logits'].append(predictions[0][bs_i]) # [bs,aa_tokens,l_max]
                    metric_values[name]['ss_logits'].append(predictions[1][bs_i]) # [bs,ss_tokens,l_max]    
                    metric_values[name]['rsa_logits'].append(predictions[2][bs_i]) # [bs,rsa_tokens,l_max]
                    metric_values[name]['targets_aa'].append(targets_aa[bs_i])
                    metric_values[name]['targets_ss'].append(targets_ss[bs_i])
                    metric_values[name]['targets_rsa'].append(targets_rsa[bs_i])
            elif name == 'save_embedding':
                ## only keep embedding of positions with residues
                for bs_i in range(len(input_mask)):
                    seq_id = seqId_list[bs_i]
                    bs_mask = input_mask[bs_i].astype(bool)
                    last_hidden = hiddenMats[bs_i,-1,bs_mask,:][1:-1]  # embedding of last layer(-1) [seq_l,hidden_d]
                    if seq_id not in metric_values[name].keys():
                        metric_values[name][seq_id] = last_hidden.tolist()
            elif name == 'mutation_embedding_umap':
                for bs_i in range(len(targets)):
                    bs_mut_rel_idx = mut_relative_idxs[bs_i]
                    bs_mut_nm = mutants[bs_i]
                    bs_fit_gt = fitness_gt[bs_i]
                    bs_mask = input_mask[bs_i].astype(bool)
                    last_hidden = lastHiddenMats[bs_i,bs_mask,:][1:-1]  # ndarray, embedding of last layer(-1) [seq_l,hidden_d]
                    mut_hidden = last_hidden[bs_mut_rel_idx]
                    aa_head_hidden = AAHeadHiddenMats[bs_i,bs_mask,:][1:-1]
                    aa_head_mut_hidden = aa_head_hidden[bs_mut_rel_idx]
                    metric_values[name]['mut_nm'].append(bs_mut_nm)
                    metric_values[name]['fit_gt'].append(bs_fit_gt)
                    metric_values[name]['all_pos_ave'].append(np.mean(last_hidden,axis=0,dtype=np.float32))
                    metric_values[name]['mut_pos_ave'].append(np.mean(mut_hidden,axis=0,dtype=np.float32))
                    metric_values[name]['all_pos_ave_AA_head'].append(np.mean(aa_head_hidden,axis=0,dtype=np.float32))
                    metric_values[name]['mut_pos_ave_AA_head'].append(np.mean(aa_head_mut_hidden,axis=0,dtype=np.float32))
            elif re.search(r'accuracy.*_subClass_AB',name) is not None:
                metric_values[name] = list(map(sum, zip(metric_values[name], metric(targets_subClassHLPair, pred_subClassHLPair_logits, normalize=False))))
            elif name in ['embed_antibody', 'embed_antibody_internal']:
                for bs_i in range(len(entityH)):
                    #if len(hidden_states_pooled[bs_i,:]) != model_config.hidden_size:
                    #    logger.info('embed size abnormal: {}'.format(len(hidden_states_pooled[bs_i,:])))
                    #    logger.info('task: {}, mlm_mask_stragy: {}'.format(task,mlm_mask_stragy))
                    ## extract hidden vec of seq
                    if task == 'antibody_embed_seqConcate':
                        token_type_lmax = token_type_ids[bs_i]
                        input_mask_lmax = input_masks[bs_i].astype(bool)
                        hidden_states_lastLayer_seq = hidden_states_encoder_lastLayer[bs_i,input_mask_lmax,:] #[n_pos,hidden_d]
                        hidden_states_transform_token_seq = hidden_states_transform_token[bs_i,input_mask_lmax,:]
                        token_type_seq = token_type_lmax[input_mask_lmax] #[n_pos,]
                        token_type_VH = token_type_seq == 0
                        token_type_VH[0] = False
                        token_type_VH[-1] = False
                        token_type_VL = token_type_seq == 1
                        token_type_VL[-1] = False
                        hidden_states_lastLayer_token_VH_seq = hidden_states_lastLayer_seq[token_type_VH,:] #[len_VH,hidden_d]
                        hidden_states_lastLayer_token_VL_seq = hidden_states_lastLayer_seq[token_type_VL,:] #[len_VL,hidden_d]
                        hidden_states_transform_token_VH_seq = hidden_states_transform_token_seq[token_type_VH,:] 
                        hidden_states_transform_token_VL_seq = hidden_states_transform_token_seq[token_type_VL,:]
                    elif task == 'antibody_embed_seqIndiv':
                        input_mask_VH = input_masks_VH[bs_i].astype(bool)
                        input_mask_VL = input_masks_VL[bs_i].astype(bool)
                        hidden_states_lastLayer_token_VH_seq = hidden_states_encoder_lastLayer_VH[bs_i,input_mask_VH,:][1:-1,:]
                        hidden_states_lastLayer_token_VL_seq = hidden_states_encoder_lastLayer_VL[bs_i,input_mask_VL,:][1:-1,:]
                        hidden_states_transform_token_VH_seq = hidden_states_transform_token_VH[bs_i,input_mask_VH,:][1:-1,:]
                        hidden_states_transform_token_VL_seq = hidden_states_transform_token_VL[bs_i,input_mask_VL,:][1:-1,:]
                    metric_values[name].append({'entityH': entityH[bs_i],
                                                'entityL': entityL[bs_i],
                                                'hidden_states_lastLayer_token_VL': hidden_states_lastLayer_token_VL_seq.tolist(),
                                                'hidden_states_lastLayer_token_VH': hidden_states_lastLayer_token_VH_seq.tolist(),
                                                'hidden_states_transform_token_VL': hidden_states_transform_token_VL_seq.tolist(),
                                                'hidden_states_transform_token_VH': hidden_states_transform_token_VH_seq.tolist(),
                                                'subClass_pair': int(targets_subClassHLPair[bs_i])})
            else:
                metric_values[name] = list(map(sum, zip(metric_values[name], metric(targets, predictions, normalize=False))))
            '''
            elif name=='att_fnorm2':
                metric_values[name] = list(map(sum, zip(metric_values[name],[np.sum(loss_fnorm2),len(loss_fnorm2)])))
            elif name=='att_fnorm2_local':
                metric_values[name] = list(map(sum, zip(metric_values[name],[np.sum(loss_fnorm2_local),len(loss_fnorm2_local)])))
            elif name=='att_fnorm2_local_nor':
                metric_values[name] = list(map(sum, zip(metric_values[name],[np.sum(loss_fnorm2_local_nor),len(loss_fnorm2_local_nor)])))
            '''

    # get final value of each metric
    metric_outputs = {}
    for name, value in metric_values.items():
        if name == 'perplexity':
            metric_outputs[f'lm_ece{mlm_maskStragy_id}'] = np.exp(value[0] / value[2])
            metric_outputs[f'lm_ppl{mlm_maskStragy_id}'] = value[1] / value[2]
        elif name == 'perplexity_subClass_AB':
            metric_outputs[f'AB_subClass_ece{mlm_maskStragy_id}'] = np.exp(value[0] / value[2])
            metric_outputs[f'AB_subClass_ppl{mlm_maskStragy_id}'] = value[1] / value[2]
        elif 'precision' in name:
            metric_outputs[name] = np.sum(value[0], axis=0) / np.sum(value[1], axis=0)
            indiv_prec_stack = np.concatenate(value[2],axis=0)
            metric_outputs[name+'_indiv_mean'] = np.mean(indiv_prec_stack,axis=0) # [n_layer,n_head]
            metric_outputs[name+'_indiv_std'] = np.std(indiv_prec_stack,axis=0) # [n_layer,n_head]
            if set_nms is not None and len(set_nms) == 42:
              for wt_idx in range(len(set_nms)):
                metric_outputs['{}_{}_wt'.format(set_nms[wt_idx],name)]=indiv_prec_stack[wt_idx]
        elif name == 'train_logisticRegression':
            metric_outputs['lgr_best_all'] = metric_values['train_logisticRegression'][0]
            metric_outputs['lgr_best_short'] = metric_values['train_logisticRegression'][1]
            metric_outputs['lgr_best_medium'] = metric_values['train_logisticRegression'][2]
            metric_outputs['lgr_best_long'] = metric_values['train_logisticRegression'][3]
        elif name == 'train_logisticRegression_layerwise':
            metric_outputs['lgr_best_all_layerwise'] = metric_values['train_logisticRegression_layerwise'][:,0]
            metric_outputs['lgr_best_short_layerwise'] = metric_values['train_logisticRegression_layerwise'][:,1]
            metric_outputs['lgr_best_medium_layerwise'] = metric_values['train_logisticRegression_layerwise'][:,2]
            metric_outputs['lgr_best_long_layerwise'] = metric_values['train_logisticRegression_layerwise'][:,3]
        elif name == 'train_logisticRegression_layersupervise':
            metric_outputs['lgr_best_all_layersupervise'] = metric_values['train_logisticRegression_layersupervise'][0]
            metric_outputs['lgr_best_short_layersupervise'] = metric_values['train_logisticRegression_layersupervise'][1]
            metric_outputs['lgr_best_medium_layersupervise'] = metric_values['train_logisticRegression_layersupervise'][2]
            metric_outputs['lgr_best_long_layersupervise'] = metric_values['train_logisticRegression_layersupervise'][3]
        elif name == 'test_logisticRegression':
            prec_out = value[0][:,:,0] / value[0][:,:,1]
            metric_outputs['lgr_test_prec_all'] = prec_out[0,:]
            metric_outputs['lgr_test_prec_short'] = prec_out[1,:]
            metric_outputs['lgr_test_prec_medium'] = prec_out[2,:]
            metric_outputs['lgr_test_prec_long'] = prec_out[3,:]
            indiv_prec_stack = np.concatenate(value[1],axis=2) # [k_range,k_top,bs]
            metric_outputs['lgr_test_prec_all_indiv_mean'] = np.nanmean(indiv_prec_stack[0,:,:],axis=1)
            metric_outputs['lgr_test_prec_all_indiv_std'] = np.nanstd(indiv_prec_stack[0,:,:],axis=1)
            metric_outputs['lgr_test_prec_short_indiv_mean'] = np.nanmean(indiv_prec_stack[1,:,:],axis=1)
            metric_outputs['lgr_test_prec_short_indiv_std'] = np.nanstd(indiv_prec_stack[1,:,:],axis=1)
            metric_outputs['lgr_test_prec_medium_indiv_mean'] = np.nanmean(indiv_prec_stack[2,:,:],axis=1)
            metric_outputs['lgr_test_prec_medium_indiv_std'] = np.nanstd(indiv_prec_stack[2,:,:],axis=1)
            metric_outputs['lgr_test_prec_long_indiv_mean'] = np.nanmean(indiv_prec_stack[3,:,:],axis=1)
            metric_outputs['lgr_test_prec_long_indiv_std'] = np.nanstd(indiv_prec_stack[3,:,:],axis=1)
        elif name == 'test_logisticRegression_layerwise':
            prec_out = value[0][:,:,:,0] / value[0][:,:,:,1]
            metric_outputs['lgr_test_prec_all_layerwise'] = prec_out[:,0,:]
            metric_outputs['lgr_test_prec_short_layerwise'] = prec_out[:,1,:]
            metric_outputs['lgr_test_prec_medium_layerwise'] = prec_out[:,2,:]
            metric_outputs['lgr_test_prec_long_layerwise'] = prec_out[:,3,:]
            indiv_prec_stack = np.concatenate(value[1],axis=3) # [n_layer,k_range,k_top,bs]
            metric_outputs['lgr_test_prec_all_layerwise_indiv_mean'] = np.nanmean(indiv_prec_stack[:,0,:,:],axis=2)
            metric_outputs['lgr_test_prec_all_layerwise_indiv_std'] = np.nanstd(indiv_prec_stack[:,0,:,:],axis=2)
            metric_outputs['lgr_test_prec_short_layerwise_indiv_mean'] = np.nanmean(indiv_prec_stack[:,1,:,:],axis=2)
            metric_outputs['lgr_test_prec_short_layerwise_indiv_std'] = np.nanstd(indiv_prec_stack[:,1,:,:],axis=2)
            metric_outputs['lgr_test_prec_medium_layerwise_indiv_mean'] = np.nanmean(indiv_prec_stack[:,2,:,:],axis=2)
            metric_outputs['lgr_test_prec_medium_layerwise_indiv_std'] = np.nanstd(indiv_prec_stack[:,2,:,:],axis=2)
            metric_outputs['lgr_test_prec_long_layerwise_indiv_mean'] = np.nanmean(indiv_prec_stack[:,3,:,:],axis=2)
            metric_outputs['lgr_test_prec_long_layerwise_indiv_std'] = np.nanstd(indiv_prec_stack[:,3,:,:],axis=2)
        elif name == 'test_logisticRegression_layersupervise':
            prec_out = value[0][:,:,0] / value[0][:,:,1]
            metric_outputs['lgr_test_prec_all_layersupervise'] = prec_out[0,:]
            metric_outputs['lgr_test_prec_short_layersupervise'] = prec_out[1,:]
            metric_outputs['lgr_test_prec_medium_layersupervise'] = prec_out[2,:]
            metric_outputs['lgr_test_prec_long_layersupervise'] = prec_out[3,:]
            indiv_prec_stack = np.concatenate(value[1],axis=2) # [k_range,k_top,bs]
            metric_outputs['lgr_test_prec_all_layersupervise_indiv_mean'] = np.nanmean(indiv_prec_stack[0,:,:],axis=1)
            metric_outputs['lgr_test_prec_all_layersupervise_indiv_std'] = np.nanstd(indiv_prec_stack[0,:,:],axis=1)
            metric_outputs['lgr_test_prec_short_layersupervise_indiv_mean'] = np.nanmean(indiv_prec_stack[1,:,:],axis=1)
            metric_outputs['lgr_test_prec_short_layersupervise_indiv_std'] = np.nanstd(indiv_prec_stack[1,:,:],axis=1)
            metric_outputs['lgr_test_prec_medium_layersupervise_indiv_mean'] = np.nanmean(indiv_prec_stack[2,:,:],axis=1)
            metric_outputs['lgr_test_prec_medium_layersupervise_indiv_std'] = np.nanstd(indiv_prec_stack[2,:,:],axis=1)
            metric_outputs['lgr_test_prec_long_layersupervise_indiv_mean'] = np.nanmean(indiv_prec_stack[3,:,:],axis=1)
            metric_outputs['lgr_test_prec_long_layersupervise_indiv_std'] = np.nanstd(indiv_prec_stack[3,:,:],axis=1)
        elif 'all_pred_distribution' in name:
            metric_outputs[name+'_corr_S-M-L'] = metric_values[name][0]
            metric_outputs[name+'_corr'] = metric_values[name][1]
            metric_outputs[name+'_total'] = metric_values[name][2]
        elif name == 'fitness_assess_supervise': # fitness assessment: mse, spearman coef
            metric_outputs['total_fit_assess_sets'] = len(metric_values[name].keys())
            for set_key, fit_value in metric_values[name].items():
                fit_value = np.array(fit_value)
                metric_outputs[set_key+'_num'] = fit_value.shape[0]
                metric_outputs[set_key+'_mse'] = np.mean(np.square(fit_value[:,0] - fit_value[:,1]))
                metric_outputs[set_key+'_spearmanr'] = scipy.stats.spearmanr(fit_value[:,0],fit_value[:,1]).correlation
        elif name == 'fitness_unsupervise_CAGI':
            metric_fun = registry.get_metric('fitness_unsupervise_CAGI')
            metric_fun(value,data_dir,split,from_pretrained,pretrained_epoch)
        elif name == 'fitness_supervise_CAGI':
            metric_fun = registry.get_metric('fitness_supervise_CAGI')
            metric_fun(value,data_dir,split,from_pretrained,pretrained_epoch,set_nm=model_config.set_nm)
        elif name == 'fitness_unsupervise_mutagenesis':
            pretrain_model = re.split('/',from_pretrained)[-1]
            metric_fun = registry.get_metric('fitness_unsupervise_mutagenesis')
            metric_fun(value,mutgsis_set,eval_save_dir,pretrain_model,task,pretrained_epoch=pretrained_epoch,save_raw_score=True)
        elif name == 'fitness_unsupervise_scanning':
            metric_fun = registry.get_metric('fitness_unsupervise_scanning')
            pretrain_setId = re.split('_ensemble_',re.split('/',from_pretrained)[-4])[0]
            metric_fun(value,data_dir,split,pretrained_epoch=pretrained_epoch,pretrain_setId=pretrain_setId)
        elif name == 'multitask_unsupervise_mutagenesis':
            pretrain_model = re.split('/',from_pretrained)[-1]
            metric_fun = registry.get_metric('multitask_unsupervise_mutagenesis')
            metric_fun(value,mutgsis_set,eval_save_dir,pretrain_model,task,pretrained_epoch=pretrained_epoch,save_raw_score=True,cls_eval=model_config.cls_eval,class_channel_last=model_config.class_channel_last)
        elif name == 'multitask_unsupervise_mutagenesis_structure':
            pretrain_model = re.split('/',from_pretrained)[-1]
            metric_fun = registry.get_metric('multitask_unsupervise_mutagenesis_structure')
            metric_fun(value,mutgsis_set,eval_save_dir,pretrain_model,task,pretrained_epoch=pretrained_epoch,cls_eval=model_config.cls_eval,class_channel_last=model_config.class_channel_last,model_config=model_config)
        elif name == 'seqModel_seq_struct_eval':  
            if model_config.label_type == 'distMap':
                pretrain_model = re.split('/',from_pretrained)[-1]
                metric_fun = registry.get_metric('seqModel_seq_struct_eval')
                metric_fun(None,set_nms[0],eval_save_dir,pretrain_model,task,label_type=model_config.label_type,split=split,pretrained_epoch=pretrained_epoch,accumulator=accumulator)
            else:
                pretrain_model = re.split('/',from_pretrained)[-1]
                metric_fun = registry.get_metric('seqModel_seq_struct_eval')
                metric_fun(value,set_nms[0],eval_save_dir,pretrain_model,task,label_type=model_config.label_type,split=split,pretrained_epoch=pretrained_epoch)
        elif name == 'multitask_seq_struct_eval':
                metric(value,split,eval_save_dir,pretrain_model,task,pretrained_epoch=pretrained_epoch,accumulator=accumulator,batch_accumu=False,class_channel_last=model_config.class_channel_last)
        elif 'contact_background' in name:
            metric_outputs[name] = value[0] / value[1]
            metric_outputs[name+'_indiv_mean'] = np.mean(value[2])
            metric_outputs[name+'_indiv_std'] = np.std(value[2])
        elif name == 'mutation_embedding_umap':
            pretrain_model = re.split('/',from_pretrained)[-1]
            metric_fun = registry.get_metric('mutation_embedding_umap')
            metric_fun(value,model_name=pretrain_model,pretrained_epoch=pretrained_epoch,save_embeddings=True,draw_fig=False,eval_path=eval_save_dir,set_name=mutgsis_set)
        elif name == 'save_embedding':
            pretrain_model = re.split('/',from_pretrained)[-1]
            embedding_save_model_path = f'{eval_path}/embedding_analysis/embedding_save/{model_name}/{set_name}_{pretrained_epoch}'
            if not os.path.isdir(embedding_save_model_path):
                os.makedirs(embedding_save_model_path,exist_ok=True)
            
            if embed_modelNm is not None:
                saveEmbed_dir = f'{data_dir}/embedding_{split}_{embed_modelNm}'
            else:
                if pretrained_epoch is None:
                    saveEmbed_dir = f'{data_dir}/embedding_{split}_{pretrain_model}'
                else:
                    saveEmbed_dir = f'{data_dir}/embedding_{split}_{pretrain_model}_{pretrained_epoch}'
            with open(f'{saveEmbed_dir}.pickle', 'wb') as jfl:
              pkl.dump(value,jfl,protocol=pkl.HIGHEST_PROTOCOL)
        elif name == 'embed_antibody_internal':
            pretrain_set = re.split('/',from_pretrained)[-3]
            antibody_straty_set = re.split('/',from_pretrained)[-2]
            Path('{}/embeddings/{}'.format(data_dir,pretrain_set)).mkdir(parents=True, exist_ok=True)  
            with open('{}/embeddings/{}/{}.json'.format(data_dir,pretrain_set,antibody_straty_set),'w') as jfl:
              json.dump(value,jfl)
        elif name == 'embed_antibody':
            Path('{}/embeddings'.format(data_dir)).mkdir(parents=True, exist_ok=True)  
            with open('{}/embeddings/{}.json'.format(data_dir,split),'w') as jfl:
              json.dump(value,jfl)
        else:
            metric_outputs[f'{name}{mlm_maskStragy_id}'] = value[0] / value[1]
    
    if output_pred:
        return (metric_outputs, save_outputs)
    else:
        return (metric_outputs, None)



def run_train(model_type: str,
              task: str,
              learning_rate: float = 1e-4,
              batch_size: int = 1024,
              num_train_epochs: int = 10,
              num_train_total_epochs: int = 300,
              num_log_iter: int = 20,
              fp16: bool = False,
              fp16_opt_level: str = 'O1',
              warmup_steps: int = 10000,
              gradient_accumulation_steps: int = 1,
              loss_scale: int = 0,
              max_grad_norm: float = 1.0,
              exp_name: typing.Optional[str] = None,
              from_pretrained: typing.Optional[str] = None,
              pretrained_epoch: int = None,
              log_dir: str = './logs',
              pytorch_profiler: bool = False,
              eval_freq: int = 1,
              save_freq: typing.Union[int, str] = 1,
              save_freq_opt_checkpoint: typing.Union[int, str] = 'improvement',
              model_config_file: typing.Optional[str] = None, # only when 'from_pretrained' is None, config load from this file 
              extra_config_file: typing.Optional[str] = None, # after config load from 'from_pretrained', extra params here
              data_dir: str = './data',
              data_format: str = 'lmdb',
              train_split: str = 'train',
              valid_split: str = 'valid',
              output_dir: str = './results',
              no_cuda: bool = False,
              seed: int = 42,
              local_rank: int = -1,
              tokenizer: str = 'pfam',
              num_workers: int = 8,
              debug: bool = False,
              log_level: typing.Union[str, int] = logging.INFO,
              patience: int = -1,
              resume_from_checkpoint: bool = False,
              mlm_mask_stragy: str = 'vanilla',
              balancing: bool = True,
              lr_scheduler: str = 'constant',
              save_checkpoint: bool = True,
              neighbor_strategy: str = 'knn',
              knn_value: int = 20,
              dist_cutoff: float = 12.0) -> None:

    # SETUP AND LOGGING CODE #
    input_args = locals() # the dictionary of current local symbol table
    device, n_gpu, is_master, is_global_master = utils.setup_distributed(local_rank, no_cuda)

    exp_dir = utils.get_expname(exp_name, task, model_type)
    
    ## if 'best' is given as pretrained_epoch
    if isinstance(pretrained_epoch, str) and pretrained_epoch.lower() == 'best':
        pretrained_epoch = None

    if is_global_master:
        save_path = Path(output_dir) / exp_dir
        # save all the hidden parameters.
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / 'args.json').open('w') as f:
            json.dump(input_args, f)
    else:
        save_path = None 

    utils.barrier_if_distributed()
    utils.setup_logging(is_master, is_global_master, save_path, log_level)
    utils.set_random_seeds(seed, n_gpu)

    if isinstance(tokenizer, str):
        tokenizer = BaseTokenizer(vocab=tokenizer)
    
    vocab_num = tokenizer.vocab_size

    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained, extra_config_file, pretrained_epoch)
    model.resize_token_embeddings(vocab_num) ## append 'X' token; take care of tie_weights, resize mlm-head bias module
    model = model.to(device)

    # setup the datasets , model_config
    train_dataset = utils.setup_dataset(task, data_dir, train_split, tokenizer, data_format, in_memory=False, mlm_mask_stragy=mlm_mask_stragy, neighbor_strategy=neighbor_strategy, knn_value=knn_value, dist_cutoff=dist_cutoff, model_config=model.config)
    valid_dataset = utils.setup_dataset(task, data_dir, valid_split, tokenizer, data_format, in_memory=False, mlm_mask_stragy=mlm_mask_stragy, neighbor_strategy=neighbor_strategy, knn_value=knn_value, dist_cutoff=dist_cutoff, model_config=model.config)
    train_loader = utils.setup_loader(
        train_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, num_workers, balancing=balancing)
    valid_loader = utils.setup_loader(
        valid_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, num_workers)

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, batch_size, num_train_total_epochs)

    optimizer = utils.setup_optimizer(model, learning_rate)
    
    # setup log recorder
    ## only master gpu of each node has valid viz, others are dummy viz
    viz = visualization.get(log_dir, exp_dir, local_rank, int(os.environ["RANK"]), debug=debug)
    viz.log_config(input_args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)

    logger.info(
        f"device: {device}, "
        f"n_gpu: {n_gpu}, "
        f"distributed_training: {local_rank != -1}, "
        f"local rank: {os.environ['LOCAL_RANK']}; world rank: {os.environ['RANK']}; world size: {os.environ['WORLD_SIZE']}, "
        f"16-bits training: {fp16}")

    runner = BackwardRunner(
        model, optimizer, gradient_accumulation_steps, device, n_gpu,
        fp16, fp16_opt_level,local_rank, max_grad_norm, warmup_steps, num_train_optimization_steps, lr_scheduler,num_train_total_epochs)

    runner.initialize_fp16()
    if resume_from_checkpoint:
        assert from_pretrained is not None
        start_epoch, start_gloStep = runner.resume_from_checkpoint(from_pretrained,pretrained_epoch)
    else:
        start_epoch, start_gloStep = 0, 0
    
    runner.initialize_distributed_model()
    runner._global_step = start_gloStep # set starting value of global steps

    if isinstance(save_freq, str) and save_freq != 'improvement':
        raise ValueError(
            f"Only recongized string value for save_freq is 'improvement'"
            f", received: {save_freq}")

    if save_freq == 'improvement' and eval_freq <= 0:
        raise ValueError("Cannot set save_freq to 'improvement' and eval_freq < 0")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num epochs (current cycle) = %d, %d to %d" % (num_train_epochs-start_epoch,start_epoch,num_train_epochs-1))
    logger.info("  Num total epochs = %d", num_train_total_epochs)
    logger.info("  Num total steps = %d", num_train_optimization_steps)
    logger.info("  Num parameters = %d", num_trainable_parameters)

    best_val_loss = float('inf')
    best_val_epoch = 0
    num_evals_no_improvement = 0
    
    def do_save(epoch_id: int, num_evals_no_improvement: int) -> bool:
        if not is_global_master:
            return False
        ## condition on 'save_freq'
        if isinstance(save_freq, int): # also save the best model so far
            return ((epoch_id + 1) % save_freq == 0) or ((epoch_id + 1) == num_train_epochs) or (num_evals_no_improvement == 0)
        else:
            return num_evals_no_improvement == 0

    utils.barrier_if_distributed()

    # ACTUAL TRAIN/EVAL LOOP #

    with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu, gradient_accumulation_steps):
        # before train, do one round of evaluation first
        # before 0th epoch, evaluate random initilized model
        # before kth epoch, a fix to last epoch's log not saved by TB in last round
        #_, _ = run_valid_epoch(start_epoch-1, valid_loader, runner, viz, is_master)

        for epoch_id in range(start_epoch, num_train_epochs):
            # save untrained model at epoch_id = 0
            if epoch_id == 0 and is_global_master:
                logger.info("** ** * Saving untrained model before epoch 0 ** ** * ")
                # Only save the model itself
                runner.save_state(save_path, epoch_id, save_freq, save_freq_opt_checkpoint, save_checkpoint, num_train_epochs, num_evals_no_improvement)
                logger.info(f"Saving model checkpoint to {save_path}")

            run_train_epoch(epoch_id, train_loader, runner,
                            viz, num_log_iter, gradient_accumulation_steps, log_dir=f'{log_dir}/{exp_dir}', pytorch_profiler=pytorch_profiler,start_epoch=start_epoch)
            if eval_freq > 0 and (epoch_id + 1) % eval_freq == 0:
                val_loss, _ = run_valid_epoch(epoch_id, valid_loader, runner, viz, is_master)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch_id
                    num_evals_no_improvement = 0
                else:
                    num_evals_no_improvement += 1
            
            # Save trained model
            if do_save(epoch_id, num_evals_no_improvement):
                logger.info("** ** * Saving trained model ** ** * ")
                # Only save the model itself
                runner.save_state(save_path, epoch_id, save_freq, save_freq_opt_checkpoint, save_checkpoint, num_train_epochs, num_evals_no_improvement)
                logger.info(f"Saving model checkpoint to {save_path}")

            utils.barrier_if_distributed()
            if patience > 0 and num_evals_no_improvement >= patience:
                logger.info(f"Finished training at epoch {epoch_id} because no "
                            f"improvement for {num_evals_no_improvement} epochs.")
                logger.log(35, f"Best Val Loss (early-stopping): {best_val_loss} at epoch {best_val_epoch}")
                
                if local_rank != -1:
                    # If you're distributed, raise this error. It sends a signal to
                    # the master process which lets it kill other processes and terminate
                    # without actually reporting an error. See utils/distributed_utils.py
                    # for the signal handling code.
                    raise errors.EarlyStopping
                else:
                    break
    logger.info(f"Finished training after {num_train_epochs} epochs.")
    
    if best_val_loss != float('inf') and is_global_master:
        logger.log(35, f"Best Val Loss: {best_val_loss} at epoch {best_val_epoch}")
    
    # close SummaryWriter in tensorBoardX
    # if tensorboard writer is not closed, EOFError will be raised in multiprocess setting
    if hasattr(viz,'close_logger'):
       viz.close_logger()


def run_eval(model_type: str,
             task: str,
             from_pretrained: str,
             pretrained_epoch: typing.Union[str, int] = None,
             split: str = 'holdout',
             batch_size: int = 1024,
             model_config_file: typing.Optional[str] = None,
             extra_config_file: typing.Optional[str] = None,
             data_dir: str = './data',
             eval_save_dir: str = './eval_results',
             data_format: str = 'lmdb',
             no_cuda: bool = False,
             local_rank: int = -1,
             seed: int = 42,
             tokenizer: str = 'pfam',
             num_workers: int = 8,
             debug: bool = False,
             metrics: typing.Tuple[str, ...] = (),
             log_level: typing.Union[str, int] = logging.INFO,
             mutgsis_set: str = None,
             mlm_mask_stragy: str = None,
             embed_modelNm: str = None,
             neighbor_strategy: str = 'knn',
             knn_value: int = 20,
             dist_cutoff: float = 8.0) -> typing.Dict[str, float]:

    # for solving `RuntimeError: received 0 items of ancdata`
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    local_rank = -1  #not support torch.distributed.launch for evaluation
    device, n_gpu, is_master, is_global_master = utils.setup_distributed(local_rank, no_cuda)
    logger.info(f"device: {device}, n_gpu: {n_gpu}")
    utils.setup_logging(is_master, is_global_master, save_path=None, log_level=log_level)
    utils.set_random_seeds(seed, n_gpu)

    if isinstance(tokenizer, str):
        tokenizer = BaseTokenizer(vocab=tokenizer)

    ## initilize useful variables ##
    vocab_num = tokenizer.vocab_size
    dt_nm = re.split('/', data_dir)[-1]
    ## if 'best' is given as pretrained_epoch
    if isinstance(pretrained_epoch, str) and pretrained_epoch.lower() == 'best':
        pretrained_epoch = None
    ## if embed_modelNm is given, at default use best epoch
    if embed_modelNm is not None:
        if re.search(r'rp75', embed_modelNm):
            pretrained_epoch = 224
        elif re.search(r'rp15', embed_modelNm):
            pretrained_epoch = 729
        else:
            Exception(f'invalid embed_modelNm {embed_modelNm}')

    

    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained, extra_config_file, pretrained_epoch)
    model.resize_token_embeddings(vocab_num) ## append 'X' token; take care of tie_weights, resize mlm-head bias module

    model = model.to(device)
    model_config = model.config # instance of BaseConfig

    runner = ForwardRunner(model, device, n_gpu)
    runner.initialize_distributed_model()
    valid_dataset = utils.setup_dataset(task, data_dir, split, tokenizer,
        data_format, in_memory=False, mutgsis_set=mutgsis_set, mlm_mask_stragy=mlm_mask_stragy, neighbor_strategy=neighbor_strategy, knn_value=knn_value, dist_cutoff=dist_cutoff, model_config=model_config)
    valid_loader = utils.setup_loader(
        valid_dataset, batch_size, local_rank, n_gpu,
        1, num_workers)

    metric_functions = []
    for name in metrics:
      if 'precision' in name:
        metric_functions.append(registry.get_metric('contact_precision'))
      elif 'contact_background' in name:
        metric_functions.append(registry.get_metric('contact_background_prec'))
      elif 'att_fnorm2' in name:
        continue
      elif 'loss' in name:
        continue
      elif 'all_pred_distribution' in name:
        metric_functions.append(registry.get_metric('all_pred_distribution'))
      else:
        metric_functions.append(registry.get_metric(name))
    metrics_to_save, outputs_to_save = run_eval_epoch(valid_loader, runner, metrics, metric_functions,
                                                      data_dir=data_dir, task=task,
                                                      from_pretrained=from_pretrained,
                                                      pretrained_epoch=pretrained_epoch,
                                                      model_config=model_config,
                                                      split=split,
                                                      eval_save_dir=eval_save_dir,
                                                      output_pred=False,
                                                      is_master=is_master,
                                                      mlm_mask_stragy=mlm_mask_stragy,
                                                      embed_modelNm=embed_modelNm,
                                                      mutgsis_set=mutgsis_set)
    
    if metrics_to_save:
      logger.info(f"eval_report*> {';'.join(f'{name}: {val}' for name, val in metrics_to_save.items())}")
    
      eval_path = f"{eval_save_dir}/{task}/predictions/{re.split('/',from_pretrained)[-1]}"
      Path(eval_path).mkdir(parents=True, exist_ok=True)
      ## antibody seq model
      mlm_maskStragy_id = f'_{mlm_mask_stragy}' if mlm_mask_stragy is not None else ''
      if pretrained_epoch is None:
        with (Path(eval_path) / f'results_metrics_{dt_nm}_{split}{mlm_maskStragy_id}.json').open('w') as f:
          json.dump(metrics_to_save, f, cls=NumpyEncoder)
      else:
        with (Path(eval_path) / f'results_metrics_{dt_nm}_{split}_{pretrained_epoch}{mlm_maskStragy_id}.json').open('w') as f:
          json.dump(metrics_to_save, f, cls=NumpyEncoder)

    if outputs_to_save is not None:
      if pretrained_epoch is None:
        with (Path(eval_path) / f'output_predictions_{dt_nm}_{split}.pkl').open('wb') as f:
          pkl.dump(outputs_to_save, f)
      else:
        with (Path(eval_path) / f'output_predictions_{dt_nm}_{split}_{pretrained_epoch}.pkl').open('wb') as f:
          pkl.dump(outputs_to_save, f)

    return metrics_to_save


def run_eval_esm(model_type: str,
                 task: str,
                 from_pretrained: str,
                 model_name: str='/.model', 
                 data_dir: str='/.data',
                 data_format: str = 'lmdb',
                 split: str = 'holdout',
                 batch_size: int = 32,
                 no_cuda: bool = False,
                 seed: int = 42,
                 repr_layers: typing.List = [12],
                 num_workers: int = 8,
                 metrics: typing.Tuple[str, ...] = (),
                 log_level: typing.Union[str, int] = logging.INFO,
                 output_pred: bool = False,
                 **kwargs) -> typing.Dict[str, float]:
    
    # for solving `RuntimeError: received 0 items of ancdata`
    torch.multiprocessing.set_sharing_strategy('file_system')
    # setting
    local_rank = -1  # TAPE does not support torch.distributed.launch for evaluation
    device, n_gpu, is_master = utils.setup_distributed(local_rank, no_cuda)
    utils.setup_logging(local_rank, save_path=None, log_level=log_level)
    utils.set_random_seeds(seed, n_gpu)

    # load model and tokenizer
    model, alphabet = torch.hub.load("facebookresearch/esm", model_name)
    model.eval()
    model = model.to(device)

    # prepare dataloader
    dataset = utils.setup_dataset(task, data_dir, split, 'none', data_format, alphabet_obj=alphabet)
    data_loader = utils.setup_loader(dataset, batch_size, local_rank, n_gpu, 1, num_workers)

    # metric function
    metric_functions = []
    for name in metrics:
        if 'logisContact_esm' in name:
            metric_functions.append(registry.get_metric('logisContact_esm'))
        elif 'precision' in name:
            metric_functions.append(registry.get_metric('contact_precision'))
        elif 'contact_background' in name:
            metric_functions.append(registry.get_metric('contact_background_prec'))
        else:
            metric_functions.append(registry.get_metric(name))
    
    repr_layers_list = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers
    ]
    # vars for metrics calculation
    metric_values = {} # Dict[str,Tuple[float,float]]
    for name in metrics:
        if 'logisContact_esm' in name:
            metric_values[name] = [0., 0., []]
        elif 'precision' in name:
            metric_values[name] = [[],[]]
        elif name == 'test_logisticRegression':
            metric_values[name] = np.zeros((4,3,2))
        else:
            metric_values[name] = [0., 0.]
    save_outputs = []
    with torch.no_grad():
        for batch_idx,batch_dict  in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(data_loader)} batches ({batch_dict['batch_tokens'].size(0)} sequences)"
             )

            batch_tokens = batch_dict['batch_tokens'].to(device) # [bs,l_max] with start token and padding
            target_contacts = batch_dict['targets_contacts'].cpu().numpy() # [bs,l_max-1,l_max-1] with padding only
            valid_masks = batch_dict['valid_masks'].cpu().numpy() # [bs,l_max-1] with padding only 
            seq_lengths = batch_dict['seq_lengths'].cpu().numpy() # [bs,]
            if "type_flag" in batch_dict.keys():
                type_flags = batch_dict['type_flag'].cpu().numpy() #size: [bs,]

            # keys in out: 
            # 'logits'- tensor, before softmax, [bs,L_max,n_token], 
            # 'representations'- dict, key: repr_layers, value: tensor, [bs, L_max, hidden_size], 
            # 'attentions'- tensor, [bs,n_layer,n_head,L_max,L_max]
            # 'contacts'- tensor, [bs,L_max-1,L_max-1] (start token is removed)
            out = model(batch_tokens, repr_layers=repr_layers_list, return_contacts=True)
            attentionMats = out['attentions'].cpu().numpy()[:,:,:,1:,1:]
            pred_contacts = out['contacts'].cpu().numpy()
            
            # record outputs
            if output_pred:
                for pred_cont, tar_cont, attenM, valMask, seq_len, type_fg in zip(pred_contacts, target_contacts, attentionMats, valid_masks, seq_lengths, type_flags):
                    save_outputs.append({"pred_contact": pred_cont, "target_contact": tar_cont, "attenMat": attenM, "val_mask": valMask, "seq_length": seq_len,"type_flag":type_fg})


            # calculate metrics
            for name, metric in zip(metrics, metric_functions):
                if 'precision' in name:
                    #name == 'max_precision_all_5':
                    name_split = re.split('_',name)
                    top_cut = int(name_split[-1])
                    symm_way = name_split[0]
                    cal_range = name_split[2]                
                    corr_list, total_list = metric(target_contacts, attentionMats, normalize=False,
                                                   valid_mask=valid_masks,seq_length=seq_lengths,
                                                   top_cut=top_cut, symm_way=symm_way,cal_range=cal_range)
                    metric_values[name][0].append(corr_list)
                    metric_values[name][1].append(total_list)
                elif 'logisContact_esm' in name:
                    #name == 'logisContact_esm_all_5':
                    name_split = re.split('_',name)
                    top_cut = int(name_split[-1])
                    cal_range = name_split[-2]
                    corr,total,indiv_prec_list = metric(target_contacts,pred_contacts, normalize=False,valid_mask=valid_masks,seq_length=seq_lengths,top_cut=top_cut,cal_range=cal_range)
                    metric_values[name][0] += corr
                    metric_values[name][1] += total
                    metric_values[name][2].append(indiv_prec_list)
                elif 'contact_background' in name:
                    #name == 'contact_background_prec_all':
                    name_split = re.split('_',name)
                    cal_range = name_split[-1]                
                    metric_values[name] = list(map(sum, zip(metric_values[name],metric(target_contacts,normalize=False,valid_mask=valid_masks,seq_length=seq_lengths,cal_range=cal_range))))
                elif name == 'train_logisticRegression':
                    metric_values[name] = metric(target_contacts,attentionMats,type_flags,valid_mask=valid_masks,seq_length=seq_lengths,data_dir=data_dir)
                elif name == 'test_logisticRegression':
                    best_mdl_set = [[16,0.001],[2,0.001],[17,0.05],[16,0.001]]
                    metric_values[name] += metric(target_contacts,attentionMats,best_mdl_set,data_dir=data_dir,valid_mask=valid_masks,seq_length=seq_lengths,
                                                  mdl_save_dir='logistic_models_esm',pretrain_model=model_name)
                else:
                    metric_values[name] = list(map(sum, zip(metric_values[name], metric(target_contacts,pred_contacts, normalize=False))))
    
    # get final value of each metric
    metric_outputs = {}
    for name, value in metric_values.items():
        if 'logisContact_esm' in name:
            metric_outputs[name] = value[0] / value[1]
            indiv_prec_stack = np.concatenate(value[2],axis=0)
            metric_outputs[name+'_indiv_mean'] = np.mean(indiv_prec_stack)
            metric_outputs[name+'_indiv_std'] = np.std(indiv_prec_stack)
        elif 'precision' in name:
            metric_outputs[name] = np.sum(value[0], axis=0) / np.sum(value[1], axis=0)
        elif name == 'train_logisticRegression':
            metric_outputs['lgr_best_all'] = metric_values['train_logisticRegression'][0]
            metric_outputs['lgr_best_short'] = metric_values['train_logisticRegression'][1]
            metric_outputs['lgr_best_medium'] = metric_values['train_logisticRegression'][2]
            metric_outputs['lgr_best_long'] = metric_values['train_logisticRegression'][3]
        elif name == 'test_logisticRegression':
            prec_out = metric_values['test_logisticRegression'][:,:,0] / metric_values['test_logisticRegression'][:,:,1]
            metric_outputs['lgr_test_prec_all'] = prec_out[0,:]
            metric_outputs['lgr_test_prec_short'] = prec_out[1,:]
            metric_outputs['lgr_test_prec_medium'] = prec_out[2,:]
            metric_outputs['lgr_test_prec_long'] = prec_out[3,:]
        else:
            metric_outputs[name] = value[0] / value[1]

    logger.info(';'.join(f'{name}: {val}' for name, val in metric_outputs.items()))
    
    dt_nm = re.split('/', data_dir)[-1]
    if dt_nm == 'pdbmap_contactData':
        save_dir = '{}/esm_models'.format(data_dir)
    elif dt_nm == 'logistic_datasets':
        save_dir = '{}/logistic_models_esm'.format(data_dir) 

    if len(metric_outputs.keys()) > 0:
        with open('{}/{}/{}_results_metrics_{}_{}.json'.format(save_dir,model_name,model_name,dt_nm,split),'w') as f:
            json.dump(metric_outputs, f, cls=NumpyEncoder)
    if len(save_outputs) > 0:
        with open('{}/{}/{}_output_predictions_{}_{}.pkl'.format(save_dir,model_name,model_name,dt_nm,split),'wb') as f:
            pkl.dump(save_outputs, f)

    return metric_outputs


def run_embed(model_type: str,
              data_file: str,
              out_file: str,
              from_pretrained: str,
              batch_size: int = 1024,
              model_config_file: typing.Optional[str] = None,
              extra_config_file: typing.Optional[str] = None,
              full_sequence_embed: bool = False,
              no_cuda: bool = False,
              seed: int = 42,
              tokenizer: str = 'iupac',
              num_workers: int = 8,
              log_level: typing.Union[str, int] = logging.INFO) -> None:

    local_rank = -1  # TAPE does not support torch.distributed.launch for embedding
    device, n_gpu, is_master = utils.setup_distributed(local_rank, no_cuda)
    utils.setup_logging(local_rank, save_path=None, log_level=log_level)
    utils.set_random_seeds(seed, n_gpu)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}")

    task_spec = registry.get_task_spec('embed')
    model = registry.get_task_model(
        model_type, task_spec.name, model_config_file, from_pretrained, extra_config_file)
    model = model.to(device)
    runner = ForwardRunner(model, device, n_gpu)
    runner.initialize_distributed_model()
    runner.eval()
    torch.set_grad_enabled(False)

    dataset = task_spec.dataset(data_file, tokenizer=tokenizer)  # type: ignore
    valid_loader = utils.setup_loader(dataset, batch_size, local_rank, n_gpu, 1, num_workers)

    with utils.IncrementalNPZ(out_file) as npzfile:
        with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu):
            for batch in tqdm(valid_loader, total=len(valid_loader)):
                outputs = runner.forward(batch, no_loss=True)
                ids = batch['ids']
                sequence_embed = outputs[0]
                pooled_embed = outputs[1]
                sequence_lengths = batch['input_mask'].sum(1)
                sequence_embed = sequence_embed.cpu().numpy()
                pooled_embed = pooled_embed.cpu().numpy()
                sequence_lengths = sequence_lengths.cpu().numpy()

                for seqembed, poolembed, length, protein_id in zip(
                        sequence_embed, pooled_embed, sequence_lengths, ids):
                    seqembed = seqembed[:length]
                    arrays = {'pooled': poolembed}
                    if not full_sequence_embed:
                        # avgpool across the sequence
                        arrays['avg'] = seqembed.mean(0)
                    else:
                        arrays['seq'] = seqembed
                    to_save = {protein_id: arrays}
                    npzfile.savez(**to_save)
