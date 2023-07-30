# coding=utf-8
"""PyTorch BERT model. Modified based on TAPE.

Author: Yuanfei Sun
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math, copy
from typing import Optional, Union, Tuple, Set

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .modeling_utils import BaseConfig
from .modeling_utils import BaseModel
from .modeling_utils import prune_linear_layer
from .modeling_utils import get_activation_fn
from .modeling_utils import LayerNorm
from .modeling_utils import MLMHead, NonContactRegHead, ContactRegHead, CEntropyRegHead, CEntropyRegWeightNorHead, Class1DHead, Class2DHead
from .modeling_utils import ABSeqIndivHead, ABSeqConcateHead, ABEmbedSeqConcateHead, ABEmbedSeqIndivHead
# down-stream tasks
from .modeling_utils import ValuePredictionHead
from .modeling_utils import SequenceClassificationHead
from .modeling_utils import SequenceToSequenceClassificationHead
from .modeling_utils import PairwiseContactPredictionHead

from .modeling_pt_performer import (
    PerformerAttention,
    PerformerAttentionConfig
)
# for registering task and model
from mapping import registry

logger = logging.getLogger(__name__)

# url for pre-trained model and configs
URL_PREFIX = "https://s3.amazonaws.com/proteindata/pytorch-models/"
BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base': URL_PREFIX + "bert-base-pytorch_model.bin",
}
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base': URL_PREFIX + "bert-base-config.json"
}

class BertConfig(BaseConfig):
    r"""
        :class:`~pytorch_transformers.BertConfig` is the configuration class to store the
        configuration of a `BertModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in
                `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Bert encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Bert encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Bert encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            *attention_type (:obj:`str`, `optional`, defaults to :obj:`'softmax'`): The type of attention mechanism to use. 
                Possibilities are :obj:`'softmax'` and :obj:`'performer'`, with the latter referring to the FAVOR+ algorithm 
                put forward in the paper "Rethinking Attention with Performers".
            *performer_attention_config (:obj:`str`, `optional`, defaults to :obj:`None`): An instance of PerformerAttentionConfig 
                carrying options for the PerformerAttention module. Only used when :obj:`attention_type` = :obj:`'performer'`.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
            head_selector: select head to apply binary contact supervision.
            lamda_contact: hyper-weight for promoting contact attention term.
            lamda_nonContact: hyper-weight for penalizing non-contact attention term.
            lamda_mlm: hyper-weight for mlm loss term.
            gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If True, use gradient checkpointing to save memory at the expense of slower backward pass.
    """
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    
    def __init__(self,
                 vocab_size: int = 28,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 4,
                 num_attention_heads: int = 8,
                 intermediate_size: int = 1024,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 attention_type='softmax', # for performer
                 performer_attention_config: Optional[Union[dict, PerformerAttentionConfig]] = None, # for performer
                 max_position_embeddings: int = 8096,
                 # type_vocab_size: length of token_type_ids [0,1]
                 # (Segment token indices to indicate first and second portions of the inputs)
                 type_vocab_size: int = 2,                 
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 head_selector: Tuple = None, # ([0]*8,[0]*8,[0]*8,[1]*8), for supervise attention                 
                 lamda_nonContact: float = 1.0, # weight for penalize non-contact
                 lamda_contact: float = 1.0, # weight for promote contact
                 lamda_mlm: float=1.0, # weight for mlm loss
                 lamda_l1: float=0.1, # weight for l1 regu term
                 lamda_ce: float=1.0, # weight for ce loss
                 weight_short: float=1.0, # balancing weight for short range
                 weight_medium: float=1.0, # balancing weight for medium range
                 weight_long: float=1.0, # balancing weight for long range
                 gamma: float=0.0, # gamma - power of focal loss
                 weight_s_con: float=1.0, # short con balancing weight
                 weight_s_non: float=1.0, # short nonCon balancing weight
                 weight_m_con: float=1.0, # medium con balancing weight
                 weight_m_non: float=1.0, # medium nonCon balancing weight
                 weight_l_con: float=1.0, # long con balancing weight
                 weight_l_non: float=1.0, # long nonCon balancing weight
                 gradient_checkpointing: bool=False,
                 subClass_dropoutProb: float=0.1,  # antibody, gene pair classcification
                 weight_subClassLoss: float=0.0, # antibody, weight for gene pair loss
                 set_nm: str=None, # set name for protein family
                 label_apply_ln: bool=True, # apply natural log to fitness labels
                 pred_head_dropout: float = 0.1, # dropout rate in prediction head
                 class_size: int = -1, # structure awareness eval, num of class for structure label
                 label_type: str = None, # structure awareness eval, structure label type: ss, rsa, distMap
                 multi_copy_num: int = 1, # structure awareness eval, copy number for single input
                 eval_phase: bool = False, # structure awareness eval, train or eval indicator
                 use_class_weights: bool = False, # whether to apply class balancing in cross-entropy
                 seq_max_length: int = 128,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.attention_type = attention_type
        self.performer_attention_config = performer_attention_config
        self.head_selector = head_selector
        self.lamda_nonContact = lamda_nonContact
        self.lamda_contact = lamda_contact
        self.lamda_mlm = lamda_mlm
        self.lamda_ce = lamda_ce
        self.lamda_l1 = lamda_l1
        self.weight_short = weight_short
        self.weight_medium = weight_medium
        self.weight_long = weight_long
        self.gamma = gamma
        self.weight_s_con = weight_s_con
        self.weight_s_non = weight_s_non
        self.weight_m_con = weight_m_con
        self.weight_m_non = weight_m_non
        self.weight_l_con = weight_l_con
        self.weight_l_non = weight_l_non
        self.gradient_checkpointing = gradient_checkpointing
        self.subClass_dropoutProb = subClass_dropoutProb
        self.weight_subClassLoss = weight_subClassLoss
        self.set_nm = set_nm
        self.label_apply_ln = label_apply_ln
        self.pred_head_dropout = pred_head_dropout
        self.class_size = class_size
        self.label_type = label_type
        self.multi_copy_num = multi_copy_num
        self.eval_phase = eval_phase
        self.use_class_weights = use_class_weights
        self.seq_max_length = seq_max_length
        
        if isinstance(self.performer_attention_config, dict):
            self.performer_attention_config = PerformerAttentionConfig(**self.performer_attention_config)

    def to_dict(self):
        output = super().to_dict()
        
        # Correct for the fact that PretrainedConfig doesn't call .__dict__ recursively on non-JSON primitives
        performer_config = output['performer_attention_config']
        if performer_config is not None:
            output['performer_attention_config'] = copy.deepcopy(performer_config.__dict__)
        
        return output

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be
        # able to load any TensorFlow checkpoint file
        # TAPE uses apex.normalization.fused_layer_norm to speed up. If not
        # available, same as nn.LayerNorm
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            # only use 0 for segment id(since only one sequence)
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

'''
Multi-head attention
'''
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        # output_attentions shows up in Class BaseConfig
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # all_head_size == hidden_size?
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    '''
    output x has shape (batch_size,num_attention_head,sequence_length,attention_head_size)
    '''
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    '''
    hidden_states: shape (batch_size, sequence_length, hidden_size)
    attention_mask: Mask to avoid performing attention on padding token
                    indices. 0 for positions to attend, -10000.0 for
                    masked/padded positions. Added to attention_score before softmax
                    shape (batch_size, num_attention_head, from_seq_length, to_seq_length)
    '''
    def forward(
        self, 
        hidden_states,
        attention_mask=None,
        cross_hidden_states=None,
        cross_attention_mask=None
    ): 
        # shape of Q/K/V_layer (batch_size,num_attention_head,sequence_length,attention_head_size)
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        # If this is instantiated as a cross-attention module, the keys
        # and values come from anothor encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = cross_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(cross_hidden_states))
            value_layer = self.transpose_for_scores(self.value(cross_hidden_states))
            attention_mask = cross_attention_mask.unsqueeze(1).unsqueeze(2)

            # Since input_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are *adding* it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            
            try:
                attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            except StopIteration:
                # start from pytorch 1.5, DataParallel replica models don't have parameters()
                # https://github.com/pytorch/pytorch/issues/38493
                # https://github.com/pytorch/pytorch/pull/33907
                # https://github.com/huggingface/transformers/pull/4300
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)  # fp16 compatibility


            attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Bert paper.
        # shape of attention_probs (batch_size,num_attention_head,seq_length,seq_length)
        attention_probs = self.dropout(attention_probs)
        # atten_probs shape (batch_size,num_attention_head,seq_length,seq_length)
        # value_layer shape (batch_size,num_attention_head,seq_length,attention_head_size)
        # context_layer shape (batch_size,num_attention_head,seq_length,attention_head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer shape (batch_size,seq_length,num_attention_head,attention_head_size)
        # self.all_head_size = self.num_attention_head * self.attention_head_size
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer shape (batch_size,seq_length,all_head_size)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

'''
add & norm of self-attention module
'''
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

'''
Self-attention module
'''
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_type = config.attention_type
        if self.attn_type == 'softmax':
            self.self = BertSelfAttention(config)
        elif self.attn_type == 'performer':
            performer_config = config.performer_attention_config or PerformerAttentionConfig()
            performer_config.attention_dropout = config.attention_probs_dropout_prob
            performer_config.d_model = config.hidden_size
            performer_config.num_heads = config.num_attention_heads 
            
            self.self = PerformerAttention(performer_config)
        else:
            raise ValueError(f"Bert: Invalid attention_type {self.att_type}")

        self.output = BertSelfOutput(config)
    '''
    ?
    '''
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask=None, cross_hidden_states=None, cross_attention_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, cross_hidden_states, cross_attention_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attention_probs if we output them
        return outputs

'''
Feed forward module
'''
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation_fn(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

'''
Add & Norm of feed forward module
'''
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


'''
One module in encoder
'''
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, cross_hidden_states=None, cross_attention_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, cross_hidden_states, cross_attention_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attention_probs if we output them
        return outputs

'''
Encoder contains multiple self-attention modules
'''
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def run_function(self, start, chunk_size):
        def custom_forward(hidden_states, attention_mask):
            all_hidden_states = ()
            all_attentions = ()
            chunk_slice = slice(start, start + chunk_size)
            for layer in self.layer[chunk_slice]:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                layer_outputs = layer(hidden_states, attention_mask)
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = (hidden_states,)
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                outputs = outputs + (all_attentions,)
            return outputs

        return custom_forward

    def forward(self, hidden_states, attention_mask=None, cross_hidden_states=None, cross_attention_mask=None, chunks=None):
        all_hidden_states = ()
        all_attentions = ()

        if chunks is not None:
            assert isinstance(chunks, int)
            chunk_size = (len(self.layer) + chunks - 1) // chunks
            for start in range(0, len(self.layer), chunk_size):
                outputs = checkpoint(self.run_function(start, chunk_size),
                                     hidden_states, attention_mask, cross_hidden_states, cross_attention_mask)
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + outputs[1]
                if self.output_attentions:
                    all_attentions = all_attentions + outputs[-1]
                hidden_states = outputs[0]
        else:
            for i, layer_module in enumerate(self.layer):
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(hidden_states, attention_mask, cross_hidden_states, cross_attention_mask)
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            # Add last layer
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = (hidden_states,) # last hidden_states
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)

'''
pool embedding of first token '[cls]'
'''
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertAbstractModel(BaseModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

@registry.register_task_model('embed', 'transformer')
@registry.register_task_model('embed_seq', 'transformer')
class BertModel(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class BaseModel in modeling_utils.py
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self,
                input_ids,
                input_mask=None,
                token_type_ids=None):
        if input_mask is None:
            input_mask = torch.ones_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)

        # Since input_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are *adding* it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        
        try:
            extended_attention_mask = extended_attention_mask.to(
                                          dtype=next(self.parameters()).dtype)  # fp16 compatibility
        except StopIteration:
            # start from pytorch 1.5, DataParallel replica models don't have parameters()
            # https://github.com/pytorch/pytorch/issues/38493
            # https://github.com/pytorch/pytorch/pull/33907
            # https://github.com/huggingface/transformers/pull/4300
            extended_attention_mask = extended_attention_mask.to(
                                          dtype=input_ids.dtype)  # fp16 compatibility


        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids,token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       chunks=None)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

@registry.register_task_model('masked_language_modeling', 'transformer')
class BertForMaskedLM(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.mlm = MLMHead(
            config.hidden_size, config.vocab_size, config.hidden_act, config.layer_norm_eps, config.use_class_weights,
            ignore_index=-1)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.mlm.decoder,
                                   self.bert.embeddings.word_embeddings)

    def resize_output_bias(self, new_num_tokens=None):
        """ resize bias tensor MLMHead

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the original bias module.
        Return: ``torch.nn.Parameter``
            Pointer to the resized MLMHead bias module or the old bias module if
            new_num_tokens is None
        """
        old_bias = self.mlm.bias
        #print(f'size: {old_bias.data.size()[0]}', flush=True)
        old_num_tokens = old_bias.data.size()[0]
        if new_num_tokens is None:
            return self.mlm.bias
        
        # Build new bias
        new_bias = nn.Parameter(data=torch.zeros(new_num_tokens))
        new_bias.to(old_bias.data.device)

        # Copy bias weights from the old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_bias.data[:num_tokens_to_copy] = \
            old_bias.data[:num_tokens_to_copy]
        
        self.mlm.bias = new_bias
        return self.mlm.bias

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None,
                token_type_ids=None):

        outputs = self.bert(input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

        sequence_output, pooled_output = outputs[:2]
        # add hidden states and attention if they are here
        outputs = self.mlm(sequence_output, targets) + outputs[2:]
        # (loss_and_metrics), prediction_scores, (hidden_states), (attentions)
        return outputs

@registry.register_task_model('antibody_mlm_seqConcate', 'transformer')
class BertForABMLMSeqConcate(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.pred = ABSeqConcateHead(
                        config.hidden_size,
                        config.vocab_size,
                        config.hidden_act,
                        config.layer_norm_eps,
                        config.subClass_dropoutProb,
                        config.weight_subClassLoss,
                        ignore_index=-1)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.pred.decoder,
                                   self.bert.embeddings.word_embeddings)

    def resize_output_bias(self, new_num_tokens=None):
        """ resize bias tensor in ABSeqConcateHead

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the original bias module.
        Return: ``torch.nn.Parameter``
            Pointer to the resized MLMHead bias module or the old bias module if
            new_num_tokens is None
        """
        old_bias = self.pred.bias
        #print(f'size: {old_bias.data.size()[0]}', flush=True)
        old_num_tokens = old_bias.data.size()[0]
        if new_num_tokens is None:
            return self.pred.bias
        
        # Build new bias
        new_bias = nn.Parameter(data=torch.zeros(new_num_tokens))
        new_bias.to(old_bias.data.device)

        # Copy bias weights from the old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_bias.data[:num_tokens_to_copy] = \
            old_bias.data[:num_tokens_to_copy]
        
        self.pred.bias = new_bias
        return self.pred.bias

    def forward(self,
                input_ids,
                input_mask=None,
                token_type_ids=None,
                targets=None,
                subClassHLPair=None):
        # multi-layer self-attention modules
        outputs = self.bert(input_ids, input_mask=input_mask, token_type_ids=token_type_ids)
        # split bert_outputs
        bert_sequence_output = outputs[0]
        # add hidden states and attention if they are here
        outputs = self.pred(bert_sequence_output,targets,subClassHLPair) + outputs[2:]
        # (loss, metric), pred_token_logits, pred_subClass_logits, (selfAtt hidden_states), (selfAtt attentions)
        return outputs

@registry.register_task_model('antibody_embed_seqConcate', 'transformer')
class BertForABEmbedSeqConcate(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.pred = ABEmbedSeqConcateHead(
            config.hidden_size, config.vocab_size, config.hidden_act, config.layer_norm_eps, ignore_index=-1)

        self.init_weights()

    def forward(self,
                input_ids,
                input_mask=None,
                token_type_ids=None):
        # multi-layer self-attention modules
        outputs = self.bert(input_ids, input_mask=input_mask, token_type_ids=token_type_ids)
        # split bert_outputs
        bert_sequence_output = outputs[0]
        # add hidden states and attention if they are here
        outputs = self.pred(bert_sequence_output) + outputs[2:]
        # hidden_states_token, hidden_states_subClassHLPair, (selfAtt hidden_states), (selfAtt attentions)
        return outputs

@registry.register_task_model('antibody_mlm_seqIndiv', 'transformer')
class BertForABMLMSeqIndiv(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.crossAtt = BertLayer(config)
        self.pred = ABSeqIndivHead(config.hidden_size,
                                   config.vocab_size,
                                   config.hidden_act,
                                   config.layer_norm_eps,
                                   config.subClass_dropoutProb,
                                   config.weight_subClassLoss,
                                   ignore_index=-1)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.pred.decoder,
                                   self.bert.embeddings.word_embeddings)

    def resize_output_bias(self, new_num_tokens=None):
        """ resize bias tensor in ABSeqIndivHead

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the original bias module.
        Return: ``torch.nn.Parameter``
            Pointer to the resized MLMHead bias module or the old bias module if
            new_num_tokens is None
        """
        old_bias = self.pred.bias
        #print(f'size: {old_bias.data.size()[0]}', flush=True)
        old_num_tokens = old_bias.data.size()[0]
        if new_num_tokens is None:
            return self.pred.bias
        
        # Build new bias
        new_bias = nn.Parameter(data=torch.zeros(new_num_tokens))
        new_bias.to(old_bias.data.device)

        # Copy bias weights from the old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_bias.data[:num_tokens_to_copy] = \
            old_bias.data[:num_tokens_to_copy]
        
        self.pred.bias = new_bias
        return self.pred.bias

    def forward(self,
                input_ids_VH,
                input_ids_VL,
                input_mask_VH=None,
                input_mask_VL=None,
                token_type_ids_VH=None,
                token_type_ids_VL=None,
                targets_VH=None,
                targets_VL=None,
                subClassHLPair=None,
                subClassHLPair_mask=None):
        # multi-layer self-attention modules
        bert_outputs_VH = self.bert(input_ids_VH, input_mask=input_mask_VH, token_type_ids=token_type_ids_VH)
        bert_outputs_VL = self.bert(input_ids_VL, input_mask=input_mask_VL, token_type_ids=token_type_ids_VL)
        # split bert_outputs
        bert_sequence_output_VH = bert_outputs_VH[0]
        bert_sequence_output_VL = bert_outputs_VL[0]
        # cross-attention module
        crossAtt_outputs_VH = self.crossAtt(bert_sequence_output_VH,cross_hidden_states=bert_sequence_output_VL,cross_attention_mask=input_mask_VL) 
        crossAtt_outputs_VL = self.crossAtt(bert_sequence_output_VL,cross_hidden_states=bert_sequence_output_VH,cross_attention_mask=input_mask_VH)
        # split_crossAttention outputs
        crossAtt_sequence_output_VH = crossAtt_outputs_VH[0]
        crossAtt_sequence_output_VL = crossAtt_outputs_VL[0]
        # add hidden states and attention if they are here
        outputs = self.pred(crossAtt_sequence_output_VH,
                            crossAtt_sequence_output_VL, 
                            targets_VH, targets_VL, subClassHLPair) + bert_outputs_VH[2:] + bert_outputs_VL[2:] + crossAtt_outputs_VH + crossAtt_outputs_VL
        # (lossi, metrics), prediction_logits_token_VH, prediction_logits_token_VL, prediction_logits_subClass_pair, 
        # (selfAtt hidden_states VH), (selfAtt attentions VH), (selfAtt hidden_states VL), (selfAtt attentions VL), 
        # (crossAtt hidden_states VH), (crossAtt attentions VH), (crossAtt hidden_states VL), (crossAtt attentions VL)

        return outputs


@registry.register_task_model('antibody_embed_seqIndiv', 'transformer')
class BertForABEmbedSeqIndiv(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.crossAtt = BertLayer(config)
        self.pred = ABEmbedSeqIndivHead(
            config.hidden_size, config.vocab_size, config.hidden_act, config.layer_norm_eps, ignore_index=-1)

        self.init_weights()

    def forward(self,
                input_ids_VH,
                input_ids_VL,
                input_mask_VH=None,
                input_mask_VL=None,
                token_type_ids_VH=None,
                token_type_ids_VL=None):
        # multi-layer self-attention modules
        bert_outputs_VH = self.bert(input_ids_VH, input_mask=input_mask_VH, token_type_ids=token_type_ids_VH)
        bert_outputs_VL = self.bert(input_ids_VL, input_mask=input_mask_VL, token_type_ids=token_type_ids_VL)
        # split bert_outputs
        bert_sequence_output_VH = bert_outputs_VH[0]
        bert_sequence_output_VL = bert_outputs_VL[0]
        # cross-attention module
        crossAtt_outputs_VH = self.crossAtt(bert_sequence_output_VH,cross_hidden_states=bert_sequence_output_VL,cross_attention_mask=input_mask_VL) 
        crossAtt_outputs_VL = self.crossAtt(bert_sequence_output_VL,cross_hidden_states=bert_sequence_output_VH,cross_attention_mask=input_mask_VH)
        # split_crossAttention outputs
        crossAtt_sequence_output_VH = crossAtt_outputs_VH[0]
        crossAtt_sequence_output_VL = crossAtt_outputs_VL[0]
        # add hidden states and attention if they are here
        outputs = self.pred(crossAtt_sequence_output_VH,
                            crossAtt_sequence_output_VL) + bert_outputs_VH[2:] + bert_outputs_VL[2:] + crossAtt_outputs_VH + crossAtt_outputs_VL
        # hidden_states_VH, hidden_states_VL, hidden_states_subClassHLPair 
        # (selfAtt hidden_states VH), (selfAtt attentions VH), (selfAtt hidden_states VL), (selfAtt attentions VL), 
        # crossAtt hidden_states VH, (crossAtt attentions VH), crossAtt hidden_states VL, (crossAtt attentions VL)

        return outputs

# add weight normalization
@registry.register_task_model('contact_ce_attention_weightnor', 'transformer')
class BertForConCEReg(BertAbstractModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.mlm = CEntropyRegWeightNorHead(config.hidden_size, config.vocab_size,
                                            config.lamda_mlm, config.lamda_ce, config.lamda_l1,
                                            config.num_hidden_layers, config.num_attention_heads,
                                            config.head_selector, config.hidden_act,
                                            config.layer_norm_eps, ignore_index=-1)
        self.init_weights()
        self.tie_weights()
   
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.mlm.decoder,
                                   self.bert.embeddings.word_embeddings)
    
    def forward(self,
                input_ids,
                input_mask=None,
                targets=None,
                targets_contact=None,
                valid_mask=None):
        outputs = self.bert(input_ids, input_mask=input_mask)
        sequence_output, pooled_output, hidden_states, attentions = outputs
        # add hidden states and attention if they are here
        outputs = self.mlm(attentions, sequence_output, targets, targets_contact, valid_mask) + outputs[2:]
        # (loss_and_metric), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('contact_ce_attention', 'transformer')
class BertForConCEReg(BertAbstractModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.mlm = CEntropyRegHead(hidden_size=config.hidden_size,
                                   vocab_size=config.vocab_size,
                                   lamda_mlm=config.lamda_mlm,
                                   lamda_ce=config.lamda_ce,
                                   lamda_l1=config.lamda_l1,
                                   weight_short=config.weight_short,
                                   weight_medium=config.weight_medium,
                                   weight_long=config.weight_long,
                                   gamma=config.gamma,
                                   weight_s_con=config.weight_s_con,
                                   weight_s_non=config.weight_s_non,
                                   weight_m_con=config.weight_m_con,
                                   weight_m_non=config.weight_m_non,
                                   weight_l_con=config.weight_l_con,
                                   weight_l_non=config.weight_l_non,
                                   layer_num=config.num_hidden_layers,
                                   head_num=config.num_attention_heads,
                                   head_selector=config.head_selector,
                                   hidden_act=config.hidden_act,
                                   layer_norm_eps=config.layer_norm_eps,
                                   ignore_index=-1)
        self.init_weights()
        self.tie_weights()
   
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.mlm.decoder,
                                   self.bert.embeddings.word_embeddings)
    
    def forward(self,
                input_ids,
                input_mask=None,
                targets=None,
                targets_contact=None,
                valid_mask=None):
        outputs = self.bert(input_ids, input_mask=input_mask)
        sequence_output, pooled_output, hidden_states, attentions = outputs
        # add hidden states and attention if they are here
        outputs = self.mlm(attentions, sequence_output, targets, targets_contact, valid_mask) + outputs[2:]
        # (loss_and_metric), prediction_scores, (hidden_states), (attentions)
        return outputs

@registry.register_task_model('penalize_nonContact_attention', 'transformer')
class BertForNonContactReg(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.mlm = NonContactRegHead(config.hidden_size, config.vocab_size, config.head_selector, 
                                     config.lamda_nonContact, config.lamda_mlm,
                                     config.weight_short, config.weight_medium, config.weight_long,
                                     config.hidden_act, config.layer_norm_eps, ignore_index=-1)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.mlm.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None,
                targets_contact=None,
                valid_mask_nc=None,
                valid_mask=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output, hidden_states, attentions= outputs
        # add hidden states and attention if they are here
        outputs = self.mlm(attentions, sequence_output, targets, targets_contact, valid_mask_nc, valid_mask) + outputs[2:]
        # (loss_and_metric), prediction_scores, (hidden_states), (attentions)
        return outputs

@registry.register_task_model('promote_contact_attention', 'transformer')
class BertForContactReg(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.mlm = ContactRegHead(config.hidden_size, config.vocab_size, config.head_selector,
                                  config.lamda_contact, config.lamda_mlm,
                                  config.weight_short, config.weight_medium, config.weight_long,
                                  config.hidden_act, config.layer_norm_eps, ignore_index=-1)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.mlm.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None,
                targets_contact=None,
                valid_mask=None):
        
        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output, hidden_states, attentions= outputs
        # add hidden states and attention if they are here
        outputs = self.mlm(attentions, sequence_output, targets, targets_contact, valid_mask) + outputs[2:]
        # (loss_and_metric), prediction_scores, (hidden_states), (attentions)
        return outputs

@registry.register_task_model('mutation_fitness_supervise_mutagenesis', 'transformer')
@registry.register_task_model('mutation_fitness_supervise_CAGI', 'transformer')
class BertForValuePrediction(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.predict = ValuePredictionHead(config.hidden_size, middle_size=128, dropout=config.pred_head_dropout)

        self.init_weights()
        self.freeze_bert_partial(config.num_hidden_layers)

    def freeze_bert_partial(self, freeze_bert_firstN):
        # freeze embedding layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # freeze pooler layers (MLP for first token's embedding)
        # keep pooling layers updated
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

        # freeze first N attention layers in bert
        for i in range(freeze_bert_firstN):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('fluorescence', 'transformer')
@registry.register_task_model('stability', 'transformer')
class BertForValuePrediction(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.predict = ValuePredictionHead(config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('remote_homology', 'transformer')
class BertForSequenceClassification(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.classify = SequenceClassificationHead(
            config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]

        outputs = self.classify(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('secondary_structure', 'transformer')
class BertForSequenceToSequenceClassification(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.classify = SequenceToSequenceClassificationHead(
            config.hidden_size, config.num_labels, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.classify(sequence_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('contact_prediction', 'transformer')
class BertForContactPrediction(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.predict = PairwiseContactPredictionHead(config.hidden_size, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, protein_length, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(sequence_output, protein_length, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs

@registry.register_task_model('mutation_fitness_UNsupervise_CAGI', 'transformer')
@registry.register_task_model('mutation_fitness_UNsupervise_mutagenesis', 'transformer')
@registry.register_task_model('mutation_fitness_UNsupervise_scanning', 'transformer')
class BertForMutUnSV(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.mlm = MLMHead(
            config.hidden_size, config.vocab_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.mlm.decoder,
                                   self.bert.embeddings.word_embeddings)

    def resize_output_bias(self, new_num_tokens=None):
        """ resize bias tensor MLMHead

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the original bias module.
        Return: ``torch.nn.Parameter``
            Pointer to the resized MLMHead bias module or the old bias module if
            new_num_tokens is None
        """
        old_bias = self.mlm.bias
        #print(f'size: {old_bias.data.size()[0]}', flush=True)
        old_num_tokens = old_bias.data.size()[0]
        if new_num_tokens is None:
            return self.mlm.bias
        
        # Build new bias
        new_bias = nn.Parameter(data=torch.zeros(new_num_tokens))
        new_bias.to(old_bias.data.device)

        # Copy bias weights from the old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_bias.data[:num_tokens_to_copy] = \
            old_bias.data[:num_tokens_to_copy]
        
        self.mlm.bias = new_bias
        return self.mlm.bias

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None,
                **kwargs):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        # add hidden states and attention if they are here
        mlm_outputs = self.mlm(sequence_output, targets)
        outputs = mlm_outputs[:2] + outputs[2:] + (mlm_outputs[2],)
        # (loss), prediction_scores, (hidden_states), (attentions), aa_head_hd
        return outputs

@registry.register_task_model('structure_awareness_1d', 'transformer')
class BertForStructure1D(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.pred_head = Class1DHead(config.hidden_size, config.class_size, config.hidden_act, config.layer_norm_eps, dropout_rate=config.pred_head_dropout,ignore_index=-1)

        self.init_weights()
        self.freeze_bert_partial(config.num_hidden_layers)
    
    def freeze_bert_partial(self, freeze_bert_firstN):
        # freeze embedding layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # freeze pooler layers (MLP for first token's embedding)
        for param in self.bert.pooler.parameters():
            param.requires_grad = False

        # freeze first N attention layers in bert
        for i in range(freeze_bert_firstN):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None,
                token_type_ids=None):

        outputs = self.bert(input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

        sequence_output, pooled_output = outputs[:2]
        # add hidden states and attention if they are here
        outputs = self.pred_head(sequence_output, targets) + outputs[2:]
        # (loss_and_metrics), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('structure_awareness_2d', 'transformer')
class BertForStructure2D(BertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.pred_head = Class2DHead(config.hidden_size, config.class_size, config.hidden_act, config.layer_norm_eps, dropout_rate=0.1,ignore_index=-1)

        self.init_weights()
        self.freeze_bert_partial(config.num_hidden_layers)
    
    def freeze_bert_partial(self, freeze_bert_firstN):
        # freeze embedding layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # freeze pooler layers (MLP for first token's embedding)
        for param in self.bert.pooler.parameters():
            param.requires_grad = False

        # freeze first N attention layers in bert
        for i in range(freeze_bert_firstN):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
    
    def forward(self,
                input_ids,
                input_mask=None,
                targets=None,
                token_type_ids=None):

        outputs = self.bert(input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

        sequence_output, pooled_output = outputs[:2]
        # add hidden states and attention if they are here
        outputs = self.pred_head(sequence_output, targets) + outputs[2:]
        # (loss_and_metrics), prediction_scores, (hidden_states), (attentions)
        return outputs