from __future__ import absolute_import, division, print_function, unicode_literals
from ast import Call
from platform import node
from re import T
from tkinter.tix import Tree
from turtle import forward
from typing import Callable, Optional, Union, Tuple, Set
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv, TransformerConv
from torch_geometric.data import Batch as pyg_Batch

from .modeling_utils import get_activation_fn, glorot_orthogonal

class graph_MPNN(nn.Module):
    """ GNN module with MP options: GATv2Conv, GINEConv, TransformerConv
    """
    # Parameter initialization
    def __init__(self,
                 input_node_size: int=None,
                 input_edge_size: int=None,
                 lm_hidden_size: int=768,
                 node_feat_channels: int=128,
                 edge_feat_channels: int=128,
                 atten_heads: int=4,
                 dropout_rate: float=0.1,
                 atten_dropout_rate: float=0.1,
                 add_self_loop: bool=True,
                 mp_layer_name: str=None,
                 num_layers: int=4,
                 activ_fn: Union[str,nn.Module]=nn.GELU(),
                 residual: bool=True,
                 return_atten_weights: bool=True,
                 apply_final_node_transform: bool=True,
                 **kwargs):
        super().__init__()
        self.input_node_size = input_node_size
        self.input_edge_size = input_edge_size
        self.lm_hidden_size = lm_hidden_size
        self.node_feat_channels = node_feat_channels
        self.edge_feat_channels = edge_feat_channels
        self.atten_heads =  atten_heads
        self.dropout_rate = dropout_rate
        self.atten_dropout_rate = atten_dropout_rate
        self.add_self_loop = add_self_loop
        self.mp_layer_name = mp_layer_name
        self.num_layers = num_layers
        self.activ_fn = activ_fn
        self.residual = residual
        self.return_atten_weights = return_atten_weights
        self.apply_final_node_transform = apply_final_node_transform

        ################
        # initial module
        # transform channel dimension of node and edge features
        ################
        self.init_node_trans = nn.Linear(input_node_size, node_feat_channels, bias=False)
        self.init_edge_trans = nn.Linear(input_edge_size, edge_feat_channels, bias=False)
        
        ################
        # graph module
        # graph message passing module to update node features
        ################
        if mp_layer_name is not None:
            if mp_layer_name.lower() == 'gatv2':
                graph_modules = [
                    GATv2Conv_module(node_feat_channels=node_feat_channels,
                                    edge_feat_channels=edge_feat_channels,
                                    atten_heads=atten_heads,
                                    dropout_rate=dropout_rate,
                                    atten_dropout_rate=atten_dropout_rate,
                                    add_self_loops=add_self_loop,
                                    activ_fn=activ_fn,
                                    residual=residual,
                                    return_atten_weights=return_atten_weights)
                    for _ in range(num_layers)]
            elif mp_layer_name.lower() == 'gine':
                graph_modules = [
                    GINEConv_module(node_feat_channels=node_feat_channels,
                                    edge_feat_channels=edge_feat_channels,
                                    dropout_rate=dropout_rate,
                                    activ_fn=activ_fn,
                                    residual=residual)
                    for _ in range(num_layers)]
            elif mp_layer_name.lower() == 'transformer':
                graph_modules = [
                    TransformerConv_module(node_feat_channels=node_feat_channels,
                                        edge_feat_channels=edge_feat_channels,
                                        atten_heads=atten_heads,
                                        dropout_rate=dropout_rate,
                                        atten_dropout_rate=atten_dropout_rate,
                                        activ_fn=activ_fn,
                                        residual=residual,
                                        return_atten_weights=return_atten_weights)
                    for _ in range(num_layers)]
            self.graph_modules = nn.ModuleList(graph_modules)
        # transform node embedding size back to LM hidden size
        self.final_node_trans = nn.Linear(node_feat_channels, lm_hidden_size, bias=False)
        ## initialize linear layer weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.init_node_trans.weight, scale=scale)
        glorot_orthogonal(self.init_edge_trans.weight, scale=scale)
        glorot_orthogonal(self.final_node_trans.weight, scale=scale)
    
    def forward(self,
                graph_batch: pyg_Batch):
        # transform graph data dtype to model dtype
        graph_batch.x = graph_batch.x.to(self.init_node_trans.weight.dtype)
        graph_batch.edge_attr = graph_batch.edge_attr.to(self.init_edge_trans.weight.dtype) 
        
        node_feats = self.init_node_trans(graph_batch.x)
        edge_feats = self.init_edge_trans(graph_batch.edge_attr)
        
        # attention weight tuple, ((edge_index_layer1, attention_weights_layer1), (edge_index_layer2, attention_weights_layer2),...)
        atten_weight_tuple = () if self.return_atten_weights else None
        if self.mp_layer_name is not None:
            for mp_layer in self.graph_modules:
                if self.return_atten_weights and self.mp_layer_name.lower() in ['gatv2','transformer']:
                    node_feats, atten_weights = mp_layer(node_feats=node_feats,edge_index=graph_batch.edge_index,edge_attr=edge_feats)
                    atten_weight_tuple = atten_weight_tuple + (atten_weights,)
                else:
                    node_feats = mp_layer(node_feats=node_feats,edge_index=graph_batch.edge_index,edge_attr=edge_feats)
        
        if self.apply_final_node_transform:
            node_feats = self.final_node_trans(node_feats)
        graph_batch.x = node_feats
        if self.return_atten_weights:
            return graph_batch, atten_weight_tuple
        else:
            return graph_batch

class GATv2Conv_module(nn.Module):
    """ one GNN module with GATv2Conv
    """
    # Parameter initialization
    def __init__(self,
                 node_feat_channels: int,
                 edge_feat_channels: int,
                 atten_heads: int=4,
                 dropout_rate: float=0.1,
                 atten_dropout_rate: float=0.1,
                 add_self_loops: bool=True,
                 activ_fn: Union[str,nn.Module]=nn.GELU(),
                 residual: bool=True,
                 return_atten_weights: bool=True,
                 **kwargs):
        super().__init__()
        self.node_feat_channels = node_feat_channels
        self.edge_feat_channels = edge_feat_channels
        self.atten_heads = atten_heads
        self.dropout_rate = dropout_rate
        self.atten_dropout_rate = atten_dropout_rate
        self.add_self_loops = add_self_loops
        self.residual = residual
        self.return_atten_weights = return_atten_weights
        if isinstance(activ_fn, str):
            self.activ_fn = get_activation_fn(activ_fn,nnModule=True)
        else:
            self.activ_fn = activ_fn

        ## first normalization layer
        self.norm_node_feats_layer1 = nn.LayerNorm(self.node_feat_channels)
        ## gat layer
        self.gat_layer = GATv2Conv(in_channels=self.node_feat_channels,
                                   out_channels=self.node_feat_channels // self.atten_heads,
                                   heads=self.atten_heads,
                                   concat=True,
                                   dropout=self.atten_dropout_rate,
                                   add_self_loops=self.add_self_loops,
                                   edge_dim=self.edge_feat_channels,
                                   fill_value='mean')
        ## dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        ## second normalization layer
        self.norm_node_feats_layer2 = nn.LayerNorm(self.node_feat_channels)
        ## MLP
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.node_feat_channels, self.node_feat_channels * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.node_feat_channels * 2, self.node_feat_channels, bias=False)
        ])

    def forward(self,
                node_feats: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor):
        """a forward pass of one GAT module
        
        gat_layer->dropout->add&norm->MLP(dropout)->add&norm
        """
        node_feats_cache1 = node_feats
        ## gat layer and dropout
        atten_weight_tuple = None # (edge_index [2,|edge|], attention_weights [|edge|, H])
        if self.return_atten_weights:
            node_feats, atten_weight_tuple = self.gat_layer(node_feats,edge_index,edge_attr,self.return_atten_weights)
        else:
            node_feats = self.gat_layer(node_feats,edge_index,edge_attr,self.return_atten_weights)
        node_feats = self.dropout(node_feats)
        ## first round of add and normalization
        if self.residual:
            node_feats = node_feats_cache1 + node_feats
        node_feats = self.norm_node_feats_layer1(node_feats)
        node_feats_cache2 = node_feats
        ## MLP layers
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        ## second round of add and normalization
        if self.residual:
            node_feats = node_feats_cache2 + node_feats
        node_feats = self.norm_node_feats_layer2(node_feats)
        
        if self.return_atten_weights:
            return node_feats, atten_weight_tuple
        else:
            return node_feats

class GINEConv_module(nn.Module):
    """ one GNN module with GINEConv
    """
    # Parameter initialization
    def __init__(self,
                 node_feat_channels: int,
                 edge_feat_channels: int,
                 dropout_rate: float=0.1,
                 activ_fn: Union[str,nn.Module]=nn.GELU(),
                 residual: bool=True,
                 **kwargs):
        super().__init__()
        self.node_feat_channels = node_feat_channels
        self.edge_feat_channels = edge_feat_channels
        self.dropout_rate = dropout_rate
        self.residual = residual
        if isinstance(activ_fn, str):
            self.activ_fn = get_activation_fn(activ_fn,nnModule=True)
        else:
            self.activ_fn = activ_fn
        ## first normalization layer
        self.norm_node_feats_layer1 = nn.LayerNorm(self.node_feat_channels)
        ## NN for GINE
        gine_nn = nn.Sequential(
                    nn.Linear(self.node_feat_channels, 2 * self.node_feat_channels),
                    nn.LayerNorm(2 * self.node_feat_channels),
                    self.activ_fn,
                    nn.Linear(2 * self.node_feat_channels, self.node_feat_channels),
                    nn.LayerNorm(self.node_feat_channels),
                    self.activ_fn,)
        ## gat layer
        self.gine_layer = GINEConv(gine_nn,train_eps=True,edge_dim=self.edge_feat_channels)
        ## dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        ## second normalization layer
        self.norm_node_feats_layer2 = nn.LayerNorm(self.node_feat_channels)
        ## MLP
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.node_feat_channels, self.node_feat_channels * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.node_feat_channels * 2, self.node_feat_channels, bias=False)
        ])

    def forward(self,
                node_feats: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor):
        """a forward pass of one GAT module
        
        gin_layer->dropout->add&norm->MLP(dropout)->add&norm
        """
        node_feats_cache1 = node_feats
        ## gat layer and dropout
        node_feats = self.gine_layer(node_feats,edge_index,edge_attr)
        node_feats = self.dropout(node_feats)
        ## first round of add and normalization
        if self.residual:
            node_feats = node_feats_cache1 + node_feats
        node_feats = self.norm_node_feats_layer1(node_feats)
        node_feats_cache2 = node_feats
        ## MLP layers
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        ## second round of add and normalization
        if self.residual:
            node_feats = node_feats_cache2 + node_feats
        node_feats = self.norm_node_feats_layer2(node_feats)

        return node_feats        

class TransformerConv_module(nn.Module):
    """ one GNN module with TransformerConv
    """
    # Parameter initialization
    def __init__(self,
                 node_feat_channels: int,
                 edge_feat_channels: int,
                 atten_heads: int=4,
                 dropout_rate: float=0.1,
                 atten_dropout_rate: float=0.1,
                 activ_fn: Union[str,nn.Module]=nn.GELU(),
                 residual: bool=True,
                 return_atten_weights: bool=True,
                 **kwargs):
        super().__init__()
        self.node_feat_channels = node_feat_channels
        self.edge_feat_channels = edge_feat_channels
        self.atten_heads = atten_heads
        self.dropout_rate = dropout_rate
        self.atten_dropout_rate = atten_dropout_rate
        self.residual = residual
        self.return_atten_weights = return_atten_weights
        if isinstance(activ_fn, str):
            self.activ_fn = get_activation_fn(activ_fn,nnModule=True)
        else:
            self.activ_fn = activ_fn

        ## first normalization layer
        self.norm_node_feats_layer1 = nn.LayerNorm(self.node_feat_channels)
        ## gat layer
        self.transfConv_layer = TransformerConv(in_channels=self.node_feat_channels,
                                        out_channels=self.node_feat_channels // self.atten_heads,
                                        heads=self.atten_heads,
                                        concat=True,
                                        beta=True,
                                        dropout=self.atten_dropout_rate,
                                        edge_dim=self.edge_feat_channels)
        ## dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        ## second normalization layer
        self.norm_node_feats_layer2 = nn.LayerNorm(self.node_feat_channels)
        ## MLP
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.node_feat_channels, self.node_feat_channels * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.node_feat_channels * 2, self.node_feat_channels, bias=False)
        ])

    def forward(self,
                node_feats: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor):
        """a forward pass of one GAT module
        
        gat_layer->dropout->add&norm->MLP(dropout)->add&norm
        """
        node_feats_cache1 = node_feats
        ## gat layer and dropout
        ## atten_weight_tuple: (edge_index [2,|edge|], attention_weights [|edge|,H])
        atten_weight_tuple = None
        if self.return_atten_weights:
            node_feats, atten_weight_tuple = self.transfConv_layer(node_feats,edge_index,edge_attr,self.return_atten_weights)
        else:
            node_feats = self.transfConv_layer(node_feats,edge_index,edge_attr,self.return_atten_weights)
        node_feats = self.dropout(node_feats)
        ## first round of add and normalization
        if self.residual:
            node_feats = node_feats_cache1 + node_feats
        node_feats = self.norm_node_feats_layer1(node_feats)
        node_feats_cache2 = node_feats
        ## MLP layers
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        ## second round of add and normalization
        if self.residual:
            node_feats = node_feats_cache2 + node_feats
        node_feats = self.norm_node_feats_layer2(node_feats)
        
        if self.return_atten_weights:
            return node_feats, atten_weight_tuple
        else:
            return node_feats
