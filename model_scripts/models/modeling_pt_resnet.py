import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from .modeling_utils import get_activation_fn, glorot_orthogonal


class SEBlock(nn.Module):
    """A squeeze-and-excitation block for PyTorch."""

    def __init__(self, ch, ratio=16):
        super().__init__()
        self.ratio = ratio
        self.linear1 = nn.Linear(ch, ch // ratio)
        self.linear2 = nn.Linear(ch // ratio, ch)
        self.act = nn.ReLU()

    def forward(self, in_block):
        x = torch.reshape(in_block, (in_block.shape[0], in_block.shape[1], -1))
        x = torch.mean(x, dim=-1)
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = torch.sigmoid(x)
        if len(in_block.size()) == 4:
            return torch.einsum('bcij,bc->bcij', in_block, x)
        elif len(in_block.size()) == 3:
            return torch.einsum('bci,bc->bci', in_block, x)


class ResNet2D(nn.Module):
    """A custom ResNet module for PyTorch."""

    # Parameter initialization
    def __init__(self,
                 num_channels,
                 num_chunks,
                 module_name,
                 activ_fn=F.elu,
                 inorm=False,
                 init_projection=False,
                 extra_blocks=False,
                 dilation_cycle=None):
        super().__init__()
        self.num_channel = num_channels
        self.num_chunks = num_chunks
        self.module_name = module_name
        self.init_projection = init_projection
        self.activ_fn = activ_fn
        self.inorm = inorm
        self.extra_blocks = extra_blocks
        self.dilation_cycle = [1, 2, 4, 8] if dilation_cycle is None else dilation_cycle

        if self.init_projection:
            self.add_module(f'resnet_{self.module_name}_init_proj', nn.Conv2d(in_channels=num_channels,
                                                                           out_channels=num_channels,
                                                                           kernel_size=(1, 1)))

        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                if self.inorm:
                    self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_1',
                                    nn.InstanceNorm2d(num_channels, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_2',
                                    nn.InstanceNorm2d(num_channels // 2, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_3',
                                    nn.InstanceNorm2d(num_channels // 2, eps=1e-06, affine=True))

                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_1',
                                nn.Conv2d(num_channels, num_channels // 2, kernel_size=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_2',
                                nn.Conv2d(num_channels // 2,
                                          num_channels // 2,
                                          kernel_size=(3, 3),
                                          dilation=dilation_rate,
                                          padding=dilation_rate))
                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_3',
                                nn.Conv2d(num_channels // 2, num_channels, kernel_size=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_se_block',
                                SEBlock(num_channels, ratio=16))

        if self.extra_blocks:
            for i in range(2):
                if self.inorm:
                    self.add_module(f'resnet_{self.module_name}_extra{i}_inorm_1',
                                    nn.InstanceNorm2d(num_channels, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_extra{i}_inorm_2',
                                    nn.InstanceNorm2d(num_channels // 2, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_extra{i}_inorm_3',
                                    nn.InstanceNorm2d(num_channels // 2, eps=1e-06, affine=True))

                self.add_module(f'resnet_{self.module_name}_extra{i}_conv2d_1',
                                nn.Conv2d(num_channels, num_channels // 2, kernel_size=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_extra{i}_conv2d_2',
                                nn.Conv2d(num_channels // 2,
                                          num_channels // 2,
                                          kernel_size=(3, 3),
                                          dilation=(1, 1),
                                          padding=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_extra{i}_conv2d_3',
                                nn.Conv2d(num_channels // 2, num_channels, kernel_size=(1, 1)))
                self.add_module(f'resnet_{self.module_name}_extra{i}_se_block',
                                SEBlock(num_channels, ratio=16))

    def forward(self, x):
        """Compute ResNet output."""
        activ_fn = self.activ_fn

        if self.init_projection:
            x = self._modules[f'resnet_{self.module_name}_init_proj'](x)

        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                _residual = x

                # Internal block
                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_1'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_1'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_2'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_2'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_3'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_conv2d_3'](x)

                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_se_block'](x)

                x = x + _residual

        if self.extra_blocks:
            for i in range(2):
                _residual = x

                # Internal block
                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_extra{i}_inorm_1'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_extra{i}_conv2d_1'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_extra{i}_inorm_2'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_extra{i}_conv2d_2'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_extra{i}_inorm_3'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_extra{i}_conv2d_3'](x)

                x = self._modules[f'resnet_{self.module_name}_extra{i}_se_block'](x)

                x = x + _residual

        return x

class MultiHeadRegionalAttention2D(nn.Module):
    """A multi-head attention block for PyTorch that operates regionally."""

    @staticmethod
    def get_stretch_weight(s):
        w = np.zeros((s * s, 1, 1, s, s))
        for i in range(s):           
            for j in range(s):
                w[s * i + j, 0, 0, i, j] = 1
        return np.asarray(w).astype(np.float32)

    def __init__(self, in_dim=3, region_size=3, d_k=16, d_v=32, n_head=4, att_drop=0.1, output_score=False):
        super().__init__()
        self.temper = int(np.sqrt(d_k))
        self.dk_per_head = d_k // n_head
        self.dv_per_head = d_v // n_head
        self.dropout_layer = nn.Dropout(att_drop)
        self.output_score = output_score
        self.q_layer = nn.Conv2d(in_dim, d_k, kernel_size=(1, 1), bias=False)
        self.k_layer = nn.Conv2d(in_dim, d_k, kernel_size=(1, 1), bias=False)
        self.v_layer = nn.Conv2d(in_dim, d_v, kernel_size=(1, 1), bias=False)
        self.softmax_layer = nn.Softmax(1)
        self.stretch_layer = nn.Conv3d(in_channels=1,
                                       out_channels=region_size * region_size,
                                       kernel_size=(1, region_size, region_size),
                                       bias=False,
                                       padding=(0, 1, 1))
        self.stretch_layer.weight = nn.Parameter(
            torch.tensor(self.get_stretch_weight(region_size)), requires_grad=False
        )

    def forward(self, x):
        """Compute attention output and attention score."""
        Q = self.stretch_layer(self.q_layer(x).unsqueeze(1))
        K = self.stretch_layer(self.k_layer(x).unsqueeze(1))
        V = self.stretch_layer(self.v_layer(x).unsqueeze(1))
        qk = torch.mul(Q, K).permute(0, 2, 1, 3, 4)
        qk1 = qk.reshape((-1, self.dk_per_head, qk.shape[2], qk.shape[3], qk.shape[4]))
        attention_score = self.softmax_layer(torch.div(torch.sum(qk1, 1), self.temper))
        attention_score2 = self.dropout_layer(attention_score)
        attention_score2 = torch.repeat_interleave(attention_score2.unsqueeze(0).permute(0, 2, 1, 3, 4),
                                                   repeats=self.dv_per_head, dim=2)
        attention_out = torch.sum(torch.mul(attention_score2, V), dim=1)
        return attention_out, attention_score if self.output_score else attention_out


class ResNet2DInputWithOptAttention(nn.Module):
    """A ResNet and (optionally) regionally-attentive convolution module for a pair of 2D feature tensors."""

    def __init__(self,
                 num_chunks=4,
                 init_channels=128,
                 num_channels=128,
                 use_region_atten=False,
                 n_head=4,
                 activ_fn=F.elu,
                 dropout=0.1):
        super().__init__()
        self.num_chunks = num_chunks
        self.init_channels = init_channels
        self.num_channels = num_channels
        self.use_region_atten = use_region_atten
        self.n_head = n_head
        if isinstance(activ_fn, str):
            self.activ_fn = get_activation_fn(activ_fn)
        else:
            self.activ_fn = activ_fn
        self.dropout = dropout

        self.add_module('conv2d_1', nn.Conv2d(in_channels=self.init_channels,
                                              out_channels=self.num_channels,
                                              kernel_size=(1, 1),
                                              padding=(0, 0)))
        self.add_module('inorm_1', nn.InstanceNorm2d(self.num_channels, eps=1e-06, affine=True))
        self.add_module('base_resnet', ResNet2D(num_channels,
                                              self.num_chunks,
                                              module_name='base_resnet2D',
                                              activ_fn=self.activ_fn,
                                              inorm=True,
                                              init_projection=True,
                                              extra_blocks=False))
        if self.use_region_atten:
            self.add_module('MHA2D_1', MultiHeadRegionalAttention2D(in_dim=self.num_channels,
                                                                  d_v=self.num_channels,
                                                                  n_head=self.n_head,
                                                                  att_drop=self.dropout,
                                                                  output_score=True))

        # Reset learnable parameters
        #self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        # Reinitialize final output layer
        final_layer_bias = self._modules['phase2_conv'].bias.clone()
        final_layer_bias[1] = -7.0  # -7 chosen as the second term's bias s.t. positives are predicted with prob=0.001
        self._modules['phase2_conv'].bias = nn.Parameter(final_layer_bias, requires_grad=True)

    def forward(self, f2d_tile: torch.Tensor):
        """Compute final convolution output."""
        activ_fun = self.activ_fn
        out_conv2d_1 = self._modules['conv2d_1'](f2d_tile)
        out_inorm_1 = activ_fun(self._modules['inorm_1'](out_conv2d_1))

        # First ResNet
        out_base_resnet = activ_fun(self._modules['base_resnet'](out_inorm_1))
        if self.use_region_atten:
            out_base_resnet, attention_scores_1 = self._modules['MHA2D_1'](out_base_resnet)
            out_base_resnet = activ_fun(out_base_resnet)

        return out_base_resnet


class ResNet1D(nn.Module):
    """A custom ResNet module (1D) for PyTorch."""

    # Parameter initialization
    def __init__(self,
                 num_channels,
                 num_chunks,
                 module_name,
                 activ_fn=F.elu,
                 inorm=False,
                 init_projection=False,
                 extra_blocks=False,
                 dilation_cycle=None):
        super().__init__()
        self.num_channel = num_channels
        self.num_chunks = num_chunks
        self.module_name = module_name
        self.init_projection = init_projection
        if isinstance(activ_fn, str):
            self.activ_fn = get_activation_fn(activ_fn)
        else:
            self.activ_fn = activ_fn
        self.inorm = inorm
        self.extra_blocks = extra_blocks
        self.dilation_cycle = [1, 2, 4, 8] if dilation_cycle is None else dilation_cycle

        if self.init_projection:
            self.add_module(f'resnet_{self.module_name}_init_proj', nn.Conv1d(in_channels=num_channels,
                                                                              out_channels=num_channels,
                                                                              kernel_size=1))

        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                if self.inorm:
                    self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_1',
                                    nn.InstanceNorm1d(num_channels, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_2',
                                    nn.InstanceNorm1d(num_channels // 2, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_3',
                                    nn.InstanceNorm1d(num_channels // 2, eps=1e-06, affine=True))

                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_conv1d_1',
                                nn.Conv1d(num_channels, num_channels // 2, kernel_size=1))
                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_conv1d_2',
                                nn.Conv1d(num_channels // 2,
                                          num_channels // 2,
                                          kernel_size=3,
                                          dilation=dilation_rate,
                                          padding=dilation_rate))
                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_conv1d_3',
                                nn.Conv1d(num_channels // 2, num_channels, kernel_size=1))
                self.add_module(f'resnet_{self.module_name}_{i}_{dilation_rate}_se_block',
                                SEBlock(num_channels, ratio=16))

        if self.extra_blocks:
            for i in range(2):
                if self.inorm:
                    self.add_module(f'resnet_{self.module_name}_extra{i}_inorm_1',
                                    nn.InstanceNorm1d(num_channels, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_extra{i}_inorm_2',
                                    nn.InstanceNorm1d(num_channels // 2, eps=1e-06, affine=True))
                    self.add_module(f'resnet_{self.module_name}_extra{i}_inorm_3',
                                    nn.InstanceNorm1d(num_channels // 2, eps=1e-06, affine=True))

                self.add_module(f'resnet_{self.module_name}_extra{i}_conv1d_1',
                                nn.Conv1d(num_channels, num_channels // 2, kernel_size=1))
                self.add_module(f'resnet_{self.module_name}_extra{i}_conv1d_2',
                                nn.Conv1d(num_channels // 2,
                                          num_channels // 2,
                                          kernel_size=3,
                                          dilation=1,
                                          padding=1))
                self.add_module(f'resnet_{self.module_name}_extra{i}_conv1d_3',
                                nn.Conv1d(num_channels // 2, num_channels, kernel_size=1))
                self.add_module(f'resnet_{self.module_name}_extra{i}_se_block',
                                SEBlock(num_channels, ratio=16))

    def forward(self, x):
        """Compute ResNet output."""
        activ_fn = self.activ_fn

        if self.init_projection:
            x = self._modules[f'resnet_{self.module_name}_init_proj'](x)

        for i in range(self.num_chunks):
            for dilation_rate in self.dilation_cycle:
                _residual = x

                # Internal block
                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_1'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_conv1d_1'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_2'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_conv1d_2'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_inorm_3'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_conv1d_3'](x)

                x = self._modules[f'resnet_{self.module_name}_{i}_{dilation_rate}_se_block'](x)

                x = x + _residual

        if self.extra_blocks:
            for i in range(2):
                _residual = x

                # Internal block
                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_extra{i}_inorm_1'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_extra{i}_conv1d_1'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_extra{i}_inorm_2'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_extra{i}_conv1d_2'](x)

                if self.inorm:
                    x = self._modules[f'resnet_{self.module_name}_extra{i}_inorm_3'](x)
                x = activ_fn(x)
                x = self._modules[f'resnet_{self.module_name}_extra{i}_conv1d_3'](x)

                x = self._modules[f'resnet_{self.module_name}_extra{i}_se_block'](x)

                x = x + _residual

        return x

class ResNet1DInputWithOptAttention(nn.Module):
    """A ResNet and (optionally) regionally-attentive convolution module for a pair of 2D feature tensors."""

    def __init__(self,
                 num_chunks=4,
                 init_channels=128,
                 num_channels=128,
                 num_interact_transf_atten = 4,
                 seq_max_length = 1000,
                 use_region_atten=False,
                 n_head=4,
                 activ_fn=F.elu,
                 dropout=0.1,
                 initializer_range=0.02):
        super().__init__()
        self.num_chunks = num_chunks
        self.init_channels = init_channels
        self.num_channels = num_channels
        self.num_interact_transf_atten = num_interact_transf_atten
        self.seq_max_length = seq_max_length
        self.use_region_atten = use_region_atten
        self.n_head = n_head
        if isinstance(activ_fn, str):
            self.activ_fn = get_activation_fn(activ_fn)
        else:
            self.activ_fn = activ_fn
        self.dropout = dropout
        self.initializer_range = initializer_range

        self.add_module('conv2d_reduct', nn.Conv2d(in_channels=self.init_channels,
                                                out_channels=self.num_channels // self.num_interact_transf_atten,
                                                kernel_size=(1,1),
                                                padding=(0,0)))
        self.transf_atten_weights = nn.Parameter(data=torch.ones(self.num_interact_transf_atten,self.seq_max_length,self.seq_max_length))
        self.transf_atten_weights.data.normal_(mean=1.0,std=self.initializer_range)
        #self.reset_parameters()
        self.transf_softmax = nn.Softmax(dim=2)

        self.add_module('inorm_reduct', nn.InstanceNorm1d(self.num_channels, eps=1e-06, affine=True))

        self.add_module('base_resnet', ResNet1D(num_channels,
                                                self.num_chunks,
                                                module_name='base_resnet1D',
                                                activ_fn=self.activ_fn,
                                                inorm=True,
                                                init_projection=True,
                                                extra_blocks=False))
        if self.use_region_atten:
            self.add_module('MHA1D_1', MultiHeadRegionalAttention1D(in_dim=self.num_channels,
                                                                    d_v=self.num_channels,
                                                                    n_head=self.n_head,
                                                                    att_drop=self.dropout,
                                                                    output_score=True))

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.transf_atten_weights, scale=scale)
    
    def forward(self, f2d_tile: torch.Tensor, aa_mask: torch.Tensor):
        """Convert pairwise embeddings to position-wise embeddings, then do 1D convolution
        Args:
            f2d_tile: intra-protein pairwise stacked embeddings, size [batch_size, C, L, L]
        """
        activ_fun = self.activ_fn
        out_conv2d_reduct = self._modules['conv2d_reduct'](f2d_tile)
        out_conv2d_reduct = out_conv2d_reduct.to(self.transf_atten_weights.dtype)
        seq_padded_length = out_conv2d_reduct.size()[-1]
        
        # apply 2d mask for feature map
        extend_aa_mask = aa_mask.unsqueeze(1).unsqueeze(2) # [bs,1,1,L]
        out_conv2d_reduct = torch.mul(out_conv2d_reduct,extend_aa_mask)

        # attention masking for pading positions is not applicable here
        # because learnable 2D to 1D transfer attention matrix is the same across examples in mini batch
        # so just use the sub-matrix for multiplication with 2D feature matrix 
        out_conv1d = torch.einsum('bcij,hij->bhci',out_conv2d_reduct,self.transf_softmax(self.transf_atten_weights[:,:seq_padded_length,:seq_padded_length]))
        out_conv1d = out_conv1d.to(f2d_tile.dtype)
        out_conv1d_concate = out_conv1d.reshape(out_conv1d.size()[0],-1,out_conv1d.size()[3]) # size: 'b(h*c)l'
        out_inorm_1 = activ_fun(self._modules['inorm_reduct'](out_conv1d_concate))

        # First ResNet
        out_base_resnet = activ_fun(self._modules['base_resnet'](out_inorm_1))
        if self.use_region_atten:
            out_base_resnet, attention_scores_1 = self._modules['MHA1D_1'](out_base_resnet)
            out_base_resnet = activ_fun(out_base_resnet)

        return out_base_resnet

class MultiHeadRegionalAttention1D(nn.Module):
    """A multi-head attention block for PyTorch that operates regionally."""
    @staticmethod
    def get_stretch_weight(s):
        w = np.zeros((s * s, 1, s, s))
        for i in range(s):           
            for j in range(s):
                w[s * i + j, 0, i, j] = 1
        return np.asarray(w).astype(np.float32)

    def __init__(self, in_dim=3, region_size=3, d_k=16, d_v=32, n_head=4, att_drop=0.1, output_score=False):
        super().__init__()
        self.temper = int(np.sqrt(d_k))
        self.dk_per_head = d_k // n_head
        self.dv_per_head = d_v // n_head
        self.dropout_layer = nn.Dropout(att_drop)
        self.output_score = output_score
        self.q_layer = nn.Conv1d(in_dim, d_k, kernel_size=1, bias=False)
        self.k_layer = nn.Conv1d(in_dim, d_k, kernel_size=1, bias=False)
        self.v_layer = nn.Conv1d(in_dim, d_v, kernel_size=1, bias=False)
        self.softmax_layer = nn.Softmax(1)
        self.stretch_layer = nn.Conv2d(in_channels=1,
                                       out_channels=region_size * region_size,
                                       kernel_size=(1, region_size),
                                       bias=False,
                                       padding=(0, 1))
        self.stretch_layer.weight = nn.Parameter(
            torch.tensor(self.get_stretch_weight(region_size)), requires_grad=False
        )

    def forward(self, x):
        """Compute attention output and attention score."""
        Q = self.stretch_layer(self.q_layer(x).unsqueeze(1))
        K = self.stretch_layer(self.k_layer(x).unsqueeze(1))
        V = self.stretch_layer(self.v_layer(x).unsqueeze(1))
        qk = torch.mul(Q, K).permute(0, 2, 1, 3)
        qk1 = qk.reshape((-1, self.dk_per_head, qk.shape[2], qk.shape[3]))
        attention_score = self.softmax_layer(torch.div(torch.sum(qk1, 1), self.temper))
        attention_score2 = self.dropout_layer(attention_score)
        attention_score2 = torch.repeat_interleave(attention_score2.unsqueeze(0).permute(0, 2, 1, 3),
                                                   repeats=self.dv_per_head, dim=2)
        attention_out = torch.sum(torch.mul(attention_score2, V), dim=1)
        return attention_out, attention_score if self.output_score else attention_out


class structureMultiTask_head(nn.Module):
    """prediction heads for multiple tasks    
    """
    def __init__(self,
                 num_chunks=1,
                 num_channels=128,
                 num_aa_classes=29,
                 num_ss_classes=3,
                 num_rsa_classes=2,
                 num_dist_classes=32,
                 use_region_atten=False,
                 n_head_region_atten=4,
                 activ_fn=F.elu,
                 dropout=0.1,
                 ignore_index=-1,
                 aa_loss_weight=1.0,
                 ss_loss_weight=1.0,
                 rsa_loss_weight=1.0,
                 dist_loss_weight=1.0,
                 use_class_weights: bool=False):
        super().__init__()
        self.num_chunk = num_chunks
        self.num_channels = num_channels
        self.num_aa_classes = num_aa_classes
        self.num_ss_classes = num_ss_classes
        self.num_rsa_classes = num_rsa_classes
        self.num_dist_classes = num_dist_classes
        self.aa_loss_weight = aa_loss_weight
        self.ss_loss_weight = ss_loss_weight
        self.rsa_loss_weight = rsa_loss_weight
        self.dist_loss_weight = dist_loss_weight
        self.use_region_atten = use_region_atten
        self.n_head_region_atten = n_head_region_atten
        if isinstance(activ_fn, str):
            self.activ_fn = get_activation_fn(activ_fn)
        else:
            self.activ_fn = activ_fn
        self.dropout = dropout
        self.ignore_index = ignore_index
        self.use_class_weights = use_class_weights
        ## AA head
        self.add_module('aa_bin_resnet', ResNet1D(num_channels,
                                                  num_chunks=num_chunks,
                                                  module_name='aa_bin_resnet',
                                                  activ_fn=self.activ_fn,
                                                  inorm=True,
                                                  init_projection=False,
                                                  extra_blocks=True))
        self.add_module('aa_bin_conv', nn.Conv1d(in_channels=self.num_channels,
                                                 out_channels=self.num_aa_classes,
                                                 kernel_size=1,
                                                 padding=0))
        if self.use_region_atten:
            self.add_module('aa_bin_MHA1D', MultiHeadRegionalAttention1D(
                                                in_dim=self.num_channels,
                                                d_v=self.num_channels,
                                                n_head=self.n_head_region_atten,
                                                att_drop=self.dropout,
                                                output_score=True))
        ## SS head
        self.add_module('ss_bin_resnet', ResNet1D(num_channels,
                                                  num_chunks=num_chunks,
                                                  module_name='ss_bin_resnet',
                                                  activ_fn=self.activ_fn,
                                                  inorm=True,
                                                  init_projection=False,
                                                  extra_blocks=True))
        self.add_module('ss_bin_conv', nn.Conv1d(in_channels=self.num_channels,
                                                 out_channels=self.num_ss_classes,
                                                 kernel_size=1,
                                                 padding=0))
        if self.use_region_atten:
            self.add_module('ss_bin_MHA1D', MultiHeadRegionalAttention1D(
                                                in_dim=self.num_channels,
                                                d_v=self.num_channels,
                                                n_head=self.n_head_region_atten,
                                                att_drop=self.dropout,
                                                output_score=True))
        ## RSA head
        self.add_module('rsa_bin_resnet', ResNet1D(num_channels,
                                                   num_chunks=num_chunks,
                                                   module_name='rsa_bin_resnet',
                                                   activ_fn=self.activ_fn,
                                                   inorm=True,
                                                   init_projection=False,
                                                   extra_blocks=True))
        self.add_module('rsa_bin_conv', nn.Conv1d(in_channels=self.num_channels,
                                                 out_channels=self.num_rsa_classes,
                                                 kernel_size=1,
                                                 padding=0))
        if self.use_region_atten:
            self.add_module('rsa_bin_MHA1D', MultiHeadRegionalAttention1D(
                                                in_dim=self.num_channels,
                                                d_v=self.num_channels,
                                                n_head=self.n_head_region_atten,
                                                att_drop=self.dropout,
                                                output_score=True))
        ## Dist head
        self.add_module('dist_bin_resnet', ResNet2D(num_channels,
                                                    num_chunks=num_chunks,
                                                    module_name='dist_bin_resnet',
                                                    activ_fn=self.activ_fn,
                                                    inorm=True,
                                                    init_projection=False,
                                                    extra_blocks=True))
        self.add_module('dist_bin_conv', nn.Conv2d(in_channels=self.num_channels,
                                                   out_channels=self.num_dist_classes,
                                                   kernel_size=(1, 1),
                                                   padding=(0, 0)))
        if self.use_region_atten:
            self.add_module('dist_bin_MHA2D', MultiHeadRegionalAttention2D(
                                                in_dim=self.num_channels,
                                                d_v=self.num_channels,
                                                n_head=self.n_head_region_atten,
                                                att_drop=self.dropout,
                                                output_score=True))

    def get_class_weight(self, target_inputs: torch.Tensor, class_num: int):
        total_num = target_inputs.reshape(-1).ge(0).sum().to('cpu').to(torch.float)
        bin_count = torch.bincount(torch.masked_select(target_inputs,target_inputs.ge(0))).to('cpu').to(torch.float)
        bin_miss = class_num - bin_count.size(0)
        if bin_miss > 0:
            bin_count = torch.cat((bin_count,torch.zeros(bin_miss).to('cpu')))
        bin_count = torch.where(bin_count == 0.,total_num,bin_count)
        class_weights = torch.reciprocal(bin_count)
        return class_weights

    def forward(self,
                fea1d,
                fea2d,
                targets_aa,
                targets_ss,
                targets_rsa,
                targets_dist):
        """Compute final convolution output."""
        activ_fun = self.activ_fn

        # aa bin ResNet
        aa_resnet_out = activ_fun(self._modules['aa_bin_resnet'](fea1d))
        if self.use_region_atten:
            aa_resnet_out, aa_attention_scores = self._modules['aa_bin_MHA1D'](aa_resnet_out)
            aa_resnet_out = activ_fun(aa_resnet_out)
        # aa logits convolution
        aa_logit_out = self._modules['aa_bin_conv'](aa_resnet_out)

        # ss bin ResNet
        ss_resnet_out = activ_fun(self._modules['ss_bin_resnet'](fea1d))
        if self.use_region_atten:
            ss_resnet_out, ss_attention_scores = self._modules['ss_bin_MHA1D'](ss_resnet_out)
            ss_resnet_out = activ_fun(ss_resnet_out)
        # ss logits convolution
        ss_logit_out = self._modules['ss_bin_conv'](ss_resnet_out)

        # rsa bin ResNet
        rsa_resnet_out = activ_fun(self._modules['rsa_bin_resnet'](fea1d))
        if self.use_region_atten:
            rsa_resnet_out, rsa_attention_scores = self._modules['rsa_bin_MHA1D'](rsa_resnet_out)
            rsa_resnet_out = activ_fun(rsa_resnet_out)
        # rsa logits convolution
        rsa_logit_out = self._modules['rsa_bin_conv'](rsa_resnet_out)

        # dist bin ResNet
        dist_resnet_out = activ_fun(self._modules['dist_bin_resnet'](fea2d))
        if self.use_region_atten:
            dist_resnet_out, dist_attention_scores = self._modules['dist_bin_MHA2D'](dist_resnet_out)
            dist_resnet_out = activ_fun(dist_resnet_out)
        # dist logits convolution
        dist_logit_out = self._modules['dist_bin_conv'](dist_resnet_out)

        aa_class_weight = None
        ss_class_weight = None
        rsa_class_weight = None
        dist_class_weight = None

        ## loss terms
        metrics = {}
        loss_sum = 0.
        if targets_aa is not None:
            if self.use_class_weights:
                aa_class_weight = self.get_class_weight(targets_aa,self.num_aa_classes).to(aa_logit_out)
            aa_loss_fct = nn.CrossEntropyLoss(weight=aa_class_weight,ignore_index=self.ignore_index)
            aa_masked_lm_loss = aa_loss_fct(
                aa_logit_out.permute(0,2,1).reshape(-1, self.num_aa_classes), targets_aa.view(-1))
            metrics['aa_ppl'] = torch.exp(aa_masked_lm_loss)
            loss_sum += self.aa_loss_weight*aa_masked_lm_loss
        if targets_ss is not None:
            if self.use_class_weights:
                ss_class_weight = self.get_class_weight(targets_ss,self.num_ss_classes).to(ss_logit_out)
            ss_loss_fct = nn.CrossEntropyLoss(weight=ss_class_weight, ignore_index=self.ignore_index)
            ss_ce_loss = ss_loss_fct(
                ss_logit_out.permute(0,2,1).reshape(-1, self.num_ss_classes), targets_ss.view(-1))
            metrics['ss_ppl'] = torch.exp(ss_ce_loss)
            loss_sum += self.ss_loss_weight*ss_ce_loss
        if targets_rsa is not None:
            if self.use_class_weights:
                rsa_class_weight = self.get_class_weight(targets_rsa,self.num_rsa_classes).to(rsa_logit_out)
            rsa_loss_fct = nn.CrossEntropyLoss(weight=rsa_class_weight, ignore_index=self.ignore_index)
            rsa_ce_loss = rsa_loss_fct(
                rsa_logit_out.permute(0,2,1).reshape(-1, self.num_rsa_classes), targets_rsa.view(-1))
            metrics['rsa_ppl'] = torch.exp(rsa_ce_loss)
            loss_sum += self.rsa_loss_weight*rsa_ce_loss
        if targets_dist is not None:
            if self.use_class_weights:
                dist_class_weight = self.get_class_weight(targets_dist,self.num_dist_classes).to(dist_logit_out)
            dist_loss_fct = nn.CrossEntropyLoss(weight=dist_class_weight, ignore_index=self.ignore_index)
            dist_ce_loss = dist_loss_fct(
                dist_logit_out.permute(0,2,3,1).reshape(-1, self.num_dist_classes), targets_dist.view(-1))
            metrics['dist_ppl'] = torch.exp(dist_ce_loss)
            loss_sum += self.dist_loss_weight*dist_ce_loss
        
        loss_and_metrics = (loss_sum, metrics)
        # outputs: ((loss, metrics), logit_tuple)
        outputs = (loss_and_metrics,) + ((aa_logit_out,ss_logit_out,rsa_logit_out,dist_logit_out),)
        return outputs