from typing import Sequence

import math
import copy
import numpy as np
import torch

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.cnn.bricks.transformer import build_dropout, AdaptivePadding
from mmcv.runner import BaseModule, ModuleList, Sequential
from mmcv.utils import to_2tuple
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.ops import Upsample, resize
from ..builder import BACKBONES
from .resnet import BasicBlock, Bottleneck

from mmcls.models.utils import resize_pos_embed
from ..utils import PatchEmbed  # ConvPatchEmbed

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


# Pair
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# Reference: https://github.com/OliverRensu/Shunted-Transformer
class ConvPatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 num_convs=0,
                 conv_type='Conv2d',
                 patch_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 init_cfg=None):
        super(ConvPatchEmbed, self).__init__(init_cfg=init_cfg)

        assert patch_size % 2 == 0

        self.embed_dims = embed_dims
        if stride is None:
            stride = patch_size // 2
        else:
            stride = stride // 2

        self.stem = torch.nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7,7), stride=(2,2), padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True))

        if num_convs > 0:
            convs = []
            for _ in range(num_convs):
                convs.append(torch.nn.Conv2d(64, 64, (3,3), (1,1), padding=1, bias=False))
                convs.append(torch.nn.BatchNorm2d(64))
                convs.append(torch.nn.ReLU(True))
            self.convs = torch.nn.Sequential(*convs)
        else:
            self.convs = None

        kernel_size = to_2tuple(patch_size//2)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)

        # self.projection = build_conv_layer(
        #     cfg=dict(
        #     type=conv_type,
        #     in_channels=64,
        #     out_channels=embed_dims,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding,
        #     dilation=dilation,
        #     bias=bias)

        self.projection = nn.Conv2d(
            in_channels=64,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None


    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """
        x = self.stem(x)
        if self.convs is not None:
            x = self.convs(x)

        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size

class SSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_size=7,
        conv_bias=True,
        bias=False,
        init_layer_scale=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert conv_size % 2 == 1
        padding = int(conv_size // 2)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=(conv_size, conv_size),
            stride=(1, 1),
            padding=(padding, padding),
            groups=self.d_inner
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)



    def forward(self, x, hw_shape):
        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner

        conv_state, ssm_state = None, None

        xz = self.in_proj(x)  # [B, L, 2 * E]
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        x, z = xz.chunk(2, dim=-1)
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + d_state * 2)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        dt = dt.permute(0, 2, 1).contiguous()  # [B, d_innter, L]
        B = B.permute(0, 2, 1).contiguous()  # [B, d_state, L]
        C = C.permute(0, 2, 1).contiguous()  # [B, d_state, L]

        assert self.activation in ["silu", "swish"]
        ys = selective_scan_fn(
                x_conv.permute(0, 2, 1).contiguous(),
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ).permute(0, 2, 1)
        y = sum(ys) * self.act(z)
        out = self.out_proj(y)

        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out


class VisionEncoderMambaBlock(nn.Module):
    """
    VisionMambaBlock is a module that implements the Mamba block from the paper
    Vision Mamba: Efficient Visual Representation Learning with Bidirectional
    State Space Model

    Args:
        dim (int): The input dimension of the input tensor.
        heads (int): The number of heads in the multi-head attention mechanism.
        dt_rank (int): The rank of the state space model.
        dim_inner (int): The dimension of the inner layer of the
            multi-head attention.
        d_state (int): The dimension of the state space model.


    Example:
    >>> block = VisionMambaBlock(dim=256, heads=8, dt_rank=32,
            dim_inner=512, d_state=256)
    >>> x = torch.randn(1, 32, 256)
    >>> out = block(x)
    >>> out.shape
    torch.Size([1, 32, 256])
    """

    def __init__(
        self,
        dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.SiLU()
        self.ssm = SSM(d_model=dim, dt_rank=dt_rank, d_state=d_state, conv_bias=True, bias=True)

        # Linear layer for z and x TODO: split x and z proj
        self.proj = nn.Linear(dim, dim)

        # Softplus
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, hw_size):
        # x is of shape [batch_size, seq_len, dim]
        b, s, d = x.shape

        # Skip connection
        skip = x

        # Normalization
        x = self.norm(x)

        # Split x into x1 and x2 with linears
        z1 = self.proj(x)
        x1 = self.proj(x)

        # forward con1d
        x1_rearranged = rearrange(x1, "b s d -> b d s")
        forward_conv_output = self.forward_conv1d(x1_rearranged)
        forward_conv_output = rearrange(
            forward_conv_output, "b d s -> b s d"
        )
        x1_ssm = self.ssm(forward_conv_output,hw_size)

        # backward conv x2
        # TODO: x2.flip([1]) compare the unflip version and flip version
        x2_rearranged = rearrange(x1, "b s d -> b d s")
        x2 = self.backward_conv1d(x2_rearranged)
        x2 = rearrange(x2, "b d s -> b s d")

        # Backward ssm
        x2 = self.ssm(x2,hw_size)

        # Activation
        z = self.activation(z1)

        # matmul with z + backward ssm
        x2 = x2 * z # @ -> *

        # Matmul with z and x1
        x1 = x1_ssm * z # @ -> *

        # Add both matmuls
        x = x1 + x2
        # TODO: There may be a additional proj(dim,dim)
        # Add skip connection
        return x + skip


@BACKBONES.register_module()
class Vim(BaseModule):
    def __init__(self,
                 embed_dims: int,
                 dt_rank: int = 32,
                 dim_inner: int = None,
                 d_state: int = None,
                 patch_size: int = 16,
                 channels: int = 3,
                 dropout: float = 0.1,
                 num_layers: int = 12,
                 out_indices: int = -1,
                 num_convs_patch_embed: int = 1,
                 final_norm=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None,
                 **kwargs):
        super(Vim, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.patch_size = patch_size
        self.channels = channels
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_convs_patch_embed = num_convs_patch_embed


        self.patch_embed = ConvPatchEmbed(
            in_channels=self.channels,
            embed_dims=self.embed_dims,
            num_convs=self.num_convs_patch_embed,
            patch_size=self.patch_size,
            stride=self.patch_size
        )


        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # class token   TODO: add in middle (and remove when decoding)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Latent
        self.to_latent = nn.Identity()

        # encoder layers
        self.layers = nn.ModuleList()

        # Append the encoder layers
        for _ in range(self.num_layers):
            self.layers.append(
                VisionEncoderMambaBlock(
                    dim=embed_dims,
                    dt_rank=dt_rank,
                    dim_inner=dim_inner,
                    d_state=d_state,
                )
            )

        self.final_norm = final_norm
        # assert final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        for i in out_indices:
            if i != self.num_layers - 1:
                if norm_cfg is not None:
                    norm_layer = build_norm_layer(norm_cfg, self.embed_dims)[1]
                else:
                    norm_layer = nn.Identity()
                self.add_module(f'norm_layer{i}', norm_layer)
    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self, x: Tensor):
        # Patch embedding
        # b, c, h, w = x.shape

        x, hw_size = self.patch_embed(x)
        # print(f"Patch embedding: {x.shape}")

        # Shape
        # b, n, _ = x.shape

        # Cls tokens
        # cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        # print(f"Cls tokens: {cls_tokens.shape}")

        # Concatenate
        # x = torch.cat((cls_tokens, x), dim=1)

        # Dropout
        x = self.dropout(x)
        # print(x.shape)

        # Forward pass with the layers
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, hw_size)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *hw_size, C)
                if i != self.num_layers - 1:
                    norm_layer = getattr(self, f'norm_layer{i}')
                    patch_token = norm_layer(patch_token)
                patch_token = patch_token.permute(0, 3, 1, 2)
                outs.append(patch_token)
        return tuple(outs)


        # Latent
        # x = self.to_latent(x)


