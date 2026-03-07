"""
Backbone network for image feature extraction.

This module provides ResNet-based backbone networks for extracting
visual features from input images.
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from other_utils.misc import NestedTensor, is_main_process
from position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Load state dict, handling the num_batches_tracked key."""
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """Base class for backbone networks."""
    
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        """
        Initialize the backbone base.
        
        Args:
            backbone: The underlying backbone network
            train_backbone: Whether to train the backbone
            num_channels: Number of output channels
            return_interm_layers: Whether to return intermediate layer outputs
        """
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer4": "0"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs


class Backbone(BackboneBase):
    """ResNet backbone with frozen batch normalization."""
    
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        """
        Initialize the ResNet backbone.
        
        Args:
            name: Name of the ResNet model (e.g., 'resnet18', 'resnet50')
            train_backbone: Whether to train the backbone
            return_interm_layers: Whether to return intermediate layer outputs
            dilation: Whether to use dilated convolutions
        """
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, return_interm_layers=False):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.return_interm_layers = return_interm_layers

        if self.return_interm_layers:
            self.conv1x1_layer1 = nn.Conv2d(64, 512, kernel_size=1)
            self.conv1x1_layer2 = nn.Conv2d(128, 512, kernel_size=1)
            self.conv1x1_layer3 = nn.Conv2d(256, 512, kernel_size=1)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        if self.return_interm_layers:
            for name, x in xs.items():
                out.append(x)
                pos.append(self.position_embedding(x).to(x.dtype))
            return out, pos
        else:
            for name, x in xs.items():
                out.append(x)
                pos.append(self[1](x).to(x.dtype))

            return out, pos


def build_backbone(args):
    """
    Build the backbone network from arguments.
    
    Args:
        args: Arguments containing backbone configuration
    
    Returns:
        Joiner module combining backbone and positional encoding
    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
