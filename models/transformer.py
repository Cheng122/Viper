"""
Transformer decoder implementation for DETRVAE.

This module implements the transformer decoder architecture with
cross-attention to image features and self-attention over proprioceptive data.

Based on PyTorch's Transformer implementation with modifications for imitation learning.
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor

import IPython
e = IPython.embed

class Transformer_decoder(nn.Module):

    def __init__(self, context_len=None, d_model=512, nhead=8, num_decoder_layers=6, dropout=0.1,
                 use_pos_embd_image=False, query_num=50, use_pos_embd_action=False,
                 self_attention=True):
        super().__init__()

        self.decoder = Transformer_BERT(context_len=context_len,
                                        latent_dim=d_model,
                                        num_head=nhead,
                                        num_layer=num_decoder_layers,
                                        dropout_rate=dropout,
                                        use_pos_embd_image=use_pos_embd_image,
                                        use_pos_embd_action=use_pos_embd_action,
                                        query_num=query_num,
                                        self_attention=self_attention)
    
        self._reset_parameters()
    
        self.d_model = d_model
        self.nhead = nhead
        self.self_attention = self_attention

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, proprio_input=None, additional_pos_embed=None, pos_embed=None):
        # TODO flatten only when input has H and W
        # if len(src.shape) == 4: # has H and W
            # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        proprio_input = proprio_input.permute(1,0,2)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        src = torch.cat([proprio_input, src], axis=0)
            
        additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
        pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

        action_input_token = torch.zeros_like(query_embed)
        input_tokens = torch.cat([src, action_input_token], axis=0)
        hs = self.decoder(input_tokens, pos_embed, query_embed)
        hs = hs.transpose(1, 0)
        return hs


class Transformer_Block(nn.Module):
    def __init__(self, latent_dim, num_head, dropout_rate, self_attention=True, query_num=50) -> None:
        super().__init__()
        self.num_head = num_head
        self.latent_dim = latent_dim
        self.ln_1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout_rate)
        self.ln_2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(dropout_rate),
        )
        self.ln_3 = nn.LayerNorm(latent_dim)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.query_num = query_num
        self.self_attention = self_attention
        
    def forward(self, x):
        if self.self_attention:
            x = self.ln_1(x)
            x2 = self.attn(x, x, x, need_weights=False)[0]
            x = x + self.dropout1(x2)
            x = self.ln_2(x)
            x = x + self.mlp(x)
            x = self.ln_3(x)
            return x
        else:
            x = self.ln_1(x)
            x_action = x[-self.query_num:].clone()
            x_condition = x[:-self.query_num].clone()
            x2 = self.attn(x_action, x_condition, x_condition, need_weights=False)[0]
            x2 = x2 + self.dropout1(x2)
            x2 = self.ln_2(x2)
            x2 = x2 + self.mlp(x2)
            x2 = self.ln_3(x2)
            x = torch.cat((x_condition, x2), dim=0)
            return x
            
    
class Transformer_BERT(nn.Module):
    def __init__(self, context_len, latent_dim=128, num_head=4, num_layer=4, dropout_rate=0.0,  
                 use_pos_embd_image=False, use_pos_embd_action=False, query_num=50,
                 self_attention=True) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_head = num_head
        self.num_layer = num_layer
        self.context_len = context_len
        self.use_pos_embd_image = use_pos_embd_image==1
        self.use_pos_embd_action = use_pos_embd_action==1
        self.query_num = query_num
        if use_pos_embd_action and use_pos_embd_image:
            self.weight_pos_embed = None
        elif use_pos_embd_image and not use_pos_embd_action:
            self.weight_pos_embed = nn.Embedding(self.query_num, latent_dim)
        elif not use_pos_embd_image and not use_pos_embd_action:
            self.weight_pos_embed = nn.Embedding(self.context_len, latent_dim)
        elif not use_pos_embd_image and use_pos_embd_action:
            raise ValueError("use_pos_embd_action is not supported")
        else:
            raise ValueError("bug ? is not supported")
        
        self.attention_blocks = nn.Sequential(
            *[Transformer_Block(latent_dim, num_head, dropout_rate, self_attention, query_num) for _ in range(num_layer)],
        )
        self.self_attention = self_attention
    
    def forward(self, x, pos_embd_image=None, query_embed=None):
        if not self.use_pos_embd_image and not self.use_pos_embd_action: #everything learned - severe overfitting
            x = x + self.weight_pos_embed.weight[:, None]
        elif self.use_pos_embd_image and not self.use_pos_embd_action: #use learned positional embedding for action 
            x[-self.query_num:] = x[-self.query_num:] + self.weight_pos_embed.weight[:, None]
            x[:-self.query_num] = x[:-self.query_num] + pos_embd_image
        elif self.use_pos_embd_action and self.use_pos_embd_image: #all use sinsoidal positional embedding
            x[-self.query_num:] = x[-self.query_num:] + query_embed
            x[:-self.query_num] = x[:-self.query_num] + pos_embd_image 
                        
        x = self.attention_blocks(x)
        # take the last token
        return x


def _get_clones(module, N):
    """Create N copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer_decoder(args):
    return Transformer_decoder(
        context_len=args.context_len,
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_decoder_layers=args.dec_layers,
        use_pos_embd_image=args.use_pos_embd_image,
        use_pos_embd_action=args.use_pos_embd_action,
        query_num=args.num_queries,
        self_attention=args.self_attention
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
