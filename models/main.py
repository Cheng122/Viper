"""
DETRVAE Decoder model for robot imitation learning.

This module implements a Transformer-based decoder architecture for
predicting action sequences from visual and proprioceptive observations.

Based on DETR (DEtection TRansformer) with modifications for imitation learning.
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse

import torch
import numpy as np

from torch import nn
from backbone import build_backbone
from transformer import build_transformer_decoder

import IPython
from einops import rearrange
e = IPython.embed


def get_args_parser():
    """Create argument parser for model configuration."""
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided", default=True)

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', default="ckpt_dir")
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default="ACT")
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default="sim_transfer_cube_scripted")
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_pos_embd_image', action='store', type=int, default=1, required=False)
    parser.add_argument('--use_pos_embd_action', action='store', type=int, default=1, required=False)
    parser.add_argument('--context_len', action='store_true', default=481, required=False)
    parser.add_argument('--self_attention', action="store", type=int, default=1)
    parser.add_argument('--feature_loss_weight', action='store', type=float, default=0.05)
    parser.add_argument('--action_dim', action='store', type=int, default=14)
    parser.add_argument('--gpu', type=int, default=0)
    return parser


def build_Viper_model_and_optimizer(args_override):
    """
    Build the Viper model and optimizer.
    
    Args:
        args_override: Dictionary of arguments to override defaults
    
    Returns:
        Tuple of (model, optimizer)
    """
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    state_dim = 14 # TODO hardcode
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)
    transformer_decoder = build_transformer_decoder(args)
    model = DETRVAE_Decoder(
        backbones,
        transformer_decoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        action_dim=args.action_dim,
        feature_loss=args.feature_loss if hasattr(args, 'feature_loss') else False,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def get_sinusoid_encoding_table(n_position, d_hid):
    """
    Generate sinusoidal positional encoding table.
    
    Args:
        n_position: Number of positions
        d_hid: Hidden dimension
    
    Returns:
        Positional encoding tensor of shape (1, n_position, d_hid)
    """
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE_Decoder(nn.Module):
    """
    Decoder-only Transformer for imitation learning.
    
    This model takes visual observations and proprioceptive data as input
    and predicts a sequence of future actions.
    
    Architecture:
    - ResNet18 backbone for image feature extraction
    - Learnable position embeddings for proprioceptive data
    - Transformer decoder with cross-attention to image features
    - Action and proprioception prediction heads
    """
    
    def __init__(self, backbones, transformer_decoder, state_dim, num_queries, camera_names, action_dim, history_size=50, feature_loss=False):
        """
        Initialize the DETRVAE Decoder.
        
        Args:
            backbones: List of backbone networks for image feature extraction
            transformer_decoder: Transformer decoder module
            state_dim: Dimension of robot state (14 for bimanual)
            num_queries: Number of query tokens (prediction horizon)
            camera_names: List of camera names
            action_dim: Dimension of action space
            history_size: Size of history buffer
            feature_loss: Whether to use feature matching loss
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer_decoder = transformer_decoder
        self.state_dim, self.action_dim = state_dim, action_dim
        self.cam_num = len(camera_names)
        self.history_size = history_size
        hidden_dim = transformer_decoder.d_model
        
        self.proprio_head = nn.Linear(hidden_dim, state_dim)
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.cam_history_size = history_size // 10
        self.emb = nn.Embedding(2, 64)
        self.linear1 = nn.Linear(1, 64)
        self.linear2 = nn.Linear(896, 512)              # 896 = 64 * 14(state_dim)
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,groups=64)
        self.part=torch.tensor([0,0,0,0,0,0,0,1,1,1,1,1,1,1]).cuda()
        self.linear3=nn.Linear(64,512)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None
        
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))
        self.additional_pos_embed = nn.Embedding(self.history_size, hidden_dim)
        self.feature_loss = feature_loss
    
    def forward(self, qpos, image, image_future, env_state=None, actions=None, is_pad=None):
        """
        Forward pass of the model.
        
        Args:
            qpos: Joint positions (batch, history_size, state_dim)
            image: Historical images (batch, num_images, channel, height, width)
            image_future: Future images for feature matching (batch, num_images, channel, height, width)
            env_state: Environment state (not used)
            actions: Ground truth actions for training (batch, horizon, action_dim)
            is_pad: Padding mask (not used)
        
        Returns:
            a_hat: Predicted actions (batch, num_queries, action_dim)
            a_proprio: Predicted proprioception (batch, history_size, state_dim)
            hs_img: Image features (or dict with feature matching info)
        """
        if len(self.backbones) > 1:
            all_cam_features = []
            all_cam_pos = []
            for cam_id, _ in enumerate(self.camera_names):
                features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
        else:
            all_cam_features = []
            all_cam_pos = []
            if self.feature_loss and self.training:
                features_list = []
                pos_list = []
                all_cam_features_future = []
                bs, _, _, h, w = image.shape
                image_total = torch.cat([image, image_future], axis=0)
                bs_t, _, _, h_t, w_t = image_total.shape
                
                for i in range(image_total.shape[1]):
                    single_image_total = image_total[:, i, :, :, :]
                    features, pos = self.backbones[0](single_image_total.reshape([-1, 3, single_image_total.shape[-2], single_image_total.shape[-1]]))
                    features = features[0]
                    pos = pos[0]
                    features_list.append(features)
                    pos_list.append(pos)
                features = torch.stack(features_list, dim=1)
                pos = torch.stack(pos_list, dim=1)

                project_feature = features
                all_cam_features_future = project_feature[bs:, :]
                project_feature = project_feature[:bs, :]
                project_feature = rearrange(project_feature, 'b n c w h -> b c (w n) h')

            else:
                bs, _, _, h, w = image.shape
                features_list = []
                pos_list = []

                for i in range(image.shape[1]):
                    single_image = image[:, i, :, :, :]
                    features, pos = self.backbones[0](single_image.reshape([-1, 3, single_image.shape[-2], single_image.shape[-1]]))
                    features = features[0]
                    pos = pos[0]
                    features_list.append(features)
                    pos_list.append(pos)
                features = torch.stack(features_list, dim=1)
                pos = torch.stack(pos_list, dim=1)

                project_feature = features
                project_feature = rearrange(project_feature, 'b n c w h -> b c (w n) h')

        qpos_process = self.linear1(qpos.unsqueeze(3))
        pos_s = self.emb(self.part).unsqueeze(0).unsqueeze(0)
        pos_t = self.conv(qpos_process.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        proprio_input = qpos_process + pos_t + 0.0001 * pos_s
        proprio_input = self.linear2(proprio_input.flatten(2))
        
        src = project_feature
        pos = rearrange(pos, 'b n c w h -> b c (w n) h')
        hs = self.transformer_decoder(
            src, 
            self.query_embed.weight,
            proprio_input=proprio_input, 
            pos_embed=pos, 
            additional_pos_embed=self.additional_pos_embed.weight
        )
        
        hs_action = hs[:, -1 * self.num_queries:, :].clone()
        hs_img = hs[:, self.history_size:-1 * self.num_queries, :].clone()
        hs_proprio = hs[:, 0:self.history_size, :].clone()
        
        a_hat = self.action_head(hs_action)
        a_proprio = self.proprio_head(hs_proprio)
        
        if self.feature_loss and self.training:
            image_feature_future = all_cam_features_future
            image_feature_future = image_feature_future.flatten(3)
            image_feature_future = rearrange(image_feature_future, 'b n c hw -> b (hw n) c')
            hs_img = {'hs_img': hs_img, 'image_feature_future': image_feature_future}
        
        return a_hat, a_proprio, hs_img
