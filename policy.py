"""
Viper Policy module for imitation learning.

This module defines the ViperPolicy class which wraps the DETRVAE_Decoder
model and provides training and inference interfaces.
"""
import torch.nn as nn
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms

from models.main import build_Viper_model_and_optimizer
import IPython
e = IPython.embed

class ViperPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_Viper_model_and_optimizer(args_override)
        self.model = model 
        self.optimizer = optimizer
        self.feature_loss_weight = args_override['feature_loss_weight'] if 'feature_loss_weight' in args_override else 0.0
    
    def __call__(self, qpos, qpos_future, image, image_future, actions=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        image_future = normalize(image_future)

        if actions is not None: # training time, including training and validation
            loss_dict = dict()
            a_hat, a_proprio, hs_img_dict = self.model(qpos, image, image_future)

            l1_actions = F.l1_loss(actions, a_hat)
            loss_dict['l1_actions'] = l1_actions
            
            if self.model.feature_loss and self.model.training:
                l1_qpos = 0.05 * F.l1_loss(qpos_future, a_proprio)
                loss_dict['l1_qpos'] = l1_qpos

                feature_loss = F.mse_loss(hs_img_dict['hs_img'], hs_img_dict['image_feature_future'])
                loss_dict['feature_loss'] = 0.1 * feature_loss

                loss_dict['loss'] = loss_dict['l1_actions'] + loss_dict['feature_loss'] + loss_dict['l1_qpos']
            else:
                loss_dict['loss'] = loss_dict['l1_actions']
            
            return loss_dict
        
        else: # inference time
            a_hat, a_proprio, _ = self.model(qpos, image, image_future, env_state)  # no action, sample from prior env state=None
            return a_hat
    
    def configure_optimizers(self):
        return self.optimizer
