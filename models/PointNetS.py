import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *


class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024

        return feature_global.reshape(bs, g, self.encoder_channel)


@MODELS.register_module()
class PointNetS(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads
        self.encoder_dims = config.encoder_dims
        self.input_format = config.input_format
        feature_dim = self.encoder_dims
        self.dropout = config.dropout

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        if config.fc_layer == 3:
            self.cls_head = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )
        elif config.fc_layer == 1:
            self.cls_head = nn.Sequential(
                nn.Linear(feature_dim, self.cls_dim),
            )

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

    def get_loss_acc(self, ret, gt, multi_hot=False, valid_multi_hot=None):
        if multi_hot:
            pred = self.sigmoid(ret)
            loss = self.criterion(pred, gt.float())
            pred_data = pred.clone().detach()

            acc, pratial_acc, recall, dice = misc.cal_multi_hot_metric(valid_multi_hot, [pred_data], [gt])
        else:
            loss = self.loss_ce(ret, gt.long())
            pred = ret.argmax(-1)
            acc = (pred == gt).sum() / float(gt.size(0)) * 100
            recall = (pred == gt).sum() / gt.sum() * 100
        return loss, acc, pratial_acc, recall, dice

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='PointNetS')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='PointNetS'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='PointNetS')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='PointNetS'
                )

            print_log(f'[PointNetS] Successful Loading the ckpt from {bert_ckpt_path}', logger='PointNetS')
        else:
            print_log('Training from scratch!!!', logger='PointNetS')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, label, fa_label):
        neighborhood = pts[np.newaxis, :]
        group_input_tokens = self.encoder(neighborhood)
        ret = self.cls_head(group_input_tokens[0])

        return ret, group_input_tokens[0]

