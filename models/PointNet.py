"""PointNet
Reference:
https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_utils.py

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import MODELS
import numpy as np
from utils import misc
from utils.logger import *
from timm.models.layers import trunc_normal_
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message


class STN3d(nn.Module):
    def __init__(self, channel=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).reshape(1, 9)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = self.iden.repeat(batchsize, 1).to(x.device)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k
        self.iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).reshape(1, self.k * self.k)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = self.iden.repeat(batchsize, 1).to(x.device)

        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """Encoder for PointNet

    Args:
        nn (_type_): _description_
    """

    def __init__(self,
                 in_channels: int,
                 input_transform: bool=True,
                 feature_transform: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input 
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
        """
        super().__init__()
        self.stn = STN3d(in_channels) if input_transform else None
        self.conv0_1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv0_2 = torch.nn.Conv1d(64, 64, 1)

        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn0_1 = nn.BatchNorm1d(64)
        self.bn0_2 = nn.BatchNorm1d(64)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fstn = STNkd(k=64) if feature_transform else None
        self.out_channels = 1024
         
    def forward_cls_feat(self, x):
        x = x.transpose(1, 2).contiguous()

        B, D, N = x.size()
        if self.stn is not None:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        x = F.relu(self.bn0_1(self.conv0_1(x)))
        x = F.relu(self.bn0_2(self.conv0_2(x)))

        if self.fstn is not None:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        return x

    def forward(self, x, features=None):
        return self.forward_cls_feat(x)


@MODELS.register_module()
class PointNet(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.cls_dim = config.cls_dim
        self.mlps = [512, 256]

        self.encoder = PointNetEncoder(in_channels=3)
        self.in_channels = config.in_channels

        self.cls_head = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
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
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
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
        feature = self.encoder(pts)
        ret = self.cls_head(feature)

        return ret, feature

