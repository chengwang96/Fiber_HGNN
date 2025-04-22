import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.logger import *
from utils import misc
import copy
from .build import MODELS


@MODELS.register_module()
class GeoMapNet(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(72, 100)
        self.cls_dim = config.cls_dim
        self.postion_embedding = Positional_Encoding(100, 3, 0.5, 'cuda')
        self.encoder = GeoMapEncoder(100, 10, 512, 0.5)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(8)])
        self.fc1 = nn.Linear(3 * 100, self.cls_dim)

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

    def forward(self, x, label, fa_label, use_transformer=False):
        out = self.embedding(x)
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out, None


class GeoMapEncoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(GeoMapEncoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)

        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)

        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)

        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)

        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        
        return out