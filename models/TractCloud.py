from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from .build import MODELS
from utils import misc


def fiber_distance_cal_efficient(set1, set2=None, num_points=20):
    set1 = set1.reshape(set1.shape[0], -1)   # set1 [N, 3*n_p]
    set2 = set2.reshape(set2.shape[0], -1) if set2 is not None else set1  # set2 [M, 3*n_p]
    set1_squ = (set1 ** 2).sum(1).view(-1, 1)  # set1_squ [N, 1]
    set2_t = torch.transpose(set2, 0, 1)   # set2_t [3*n_p, M]
    set2_squ = (set2 ** 2).sum(1).view(1, -1)   # set2_squ [1, M]
    dist = set1_squ + set2_squ - 2.0 * torch.mm(set1, set2_t)  # dist [N, M]
    # Ensure diagonal is zero if set1=set2
    if set2 is None:
       dist = dist - torch.diag(dist.diag())  
    dist = torch.sqrt(torch.clamp(dist, 0.0, np.inf))
    
    mean_dist = torch.div(dist, num_points)
    
    return mean_dist


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
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

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
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
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

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

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, k=20, k_global=500, global_feat = True):
        super(PointNetfeat, self).__init__()
        if k+k_global == 0:  
            self.conv1 = torch.nn.Conv1d(3, 64, 1)
        else:
            self.info_conv = torch.nn.Conv2d(3*2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        if k+k_global == 0:
            self.bn1 = nn.BatchNorm1d(64)
        else:
            self.info_bn = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        
        self.k = k
        self.k_global = k_global

    def forward(self, x, info_point_set):
        n_pts = x.size()[2]
        trans = None
        
        #* input local+global info for each streamline
        x = x[:,:,:,None].repeat(1, 1, 1, self.k + self.k_global)    # (num_fiber, 3, num_points) -> (num_fiber, 3, num_points, fiber_k)
        x = torch.cat((info_point_set-x, x), dim=1)   #  (num_fiber, 3*2, num_points, fiber_k)
        x = F.relu(self.info_bn(self.info_conv(x)))      # (num_fiber, 3*2, num_points, fiber_k) -> (num_fiber, 64, num_points, fiber_k)
        x = x.max(dim=-1, keepdim=False)[0]    # (num_fiber, 64,num_points, fiber_k) -> (num_fiber,64, num_points)
        
        trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


@MODELS.register_module()
class TractCloud(nn.Module):
    def __init__(self, config, **kwargs):
        super(TractCloud, self).__init__()
        self.k = config.k
        self.k_global = config.k_global
        self.feat = PointNetfeat(k=self.k, k_global=self.k_global, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, config.cls_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

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

    def forward(self, pts, label, info_point_set):
        pts = pts.transpose(1, 2)
        info_point_set = info_point_set.permute(0, 3, 2, 1)
        x, _, _ = self.feat(pts, info_point_set)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        ret = self.fc3(x)

        return ret, x
