import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from .build import MODELS
from utils import misc
from utils.logger import *
from knn_cuda import KNN
import numpy as np


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w.squeeze(), dim=1)
        
        return torch.bmm(beta.unsqueeze(1), z).squeeze() 
    

## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, neighbor_idx):
        B = 1
        N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q.permute(2, 1, 0, 3)
        k = k.permute(2, 1, 0, 3)
        v = v.permute(2, 1, 0, 3)
        k = k[neighbor_idx].squeeze()
        v = v[neighbor_idx].squeeze()
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x[0]


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x, neighbor_idx):
        x = x + self.drop_path(self.attn(self.norm1(x), neighbor_idx))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, neighbor_idx):
        for _, block in enumerate(self.blocks):
            x = block(x, neighbor_idx)
        return x


@MODELS.register_module()
class HGNN(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dim
        self.cls_dim = config.cls_dim
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.drop_path_rate = config.drop_path_rate
        self.use_fa = config.use_fa
        self.use_dfa = config.use_dfa
        self.use_ls = config.use_ls
        self.use_dls = config.use_dls
        self.use_select = config.use_select
        self.keypoint_num = config.keypoint_num
        self.max_fa = config.max_fa
        self.knn_num = config.knn
        self.has_pred = {}

        self.semantic_attention = SemanticAttention(in_size=self.feature_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.feature_dim)
        )

        self.fa_embed = nn.Sequential(
            nn.Linear(self.max_fa + 1, 128),
            nn.GELU(),
            nn.Linear(128, self.feature_dim)
        )

        self.ls_embed = nn.Sequential(
            nn.Linear(self.cls_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.feature_dim)
        )
        
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        self.blocks = TransformerEncoder(
            embed_dim=self.feature_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        if config.fc_layer == 3:
            self.cls_head = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
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
                nn.Linear(self.feature_dim, self.cls_dim),
            )
        
        if config.fc_layer == 3:
            self.fa_pred_head = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.max_fa + 1)
            )
        elif config.fc_layer == 1:
            self.fa_pred_head = nn.Sequential(
                nn.Linear(self.feature_dim, self.max_fa + 1),
            )

        self.knn = KNN(k=self.knn_num, transpose_mode=True)
        self.build_loss_func()

    def build_loss_func(self):
        self.criterion = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.ce_loss = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, fa_ret, gt, fa_gt, valid_multi_hot):
        pred = self.sigmoid(ret)
        loss = self.criterion(pred, gt.float())
        fa_loss = None
        if self.use_dfa and self.use_fa:
            if fa_gt.shape[1] < self.keypoint_num:
                import ipdb; ipdb.set_trace()
            elif fa_gt.shape[1] > self.keypoint_num:
                interval = (fa_gt.shape[1] - 1) / (self.keypoint_num - 1)
                sample_list = [round(a*interval) for a in range(self.keypoint_num)]
                fa_gt = fa_gt[:, sample_list, :]

            fa_loss = self.ce_loss(fa_ret, torch.where(fa_gt.reshape(fa_ret.shape)==1)[1])
        pred_data = pred.clone().detach()

        acc, pratial_acc, recall, dice = misc.cal_multi_hot_metric(valid_multi_hot, [pred_data], [gt])
        return loss, fa_loss, acc, pratial_acc, recall, dice

    def load_classifier_from_ckpt(self, ckpt_path):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if 'cls_head_finetune' not in k:
                    del base_ckpt[k]
            
            self.load_state_dict(base_ckpt, strict=False)
        else:
            import ipdb; ipdb.set_trace()

    def perturb_fa_matrix(self, fa_label, ratio=0.1, keypoint_num=3):
        fa_label = fa_label.reshape(-1, fa_label.shape[2])
        perturb_index = np.random.randint(low=0, high=fa_label.shape[0], size=int(fa_label.shape[0]*ratio))
        perturb_index = np.unique(perturb_index)
        perturb_result = np.random.randint(low=0, high=self.max_fa, size=len(perturb_index))
        fa_label[perturb_index] = 0
        fa_label[perturb_index, perturb_result] = 1
        fa_label = fa_label.reshape(-1, keypoint_num, fa_label.shape[1])

        return fa_label

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

    def forward(self, subject_id, point_set, point_feature, pred_data, fa_label):
        # the point feature is an attribute of fiber
        # node type: fiber, keypoint, function area
        # select keypoint, the 3d position is an attribute of keypoint
        point_set = point_set.cuda()
        point_feature = point_feature.cuda()
        pred_data = pred_data.cuda()
        fa_label = fa_label.cuda()
        neighbor_list = []
        keypoint_num = self.keypoint_num
        interval = (point_set.shape[1] - 1) / (keypoint_num - 1)
        sample_list = [round(a*interval) for a in range(keypoint_num)]
        point_list = [point_set[:, a, :] for a in sample_list]

        if fa_label.shape[1] < keypoint_num:
            import ipdb; ipdb.set_trace()
        elif fa_label.shape[1] > keypoint_num:
            interval = (fa_label.shape[1] - 1) / (keypoint_num - 1)
            sample_list = [round(a*interval) for a in range(keypoint_num)]
            fa_label = fa_label[:, sample_list, :]

        # define neighbor based on keypoint
        for points in point_list:
            _, neighbor_idx = self.knn(points.unsqueeze(0), points.unsqueeze(0))
            neighbor_list.append(neighbor_idx[0])
        
        # add function area node, fa label is an attribute of fa node
        if self.use_dfa and self.use_fa:
            # perturbation for training
            fa_label = self.perturb_fa_matrix(fa_label, keypoint_num=keypoint_num)
            all_neighbor = torch.cat(neighbor_list, axis=0)

        aggregated_xf_feature = []
        pos_list = []
        fa_label = fa_label.transpose(0, 1)
        fa_features = self.fa_embed(fa_label.float())

        # add local semantic, which is an attribute of keypoint node
        if self.use_ls:
            if self.use_dls:
                if subject_id in self.has_pred and self.use_dls:
                    ret = self.has_pred[subject_id]
                    last_pred = self.sigmoid(ret).detach()
                else:
                    last_pred = pred_data
            else:
                last_pred = pred_data

        x_list = []
        for i in range(keypoint_num):
            pos_list.append(self.pos_embed(point_list[i]))
            x_list.append(point_feature + pos_list[i])

        xk = torch.cat(x_list, axis=0)
        fa_ret = None

        # meta-path, keypoint - function area - keypoint, update function area
        if self.use_dfa and self.use_fa:
            xk = xk + fa_features.reshape(xk.shape)
        
            if self.use_ls or self.use_dls:
                ls_feature = self.ls_embed(last_pred[all_neighbor].mean(1))
                aggregated_xk_feature = self.blocks(xk + ls_feature, all_neighbor)
            else:
                aggregated_xk_feature = self.blocks(xk, all_neighbor)

            fa_ret = self.fa_pred_head(aggregated_xk_feature)
            new_fa_label = self.softmax(fa_ret)
            new_fa_label = new_fa_label.reshape(-1, keypoint_num, new_fa_label.shape[1]).transpose(0, 1)
            new_fa_features = self.fa_embed(new_fa_label.float())


        #  meta-path, fiber - keypoint - function area - keypoint - fiber, update fiber representation
        for i in range(keypoint_num):
            xf = x_list[i]
            
            if self.use_fa:
                if self.use_dfa:
                    xf = xf + new_fa_features[i]
                else:
                    xf = xf + fa_features[i]
            
            if self.use_ls or self.use_dls:
                ls_feature = self.ls_embed(last_pred[neighbor_list[i]].mean(1))
                xf = xf + ls_feature
                if self.use_select:
                    aggregated_xf_feature.append(self.blocks(xf + ls_feature, neighbor_list[i]))
                else:
                    aggregated_xf_feature.append(xf)
            else:
                if self.use_select:
                    aggregated_xf_feature.append(self.blocks(xf, neighbor_list[i]))
                else:
                    aggregated_xf_feature.append(xf)
        
        # selective module
        if self.use_select:
            semantic_embeddings = torch.stack(aggregated_xf_feature, dim=1)
            aggregated_xf_feature = self.semantic_attention(semantic_embeddings)
        else:
            aggregated_xf_feature = aggregated_xf_feature[1]
        
        # fiber classification
        ret = self.cls_head(aggregated_xf_feature)
        if self.use_ls and self.use_dls:
            self.has_pred[subject_id] = ret.detach()

        return ret, fa_ret
