import os
import numpy as np
import warnings
import pickle
import torch
import tqdm

from .build import DATASETS
from torch.utils.data import Dataset
from trk_loader import HCP105_name2id

warnings.filterwarnings('ignore')


@DATASETS.register_module()
class SimpleHCP105(Dataset):
    def __init__(self, config, fold=-1):
        self.npoints = config.N_POINTS
        self.root = config.DATA_PATH
        if 'ckpt' in self.root:
            self.root = self.root.split('ckpt')[0]
        self.split = config.subset
        self.downsample_rate = config.DOWNSAMPLE_RATE
        self.use_gtfa = config.USE_GTFA
        self.subject_list = []

        with open(os.path.join(self.root, 'ckpt-best_features_{}.pkl'.format(self.downsample_rate)), 'rb') as f:
            self.packed_data = pickle.load(f)

        if self.split == 'train':
            self.subject_list = self.packed_data['train_pid']
        elif self.split == 'val':
            self.subject_list = self.packed_data['val_pid']
        elif self.split == 'test':
            # self.subject_list = ['904044']
            self.subject_list = self.packed_data['test_pid']

        self.all_classes = HCP105_name2id
        self.classes = {}

        counter = 0
        for key in HCP105_name2id:
            if '+' in key:
                counter += 1
                continue
            self.classes[key] = HCP105_name2id[key]

        self.cat = list(self.classes.keys())
        self.classes_num = len(self.classes)

        self.valid_multi_hot = np.zeros((counter, self.classes_num))
        counter = 0
        for key in HCP105_name2id:
            if '+' in key:
                two_classes = key.split('+')
                self.valid_multi_hot[counter][HCP105_name2id[two_classes[0]]] = 1
                self.valid_multi_hot[counter][HCP105_name2id[two_classes[1]]] = 1
                counter += 1

        self.load_cls_data()

    def load_cls_data(self):
        self.data_list = []
        tbar = tqdm.tqdm(total=len(self.subject_list))

        for subject_id in self.subject_list:
            tbar.update(1)

            pkl_path = os.path.join(self.root, 'ckpt-best_data_{}_{}.pkl'.format(self.downsample_rate, subject_id))
            with open(pkl_path, 'rb') as f:
                subject_data = pickle.load(f)

            subject_data['subject_id'] = subject_id
            self.data_list.append(subject_data)
            
        tbar.close()

        postfix = self.downsample_rate
        pkl_name = os.path.join('data', 'HCP105', '82_ar_data_1000.pkl')
        # pkl_name = os.path.join('data', 'HCP105', '82_ar_data_{}.pkl'.format(postfix))

        with open(pkl_name, 'rb') as f:
            fa_dict = pickle.load(f)
        
        self.key_list = fa_dict['key_list']
        self.fa_index_dict = fa_dict['fa_index_dict']
        self.max_fa_class = 0

        for i in range(len(self.fa_index_dict)):
            if self.fa_index_dict[i] > self.max_fa_class:
                self.max_fa_class = self.fa_index_dict[i]

    def __len__(self):
        return len(self.data_list)
    
    def get_cls_item(self, index):
        data_dict = self.data_list[index]

        point_set = data_dict['point_set']
        trk_label = data_dict['label']
        point_feature = data_dict['feature_vector']
        subject_id = data_dict['subject_id']
        pred = data_dict['pred']
        local_fiber_id = data_dict['local_fiber_id']
        pred_data = torch.sigmoid(pred)
        
        if self.split == 'train' or self.use_gtfa:
            fa_label = data_dict['fa_label']
        else:
            fa_label = torch.zeros_like(data_dict['fa_label'])
            pred_label = pred_data.max(1)[1]
            
            for i in range(len(pred_label)):
                for j in range(3):
                    fa_label[0][i][j][self.fa_index_dict[pred_label[i].item() * 3 + j]] = 1 

        return point_set, trk_label, point_feature, pred_data, fa_label, subject_id, local_fiber_id

    def __getitem__(self, index):
        return self.get_cls_item(index)

