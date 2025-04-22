import os
import numpy as np
import warnings
import pickle
import torch
from dipy.tracking.streamlinespeed import set_number_of_points
import tqdm
from sklearn.metrics import euclidean_distances, pairwise_distances
from tqdm import trange

from .build import DATASETS
from torch.utils.data import Dataset
from trk_loader import HCP105_names, HCP105_name2id, HCP105_id2name, is_reverse
warnings.filterwarnings('ignore')


@DATASETS.register_module()
class OldHCP105(Dataset):
    def __init__(self, config, fold=4):
        self.npoints = config.N_POINTS
        self.root = config.DATA_PATH
        self.downsample_rate = config.DOWNSAMPLE_RATE
        self.use_subject_scale = config.USE_SUBJECT_SCALE
        self.split = config.subset
        self.subject_list = []
        self.input_format = config.INPUT_FORMAT
        self.multi_hot = config.USE_MULTI_HOT
        
        with open(os.path.join(self.root, 'all_pid.pkl'), 'rb') as f:
            self.all_pid = pickle.load(f)
        
        self.fx_list = [
            '912447', '887373', '680957', '922854', '984472'
        ]
        
        if self.split == 'val':
            self.subject_list = self.all_pid[fold*21:]
        elif self.split == 'test':
            self.subject_list = self.all_pid[:]
        elif self.split == 'train':
            self.subject_list = self.all_pid[:fold*21]

        self.all_classes = HCP105_name2id
        self.classes = {}

        counter = 0
        for key in HCP105_name2id:
            if '+' in key:
                counter += 1
                continue
            self.classes[key] = HCP105_name2id[key]

        if not self.multi_hot:
            self.classes = HCP105_name2id
        
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
        postfix = self.downsample_rate
        self.data_list = []
        self.point_num, self.point_mean = {}, {}
        tbar = tqdm.tqdm(total=len(self.subject_list))

        pkl_name = os.path.join(self.root, '81_subject_fa_dict_{}.pkl'.format(postfix))
        fa_relation = {}
            
        with open(pkl_name, 'rb') as f:
            fa_dict = pickle.load(f)
        
        key_list = fa_dict['key_list']
        fa_index_dict = fa_dict['fa_index_dict']
        fa_points_list = fa_dict['fa_points_list']

        self.max_fa_class = 0

        for i in range(len(fa_index_dict)):
            if fa_index_dict[i] > self.max_fa_class:
                self.max_fa_class = fa_index_dict[i]

        for i in range(len(key_list)):
            fa_relation[key_list[i]] = {
                'label': [fa_index_dict[3*i], fa_index_dict[3*i+1], fa_index_dict[3*i+2]],
                'position': [fa_points_list[3*i], fa_points_list[3*i+1], fa_points_list[3*i+2]],
            }

        self.each_class_num = {key:0 for key, _ in HCP105_name2id.items()}
        for subject_id in self.subject_list:
            tbar.update(1)
            pkl_name = os.path.join(self.root, subject_id, '82_fiber_based_data_list_{}.pkl'.format(postfix))
            
            with open(pkl_name, 'rb') as f:
                subject_data = pickle.load(f)
            
            processed_data = []
            for item in subject_data:
                point_set = item['data']

                if 'CC+' in item['label']:
                    item['label'] = item['label'][3:]
                
                trk_label = item['label']

                if trk_label not in HCP105_names:
                    continue
                if subject_id in self.fx_list and (trk_label == 'FX_left' or trk_label == 'FX_right'):
                    continue

                self.each_class_num[trk_label] += 1
                if self.input_format != 'fibergeomap':
                    point_set = set_number_of_points(point_set, self.npoints)
                item['sampled_data'] = point_set
                item['fa_label'] = fa_relation[trk_label]

                processed_data.append(item)
                
                if subject_id in self.point_mean:
                    self.point_mean[subject_id] += point_set.sum(axis=0)
                    self.point_num[subject_id] += 1
                else:
                    self.point_mean[subject_id] = point_set.sum(axis=0)
                    self.point_num[subject_id] = 1
            
            if self.input_format == 'subject':
                self.data_list.append(processed_data)
            elif self.input_format == 'fiber' or self.input_format == 'fibergeomap':
                self.data_list.extend(processed_data)

            self.point_mean[subject_id] /= (self.point_num[subject_id] * self.npoints)
            # print('{} = {}'.format(subject_id, self.point_mean[subject_id]))

        tbar.close()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_cls_item(self, index):
        data_dict = self.data_list[index]
        point_set = data_dict['sampled_data']
        subject_id = data_dict['pid']

        point_set = point_set - self.point_mean[subject_id]
        if self.use_subject_scale:
            point_set = point_set / self.dist[subject_id]

        point_set = torch.from_numpy(point_set)
        point_set = point_set.type(torch.FloatTensor)
        if self.multi_hot:
            cls = np.zeros((self.classes_num))
            if '+' in data_dict['label']:
                two_classes = data_dict['label'].split('+')
                cls[self.classes[two_classes[0]]] = 1
                cls[self.classes[two_classes[1]]] = 1
            else:
                cls[self.classes[data_dict['label']]] = 1
            cls = torch.from_numpy(np.array(cls).astype(np.int64))
        else:
            cls = torch.from_numpy(np.array(HCP105_name2id[data_dict['label']]).astype(np.int64))

        return subject_id, (point_set, cls, 'None')
    
    def get_subject_item(self, index):
        subject_data_dict = self.data_list[index]
        fiber_nums = len(subject_data_dict)
        point_set = np.zeros((fiber_nums, self.npoints, 3))
        fa_label = np.zeros((fiber_nums, 3, self.max_fa_class+1))

        if self.multi_hot:
            cls = np.zeros((fiber_nums, self.classes_num))
        else:
            cls = np.zeros(fiber_nums)
        for i in range(fiber_nums):
            subject_id = subject_data_dict[i]['pid']
            point_set[i] = subject_data_dict[i]['sampled_data']
            point_set[i] = point_set[i] - self.point_mean[subject_id]
            fiber_fa = subject_data_dict[i]['fa_label']['label']
            fa_points = subject_data_dict[i]['fa_label']['position']
            for j in range(3):
                fa_label[i][j][fiber_fa[j]] = 1
            
            if is_reverse(point_set[i], fa_points):
                point_set[i] = point_set[i][::-1]

            if self.multi_hot:
                if '+' in subject_data_dict[i]['label']:
                    two_classes = subject_data_dict[i]['label'].split('+')
                    cls[i][self.classes[two_classes[0]]] = 1
                    cls[i][self.classes[two_classes[1]]] = 1
                else:
                    cls[i][self.classes[subject_data_dict[i]['label']]] = 1
            else:
                cls[i] = HCP105_name2id[subject_data_dict[i]['label']] 
        
        if self.use_subject_scale:
            dist = np.max(np.sqrt(np.sum(point_set.reshape(-1, 3) ** 2, axis = 1)), 0)
            point_set = point_set / dist
        
        point_set = torch.from_numpy(point_set)
        point_set = point_set.type(torch.FloatTensor)
        cls = torch.from_numpy(np.array(cls).astype(np.int64))

        return subject_id, (point_set, cls, fa_label)
    
    def get_fibergeomap_item(self, index):
        def compute_fibergeomap(point_set):
            dist = np.max(np.sqrt(np.sum(point_set.reshape(-1, 3) ** 2, axis = 1)), 0)
            fibergeomap = np.zeros((3, 36))
            r_matrix = np.sqrt(np.sum(point_set.reshape(-1, 3) ** 2, axis = 1))
            i_matrix = np.arccos(point_set[:, 2] / r_matrix)
            z_matrix = np.arctan2(point_set[:, 1], point_set[:, 0]) + np.pi
            for i in range(36):
                r_interval = dist / 36
                fibergeomap[0][i] = ((r_matrix >= i*r_interval) * (r_matrix < (i+1)*r_interval)).sum() / point_set.shape[0]
                i_interval = 5 * np.pi / 180
                fibergeomap[1][i] = ((i_matrix >= i*i_interval) * (i_matrix < (i+1)*i_interval)).sum() / point_set.shape[0]
                z_interval = 10 * np.pi / 180
                fibergeomap[2][i] = ((z_matrix >= i*z_interval) * (z_matrix < (i+1)*z_interval)).sum() / point_set.shape[0]
            return fibergeomap
        data_dict = self.data_list[index]
        point_set = data_dict['data']
        subject_id = data_dict['pid']
        fibergeomap = np.zeros((3, 72))
        
        global_norm = point_set - self.point_mean[subject_id]
        fibergeomap[:, :36] = compute_fibergeomap(global_norm)
        local_norm = point_set - point_set.mean(0)
        fibergeomap[:, 36:] = compute_fibergeomap(local_norm)
        fibergeomap = torch.from_numpy(fibergeomap)
        fibergeomap = fibergeomap.type(torch.FloatTensor)
        if self.multi_hot:
            cls = np.zeros((self.classes_num))
            if '+' in data_dict['label']:
                two_classes = data_dict['label'].split('+')
                cls[self.classes[two_classes[0]]] = 1
                cls[self.classes[two_classes[1]]] = 1
            else:
                cls[self.classes[data_dict['label']]] = 1
            cls = torch.from_numpy(np.array(cls).astype(np.int64))
        else:
            cls = torch.from_numpy(np.array(HCP105_name2id[data_dict['label']]).astype(np.int64))
        
        return subject_id, (fibergeomap, cls, 'None')
    
    def __getitem__(self, index):
        if self.input_format == 'fiber':
            return self.get_cls_item(index)
        elif self.input_format == 'subject':
            return self.get_subject_item(index)
        elif self.input_format == 'fibergeomap':
            return self.get_fibergeomap_item(index)
