import os
import numpy as np
import warnings
import pickle
import torch
import tqdm

from .build import DATASETS
from torch.utils.data import Dataset
from trk_loader import HCP105_names, HCP105_name2id

warnings.filterwarnings('ignore')


def fiber_distance_cal_Efficient(set1, set2=None, num_points=20):
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


@DATASETS.register_module()
class HCP105(Dataset):
    def __init__(self, config):
        self.npoints = config.N_POINTS
        self.root = config.DATA_PATH
        self.downsample_rate = config.DOWNSAMPLE_RATE
        self.use_subject_scale = config.USE_SUBJECT_SCALE
        self.split = config.subset
        self.subject_list = []
        self.input_format = config.INPUT_FORMAT
        self.multi_hot = config.USE_MULTI_HOT
        self.one_shot = config.ONE_SHOT
        self.k_global = 500
        self.k = 20
        
        with open(os.path.join(self.root, 'all_pid.pkl'), 'rb') as f:
            self.all_pid = pickle.load(f)
        
        self.fx_list = [
            '912447', '887373', '680957', '922854', '984472'
        ]

        if self.split == 'train':
            if self.one_shot:
                self.subject_list = ['992774']
            else:
                self.subject_list = self.all_pid[:63]
        elif self.split == 'val':
            if self.one_shot:
                self.subject_list = ['992774']
            else:
                self.subject_list = self.all_pid[63:84]
        elif self.split == 'test':
            if self.one_shot:
                self.subject_list = ['687163', '685058', '683256', '680957', '679568', '677968', '673455', '672756', '665254', '654754', '645551', '644044', '638049', '627549', '623844', '622236', '620434', '613538', '601127', '599671', '599469']
            else:
                self.subject_list = self.all_pid[84:]

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
    
    def cal_info_point_set(self, point_set):
        point_num = len(point_set)
        global_idx = np.random.randint(0, point_num, (point_num, self.k_global))
        global_idx = torch.Tensor(global_idx)
        point_set = torch.from_numpy(point_set)
        distance_matrix = fiber_distance_cal_Efficient(point_set, point_set)
        local_idx = torch.sort(distance_matrix)[1][:, :self.k]

        all_idx = torch.cat([global_idx, local_idx], axis=1)

        return all_idx

    def load_cls_data(self):
        self.data_list, self.subject_num = [], []
        tbar = tqdm.tqdm(total=len(self.subject_list))
        postfix = self.downsample_rate

        pkl_name = os.path.join(self.root, '82_ar_data_1000.pkl')
        # pkl_name = os.path.join(self.root, '82_ar_data_{}.pkl'.format(postfix))
        fa_relation, self.info_point_set = {}, {}

        with open(pkl_name, 'rb') as f:
            fa_dict = pickle.load(f)
        
        key_list = fa_dict['key_list']
        fa_index_dict = fa_dict['fa_index_dict']
        self.max_fa_class = 0

        for i in range(len(fa_index_dict)):
            if fa_index_dict[i] > self.max_fa_class:
                self.max_fa_class = fa_index_dict[i]
        
        self.kp_num = int(len(fa_index_dict)/len(key_list))

        for i in range(len(key_list)):
            fa_relation[key_list[i]] = []
            for j in range(self.kp_num):
                fa_relation[key_list[i]].append(fa_index_dict[self.kp_num*i+j])

        self.point_mean = {}
        start_id, end_id = 0, 0
        for subject_id in self.subject_list:
            # pkl_name = os.path.join(self.root, subject_id, '82_subject_based_data_dict_{}.pkl'.format(postfix))
            pkl_name = os.path.join(self.root, subject_id, '82_processed_data_{}.pkl'.format(postfix))
            
            with open(pkl_name, 'rb') as f:
                subject_data = pickle.load(f)

            self.point_mean[subject_id] = subject_data['points_mean']
            fiber_points_list = subject_data['fiber_points_list']

            processed_data, point_set = [], []
            fiber_id = 0

            for item in fiber_points_list:
                trk_label = item['label']
                item['pid'] = subject_id

                if not trk_label in HCP105_names:
                    continue
                if subject_id in self.fx_list and (trk_label == 'FX_left' or trk_label == 'FX_right'):
                    continue

                item['fa_label'] = fa_relation[trk_label]
                item['fiber_id'] = fiber_id
                point_set.append(item['data'])
                fiber_id += 1
                processed_data.append(item)
            end_id += fiber_id
            self.subject_num.append([start_id, end_id])
            start_id = end_id

            if self.input_format == 'fiber_w_info':
                pkl_name = os.path.join(self.root, subject_id, '82_info_point_set_{}.pkl'.format(postfix))
                # if os.path.exists(pkl_name):
                #     with open(pkl_name, 'rb') as f:
                #         self.info_point_set[subject_id] = pickle.load(f)
                # else:
                point_set = np.array(point_set)
                all_idx = self.cal_info_point_set(point_set)
                self.info_point_set[subject_id] = all_idx.long()
                # self.info_point_set[subject_id] = point_set[all_idx.long()]
                # with open(pkl_name, 'wb') as f:
                #     pickle.dump(self.info_point_set[subject_id], f, pickle.HIGHEST_PROTOCOL)
            if self.input_format == 'subject':
                self.data_list.append(processed_data)
            elif self.input_format in ['fiber', 'fibergeomap', 'fiber_w_info']:
                self.data_list.extend(processed_data)

            tbar.update(1)
        tbar.close()

        pkl_name = os.path.join(self.root, '82_point_mean_{}.pkl'.format(postfix))
        with open(pkl_name, 'wb') as f:
            pickle.dump(self.point_mean, f, pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        if self.split in ['val', 'test']:
            return len(self.subject_list)
        return len(self.data_list)
    
    def get_single_cls_item(self, index):
        data_dict = self.data_list[index]
        point_set = data_dict['data']
        subject_id = data_dict['pid']
        fiber_id = data_dict['fiber_id']
        local_fiber_id = data_dict['local_fiber_id']
        info_point_set = torch.Tensor(0)
        if self.input_format == 'fiber_w_info':
            all_idx = self.info_point_set[subject_id][fiber_id]
            info_point_set = np.zeros((len(all_idx), 20, 3))
            for i, idx in enumerate(all_idx):
                info_point_set[i] = self.data_list[idx]['data']
            info_point_set = torch.from_numpy(info_point_set)
            info_point_set = info_point_set.type(torch.FloatTensor)
            # info_point_set = self.info_point_set[subject_id][fiber_id]
            # info_point_set = torch.from_numpy(info_point_set)

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

        return subject_id, (point_set, cls, info_point_set, local_fiber_id)

    def get_cls_item(self, index):
        if self.split in ['val', 'test']:
            start_id, end_id = self.subject_num[index]
            point_set_list = np.zeros((end_id-start_id, self.npoints, 3))
            local_fiber_id_list = np.zeros(end_id-start_id)
            cls_list = np.zeros((end_id-start_id, self.classes_num))
            info_point_set_list = np.zeros((end_id-start_id, self.k+self.k_global, self.npoints, 3))
            
            subject_idx = [i for i in range(start_id, end_id)]
            for idx, global_idx in enumerate(subject_idx):
                subject_id, (point_set, cls, info_point_set, local_fiber_id) = self.get_single_cls_item(global_idx)
                point_set_list[idx] = point_set
                local_fiber_id_list[idx] = local_fiber_id
                cls_list[idx] = cls
                if self.input_format == 'fiber_w_info':
                    info_point_set_list[idx] = info_point_set

            all_point_set = torch.from_numpy(point_set_list)
            all_point_set = all_point_set.type(torch.FloatTensor)
            all_cls = torch.from_numpy(np.array(cls_list).astype(np.int64))
            all_info_point_set = torch.from_numpy(np.array(info_point_set_list))
            all_info_point_set = all_info_point_set.type(torch.FloatTensor)
            all_local_fiber_id = torch.from_numpy(np.array(local_fiber_id_list).astype(np.int64))

            return subject_id, (all_point_set, all_cls, all_info_point_set, all_local_fiber_id)
        else:
            return self.get_single_cls_item(index)

    def get_subject_item(self, index):
        subject_data_dict = self.data_list[index]
        fiber_nums = len(subject_data_dict)
        point_set = np.zeros((fiber_nums, self.npoints, 3))
        fa_label = np.zeros((fiber_nums, self.kp_num, self.max_fa_class+1))
        local_fiber_id = np.zeros(fiber_nums)

        if self.multi_hot:
            cls = np.zeros((fiber_nums, self.classes_num))
        else:
            cls = np.zeros(fiber_nums)
        for i in range(fiber_nums):
            point_set[i] = subject_data_dict[i]['data']
            local_fiber_id[i] = subject_data_dict[i]['local_fiber_id']
            # local_fiber_id[i] = 0
            fiber_fa = subject_data_dict[i]['fa_label']
            for j in range(self.kp_num):
                fa_label[i][j][fiber_fa[j]] = 1

            if self.multi_hot:
                if '+' in subject_data_dict[i]['label']:
                    two_classes = subject_data_dict[i]['label'].split('+')
                    cls[i][self.classes[two_classes[0]]] = 1
                    cls[i][self.classes[two_classes[1]]] = 1
                else:
                    cls[i][self.classes[subject_data_dict[i]['label']]] = 1
            else:
                cls[i] = HCP105_name2id[subject_data_dict[i]['label']]
            subject_id = subject_data_dict[i]['pid']
        
        point_set = point_set - self.point_mean[subject_id]
        if self.use_subject_scale:
            dist = np.max(np.sqrt(np.sum(point_set.reshape(-1, 3) ** 2, axis = 1)), 0)
            point_set = point_set / dist

        point_set = torch.from_numpy(point_set)
        point_set = point_set.type(torch.FloatTensor)
        cls = torch.from_numpy(np.array(cls).astype(np.int64))
        local_fiber_id = torch.from_numpy(np.array(local_fiber_id).astype(np.int64))
        fa_label = torch.from_numpy(np.array(fa_label).astype(np.int64))

        return subject_id, (point_set, cls, fa_label, local_fiber_id)
    
    def get_single_fibergeomap_item(self, index):
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
        local_fiber_id = data_dict['local_fiber_id']
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

        return subject_id, (fibergeomap, cls, point_set, local_fiber_id)

    def get_fibergeomap_item(self, index):
        if self.split in ['val', 'test']:
            start_id, end_id = self.subject_num[index]
            fibergeomap_list = np.zeros((end_id-start_id, 3, 72))
            point_set_list = np.zeros((end_id-start_id, self.npoints, 3))
            local_fiber_id_list = np.zeros(end_id-start_id)
            cls_list = np.zeros((end_id-start_id, self.classes_num))
            
            subject_idx = [i for i in range(start_id, end_id)]
            for idx, global_idx in enumerate(subject_idx):
                subject_id, (fibergeomap, cls, point_set, local_fiber_id) = self.get_single_fibergeomap_item(global_idx)
                fibergeomap_list[idx] = fibergeomap
                local_fiber_id_list[idx] = local_fiber_id
                cls_list[idx] = cls
                point_set_list[idx] = point_set

            all_fibergeomap = torch.from_numpy(fibergeomap_list)
            all_fibergeomap = all_fibergeomap.type(torch.FloatTensor)
            all_point_set =torch.from_numpy(point_set_list)
            all_point_set = all_point_set.type(torch.FloatTensor)
            all_cls = torch.from_numpy(np.array(cls_list).astype(np.int64))
            all_local_fiber_id = torch.from_numpy(np.array(local_fiber_id_list).astype(np.int64))

            return subject_id, (all_fibergeomap, all_cls, all_point_set, all_local_fiber_id)
        else:
            return self.get_single_fibergeomap_item(index)

    def __getitem__(self, index):
        if self.input_format in ['fiber', 'fiber_w_info']:
            return self.get_cls_item(index)
        elif self.input_format == 'subject':
            return self.get_subject_item(index)
        elif self.input_format == 'fibergeomap':
            return self.get_fibergeomap_item(index)
