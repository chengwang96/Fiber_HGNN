import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import tqdm
import os

import numpy as np
import pickle
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms


def print_bad_case(pred, label, CLASS_NAMES, multi_hot):
    if multi_hot:
        pred = pred.ge(0.5)

    print_string = ''
    fiber_classes = [a for a in CLASS_NAMES.keys()]
    for i, key in enumerate(CLASS_NAMES.keys()):
        if multi_hot:
            fiber_num = (label[:, i]).sum()
        else:
            fiber_num = (label==i).sum()
        if fiber_num > 50:
            print_string += key + ' ({}), '.format(fiber_num)
    print(print_string[:-2])

    print_string = ''
    for i, key in enumerate(CLASS_NAMES.keys()):   
        print_string = '{}: '.format(i)     
        if not multi_hot:
            if (label==i).sum() < 50:
                continue
            pred_i = pred[label==i]
            pred_num_i = len(pred_i)
            need_attention = False
            for j, key in enumerate(CLASS_NAMES.keys()):
                ratio = (pred_i==j).sum()/pred_num_i
                if ratio > 0.1:
                    print_string += '{} ({:.2f}), '.format(j, ratio)
                    if i != j:
                        need_attention = True
            if need_attention:
                print(print_string[:-2])
        else:
            pred_i = pred[:, i]
            label_i = label[:, i]
            if (label_i==1).sum() < 50:
                continue
            label_i = (label_i==1)
            result = pred[label_i].sum(0)
            sort_result = result.sort(descending=True)
            acc = sort_result[0] / float((label_i).sum())

            need_attention = False
            if sort_result[1][0] != i or acc[0] < 0.9:
                print_string = '{}: 1. {} ({:.2f}); 2. {}: ({:.2f}); 3 {}: ({:.2f})'.format(key, fiber_classes[sort_result[1][0]], acc[0], fiber_classes[sort_result[1][1]], acc[1], fiber_classes[sort_result[1][2]], acc[2])
                need_attention = True
            if acc[0] > 1.0:
                import ipdb; ipdb.set_trace()

            if need_attention:
                print(print_string)
                

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def find_origin_point_set(local_fiber_id_list, pkl_name):
    orginal_point_set = []
    with open(pkl_name, 'rb') as f:
        fiber_id_2_original_fiber = pickle.load(f)

    for local_fiber_id in local_fiber_id_list:
        orginal_point_set.append(fiber_id_2_original_fiber[local_fiber_id.item()])
    
    return orginal_point_set


def test_net(args, config):
    args_dict = vars(args)
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)

    for key in ['USE_MULTI_HOT', 'INPUT_FORMAT', 'USE_SUBJECT_SCALE', 'USE_GTFA', 'ONE_SHOT', 'DOWNSAMPLE_RATE']:
        config.dataset['test']._base_[key] = args_dict[key.lower()]
    if args.run_graph:
        config.dataset.test._base_.DATA_PATH = args.pkl_path

    test_dataloader = builder.dataset_builder(args, config.dataset.test)

    for key in ['input_format', 'depth', 'num_heads', 'fc_layer', 'dropout', 'use_fa', 'use_dfa', 'use_ls', 'use_dls', 'use_select', 'keypoint_num', 'knn']:
        config.model[key] = args_dict[key]
    config.model.cls_dim = len(test_dataloader.dataset.classes)
    config.model.max_fa = test_dataloader.dataset.max_fa_class
    base_model = builder.model_builder(config.model)
    base_model = base_model.cuda()
    
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger) # for finetuned transformer
    
    if args.run_graph:
        test_graph(base_model, test_dataloader, args, config, logger=logger)
    else:
        test_others(base_model, test_dataloader, args, config, logger=logger)


def test_graph(base_model, test_dataloader, args, config, logger = None):
    base_model.eval()  # set model to eval mode
    valid_multi_hot = test_dataloader.dataset.valid_multi_hot
    tbar = tqdm.tqdm(total=len(test_dataloader))

    # point_set, cls, fa_label, local_fiber_id
    with torch.no_grad():
        for idx, (point_set, trk_label, point_feature, pred_data, fa_label, subject_id, local_fiber_id) in enumerate(test_dataloader):
            point_set = point_set[0]
            trk_label = trk_label[0]
            fa_label = fa_label[0][0].cuda()
            point_feature = point_feature[0]
            pred_data = pred_data[0].cuda()
            subject_id = subject_id[0]

            batch_size = point_set.shape[0]
            test_logits_list = []
            test_bs = 4096 * 2
            subject_done = False

            for i in range(batch_size//test_bs+1):
                if subject_done:
                    continue
                start_index = i*test_bs
                end_index = min((i+1)*test_bs, batch_size)
                if start_index==end_index:
                    import ipdb; ipdb.set_trace()
                if i == batch_size//test_bs - 1 and batch_size - (i+1)*test_bs < 1000:
                    end_index = batch_size
                    subject_done = True
                print('{} - {}'.format(start_index, end_index))
                logits, features = base_model(subject_id, point_set[start_index:end_index].cuda(), point_feature[start_index:end_index].cuda(), pred_data[start_index:end_index].cuda(), fa_label[start_index:end_index])
                logits = logits.cpu()
                test_logits_list.append(logits)

            logits = torch.cat(test_logits_list, axis=0)
            trk_label = trk_label.cpu()
            
            if not args.use_multi_hot:
                pred = logits.argmax(-1).view(-1)
            else:
                pred = logits

            prefix = args.ckpts.split('ckpt')[0]
            misc.save_as_vtk_file(valid_multi_hot, pred.cpu(), point_set.cpu(), trk_label.cpu(), prefix, subject_id)
            
            import ipdb; ipdb.set_trace()
            tbar.update(1)

    tbar.close()
    

def test_others(base_model, test_dataloader, args, config, logger = None):
    base_model.eval()
    valid_multi_hot = test_dataloader.dataset.valid_multi_hot
    tbar = tqdm.tqdm(total=len(test_dataloader))

    with torch.no_grad():
        for idx, (subject_ids, data) in enumerate(test_dataloader):
            points = data[0]
            label = data[1]
            subject_ids = subject_ids[0]
            points = points[0]
            label = label[0]

            if args.input_format == 'subject':
                fa_label = data[2].cuda()
            
            if args.input_format in ['fiber', 'fiber_w_info', 'fibergeomap']:
                fa_label = data[2][0].cuda()
            
            batch_size = points.shape[0]
            test_logits_list = []
            test_bs = 2048

            for i in range(batch_size//test_bs+1):
                start_index = i*test_bs
                end_index = min((i+1)*test_bs, batch_size)
                if start_index==end_index:
                    import ipdb; ipdb.set_trace()
                logits, features = base_model(points[start_index:end_index].cuda(), label[start_index:end_index].cuda(), fa_label[start_index:end_index])
                logits = logits.cpu()
                test_logits_list.append(logits)
            
            logits = torch.cat(test_logits_list, axis=0)
            label = label.cpu()
            # else:
            #     logits, features = base_model(points, label, fa_label)

            if not args.use_multi_hot:
                pred = logits.argmax(-1).view(-1)
            else:
                pred = logits
            
            # original_point_set = find_origin_point_set(local_fiber_id, os.path.join(config.dataset.test._base_.DATA_PATH, subject_ids, '82_fiber_id_2_original_fiber_{}.pkl'.format(args.downsample_rate)))
            
            prefix = args.ckpts.split('ckpt')[0]
            if args.input_format == 'fibergeomap':
                misc.save_as_vtk_file(valid_multi_hot, pred.cpu(), fa_label.cpu(), label.cpu(), prefix, subject_ids)
            else:
                misc.save_as_vtk_file(valid_multi_hot, pred.cpu(), points.cpu(), label.cpu(), prefix, subject_ids)
            
            import ipdb; ipdb.set_trace()
            tbar.update(1)
    
    tbar.close()
