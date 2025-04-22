import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from tqdm import trange

import numpy as np
import pickle
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms


def print_bad_case(pred, label, CLASS_NAMES, multi_hot, log_name):
    logger = get_logger(log_name)
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
                print_log(print_string[:-2], logger = logger)
        else:
            pred_i = pred[:, i]
            label_i = label[:, i]
            intersection_i = (pred * (label==1))[:, i]
            dice_i = 2 * intersection_i.sum() / (pred_i.sum() + label_i.sum())
            if (label_i==1).sum() < 50:
                continue
            label_i = (label_i==1)
            result = pred[label_i].sum(0)
            sort_result = result.sort(descending=True)
            acc = sort_result[0] / float((label_i).sum())

            print_string = '{}: 1. {} ({:.2f}); 2. {}: ({:.2f}); 3 {}: ({:.2f}); dice={:.2f}'.format(key, fiber_classes[sort_result[1][0]], acc[0]*100, fiber_classes[sort_result[1][1]], acc[1]*100, fiber_classes[sort_result[1][2]], acc[2]*100, dice_i*100)
            if acc[0] > 1.0:
                import ipdb; ipdb.set_trace()

            print_log(print_string, logger = logger)
                

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


def run_graph(args, config, train_writer=None, val_writer=None, test_writer=None):
    args_dict = vars(args)
    logger = get_logger(args.log_name)

    # build dataset
    for split in ['train', 'val', 'test']:
        config.dataset[split]._base_.DATA_PATH = args.ckpts
        for key in ['USE_GTFA', 'DOWNSAMPLE_RATE']:
            config.dataset[split]._base_[key] = args_dict[key.lower()]
    train_dataloader, val_dataloader, test_dataloader = builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(args, config.dataset.val), builder.dataset_builder(args, config.dataset.test)

    # build model
    for key in ['depth', 'num_heads', 'fc_layer', 'use_fa', 'use_dfa', 'use_ls', 'use_dls', 'use_select', 'keypoint_num', 'knn']:
        config.model[key] = args_dict[key]
    config.model.max_fa = test_dataloader.dataset.max_fa_class
    config.model.cls_dim = len(test_dataloader.dataset.classes)
    base_model = builder.model_builder(config.model)
    base_model = base_model.cuda()
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    base_model.load_classifier_from_ckpt(args.ckpts)
    
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    best_emr, best_precision, best_recall, best_dice = 0.0, 0.0, 0.0, 0.0
    test_emr, test_precision, test_recall, test_dice = 0.0, 0.0, 0.0, 0.0
    base_model.zero_grad()
    torch.autograd.set_detect_anomaly(True)
    print('n_batches = {}'.format(len(train_dataloader)))

    for epoch in range(start_epoch, config.max_epoch + 1):
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        if args.use_dfa and args.use_fa:
            losses = AverageMeter(['loss', 'fa_loss', 'acc'])
        else:
            losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        for idx, (point_set, trk_label, point_feature, pred_data, fa_label, subject_id, _) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            data_time.update(time.time() - batch_start_time)
            
            point_set = point_set[0].cuda()
            trk_label = trk_label[0].cuda()
            fa_label = fa_label[0][0].cuda()
            point_feature = point_feature[0].cuda()
            pred_data = pred_data[0].cuda()
            subject_id = subject_id[0]

            ret, fa_ret = base_model(subject_id, point_set, point_feature, pred_data, fa_label)

            valid_multi_hot = train_dataloader.dataset.valid_multi_hot
            loss, fa_loss, acc, precision, recall, dice = base_model.get_loss_acc(ret, fa_ret, trk_label, fa_label, valid_multi_hot)
            if args.use_dfa and args.use_fa:
                _loss = loss + fa_loss
            else:
                _loss = loss
            _loss.backward()
            
            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.use_dfa and args.use_fa:
                losses.update([loss.item(), fa_loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                if args.use_dfa and args.use_fa:
                    train_writer.add_scalar('Loss/Batch/FALoss', fa_loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)
        
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()], optimizer.param_groups[0]['lr']), logger = logger)

        if (epoch+1) % args.val_freq == 0:
            # Validate the current model
            metrics, precision, recall, dice = validate(base_model, val_dataloader, epoch, val_writer, args, config, logger=logger)

            if metrics.acc.item() > best_emr:
                best_emr = metrics.acc.item()
                best_dice = dice
                best_recall = recall
                best_precision = precision

                test_metrics, test_precision, test_recall, test_dice = validate(base_model, test_dataloader, epoch, test_writer, args, config, logger=logger)
                test_emr = test_metrics.acc.item()

            print_log('VAL: dice = {:.2f}, emr = {:.2f}, precision = {:.2f}, recall = {:.2f}'.format(dice, metrics.acc.item(), precision, recall), logger = logger)
            print_log('VAL: best_dice = {:.2f}, best_emr = {:.2f}, best_precision = {:.2f}, best_recall = {:.2f}'.format(best_dice, best_emr, best_precision, best_recall), logger = logger)
            print_log('TEST: test_dice = {:.2f}, test_emr = {:.2f}, test_precision = {:.2f}, test_recall = {:.2f}'.format(test_dice, test_emr, test_precision, test_recall), logger = logger)
            
            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, dataloader, epoch, val_writer, args, config, logger = None):
    base_model.eval()  # set model to eval mode
    valid_multi_hot = dataloader.dataset.valid_multi_hot

    test_pred, test_label = [], []
    with torch.no_grad():
        for idx, (point_set, trk_label, point_feature, pred_data, fa_label, subject_id, _) in enumerate(dataloader):
            point_set = point_set[0].cuda()
            trk_label = trk_label[0].cuda()
            point_feature = point_feature[0].cuda()
            pred_data = pred_data[0].cuda()
            fa_label = fa_label[0][0].cuda()
            subject_id = subject_id[0]

            ret, _ = base_model(subject_id, point_set, point_feature, pred_data, fa_label)

            if args.use_multi_hot:
                ret = torch.sigmoid(ret)
            test_pred.append(ret.detach())
            test_label.append(trk_label.detach())
        
        acc, precision, recall, dice = misc.cal_multi_hot_metric(valid_multi_hot, test_pred, test_label)

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        print_bad_case(test_pred, test_label, dataloader.dataset.classes, args.use_multi_hot, args.log_name)
        print_log('[Validation] EPOCH: %d exact acc = %.2f, precision = %.2f, recall = %.2f, dice = %.2f' % (epoch, acc, precision, recall, dice), logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc), precision, recall, dice
