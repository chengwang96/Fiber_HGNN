import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import pickle

from datasets import data_transforms


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

def train_net(args, config, train_writer=None, val_writer=None, test_writer=None):
    args_dict = vars(args)
    logger = get_logger(args.log_name)

    # build dataset
    for split in ['train', 'val', 'test']:
        for key in ['USE_MULTI_HOT', 'INPUT_FORMAT', 'USE_SUBJECT_SCALE', 'ONE_SHOT', 'DOWNSAMPLE_RATE']:
            config.dataset[split]._base_[key] = args_dict[key.lower()]
    train_dataloader, val_dataloader, test_dataloader = builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(args, config.dataset.val), builder.dataset_builder(args, config.dataset.test)

    # build model
    for key in ['input_format', 'depth', 'num_heads', 'fc_layer', 'dropout']:
        config.model[key] = args_dict[key]
    config.model.cls_dim = len(train_dataloader.dataset.classes)
    base_model = builder.model_builder(config.model)
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, _ = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    print_log('Using Data parallel ...' , logger = logger)
    base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    best_emr, best_precision, best_recall, best_dice = 0.0, 0.0, 0.0, 0.0
    test_emr, test_precision, test_recall, test_dice = 0.0, 0.0, 0.0, 0.0
    base_model.zero_grad()
    print('n_batches = {}'.format(len(train_dataloader)))

    for epoch in range(start_epoch, config.max_epoch):
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        n_batches = len(train_dataloader)

        rotation = data_transforms.PointcloudRotate()
        scale = data_transforms.PointcloudScale()
        translate = data_transforms.PointcloudTranslate()

        for idx, (subject_ids, data) in enumerate(train_dataloader):
            n_itr = epoch * n_batches + idx
            data_time.update(time.time() - batch_start_time)
            
            points = data[0].cuda()
            label = data[1].cuda()
            fa_label = None

            if args.input_format == 'subject':
                points = points[0]
                label = label[0]
                fa_label = data[2]

            if args.input_format == 'fiber_w_info':
                fa_label = data[2]

            if args.train_transform:
                points, rotation_label = rotation(points)
                points, scale_label = scale(points)
                points, translate_label = translate(points)

            ret, _ = base_model(points, label, fa_label)

            valid_multi_hot = train_dataloader.dataset.valid_multi_hot
            loss, acc, precision, recall, dice = base_model.module.get_loss_acc(ret, label, args.use_multi_hot, valid_multi_hot)
            _loss = loss
            _loss.backward()

            if config.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
            
            optimizer.step()
            base_model.zero_grad()

            losses.update([loss.item(), acc.item()])

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
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
            (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()], optimizer.param_groups[0]['lr']), logger = logger)

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

    test_pred, test_label = [], []
    with torch.no_grad():
        for idx, (subject_id, data) in enumerate(dataloader):
            points = data[0]
            label = data[1]
            subject_id = subject_id[0]
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
                if start_index >= end_index:
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
                pred = torch.sigmoid(logits)

            test_pred.append(pred.detach())
            test_label.append(label.detach())

        if args.use_multi_hot:
            valid_multi_hot = dataloader.dataset.valid_multi_hot
            acc, precision, recall, dice = misc.cal_multi_hot_metric(valid_multi_hot, test_pred, test_label)
            pred_data = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)
        else:
            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)
            pred_data = test_pred
            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            precision = recall = dice = 0

        print_bad_case(pred_data, test_label, dataloader.dataset.classes, args.use_multi_hot, args.log_name)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc), precision, recall, dice


def save_attribute(args, config):
    logger = get_logger(args.log_name)
    args_dict = vars(args)
    print_log('Tester start ... ', logger = logger)

    # build dataset
    for split in ['train', 'val', 'test']:
        for key in ['USE_MULTI_HOT', 'INPUT_FORMAT', 'USE_SUBJECT_SCALE', 'ONE_SHOT', 'DOWNSAMPLE_RATE']:
            config.dataset[split]._base_[key] = args_dict[key.lower()]
    test_dataloader = builder.dataset_builder(args, config.dataset.test)
    # train_dataloader, val_dataloader, test_dataloader = builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(args, config.dataset.val), builder.dataset_builder(args, config.dataset.test)
    # build model
    for key in ['input_format', 'depth', 'num_heads', 'fc_layer', 'dropout']:
        config.model[key] = args_dict[key]
    config.model.cls_dim = len(test_dataloader.dataset.classes)
    base_model = builder.model_builder(config.model)
    base_model = base_model.cuda()

    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger) # for finetuned transformer

    packed_data = {
        'checkpoint_file': args.ckpts,
        'train_pid': None,
        'val_pid': None,
        'test_pid': None
    }
    point_data = {}

    # packed_data, point_data = save_attribute_one_split(base_model, train_dataloader, args, 'train', packed_data, point_data, logger=logger)
    # packed_data, point_data = save_attribute_one_split(base_model, val_dataloader, args, 'val', packed_data, point_data, logger=logger)
    packed_data, point_data = save_attribute_one_split(base_model, test_dataloader, args, 'test', packed_data, point_data, logger=logger)

    pkl_path = args.ckpts.replace('.pth', '_features_{}.pkl'.format(args.downsample_rate))
    print(pkl_path)
    with open(pkl_path, 'wb') as f:
        pickle.dump(packed_data, f, pickle.HIGHEST_PROTOCOL)
    
    for key in point_data.keys():
        pkl_path = args.ckpts.replace('.pth', '_data_{}_{}.pkl'.format(args.downsample_rate, key))
        print(pkl_path)
        with open(pkl_path, 'wb') as f:
            pickle.dump(point_data[key], f, pickle.HIGHEST_PROTOCOL)


def save_attribute_one_split(base_model, dataloader, args, split, packed_data, point_data, logger = None):
    base_model.eval()  # set model to eval mode
    test_pred, test_label = [], []
    valid_multi_hot = dataloader.dataset.valid_multi_hot
    split_pid = []

    with torch.no_grad():
        for idx, (subject_id, data) in enumerate(dataloader):
            points = data[0][0]
            label = data[1][0]
            subject_id = subject_id[0]
            local_fiber_id = data[3][0]
            split_pid.append(subject_id)

            if args.input_format == 'subject':
                points = points.cuda()
                label = label.cuda()
                fa_label = data[2]

            if args.input_format in ['fiber', 'fiber_w_info', 'fibergeomap']:
                fa_label = data[2][0].cuda()

            batch_size = points.shape[0]
            test_logits_list, test_feature_list = [], []
            test_bs = 2048

            for i in range(batch_size//test_bs+1):
                start_index = i*test_bs
                end_index = min((i+1)*test_bs, batch_size)
                if start_index >= end_index:
                    import ipdb; ipdb.set_trace()
                logits, features = base_model(points[start_index:end_index].cuda(), label[start_index:end_index].cuda(), fa_label[start_index:end_index])
                logits = logits.cpu()
                features = features.cpu()
                test_logits_list.append(logits)
                test_feature_list.append(features)
            
            logits = torch.cat(test_logits_list, axis=0)
            all_features = torch.cat(test_feature_list, axis=0)
            label = label.cpu()
            # else:
            #     points = points.cuda()
            #     label = label.cuda()
            #     logits, features = base_model(points, label, fa_label)
            #     logits = logits.cpu()

            if not args.use_multi_hot:
                pred = logits.argmax(-1).view(-1)
            else:
                pred = logits

            point_data[subject_id] = {
                'feature_vector': all_features.detach().cpu(),
                'point_set': points.cpu(),
                'label': label.cpu(),
                'fa_label': fa_label.cpu(),
                'pred': pred.cpu(),
                'local_fiber_id': local_fiber_id
            }

            if args.use_multi_hot:
                pred = torch.sigmoid(pred)
            test_pred.append(pred.detach())
            test_label.append(label.detach())

        packed_data['{}_pid'.format(split)] = split_pid

        if args.use_multi_hot:
            acc, precision, recall, dice = misc.cal_multi_hot_metric(valid_multi_hot, test_pred, test_label)
        else:
            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)
            pred_data = test_pred
            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            precision = recall = dice = 0
        
        print_log('[{}]'.format(split) + ' dice = %.2f, emr = %.2f, precision = %.2f, recall = %.2f' % (dice, acc, precision, recall), logger=logger)

        return packed_data, point_data

        
