import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--num_workers', type=int, default=16)
    # seed exp_name
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')      
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--loss', type=str, default='cd1', help='loss name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--pkl_path', type = str, default=None, help = 'test used pkl path')
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument('--downsample_rate', type=int, default=1000, help = 'downsample rate')
    parser.add_argument(
        '--vote',
        action='store_true',
        default=False,
        help = 'vote acc')
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--save_attribute', 
        action='store_true', 
        default=False, 
        help = 'save attribute for graph model')
    parser.add_argument(
        '--use_multi_hot', 
        action='store_true', 
        default=False)
    parser.add_argument(
        '--use_subject_scale', 
        action='store_true', 
        default=False)
    parser.add_argument(
        '--train_transform', 
        action='store_true', 
        default=False)
    parser.add_argument(
        '--use_fa', 
        action='store_true', 
        default=False)
    parser.add_argument(
        '--use_dfa', 
        action='store_true', 
        default=False)
    parser.add_argument(
        '--use_gtfa', 
        action='store_true', 
        default=False)
    parser.add_argument(
        '--use_ls', 
        action='store_true', 
        default=False)
    parser.add_argument(
        '--use_dls', 
        action='store_true', 
        default=False)
    parser.add_argument(
        '--use_select', 
        action='store_true', 
        default=False)
    parser.add_argument('--keypoint_num', type=int, default=3)
    parser.add_argument(
        '--one_shot', 
        action='store_true', 
        default=False)
    parser.add_argument('--input_format', type = str, default='fiber')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--fc_layer', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--total_bs', type=int, default=1)
    parser.add_argument('--knn', type = int, default=10, help = 'number of knn neighbor')
    parser.add_argument(
        '--run_graph', 
        action='store_true', 
        default=False)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument(
        '--fold', type=int, default=-1)
    
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, 'TFBoard',args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

