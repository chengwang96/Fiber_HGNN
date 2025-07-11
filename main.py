from tools import train_net, run_graph, save_attribute, test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter


def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
        val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'val'))
        test_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))

    # config
    config = get_config(args, logger = logger)
    config.total_bs = args.total_bs
    # batch size
    config.dataset.train.others.bs = config.total_bs
    config.dataset.val.others.bs = 1
    config.dataset.test.others.bs = 1
    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed, deterministic=args.deterministic) # seed + rank, for augmentation
        
    # run
    if args.test:
        test_net(args, config)
    elif args.save_attribute:
        save_attribute(args, config)
    else:
        if args.run_graph:
            run_graph(args, config, train_writer, val_writer, test_writer)
        else:
            train_net(args, config, train_writer, val_writer, test_writer)


if __name__ == '__main__':
    main()
