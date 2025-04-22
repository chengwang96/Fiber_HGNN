# 1fc
CUDA_VISIBLE_DEVICES=2 python main.py --config cfgs/pointnets_hcp105.yaml --exp_name test_82fx_subject_mh_pointnets_1fc --input_format subject --use_multi_hot --save_attribute --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_1fc/ckpt-best.pth --fc_layer 1


# 3fc
CUDA_VISIBLE_DEVICES=7 python main.py --config cfgs/pointnets_hcp105.yaml --exp_name 82fx_subject_mh_pointnets_3fc_dp0.5 --input_format subject --use_multi_hot --save_attribute --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --fc_layer 3 --dropout 0.5 --downsample_rate 10

CUDA_VISIBLE_DEVICES=2 python main.py --config cfgs/pointnets_hcp105.yaml --exp_name 82fx_subject_mh_pointnets_3fc_dp0.0 --input_format subject --use_multi_hot --save_attribute --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.0/ckpt-best.pth --fc_layer 3 --dropout 0.0


# pointnet
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/pointnet_hcp105.yaml --exp_name 82fx_subject_mh_pointnet --input_format subject --use_multi_hot --save_attribute --ckpts experiments/pointnet_hcp105/cfgs/82fx_subject_mh_pointnet/ckpt-best.pth
