# PointNetS
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pointnets_hcp105.yaml --exp_name test_82fx_subject_mh_pointnets_1fc --input_format subject --use_multi_hot --test --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_1fc/ckpt-best.pth --fc_layer 1 --downsample_rate 1

CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/pointnets_hcp105.yaml --exp_name test_82fx_subject_mh_pointnets_3fc --input_format subject --use_multi_hot --test --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --fc_layer 3 --downsample_rate 10


# HGNN
CUDA_VISIBLE_DEVICES=2 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name test_82fx_subject_mh_hgnn_3fc_knn20 --input_format subject --use_multi_hot --run_graph --test --ckpts experiments/hgnn_hcp105/cfgs/82fx_subject_mh_hgnn_3fc_knn20/ckpt-best.pth --pkl_path experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5 --fc_layer 3 --knn 20 --downsample_rate 10

CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name test_82fx_subject_mh_hgnn_3fc_knn20_select --input_format subject --use_multi_hot --run_graph --test --ckpts experiments/hgnn_hcp105/cfgs/82fx_subject_mh_hgnn_3fc_knn20_select/ckpt-best.pth --pkl_path experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5 --fc_layer 3 --knn 20 --use_select --downsample_rate 10

CUDA_VISIBLE_DEVICES=2 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name test_82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_select_d3kp3 --input_format subject --use_multi_hot --run_graph --test --ckpts experiments/hgnn_hcp105/cfgs/82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_select_d3kp3/ckpt-best.pth --pkl_path experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5 --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20 --use_select --downsample_rate 10


# FiberGeoMap
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fibergeomap_hcp105.yaml --exp_name test_82fx_subject_mh_fibergeomap --input_format fibergeomap --use_multi_hot --test --ckpts experiments/fibergeomap_hcp105/cfgs/82fx_subject_mh_fibergeomap/ckpt-best.pth --total_bs 1 --downsample_rate 10


# TractCloud
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tractcloud_hcp105.yaml --exp_name test_82fx_subject_mh_tractcloud --input_format fiber_w_info --use_multi_hot --test --ckpts experiments/tractcloud_hcp105/cfgs/82fx_subject_mh_tractcloud/ckpt-best.pth --total_bs 1 --downsample_rate 1000
