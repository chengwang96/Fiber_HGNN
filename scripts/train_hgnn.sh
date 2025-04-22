# HGNN
# CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_1fc --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_1fc/ckpt-best.pth --fc_layer 1

CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20 --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --fc_layer 3 --knn 20

# CUDA_VISIBLE_DEVICES=2 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_knn20_pointnet --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnet_hcp105/cfgs/82fx_subject_mh_pointnet/ckpt-best.pth --knn 20


# HGNN + select
CUDA_VISIBLE_DEVICES=7 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_select --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --fc_layer 3 --knn 20 --use_select


# HGNN + FA + LS
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_fa_ls --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_ls --fc_layer 3 --knn 20


# HGNN + FA + LS + select
CUDA_VISIBLE_DEVICES=2 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_fa_ls_select --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_ls --fc_layer 3 --knn 20 --use_select


# HGNN + GTFA + LS
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_gtfa_ls --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_gtfa --use_ls --fc_layer 3 --knn 20


# HGNN + GTFA + LS + select
CUDA_VISIBLE_DEVICES=4 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_gtfa_ls_select --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_gtfa --use_ls --fc_layer 3 --knn 20 --use_select


# HGNN + DFA + LS
CUDA_VISIBLE_DEVICES=5 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20


# HGNN + DFA + LS + select
# knn
CUDA_VISIBLE_DEVICES=6 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_select --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20 --use_select


CUDA_VISIBLE_DEVICES=7 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn10_dp0.5_dfa_ls_select --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 10 --use_select


CUDA_VISIBLE_DEVICES=4 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn30_dp0.5_dfa_ls_select --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 30 --use_select


CUDA_VISIBLE_DEVICES=5 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn5_dp0.5_dfa_ls_select --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 5 --use_select


# depth + keypoint
# kp = 1
CUDA_VISIBLE_DEVICES=6 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_d1kp1 --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20 --depth 1

CUDA_VISIBLE_DEVICES=6 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_d2kp1 --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20 --depth 2

CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_d3kp1 --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20 --depth 3

CUDA_VISIBLE_DEVICES=2 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_d4kp1 --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20 --depth 4

# kp = 3
CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_select_d1kp3 --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20 --use_select --depth 1 --keypoint_num 3

CUDA_VISIBLE_DEVICES=4 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_select_d2kp3 --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20 --use_select --depth 2 --keypoint_num 3

CUDA_VISIBLE_DEVICES=6 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_select_d3kp3 --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20 --use_select --depth 3 --keypoint_num 3

CUDA_VISIBLE_DEVICES=5 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_select_d4kp3 --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20 --use_select --depth 4 --keypoint_num 3

# kp = 5
CUDA_VISIBLE_DEVICES=2 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_select_d1kp5 --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20 --use_select --depth 1 --keypoint_num 5

CUDA_VISIBLE_DEVICES=6 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_select_d2kp5 --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20 --use_select --depth 2 --keypoint_num 5

CUDA_VISIBLE_DEVICES=6 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_select_d3kp5 --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20 --use_select --depth 3 --keypoint_num 5

CUDA_VISIBLE_DEVICES=7 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_ls_select_d4kp5 --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --fc_layer 3 --knn 20 --use_select --depth 4 --keypoint_num 5


# HGNN + DFA + DLS
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_1fc_dfa_dls --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_1fc/ckpt-best.pth --use_fa --use_dfa --use_ls --use_dls --fc_layer 1

CUDA_VISIBLE_DEVICES=6 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa_dls --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --use_ls --use_dls --fc_layer 3 --knn 20

CUDA_VISIBLE_DEVICES=2 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_knn20_pointnet_dfa_dls --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnet_hcp105/cfgs/82fx_subject_mh_pointnet/ckpt-best.pth --use_fa --use_dfa --use_ls --use_dls --knn 20


# HGNN + GTFA
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_1fc_gtfa --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_1fc/ckpt-best.pth --use_fa --use_gtfa --fc_layer 1

CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_gtfa --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_gtfa --fc_layer 3 --knn 20

CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_knn20_pointnet_gtfa --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnet_hcp105/cfgs/82fx_subject_mh_pointnet/ckpt-best.pth --use_fa --use_gtfa --knn 20


# HGNN + FA
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_1fc_fa --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_1fc/ckpt-best.pth --use_fa --fc_layer 1

CUDA_VISIBLE_DEVICES=4 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_fa --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --fc_layer 3 --knn 20

CUDA_VISIBLE_DEVICES=4 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_knn20_pointnet_fa --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnet_hcp105/cfgs/82fx_subject_mh_pointnet/ckpt-best.pth --use_fa --knn 20


# HGNN + DFA
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_1fc_dfa --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_1fc/ckpt-best.pth --use_fa --use_dfa --fc_layer 1

CUDA_VISIBLE_DEVICES=4 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dfa --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_fa --use_dfa --fc_layer 3 --knn 20

CUDA_VISIBLE_DEVICES=5 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_knn20_pointnet_dfa --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnet_hcp105/cfgs/82fx_subject_mh_pointnet/ckpt-best.pth --use_fa --use_dfa --knn 20


# HGNN + LS
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_1fc_ls --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_1fc/ckpt-best.pth --use_ls --fc_layer 1

CUDA_VISIBLE_DEVICES=5 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_ls --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_ls --fc_layer 3 --knn 20

CUDA_VISIBLE_DEVICES=6 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_knn20_pointnet_ls --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnet_hcp105/cfgs/82fx_subject_mh_pointnet/ckpt-best.pth --use_ls --knn 20


# HGNN + DLS
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_1fc_dls --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_1fc/ckpt-best.pth --use_ls --use_dls --fc_layer 1

CUDA_VISIBLE_DEVICES=6 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_3fc_knn20_dp0.5_dls --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnets_hcp105/cfgs/82fx_subject_mh_pointnets_3fc_dp0.5/ckpt-best.pth --use_ls --use_dls --fc_layer 3 --knn 20

CUDA_VISIBLE_DEVICES=7 python main.py --config cfgs/hgnn_hcp105.yaml --exp_name 82fx_subject_mh_hgnn_knn20_pointnet_dls --input_format subject --use_multi_hot --run_graph --ckpts experiments/pointnet_hcp105/cfgs/82fx_subject_mh_pointnet/ckpt-best.pth --use_ls --use_dls --knn 20
