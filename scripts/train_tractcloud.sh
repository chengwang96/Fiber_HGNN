CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/tractcloud_hcp105.yaml --exp_name 82fx_subject_mh_tractcloud_1 --input_format fiber_w_info --use_multi_hot --total_bs 1024

CUDA_VISIBLE_DEVICES=2 python main.py --config cfgs/tractcloud_hcp105.yaml --exp_name 82fx_subject_mh_tractcloud_2 --input_format fiber_w_info --use_multi_hot --total_bs 1024

CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/tractcloud_hcp105.yaml --exp_name 82fx_subject_mh_tractcloud_3 --input_format fiber_w_info --use_multi_hot --total_bs 1024
