CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pointnets_hcp105.yaml --exp_name 82fx_subject_mh_pointnets_1fc --input_format subject --use_multi_hot --fc_layer 1

CUDA_VISIBLE_DEVICES=2 python main.py --config cfgs/pointnets_hcp105.yaml --exp_name 82fx_subject_mh_pointnets_3fc_dp0.5 --input_format subject --use_multi_hot --fc_layer 3 --dropout 0.5

CUDA_VISIBLE_DEVICES=3 python main.py --config cfgs/pointnets_hcp105.yaml --exp_name 82fx_subject_mh_pointnets_3fc_dp0.0 --input_format subject --use_multi_hot --fc_layer 3 --dropout 0.0
