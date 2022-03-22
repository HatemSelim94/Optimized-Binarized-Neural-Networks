python search/darts/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 4 --image_size 224 --num_of_classes 3 --train_subset 200 --val_subset 100 \
--arch_optim 'Adam' --arch_optim_lr 0.0001 --arch_optim_beta0 0.9 --arch_optim_beta1 0.999 \
--arch_optim_eps 1e-08 --arch_optim_weight_decay 5e-4 --arch_optim_amsgrad 0 \
--epochs 50 --network_optim 'Adam' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 1e-5 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.9 --network_sequence 'r,r,u,u' \
--stem_channels 10 --nodes_num 4 --edge_num 2 --ops_num 6 --network_final_layer 'bin' \
--experiment_path 'search/darts/experiments/' --experiment_name 'old_test' --device 'cuda' \
--seed 4 --arch_start 5 --both 1 --affine 0 --binary 1 --ops_obj_beta 0.0 --params_obj_gamma 0.0 \
--latency_obj_delta 0.0 --last_layer_binary 1 --last_layer_kernel_size 3 \
--network_type 'cells' --first_layer_activation 'htanh' --activation 'htanh' --use_skip 0 \
--dropout2d_prob 0.001 --seaborn_style 0 --use_old_ver 1

exp_path='search/darts/experiments/'
exp_name='old_test'
exp_path+=$exp_name
cp run_darts_search.bash $exp_path