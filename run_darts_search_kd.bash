python search/darts/run_kd.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 4 --image_size 448 --num_of_classes 3 --train_subset 300 --val_subset 300 \
--arch_optim 'Adam' --arch_optim_lr 0.01 --arch_optim_beta0 0.9 --arch_optim_beta1 0.999 \
--arch_optim_eps 1e-08 --arch_optim_weight_decay 5e-4 --arch_optim_amsgrad 0 \
--epochs 40 --network_optim 'Adam' --network_optim_bin_lr 1e-2 --network_optim_fp_lr 1e-3 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.9 --network_sequence 'r,r,r,n,n,u' \
--stem_channels 30 --nodes_num 4 --edge_num 2 --ops_num 6 \
--experiment_path 'search/darts/experiments/' --experiment_name 'kd_exp4' --device 'cuda' \
--seed 4 --arch_start 5 --both 1 --affine 0 --binary 1 --ops_obj_beta 0.0 --params_obj_gamma 0.0 \
--latency_obj_delta 0.0 --last_layer_binary 1 --last_layer_kernel_size 1 \
--network_type 'cells' --first_layer_activation 'htanh' --activation 'htanh' --use_skip 0 \
--dropout2d_prob 0.001 --seaborn_style 0 --use_old_ver 1 --channel_expansion_ratio_r 0.5 --channel_reduction_ratio_u 2 \
--channel_normal_ratio_n 0.25  --binary_aspp 1 --decay_step 10 --decay_val 0.9

exp_path='search/darts/experiments/'
exp_name='kd_exp4'
exp_path+=$exp_name
cp run_darts_search_kd.bash $exp_path