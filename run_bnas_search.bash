python search/bnas/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 4 --image_size 448 --num_of_classes 3 --train_subset 250 --val_subset 250 \
--arch_optim 'Adam' --arch_optim_lr 0.01 --arch_optim_beta0 0.9 --arch_optim_beta1 0.999 \
--arch_optim_eps 1e-08 --arch_optim_weight_decay 5e-4 --arch_optim_amsgrad 0 \
--epochs 10 --network_optim 'Adam' --network_optim_bin_lr 1e-2 --network_optim_fp_lr 1e-5 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.9 --network_sequence 'r,r,r,n,n,u' \
--stem_channels 15 --nodes_num 4 --edge_num 2 --ops_num 8 --network_final_layer 'bin' \
--experiment_path 'search/bnas/experiments/' --experiment_name 'af_bnas_4_n' --device 'cuda' \
--seed 4 --arch_start 4 --both 1 --affine 0 --binary 1 \
--last_layer_binary 1 --last_layer_kernel_size 3 \
--network_type 'cells' --first_layer_activation 'htanh' --activation 'htanh' --use_skip 0 \
--dropout2d_prob 0.001 --seaborn_style 0 --use_old_ver 1 --channel_expansion_ratio_r 0.5 --channel_reduction_ratio_u 2 \
--channel_normal_ratio_n 0.25 --binary_aspp 1 \
--bnas_t 4 --bnas_sample_epochs 4 --latency_gamma 0  --params_delta 0 --theta_ops 0 \
--required_latency_ms 5 --required_params_size_kb 100 --required_ops_mops 100 \
--arch_v_epochs 4 --arch_t_epochs 4

exp_path='search/bnas/experiments/'
exp_name='af_bnas_4_n'
exp_path+=$exp_name
cp run_bnas_search.bash $exp_path