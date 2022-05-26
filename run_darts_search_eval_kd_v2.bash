python eval/darts/run_kd_v2.py --data_name 'kitti' \
--batch_size 16 --image_size 448 --num_of_classes 3 --train_subset 150 --val_subset 50 \
--epochs 100 --network_optim 'Adam' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 1e-4 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.9 --network_sequence 'r,r,r,n,n,u' \
--stem_channels 60 --nodes_num 4 --edge_num 2 \
--experiment_path 'eval/darts/experiments/' --experiment_name 'kd_fp_best_nearest_2' --device 'cuda' \
--seed 4 --affine 1 --binary 1 --last_layer_binary 1 --last_layer_kernel_size 3 \
--genotype_path 'search/darts/experiments/' --search_exp_name 'final_darts_full_3c_2' --generate_jit 1 --generate_onnx 1 \
--padding_mode 'zeros' --dropout2d_prob 0.01 --network_type 'cells' --binarization 1 --use_skip 0 \
--activation 'htanh' --first_layer_activation 'htanh' --step_two 0 --seaborn_style 0 --use_old_ver 1 \
--channel_expansion_ratio_r 0.5 --channel_reduction_ratio_u 2 --channel_normal_ratio_n 0.25 \
--poly_scheduler 0 --lr_auto 0 --decay_val 0.9 --decay_step 20  --binary_aspp 1 --use_weights 1  --use_kd 1 \
--teacher_model_path 'eval/darts/experiments/fp_best_model/model.pt' --teacher_binary 0 --teacher_affine 1 \
--teacher_nodes_num 4 --teacher_edge_num 2 --teacher_ops_num 6 --teacher_cells_sequence 'r,r,r,n,n,u' \
--teacher_stem_channels 60 --teacher_genotype_path 'search/darts/experiments/' --teacher_use_old_ver 1 \
--teacher_search_exp_name 'final_darts_full_3c_2' --teacher_dropout2d 0.01 --teacher_padding_mode 'zeros' \
--teacher_binarization 1 --teacher_last_layer_binary 0 --teacher_activation 'htanh' --teacher_first_layer_activation 'htanh' --teacher_use_skip 0 \
--teacher_use_kd 1 --teacher_last_layer_kernel_size 3 --teacher_channel_expansion_ratio_r 0.5 --teacher_channel_reduction_ratio_u 2 \
--teacher_channel_normal_ratio_n 0.25 --upsample_mode 'nearest' --teacher_upsample_mode 'nearest' --load_model 0
exp_path='eval/darts/experiments/'
exp_name='kd_fp_best_nearest'
exp_path+=$exp_name
cp run_darts_eval.bash $exp_path