# to run this file, run this in the terminal: CUBLAS_WORKSPACE_CONFIG=:4096:8 bash run_darts_eval.bash
# activation accepted values [relu, htanh]: bin_input -> binconv -> relu -> batchnorm | bin_input -> binconv -> batchnorm -> htanh 
# first_layer_activation accepted values [relu, tanh]
# data_name [cityscapes, kitti][700, 500| 150, 50][decay_0.5, decay_10|0.9, 20]
# binarizaton accepted values [1, 2]
# network_type accepted values [cells, aspp]. If aspp, network upsamples by scale of 4
# network_sequence: any sequence but expected to have r as much as u  ex: r,n,r,n,u,n,u or r,r,u,u
#stem channels 40
#final_darts_full_3c_2
python eval/darts/fastinf_test.py --data_name 'kitti' \
--batch_size 16 --image_size 448 --num_of_classes 8 --train_subset 150 --val_subset 50 \
--epochs 200 --network_optim 'Adam' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 1e-4 \
--network_optim_fp_weight_decay 5e-2 --network_optim_bin_betas 0.9 --network_sequence 'r,r,r,n,n,u' \
--stem_channels 60 --nodes_num 3 --edge_num 2 \
--experiment_path 'eval/darts/experiments/' --experiment_name 'fast_inf_test' --device 'cuda' \
--seed 4 --affine 1 --binary 1 --last_layer_binary 1 --last_layer_kernel_size 3 \
--genotype_path 'search/darts/experiments/' --search_exp_name 'city_8_cls_3_nodes' --generate_jit 1 --generate_onnx 1 \
--padding_mode 'zeros' --dropout2d_prob 0.01 --network_type 'cells' --binarization 1 --use_skip 0 \
--activation 'htanh' --first_layer_activation 'htanh' --step_two 0 --seaborn_style 0 --use_old_ver 1 \
--channel_expansion_ratio_r 0.75 --channel_reduction_ratio_u 8 --channel_normal_ratio_n 0.5 \
--poly_scheduler 0 --lr_auto 0 --decay_val 0.9 --decay_step 20  --binary_aspp 1 --use_weights 1 \
--load_model 1 --load_experiment_name 'model_compare_kitti_8_cls_3_nodes'

exp_path='eval/darts/experiments/'
exp_name='fast_inf_test'
exp_path+=$exp_name
cp run_darts_eval_fastinf.bash $exp_path