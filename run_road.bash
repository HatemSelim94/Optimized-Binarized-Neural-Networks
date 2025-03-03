# to run this file, run this in the terminal: CUBLAS_WORKSPACE_CONFIG=:4096:8 bash run_darts_eval.bash
# activation accepted values [relu, htanh]: bin_input -> binconv -> relu -> batchnorm | bin_input -> binconv -> batchnorm -> htanh 
# first_layer_activation accepted values [relu, tanh]
# data_name [cityscapes, kitti][700, 500| 150, 50][decay_0.5, decay_10|0.9, 20]
# binarizaton accepted values [1, 2]
# network_type accepted values [cells, aspp]. If aspp, network upsamples by scale of 4
# network_sequence: any sequence ex: r,n,r,n,u,n,u or r,r,u,u or r,r,r,n,n,u
#stem channels 40
#final_darts_full_3c_2
# merge_type ['sum','conv']
python eval/darts/run_road.py --batch_size 8 --num_of_classes 2 \
--image_size_w 1280 --image_size_h 384 \
--epochs 150 --network_optim 'Adam' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 1e-4 \
--network_optim_fp_weight_decay 5e-2 --network_optim_bin_betas 0.9 --network_sequence 'r,r,r,n,n,u' \
--stem_channels 25 --nodes_num 5 --edge_num 2 \
--experiment_path 'eval/darts/road_experiments/' --experiment_name 'final_paper_model_b_1' --device 'cuda' \
--seed 4 --affine 1 --binary 1 --last_layer_binary 1 --last_layer_kernel_size 3 \
--genotype_path 'search/bnas/experiments/' --search_exp_name 'bnas_s_exp_5_nodes_1' --generate_jit 0 --generate_onnx 0 \
--padding_mode 'zeros' --dropout2d_prob 0.01 --network_type 'cells' --binarization 1 --use_skip 0 \
--activation 'htanh' --first_layer_activation 'htanh' --step_two 0 --seaborn_style 0 --use_old_ver 1 \
--channel_expansion_ratio_r 0.35 --channel_reduction_ratio_u 1 --channel_normal_ratio_n 0.2 \
--poly_scheduler 0 --lr_auto 0 --decay_val 0.5 --decay_step 20  --binary_aspp 1 --use_weights 1 \
--load_model 0 --load_experiment_name 'complete_test_sum' --upsample_mode 'bilinear' \
--use_maxpool 0 --merge_type 'sum' --optimize_merge 0

exp_path='eval/darts/road_experiments/'
exp_name='final_paper_model_b_1'
exp_path+=$exp_name
cp run_road.bash $exp_path