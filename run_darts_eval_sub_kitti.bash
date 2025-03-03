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
exp_path='eval/darts/kitti_experiments/'
exp_name='final_paper_kitti_sub_arch_a_1_crossval_bin_last_layer_1_kern_small_1'
exp_path+=$exp_name
rm -r $exp_path
mkdir -p $exp_path
cp run_darts_eval_sub_kitti.bash $exp_path

python eval/darts/run_kitti_sub_cross_val.py --data_name 'kitti' \
--image_size_w 640 --image_size_h 384 \
--batch_size 12 --image_size 448 --num_of_classes 19 --train_subset 170 --val_subset 30 \
--epochs 200 --network_optim 'Adam' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 1e-4 \
--network_optim_fp_weight_decay 5e-2 --network_optim_bin_betas 0.9 --network_sequence 'r,r,r,n,n,u' \
--stem_channels 60 --nodes_num 4 --edge_num 2 \
--experiment_path 'eval/darts/kitti_experiments/' --experiment_name 'final_paper_kitti_sub_arch_a_1_crossval_bin_last_layer_1_kern_small_1' --device 'cuda' \
--seed 4 --affine 1 --binary 1 --last_layer_binary 1 --last_layer_kernel_size 1 \
--genotype_path 'search/darts/experiments/' --search_exp_name 'final_darts_full_3c_2' --generate_jit 0 --generate_onnx 0 \
--padding_mode 'zeros' --dropout2d_prob 0.5 --network_type 'cells' --binarization 1 --use_skip 0 \
--activation 'htanh' --first_layer_activation 'htanh' --step_two 0 --seaborn_style 0 --use_old_ver 1 \
--channel_expansion_ratio_r 0.6 --channel_reduction_ratio_u 10 --channel_normal_ratio_n 0.3 \
--poly_scheduler 0 --lr_auto 0 --decay_val 0.5 --decay_step 10  --binary_aspp 1 --use_weights 1 \
--load_model 0 --load_experiment_name 'kitti_sub_arch_a_test_1' --upsample_mode 'bilinear' \
--use_maxpool 0 --merge_type 'sum' --train 1 --k 5 

