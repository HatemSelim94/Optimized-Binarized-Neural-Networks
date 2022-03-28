# to run this file, run this in the terminal: CUBLAS_WORKSPACE_CONFIG=:4096:8 bash run_darts_eval.bash
# activation accepted values [relu, htanh]: bin_input -> binconv -> relu -> batchnorm | bin_input -> binconv -> batchnorm -> htanh 
# first_layer_activation accepted values [relu, tanh]
# binarizaton accepted values [1, 2]
# network_type accepted values [cells, aspp]. If aspp, network upsamples by scale of 4
# network_sequence: any sequence but expected to have r as much as u  ex: r,n,r,n,u,n,u or r,r,u,u
#stem channels 40

python eval/darts/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 10 --image_size 448 --num_of_classes 3 --train_subset 700 --val_subset 500 \
--epochs 60 --network_optim 'Adam' --network_optim_bin_lr 1e-2 --network_optim_fp_lr 1e-4 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.9 --network_sequence 'r,r,n,n,n,u' \
--stem_channels 80 --nodes_num 4 --edge_num 2 \
--experiment_path 'eval/darts/experiments/' --experiment_name 'bnas_eval' --device 'cuda' \
--seed 4 --affine 1 --binary 1 --last_layer_binary 1 --last_layer_kernel_size 1 \
--genotype_path 'search/bnas/experiments/' --search_exp_name 'test_1' --generate_jit 1 --generate_onnx 1 \
--padding_mode 'zeros' --dropout2d_prob 0.0 --network_type 'cells' --binarization 1 --use_skip 0 \
--activation 'htanh' --first_layer_activation 'htanh' --step_two 0 --seaborn_style 0 --use_old_ver 1 \
--channel_expansion_ratio_r 0.25 --channel_reduction_ratio_u 14 --channel_normal_ratio_n 0.25 \
--poly_scheduler 0 --lr_auto 0 --decay_val 0.5 --decay_step 10  --binary_aspp 1

exp_path='eval/darts/experiments/'
exp_name='bnas_eval'
exp_path+=$exp_name
cp run_darts_eval.bash $exp_path