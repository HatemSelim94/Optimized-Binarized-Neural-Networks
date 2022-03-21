# to run this file, run this in ther terminal: CUBLAS_WORKSPACE_CONFIG=:4096:8 bash run_darts_eval.bash
# activation accepted values [relu, htanh]
# first_layer_activation accepted values [relu, tanh]
# binarizaton accepted values [1, 2]
# network_type accepted values [cells, aspp]. If aspp, network upsamples by scale of 4
# sequence r,r,u,u
#stem channels 40

python eval/darts/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 6 --image_size 224 --num_of_classes 3 --train_subset 18 --val_subset 6 \
--epochs 1 --network_optim 'Adam' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 1e-8 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.9 --network_sequence 'r,u' \
--stem_channels 10 --nodes_num 3 --edge_num 2 \
--experiment_path 'eval/darts/experiments/' --experiment_name 'new_test' --device 'cuda' \
--seed 10 --affine 1 --binary 1 --last_layer_binary 1 --last_layer_kernel_size 3 \
--genotype_path 'search/darts/experiments/' --search_exp_name 'r_n_u_test' --generate_jit 1 --generate_onnx 1 \
--padding_mode 'zeros' --dropout2d_prob 0.1 --network_type 'cells' --binarization 2 --use_skip 0 \
--activation 'relu' --first_layer_activation 'htanh' 

exp_path='eval/darts/experiments/'
exp_name='new_test'
exp_path+=$exp_name
cp run_darts_eval.bash $exp_path