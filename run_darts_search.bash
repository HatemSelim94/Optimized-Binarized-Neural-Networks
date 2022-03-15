python search/darts/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 4 --image_size 224 --num_of_classes 3 --train_subset 10 --val_subset 10 \
--arch_optim 'adam' --arch_optim_lr 0.001 --arch_optim_beta0 0.5 --arch_optim_beta1 0.5 \
--arch_optim_eps 1e-08 --arch_optim_weight_decay 5e-4 --arch_optim_amsgrad False \
--epochs 40 --network_optim 'adamax' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 1e-5 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.9 --network_sequence 'r,n,u' \
--stem_channels 10 --nodes_num 4 --edge_num 2 --ops_num 6 --network_final_layer 'bin' \
--experiment_path 'search/darts/experiments/' --experiment_name 'r_n_u_test' --device 'cuda' \
--seed 4 --arch_start 5 --both True --affine False --binary True --ops_obj_beta 0.0 --params_obj_gamma 0.0 \
--latency_obj_delta 0.0 --last_layer_binary True --last_layer_kernel_size 3

exp_path='search/darts/experiments/'
exp_name='r_n_u_test'
exp_path+=$exp_name
cp run_darts_search.bash $exp_path