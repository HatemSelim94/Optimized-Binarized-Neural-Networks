python eval/darts/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 6 --image_size 320 --num_of_classes 3 --train_subset 20 --val_subset 20 \
--epochs 100 --network_optim 'adam' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 1e-5 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.2 --network_sequence 'r,n,r,u,n,u' \
--stem_channels 20 --nodes_num 3 --edge_num 2 --network_final_layer 'bin' \
--experiment_path 'eval/darts/experiments/' --experiment_name 'r_n_r_u_n_u' --device 'cuda' \
--seed 4 --affine True --binary False --last_layer_binary True --last_layer_kernel_size 3 \
--genotype_path 'search/darts/experiments/' --search_exp_name 'r_n_u_test' --jit False \
--padding_mode 'zeros' --dropout2d_prob 0.0
exp_path='eval/darts/experiments/'
exp_name='r_n_r_u_n_u'
exp_path+=$exp_name
cp run_darts_eval.bash $exp_path