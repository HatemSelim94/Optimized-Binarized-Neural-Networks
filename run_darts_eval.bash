python eval/darts/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 4 --image_size 224 --num_of_classes 3 --train_subset 150 --val_subset 100 \
--epochs 40 --network_optim 'adamax' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 1e-5 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.9 --network_sequence 'r,n,u' \
--stem_channels 10 --nodes_num 4 --edge_num 2 --network_final_layer 'bin' \
--experiment_path 'eval/darts/experiments/' --experiment_name 'r_n_u_test' --device 'cuda' \
--seed 4 --affine True --binary True --last_layer_binary True --last_layer_kernel_size 3
--genotype_path 'search/darts/experiments/' --search_exp_name 'r_n_u_test'

exp_path='eval/darts/experiments/'
exp_name='r_n_u_test'
exp_path+=$exp_name
cp run_darts_eval.bash $exp_path