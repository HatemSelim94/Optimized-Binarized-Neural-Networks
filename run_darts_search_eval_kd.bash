python search/darts/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 4 --image_size 224 --num_of_classes 3 --train_subset 200 --val_subset 100 \
--arch_optim 'adam' --arch_optim_lr 0.001 --arch_optim_beta0 0.5 --arch_optim_beta1 0.5 \
--arch_optim_eps 1e-08 --arch_optim_weight_decay 5e-4 --arch_optim_amsgrad False \
--epochs 40 --network_optim 'adamax' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 1e-5 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.9 --network_sequence 'r,r,u,u' \
--stem_channels 10 --nodes_num 4 --edge_num 2 --ops_num 6 --network_final_layer 'bin' \
--experiment_path 'search/darts/experiments/' --experiment_name 'exp5' --device 'cuda' \
--seed 4 --arch_start 5 --both True --affine False

exp_path='search/darts/experiments/'
exp_name='exp5'
exp_path+=$exp_name
cp run_darts.bash $exp_path