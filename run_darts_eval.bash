python eval/darts/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 6 --image_size 224 --num_of_classes 3 --train_subset 300 --val_subset 100 \
--epochs 200 --network_optim 'Adam' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 1e-8 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.9 --network_sequence 'r,r,u,u' \
--stem_channels 40 --nodes_num 4 --edge_num 2 \
--experiment_path 'eval/darts/experiments/' --experiment_name 'full_net_224_slow_lr' --device 'cuda' \
--seed 4 --affine 1 --binary 1 --last_layer_binary 1 --last_layer_kernel_size 3 \
--genotype_path 'search/darts/experiments/' --search_exp_name 'r_n_u_test' --jit 0 --onnx 0\
--padding_mode 'zeros' --dropout2d_prob 0.1 --network_type 'cells' --binarization 1

exp_path='eval/darts/experiments/'
exp_name='full_net_224_slow_lr'
exp_path+=$exp_name
cp run_darts_eval.bash $exp_path