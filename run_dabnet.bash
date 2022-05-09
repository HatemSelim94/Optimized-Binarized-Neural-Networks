python fp_models/dabnet/run.py --data_name 'kitti' \
--batch_size 16 --image_size 448 --num_of_classes 3 --train_subset 150 --val_subset 50 \
--epochs 200 --network_optim 'Adam' --network_optim_bin_lr 1e-2 --network_optim_fp_lr 1e-3 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.2 \
--experiment_path 'fp_models/dabnet/experiments/' --experiment_name 'enet_bin_kitti_3' --device 'cuda' \
--seed 4 --decay_val 0.9 --decay_step 10 --use_weights 1 --binary 1

exp_path='fp_models/dabnet/experiments/'
exp_name='enet_bin_kitti_3'
exp_path+=$exp_name
cp run_dabnet.bash $exp_path