python fp_models/dabnet/run_kd.py --data_name 'kitti' \
--batch_size 16 --image_size 448 --num_of_classes 3 --train_subset 150 --val_subset 50 \
--epochs 240 --network_optim 'Adam' --network_optim_bin_lr 1e-2 --network_optim_fp_lr 1e-3 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.2 \
--experiment_path 'fp_models/dabnet/experiments/' --experiment_name 'kd_exp4' --device 'cuda' \
--seed 4 --decay_val 0.9 --decay_step 10 --use_weights 1 --binary 0 --kd_epoch 40 --use_kd 1

exp_path='fp_models/dabnet/experiments/'
exp_name='kd_exp4'
exp_path+=$exp_name
cp run_dabnet_kd.bash $exp_path