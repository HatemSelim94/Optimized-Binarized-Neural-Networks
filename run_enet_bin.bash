python fp_models/enet/run_bin.py --data_name 'cityscapes' \
--batch_size 16 --image_size 448 --num_of_classes 3 --train_subset 700 --val_subset 500 \
--epochs 200 --network_optim 'Adam' --network_optim_bin_lr 1e-2 --network_optim_fp_lr 1e-2 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.2 \
--experiment_path 'fp_models/enet/experiments/' --experiment_name 'pure_bin' --device 'cuda' \
--seed 4 --decay_val 0.5 --decay_step 10 --use_weights 1

exp_path='fp_models/enet/experiments/'
exp_name='pure_bin_3_cityscapes'
exp_path+=$exp_name
cp run_enet.bash $exp_path