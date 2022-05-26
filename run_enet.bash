python fp_models/enet/run.py --data_name 'cityscapes' \
--batch_size 16 --image_size 448 --num_of_classes 3 --train_subset 1500 --val_subset 500 \
--epochs 200 --network_optim 'Adam' --network_optim_bin_lr 1e-2 --network_optim_fp_lr 1e-4 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.2 \
--experiment_path 'fp_models/enet/experiments/' --experiment_name 'enet_cityscapes_3_bin_2' --device 'cuda' \
--seed 4 --decay_val 0.9 --decay_step 10 --use_weights 1

exp_path='fp_models/enet/experiments/'
exp_name='enet_cityscapes_3_bin_2'
exp_path+=$exp_name
cp run_enet.bash $exp_path