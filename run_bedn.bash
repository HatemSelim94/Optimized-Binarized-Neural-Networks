python fp_models/bedn/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 16 --image_size 448 --num_of_classes 3 --train_subset 700 --val_subset 500 \
--epochs 100 --network_optim 'Adam' --network_optim_bin_lr 1e-2 --network_optim_fp_lr 1e-3 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.2 \
--experiment_path 'fp_models/bedn/experiments/' --experiment_name 'bedn_bops_diff_opt_2' --device 'cuda' \
--seed 4 --decay_val 0.9 --decay_step 10

exp_path='fp_models/bedn/experiments/'
exp_name='bedn_bops_diff_opt_2'
exp_path+=$exp_name
cp run_bedn.bash $exp_path