python fp_models/bedn/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 16 --image_size 448 --num_of_classes 3 --train_subset 300 --val_subset 300 \
--epochs 200 --network_optim 'SGD' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 1e-3 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.2 \
--experiment_path 'fp_models/bedn/experiments/' --experiment_name 'bedn_bops' --device 'cuda' \
--seed 4 --decay_val 1 --decay_step 5

exp_path='fp_models/bedn/experiments/'
exp_name='bedn_bops'
exp_path+=$exp_name
cp run_bedn.bash $exp_path