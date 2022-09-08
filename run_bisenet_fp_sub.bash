python fp_models/bisenet/run_fp_sub.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--image_size_w 512 --image_size_h 512 \
--batch_size 16 --image_size 448 --num_of_classes 19 --train_subset 2900 --val_subset 500 \
--epochs 300 --network_optim 'Adam' --network_optim_bin_lr 1e-2 --network_optim_fp_lr 1e-4 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.2 \
--experiment_path 'fp_models/bisenet/sub_fp_experiments/' --experiment_name 'bisenet_sub_cityscapes_test_final_4' --device 'cuda' \
--seed 4 --decay_val 0.9 --decay_step 10 --use_weights 1

exp_path='fp_models/bisenet/sub_fp_experiments/'
exp_name='bisenet_sub_cityscapes_test_final_4'
exp_path+=$exp_name 
cp run_bisenet_fp_sub.bash $exp_path