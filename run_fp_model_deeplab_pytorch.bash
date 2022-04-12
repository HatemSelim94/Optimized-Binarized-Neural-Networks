python fp_models/deeplab_pytorch/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 16 --image_size 448 --num_of_classes 3 --train_subset 700 --val_subset 500 \
--epochs 50 --network_optim 'adam' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 5e-6 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.2 \
--stem_channels 100 \
--experiment_path 'fp_models/deeplab_pytorch/experiments/' --experiment_name 'exp6' --device 'cuda' \
--seed 4 --device 'cuda'

exp_path='fp_models/deeplab_pytorch/experiments/'
exp_name='exp6'
exp_path+=$exp_name
cp run_fp_model_deeplab_pytorch.bash $exp_path
