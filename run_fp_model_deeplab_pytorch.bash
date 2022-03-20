python fp_models/deeplab_pytorch/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 10 --image_size 512 --num_of_classes 3 --train_subset 500 --val_subset 200 \
--epochs 200 --network_optim 'adam' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 5e-4 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.2 \
--stem_channels 100 \
--experiment_path 'fp_models/deeplab_pytorch/experiments/' --experiment_name 'exp1' --device 'cuda' \
--seed 4 --device 'cuda'

exp_path='fp_models/deeplab_pytorch/experiments/'
exp_name='exp1'
exp_path+=$exp_name
cp run_fp_model_deeplab_pytorch.bash $exp_path