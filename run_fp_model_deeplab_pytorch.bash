python fp_models/deeplab_pytorch/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 16 --image_size 448 --num_of_classes 3 --train_subset 2 --val_subset 2 \
--epochs 400 --network_optim 'adam' --network_optim_bin_lr 1e-4 --network_optim_fp_lr 5e-6 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.2 \
--stem_channels 100 \
--experiment_path 'fp_models/deeplab_pytorch/experiments/' --experiment_name 'overfit' --device 'cuda' \
--seed 4 --device 'cuda'

exp_path='fp_models/deeplab_pytorch/experiments/'
exp_name='overfit'
exp_path+=$exp_name
cp run_fp_model_deeplab_pytorch.bash $exp_path
