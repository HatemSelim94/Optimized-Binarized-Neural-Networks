python fp_models/deeplab/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 4 --image_size 448 --num_of_classes 3 --train_subset 1000 --val_subset 500 \
--epochs 50 --network_optim 'adam' --network_optim_fp_lr 1e-3 \
--network_optim_fp_weight_decay 5e-4 \
--experiment_path 'fp_models/deeplab/experiments/' --experiment_name 'batch_size_4' --device 'cuda' \
--seed 4 --device 'cuda'

exp_path='fp_models/deeplab/experiments/'
exp_name='batch_size_4'
exp_path+=$exp_name
cp run_fp_model_deeplab.bash $exp_path