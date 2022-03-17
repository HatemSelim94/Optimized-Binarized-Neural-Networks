python fp_models/enc_dec/run.py --data_name 'cityscapes' --data_path '../../data/cityscapes/' \
--batch_size 10 --image_size 420 --num_of_classes 3 --train_subset 150 --val_subset 50 \
--epochs 200 --network_optim 'adam' --network_optim_bin_lr 1e-3 --network_optim_fp_lr 1e-6 \
--network_optim_fp_weight_decay 5e-4 --network_optim_bin_betas 0.2 \
--stem_channels 100 \
--experiment_path 'fp_models/enc_dec/experiments/' --experiment_name 'exp1' --device 'cuda' \
--seed 4 --device 'cuda' --encoder_layers 2 --decoder_layers 2 --activation 'relu' --kernel_size 3

exp_path='fp_models/enc_dec/experiments/'
exp_name='exp1'
exp_path+=$exp_name
cp run_fp_model_enc_dec.bash $exp_path