import sys
from sklearn.utils import shuffle
sys.path.append("/home/hatem/Projects/server/final_repo/final_repo")
import torch
import torch.nn as nn
from torch.utils.data import Subset, ConcatDataset, SubsetRandomSampler
import argparse
import os
import numpy as np

from architecture_eval import Network
from processing import Transformer, DataSets, KittiRoad, CityScapes  
from utilities import train, infer, set_seeds, Clipper, DataPlotter, Tracker, model_info, clean_dir, prepare_ops_metrics, jit_save, onnx_save, layers_state_setter, LR_Scheduler, Logger

parser  = argparse.ArgumentParser('ROAD')
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--image_size_w', type=int, default=1280)
parser.add_argument('--image_size_h', type=int, default=384)
parser.add_argument('--num_of_classes', type=int, default=2)

parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--network_optim', type=str, default='adam')
parser.add_argument('--network_optim_bin_lr', type=float, default=1e-3)
parser.add_argument('--network_optim_fp_lr', type=float, default=1e-5)
parser.add_argument('--network_optim_fp_weight_decay', type=float, default=5e-4)
parser.add_argument('--network_optim_bin_betas', type=float, default=0.5)
parser.add_argument('--network_sequence', type=str, default='r,r,u,u')
parser.add_argument('--stem_channels', type=int, default=64)
parser.add_argument('--nodes_num', type=int, default=4)
parser.add_argument('--edge_num', type=int, default=2)
parser.add_argument('--ops_num', type=int, default=6)
#parser.add_argument('--network_scheduler', type=str, default='lambda')
parser.add_argument('--experiment_path', type=str, default='eval/darts/sub_experiments/')
parser.add_argument('--experiment_name', type=str, default='exp1')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--affine', type=int, default=1)
parser.add_argument('--binary', type=int, default=1)
parser.add_argument('--last_layer_binary', type=int,default=1)
parser.add_argument('--last_layer_kernel_size', type=int, default=3)
parser.add_argument('--genotype_path', type=str, default='search/darts/experiments/')
parser.add_argument('--search_exp_name', type=str, default='exp1')
parser.add_argument('--jit', type=int, default=0)
parser.add_argument('--padding_mode', type=str, default='zeros')
parser.add_argument('--dropout2d_prob', type=float, default=0.5)
parser.add_argument('--network_type', type=str, default='cells')
parser.add_argument('--binarization', type=int, default=1)
parser.add_argument('--first_layer_activation', type=str, default='htanh')
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--use_skip', type=int, default=1)
parser.add_argument('--onnx', type=int, default=0)
parser.add_argument('--generate_onnx', type=int, default=0)
parser.add_argument('--generate_jit', type=int, default=0)
parser.add_argument('--use_kd', type=int, default=0)
parser.add_argument('--step_two', type=int, default=30)
parser.add_argument('--seaborn_style', type=int, default=0)
parser.add_argument('--use_old_ver', type=int, default=0)
parser.add_argument('--channel_expansion_ratio_r', type= float, default=2)
parser.add_argument('--channel_reduction_ratio_u', type=float, default=14)
parser.add_argument('--channel_normal_ratio_n', type=float, default=0.25)
parser.add_argument('--poly_scheduler', type=int, default=0)
parser.add_argument('--lr_auto',type=int, default=1)
parser.add_argument('--decay_val', type= float, default=0.01)
parser.add_argument('--decay_step', type=int, default=20)
parser.add_argument('--binary_aspp', type=int, default=1)
parser.add_argument('--use_weights', type=int, default=0)
parser.add_argument('--teacher', type=int, default=0)
parser.add_argument('--load_experiment_name', type=str, default='exp1')
parser.add_argument('--load_model', type=int, default=0)
parser.add_argument('--upsample_mode', type=str, default='bilinear')
parser.add_argument('--use_maxpool', type=int, default=0)
parser.add_argument('--merge_type', type=str,default='sum')
args = parser.parse_args()
torch.cuda.empty_cache()

assert(args.use_old_ver != args.use_skip)


def main():
    set_seeds(args.seed)
    clean_dir(args)
    if not torch.cuda.is_available():
        sys.exit(1)
    train_transforms = Transformer.get_transforms({'normalize':{'mean':CityScapes.mean,'std':CityScapes.std}, 
                                                    'resize':{'size':[args.image_size_h,args.image_size_w]},
                                                    'random_horizontal_flip':{'flip_prob':0.2}})
    test_transforms = Transformer.get_transforms({'normalize':{'mean':CityScapes.mean,'std':CityScapes.std},
                                                    'resize':{'size':[args.image_size_h,args.image_size_w]}})
    dataset = KittiRoad(transforms=train_transforms) 
    print(dataset[0][0].shape)
    classes_weights = dataset.weights.cuda() if args.use_weights else None
    criterion = nn.CrossEntropyLoss(ignore_index = dataset.ignore_index,weight=classes_weights, reduction='none')
    criterion = criterion.to(args.device)  
    net = Network(args).to(args.device) 
    net._set_criterion(criterion)
    input_shape = (1, 3, args.image_size_h, args.image_size_w)
    if args.load_model:
        net = Network(args)
        net._set_criterion(criterion) 
        net.load_state_dict(torch.load(os.path.join(args.experiment_path, args.load_experiment_name,'model.pt')), strict=False)
        net.to(args.device)
    indices = [i for i in range(len(dataset))]
    np.random.shuffle(indices)
    train_idx, val_idx = indices[60:], indices[:60]
    train_generator = torch.Generator()
    val_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    val_generator.manual_seed(args.seed)
    train_sampler = SubsetRandomSampler(train_idx, generator=train_generator)
    val_sampler = SubsetRandomSampler(val_idx, generator=val_generator)
    train_loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, pin_memory = True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size= args.batch_size, pin_memory = True, sampler=val_sampler)
    # TEST SET
    test_dataset = KittiRoad(split='test', transforms=test_transforms)
    test_loader = torch.utils.data.DataLoader(
                    test_dataset, 
                    batch_size= 1, pin_memory = True)
    model_info(net, input_shape, save=True, dir=os.path.join(args.experiment_path, args.experiment_name), verbose=True)
    #prepare_ops_metrics(net, input_shape)
    logger = Logger(os.path.join(args.experiment_path, args.experiment_name,'logs.log'))
    num_of_classes = args.num_of_classes
    if args.lr_auto:
        lr  = 0.00004*args.batch_size/16
        params_list = [{'params': net.first_layer.parameters(), 'lr': lr},]
        params_list.append({'params': net.cells[:-1].parameters(), 'lr': lr})
        params_list.append({'params': net.cells[-1].parameters(), 'lr': lr*10})
        if hasattr(net, 'last_layer'):
            params_list.append({'params': net.last_layer.parameters(), 'lr': lr*10})
        if hasattr(net, 'binaspp'):
            params_list.append({'params': net.binaspp.parameters(), 'lr': lr*10})
        if hasattr(net, 'auxlayer'):
            params_list.append({'params': net.auxlayer.parameters(), 'lr': lr*10})
        optimizer = Clipper.get_clipped_optim(args.network_optim, [params_list, lr], {'weight_decay':args.network_optim_fp_weight_decay})

    else:
        fp_params = [p for p in net.parameters() if not hasattr(p,'bin')]
        bin_params = [p for p in net.parameters() if hasattr(p,'bin')]
        optim_args = [[{'params':fp_params, 'weight_decay':args.network_optim_fp_weight_decay,'lr':args.network_optim_fp_lr},{'params':bin_params, 'weight_decay':0,'lr':args.network_optim_bin_lr, 'betas':[args.network_optim_bin_betas, args.network_optim_bin_betas ]}]]
        optimizer = Clipper.get_clipped_optim(args.network_optim, optim_args)
    if args.poly_scheduler:
        scheduler= LR_Scheduler(args.epochs,optimizer, len(train_loader))
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(20, args.epochs, args.decay_step)], gamma=args.decay_val)
    data = DataPlotter(os.path.join(args.experiment_path, args.experiment_name))
    tracker = Tracker(args.epochs)
    tracker.start()
    if args.step_two:
        layers_state_setter(net, input=True, weight=False) # binarized input, fp weights
    for epoch in range(args.epochs):
        # training
        train_miou, train_loss = train(train_loader, net, criterion, optimizer, num_of_classes, scheduler=scheduler, epoch=epoch, poly_scheduler=args.poly_scheduler)
        miou, loss= infer(val_loader, net, criterion, num_of_classes=num_of_classes, logger= logger)
        if not args.poly_scheduler:
            scheduler.step()
        tracker.print(train_loss,train_miou, loss, miou, epoch=epoch)
        data.store(epoch, train_loss, loss, train_miou, miou)
        data.plot(mode='all', save=True, seaborn=args.seaborn_style)
        data.save_as_json()
        if args.step_two:
            if epoch == args.step_two:
                layers_state_setter(net, input=True, weight=True)

    tracker.end()
    if args.generate_onnx:
        args.onnx = 1
        input_shape = (1, 3, 376, 672)
        new_net=Network(args).to(args.device)
        new_net.load_state_dict(net.state_dict(), strict=False)
        onnx_save(new_net.eval(), input_shape,os.path.join(args.experiment_path, args.experiment_name))
    if args.generate_jit:
        args.jit= 1
        input_shape = (1, 3, 376, 672)
        new_net=Network(args).to(args.device)
        new_net.load_state_dict(net.state_dict(), strict=False)
        jit_save(new_net.eval(),input_shape,os.path.join(args.experiment_path, args.experiment_name))
    torch.save(net.state_dict(), os.path.join(args.experiment_path, args.experiment_name,'model.pt'))
    net.eval()
    test_loader = torch.utils.data.DataLoader(
                test_dataset, 
                batch_size=1)
    with torch.no_grad():
        for i, (img, image_name, w, h) in enumerate(test_loader):
            img = img.cuda()
            output = net(img)
            #print(output.shape)
            #prediction = torch.argmax(output, 1)
            #DataSets.plot_image_label(prediction.cpu().squeeze(dim=0), id, val_dataset, transforms=plot_transforms,show_titles=False,show=False ,save=True, save_dir=os.path.join(args.experiment_path, args.experiment_name,'prediction_samples'), plot_name= f'sample_output_{id.item()}')
            DataSets.save_as_prob(w,h,output,name=image_name,path=os.path.join(args.experiment_path, args.experiment_name,'probs'))

if __name__ == '__main__':
    main()
    
