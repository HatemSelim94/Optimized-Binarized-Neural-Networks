import sys
sys.path.append("/home/hatem/Projects/server/final_repo/final_repo")
import torch
import torch.nn as nn
from torch.utils.data import Subset
import argparse
import os
from model_bin import DABNetBinKD as BinNetwork
from model import DABNetKD as FPNetwork
from processing import Transformer, DataSets
from processing.datasets import CityScapes, KittiDataset
from utilities.latency import get_latency
from utilities import infer, set_seeds, Clipper, DataPlotter, Tracker, train_kd_rl, model_info, clean_dir, prepare_ops_metrics,infer_kd
from thop import profile, clever_format
parser  = argparse.ArgumentParser('DABNET')
parser.add_argument('--data_name', type=str, default='cityscapes')
#parser.add_argument('--genotype_path', type=str, default='experiments/search/three_cells_stem_1/exp1/best_genotype')
#parser.add_argument('--model_path')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--num_of_classes', type=int, default=3)
parser.add_argument('--train_subset', type=int, default=300)
parser.add_argument('--val_subset', type=int, default=200)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--network_optim', type=str, default='adamax')
parser.add_argument('--network_optim_bin_lr', type=float, default=1e-3)
parser.add_argument('--network_optim_fp_lr', type=float, default=1e-5)
parser.add_argument('--network_optim_fp_weight_decay', type=float, default=5e-4)
parser.add_argument('--network_optim_bin_betas', type=float, default=0.5)
parser.add_argument('--experiment_path', type=str, default='fp_models/enet/experiments/')
parser.add_argument('--experiment_name', type=str, default='exp1')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--decay_val', type=float, default=0.7)
parser.add_argument('--decay_step', type=int, default=5)
parser.add_argument('--use_weights', type=int, default=0)
parser.add_argument('--binary', type=int, default=0)
parser.add_argument('--kd_epoch', type=int, default=200)
parser.add_argument('--use_kd', type=int, default=1)
args = parser.parse_args()
torch.cuda.empty_cache()
#FP
# 6.03 MB, 3.96GFLOPs | 8#cls miou: 84%  #3cls 97%
# latency: 3.61 | FPS: 277.38
#Bin
# 0.294 MB   8cls miou 40%  | 3cls 53%
# latency: 7.51 | FPS: 133.17

def main():
    set_seeds(args.seed)
    clean_dir(args)
    if not torch.cuda.is_available():
        sys.exit(1)
    train_transforms = Transformer.get_transforms({'normalize':{'mean':CityScapes.mean,'std':CityScapes.std}, 'resize':{'size':[args.image_size,args.image_size]},'random_horizontal_flip':{'flip_prob':0.2}})
    val_transforms = Transformer.get_transforms({'normalize':{'mean':CityScapes.mean,'std':CityScapes.std},'resize':{'size':[args.image_size,args.image_size]}})
    plot_transforms = Transformer.get_transforms({'resize':{'size':[args.image_size,args.image_size]}})
    train_dataset = DataSets.get_dataset(args.data_name, no_of_classes=args.num_of_classes, transforms=train_transforms)
    classes_weights = train_dataset.loss_weights_3 if args.num_of_classes ==3 else train_dataset.loss_weights_8
    classes_weights = classes_weights.cuda() if args.use_weights else None
    criterion = nn.CrossEntropyLoss(ignore_index = CityScapes.ignore_index,weight=classes_weights, reduction='none')
    criterion = criterion.to(args.device)
    bin_net = BinNetwork(classes=args.num_of_classes).to(args.device)  
    net = FPNetwork(classes=args.num_of_classes).to(args.device)
    #net._set_criterion(criterion)
    input_shape = (1, 3, args.image_size, args.image_size)
    l_ms, fps =get_latency(bin_net, input_shape)
    print(f'Latency: {l_ms} ms, FPS: {fps}')
    macs, params = profile(bin_net, inputs=(torch.randn(input_shape).cuda(),))
    macs, params = clever_format([macs, params], "%.3f")
    print(f'macs: {macs},  params: {params}')
    #train_transforms = Transformer.get_transforms({'normalize':{'mean':CityScapes.mean,'std':CityScapes.std}, 'resize':{'size':[448,448]},'center_crop':{'size':[args.image_size,args.image_size]},'random_horizontal_flip':{'flip_prob':0.2}})
    #val_transforms = Transformer.get_transforms({'normalize':{'mean':CityScapes.mean,'std':CityScapes.std},'resize':{'size':[448,448]},'center_crop':{'size':[args.image_size,args.image_size]}})
   
    if args.data_name == 'cityscapes':
        val_dataset = DataSets.get_dataset(args.data_name, no_of_classes=args.num_of_classes,split='val',transforms=val_transforms)
        train_idx = range(args.train_subset)
        val_idx = range(args.val_subset) 
        sub_train_dataset = Subset(train_dataset, [i for i in train_idx])
        sub_val_dataset = Subset(val_dataset, [i for i in val_idx])
    elif args.data_name == 'kitti':
        assert  args.train_subset + args.val_subset <= 200
              
        train_idx = range(args.train_subset)
        val_idx = range(args.train_subset, min(200,args.train_subset+args.val_subset)) 
        val_dataset = DataSets.get_dataset(args.data_name, no_of_classes=args.num_of_classes,split='train',transforms=val_transforms)
        train_idx = range(args.train_subset)
        val_idx = range(args.val_subset) 
        sub_train_dataset = Subset(train_dataset, [i for i in train_idx])
        sub_val_dataset = Subset(val_dataset, [i for i in val_idx])
    train_loader = torch.utils.data.DataLoader(
                    sub_train_dataset, 
                    batch_size=args.batch_size, pin_memory = True)
    val_loader = torch.utils.data.DataLoader(
                    sub_val_dataset, 
                    batch_size= args.batch_size, pin_memory = True)
    fp_params = [p for p in net.parameters()]
    print('size')
    print(sum([p.numel() for p in bin_net.parameters() if not hasattr(p, 'bin')])*32/4/1e6 + sum([p.numel() for p in bin_net.parameters() if hasattr(p, 'bin')])*1/4/1e6, 'MB')

    bin_fp_params = [p for p in bin_net.parameters() if not hasattr(p,'bin')]
    bin_params = [p for p in bin_net.parameters() if hasattr(p,'bin')]
    optim_args = [[{'params':bin_fp_params, 'weight_decay':args.network_optim_fp_weight_decay,'lr':args.network_optim_fp_lr},{'params':bin_params, 'weight_decay':0,'lr':args.network_optim_bin_lr, 'betas':[args.network_optim_bin_betas, args.network_optim_bin_betas ]}]]
    #optim_args = [[{'params':fp_params, 'weight_decay':args.network_optim_fp_weight_decay,'lr':args.network_optim_fp_lr},{'params':bin_params, 'weight_decay':0.001,'lr':args.network_optim_bin_lr}]]
    optimizer = Clipper.get_clipped_optim('Adam', optim_args)
    fp_optimizer = torch.optim.Adam(fp_params, lr=args.network_optim_fp_lr) 
    #optimizer=Clipper.get_clipped_optim('SGD',optim_args)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:((step/args.epochs))**2)
    fp_scheduler = torch.optim.lr_scheduler.MultiStepLR(fp_optimizer, milestones=[i for i in range(20, args.epochs, args.decay_step)], gamma=args.decay_val)
    bin_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(20, args.epochs, args.decay_step)], gamma=args.decay_val)
    data = DataPlotter(os.path.join(args.experiment_path, args.experiment_name))
    tracker = Tracker(args.epochs)
    tracker.start()
    start_kd = False
    for epoch in range(args.epochs):
        # training
        if epoch >= args.kd_epoch:
            start_kd = True
        else:
            inf_miou, inf_loss= infer_kd(val_loader, net, criterion, num_of_classes=args.num_of_classes)
            print(inf_miou, inf_loss)
        train_miou, train_loss = train_kd_rl(train_loader, bin_net,net, criterion, optimizer, fp_optimizer, args.num_of_classes, start_kd=start_kd, use_kd=args.use_kd)
        
        fp_scheduler.step()
        if epoch >= args.kd_epoch:
            bin_scheduler.step()
            miou, loss= infer(val_loader, bin_net, criterion, num_of_classes=args.num_of_classes)
            tracker.print(train_loss,train_miou, loss, miou, epoch=epoch-args.kd_epoch)
            data.store(epoch, train_loss, loss, train_miou, miou)
            data.plot(mode='all', save=True, seaborn=False)
            data.save_as_json()
    tracker.end()
    #save_model(jit=args.jit)
    torch.save(bin_net.state_dict(), os.path.join(args.experiment_path, args.experiment_name,'bin_model.pt'))
    torch.save(net.state_dict(), os.path.join(args.experiment_path, args.experiment_name,'model.pt'))
    bin_net.eval()
    test_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=1)
    for i, (img, label, id) in enumerate(test_loader):
        if args.data_name == 'cityscapes':
            if i >20 and i < 300:
                continue
        else:
            if i < args.train_subset:
                continue
        img = img.cuda()
        output = bin_net(img)
        prediction = torch.argmax(output, 1)
        DataSets.plot_image_label(prediction.cpu().squeeze(dim=0), id, val_dataset, transforms=plot_transforms,show_titles=False,show=False ,save=True, save_dir=os.path.join(args.experiment_path, args.experiment_name,'prediction_samples'), plot_name= f'sample_output_{id.item()}')
        if i == 320:
            break


if __name__ == '__main__':
    main()
    
