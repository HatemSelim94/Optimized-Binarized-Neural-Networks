import sys
sys.path.append("/home/hatem/Projects/server/final_repo/final_repo")
import torch
import torch.nn as nn
from torch.utils.data import Subset
import argparse
import os

from layers import Model as Network
from processing import Transformer, DataSets
from processing.datasets import CityScapes, KittiDataset
from utilities import infer, set_seeds, Clipper, DataPlotter, Tracker, train, model_info, clean_dir, prepare_ops_metrics

parser  = argparse.ArgumentParser('BEDN')
parser.add_argument('--data_name', type=str, default='cityscapes')
parser.add_argument('--data_path', type=str, default='../../data/cityscapes/')
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
parser.add_argument('--experiment_path', type=str, default='fp_models/bedn/experiments/')
parser.add_argument('--experiment_name', type=str, default='exp1')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--decay_val', type=float, default=0.7)
parser.add_argument('--decay_step', type=int, default=5)
args = parser.parse_args()
torch.cuda.empty_cache()




def main():
    set_seeds(args.seed)
    clean_dir(args)
    if not torch.cuda.is_available():
        sys.exit(1)
    classes_weights = KittiDataset.loss_weights_3 if args.num_of_classes ==3 else KittiDataset.loss_weights_8
    criterion = nn.CrossEntropyLoss(ignore_index = CityScapes.ignore_index, label_smoothing=0.2, weight=classes_weights.cuda())
    criterion = criterion.to(args.device)  
    net = Network(args.num_of_classes).to(args.device)
    input_shape = (1, 3, args.image_size, args.image_size)
    model_info(net, input_shape, save=True, dir=os.path.join(args.experiment_path, args.experiment_name), verbose=True)
    #prepare_ops_metrics(net, input_shape)
    train_transforms = Transformer.get_transforms({'normalize':{'mean':CityScapes.mean,'std':CityScapes.std}, 'resize':{'size':[args.image_size,args.image_size]},'random_horizontal_flip':{'flip_prob':0.2}})
    val_transforms = Transformer.get_transforms({'normalize':{'mean':CityScapes.mean,'std':CityScapes.std},'resize':{'size':[args.image_size,args.image_size]}})
    plot_transforms = Transformer.get_transforms({'resize':{'size':[args.image_size,args.image_size]}})
    train_dataset = DataSets.get_dataset(args.data_name, no_of_classes=args.num_of_classes, transforms=train_transforms)
    val_dataset = DataSets.get_dataset(args.data_name, no_of_classes=args.num_of_classes,split='val',transforms=val_transforms)
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
    num_of_classes = args.num_of_classes
    fp_params = [p for p in net.parameters() if not hasattr(p,'bin')]
    bin_params = [p for p in net.parameters() if hasattr(p,'bin')]
    optim_args = [{'params':fp_params, 'weight_decay':args.network_optim_fp_weight_decay,'lr':args.network_optim_fp_lr}]
    optim_args = [[{'params':fp_params, 'weight_decay':args.network_optim_fp_weight_decay,'lr':args.network_optim_fp_lr},{'params':bin_params, 'weight_decay':0.001,'lr':args.network_optim_bin_lr}]]
    optimizer = Clipper.get_clipped_optim(args.network_optim, optim_args)
    optimizer = torch.optim.Adam(fp_params, lr=args.network_optim_fp_lr) 
    #optimizer=Clipper.get_clipped_optim('SGD',optim_args)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:((step/args.epochs))**2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(20, args.epochs, args.decay_step)], gamma=args.decay_val)
    data = DataPlotter(os.path.join(args.experiment_path, args.experiment_name))
    tracker = Tracker(args.epochs)
    tracker.start()
    for epoch in range(args.epochs):
        # training
        train_miou, train_loss = train(train_loader, net, criterion, optimizer, num_of_classes)
        scheduler.step()
        miou, loss= infer(val_loader, net, criterion, num_of_classes=num_of_classes)
        tracker.print(train_loss,train_miou, loss, miou, epoch=epoch)
        data.store(epoch, train_loss, loss, train_miou, miou)
        data.plot(mode='all', save=True, seaborn=False)
        data.save_as_json()
    tracker.end()
    #save_model(jit=args.jit)
    torch.save(net, os.path.join(args.experiment_path, args.experiment_name,'model.pt'))
    net.eval()
    test_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=1)
    for i, (img, label, id) in enumerate(test_loader):
        
        if i >20 and i < 300:
            continue
        img = img.cuda()
        output = net(img)
        prediction = torch.argmax(output, 1)
        DataSets.plot_image_label(prediction.cpu().squeeze(dim=0), id, val_dataset, transforms=plot_transforms,show_titles=False,show=False ,save=True, save_dir=os.path.join(args.experiment_path, args.experiment_name,'prediction_samples'), plot_name= f'sample_output_{id.item()}')
        if i == 330:
            break
  
if __name__ == '__main__':
    main()
    
    
