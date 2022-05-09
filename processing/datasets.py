import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import shutil

from .data_utils import decode_segmap
from .transforms import ToTensor
from .helpers import labels
import time

class KittiDataset(Dataset):
    data_url = 'http://www.cvlibs.net/download.php?file=data_semantics.zip'
    image_dir_name = 'image_2'
    label_dir_name = 'semantic'
    name = 'kitti'
    ignore_index = 255
    mean = [0.379, 0.4, 0.386]
    std = [0.309, 0.319, 0.329]
    loss_weights_8 = torch.tensor([7.8749, 2.9687, 4.8369, 7.5210, 2.3980, 4.8945, 8.7403, 5.8373])
    loss_weights_3 = torch.tensor([1.666, 3.3242, 5.8373])
    Id2CategoryID   = { label.id : label.categoryId for label in labels }
    Id2CustomID   = { label.id : label.customId for label in labels }
    Id2TrainID   = { label.id : label.trainId for label in labels }
    def __init__(self, transforms=ToTensor(), no_of_classes=34, train=True,dataset_path='data/kitti/', download=False, save_annotation=True):
        """Kitti dataset class returns kitti dataset.w
        A subset from training data forms the validation dataset.

        Args:
            dataset_path (str, optional): [description]. Defaults to 'data/kitti/data_semantics'.
            mode (int, optional): 1 represents training, 2 represents validation and 0
            represents validation. Defaults to 1.
            dataset_mode: an integer represents the number of classes in the dataset
            transforms (nn.Module, optional): transfroms to be performed on the dataset
            . Defaults to None.
            download (bool, optional): download tha dataset. Defaults to False.
        """
        assert no_of_classes == 3 or no_of_classes == 8 or no_of_classes == 20 or no_of_classes == 34
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.no_of_classes = no_of_classes
        self.save_annotation = save_annotation
        self.images = []
        self.targets = []
        if train:
            self.location = 'training/'
        else:
            self.location = 'testing/'
            raise NotImplemented # (kitti)useful only for submission as it has no labels
        
        if download:
            self._download()
        
        self.images_dir = os.path.join(self.dataset_path, self.location ,self.image_dir_name)
        self.targets_dir= os.path.join(self.dataset_path, self.location, self.label_dir_name)
        
        if save_annotation:
            self._save_annotation() # changes self.targets_dir
        
        for img_file in os.listdir(self.images_dir):
            self.images.append(os.path.join(self.images_dir, img_file))
            self.targets.append(os.path.join(self.targets_dir, img_file))
        
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        """access dataset images

        Args:
            index (int): index of the image

        Returns:
            [image, target]: return the index-th image and the associated target
             in the dataset as tuple of tensors
        """
        # torchvision.io.read_image outputs a tensor (channel * h * w) in  range [0, 255] 
        # torchvision.io has some dependency problems(not stable)
        image = Image.open(self.images[index])
        label = Image.open(self.targets[index])
        if not self.save_annotation:
            if self.no_of_classes !=34:
                label = self._change_annotation(np.array(label).astype(np.int16)) # int16: there is a -1 label (ignored label) 

        if self.transforms:
            image, label = self.transforms(image, label)
        return image, label, index
            
    def _download(self):
        raise NotImplementedError
    
    def get_class_weights(self, num_of_classes= None):
        num_of_classes_ = self.no_of_classes if num_of_classes is None else num_of_classes
        if num_of_classes_ == 3:
            return self.loss_weights_3
        elif num_of_classes_ == 8:
            return self.loss_weights_8

    
    def _change_annotation(self, label):
        if self.no_of_classes == 3:
            for id, custom_id in self.Id2CustomID.items():
                label[label==id] = custom_id
        elif self.no_of_classes == 8: 
            for id, category_id in self.Id2CategoryID.items():
                label[label==id] = category_id
        elif self.no_of_classes == 20:
            for id, train_id in self.Id2TrainID.items():
                label[label==id] = train_id
        return label.astype(np.uint8) # range 0-255
    
    def _save_annotation(self):
        new_dir_path = os.path.join(self.dataset_path, self.location,self.label_dir_name+f'_{self.no_of_classes}')
        if not os.path.exists(new_dir_path):
            if self.no_of_classes == 3:
                labels_dict = self.Id2CustomID
            elif self.no_of_classes == 8:
                labels_dict = self.Id2CategoryID
            elif self.no_of_classes == 20:
                labels_dict = self.Id2TrainID
            else:
                raise NotImplementedError

            shutil.copytree(self.targets_dir, new_dir_path)
            self.targets_dir = new_dir_path
            for target_file in os.listdir(self.targets_dir):
                target_file_path = os.path.join(self.targets_dir, target_file)
                target_img = Image.open(target_file_path)
                px = list(target_img.getdata())
                for i in range(len(px)):
                    px[i] = labels_dict[px[i]]
                target_img.putdata(px)
                target_img.save(target_file_path)

        else:
            self.targets_dir = new_dir_path



# mean and std
#CITYSCAPES_MEAN = [0.28689554, 0.32513303, 0.28389177]
#CITYSCAPES_STD = [0.18696375, 0.19017339, 0.18720214]

class CityScapes(Dataset):
    data_url = ['https://www.cityscapes-dataset.com/file-handling/?packageID=1'] # packageID 1 and 3  for fine, and 2 for coarse annotated images
    name = 'cityscapes'
    ignore_index = 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    loss_weights_3 = torch.tensor([1.8845, 2.6459, 5.4678])
    loss_weights_8 = torch.tensor([4.7437, 2.4198, 3.6643, 7.5959, 4.2248, 6.8981, 7.8759, 5.4678])
    label_dir = 'gtFine'
    raw_dir = 'leftImg8bit'
    coarse_label_dir = 'gtCoarse'
    Id2CategoryID   = { label.id : label.categoryId for label in labels }
    Id2CustomID   = { label.id : label.customId for label in labels }
    Id2TrainID   = { label.id : label.trainId for label in labels }
    def __init__(self, transforms=ToTensor(),no_of_classes=34, split='train', mode='fine',dataset_dir = 'data/cityscapes/', download=False, save_annotation=True):
        """ Cityscapes datasset
        Args:
            mode (str): fine or coarse
            split (str): train, val, or test
            no_of_classe int: 3, 8, or 20
        """
        assert no_of_classes == 3 or no_of_classes == 8 or no_of_classes == 20 or no_of_classes == 34
        assert split == 'train' or split == 'val' or split == 'test'
        assert mode == 'fine' or mode == 'coarse'

        if download:
            self._download()
        self.no_of_classes = no_of_classes
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.split = split
        self._location = split
        self.save_annotation = save_annotation
        if split == 'test':
            assert mode == 'fine'
        
        if mode == 'coarse':
            self.label_dir = self.coarse_label_dir
            self.label_file_name = 'gtCoarse_labelids.png'
            self.no_of_classes = 34 # not necessary 
        else:
            self.label_file_name = 'gtFine_labelIds.png'
        
        if save_annotation:
            if mode == 'coarse':
                raise NotImplementedError
            self._save_annotation() # changes self.label_dir
        
        self.raw_images =[]
        self.labels = []
        self.transforms = transforms

        self.images_dir = os.path.join(self.dataset_dir, self.raw_dir, self._location)
        for city in os.listdir(self.images_dir):
            files_dir = os.path.join(self.images_dir, city)
            for image_file in os.listdir(files_dir):
                self.raw_images.append(os.path.join(
                    files_dir,image_file
                    ))
                label_file_id = image_file.split('leftImg8bit.png')[0]
                self.labels.append(os.path.join(self.dataset_dir,
                    self.label_dir,self._location, city, label_file_id+self.label_file_name
                ))
                
    def __len__(self):
        return len(self.raw_images)
    
    def __getitem__(self, index):
        image = Image.open(self.raw_images[index])
        label = Image.open(self.labels[index])
        if not self.save_annotation:
            if self.mode == 'fine':
                if self.no_of_classes != 34:
                    label = self._change_annotation(np.array(label).astype(np.int16)) # int16: there is a -1 label (ignored label) 

        if self.transforms:
            image, label = self.transforms(image, label)

        return image, label, index
    
    def _change_annotation(self, label):
        if self.no_of_classes == 3:
            for id, custom_id in self.Id2CustomID.items():
                label[label==id] = custom_id
        elif self.no_of_classes == 8: 
            for id, category_id in self.Id2CategoryID.items():
                label[label==id] = category_id
        elif self.no_of_classes == 20:
            for id, train_id in self.Id2TrainID.items():
                label[label==id] = train_id
        return label.astype(np.uint8) # range 0-255
    
    def _download(self):
        raise NotImplementedError
    
    def _save_annotation(self):
        new_dir_path = os.path.join(self.dataset_dir, self.label_dir+f'_{self.no_of_classes}', self.split)
        old_dir_path = os.path.join(self.dataset_dir, self.label_dir, self.split)
        if not os.path.exists(new_dir_path):
            shutil.copytree(old_dir_path, new_dir_path)
            time.sleep(30)
            if self.no_of_classes == 3:
                labels_dict = self.Id2CustomID
            elif self.no_of_classes == 8:
                labels_dict = self.Id2CategoryID
            elif self.no_of_classes == 20:
                labels_dict = self.Id2TrainID
            else:
                raise NotImplementedError
            
            for city in os.listdir(new_dir_path):
                files_dir = os.path.join(new_dir_path, city)
                for target_file in os.listdir(files_dir):
                    if target_file[-12:]=='labelIds.png':
                        target_file_path = os.path.join(files_dir, target_file)
                        target_img = Image.open(target_file_path)
                        px = list(target_img.getdata())
                        for i in range(len(px)):
                            px[i] = labels_dict[px[i]]
                        target_img.putdata(px)
                        target_img.save(target_file_path)
        
        self.label_dir = self.label_dir+f'_{self.no_of_classes}'


class DataSets:
    @staticmethod
    def get_dataset(name, no_of_classes, transforms=ToTensor(),split='train',mode='fine',save_annotation=True):
        if name == 'kitti':
            assert split == 'train'
            return KittiDataset(transforms=transforms, no_of_classes=no_of_classes, train=True)
        elif name == 'cityscapes':
            return CityScapes(transforms=transforms, no_of_classes=no_of_classes, split=split, mode = mode, save_annotation=save_annotation)
    
    @staticmethod
    def plot_image_label(pred, id, dataset, miou = None, show=True,transforms= ToTensor(), show_titles= True,save=False, save_dir= None, dpi=1080, plot_name=None):
        if dataset.name == 'kitti':
            orig_dataset =  KittiDataset(transforms=transforms, no_of_classes=dataset.no_of_classes)
        else:
            orig_dataset = CityScapes(transforms=transforms, mode=dataset.mode, split=dataset.split, no_of_classes=dataset.no_of_classes)
        orig_img, label, _ = orig_dataset[id]
        fig,ax = plt.subplots(1,3, sharex='all', sharey='all')
        ax[0].imshow(orig_img.permute(1,2,0))
        if show_titles:
            ax[0].set_title('Image')
        ax[0].axis('off')
        ax[1].imshow(decode_segmap(label, dataset.no_of_classes))
        if show_titles:
            ax[1].set_title('Label')
        ax[1].axis('off')
        #ax[2].imshow(processed_image.permute(1,2,0))
        #if show_titles:
        #    ax[2].set_title('Processed Image')
        #ax[2].axis('off')
        ax[2].imshow(decode_segmap(pred, dataset.no_of_classes))
        if show_titles:
            if miou is not None:
                ax[2].set_title(f'Prediction\n MIoU:{miou}')
            else:
                ax[2].set_title('Prediction')
        ax[2].axis('off')
        fig.subplots_adjust(wspace=0.05, hspace=0)
        
        if save:
            assert save_dir is not None
            assert dpi >= 0
            if not os.path.exists(save_dir):
                os.makedirs(save_dir) 
            plt.draw()
            plt.savefig(os.path.join(save_dir,plot_name+'_plot.png'), dpi = 1080)

        if show:
            plt.show()
        
        plt.clf()
        plt.close('all')

        

