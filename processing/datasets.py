import matplotlib.pyplot as plt
import torch
import torchvision.transforms as VisionTrans
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import shutil
import cv2
import random 
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
        self.train= train
        if train:
            self.location = 'training/'
        else:
            self.location = 'testing/'
            #raise NotImplemented # (kitti)useful only for submission as it has no labels
        
        if download:
            self._download()
        
        self.images_dir = os.path.join(self.dataset_path, self.location ,self.image_dir_name)
        self.targets_dir= os.path.join(self.dataset_path, self.location, self.label_dir_name)
        
        if train:
            if save_annotation:
                self._save_annotation() # changes self.targets_dir
        if train:
            for img_file in os.listdir(self.images_dir):
                self.images.append(os.path.join(self.images_dir, img_file))
                self.targets.append(os.path.join(self.targets_dir, img_file))
        else:
            for img_file in os.listdir(self.images_dir):
                self.images.append(os.path.join(self.images_dir, img_file))
            
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
        if self.train:
            image = Image.open(self.images[index])
            label = Image.open(self.targets[index])
            if not self.save_annotation:
                if self.no_of_classes !=34:
                    label = self._change_annotation(np.array(label).astype(np.int16)) # int16: there is a -1 label (ignored label) 

            if self.transforms:
                image, label = self.transforms(image, label)
            return image, label, index
        else:
            image = Image.open(self.images[index])
            if self.transforms:
                image, _= self.transforms(image)
            return image, index
            
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

    def convert_to_orig_ids(self, pred, id, ncls = 20,save=True, save_dir=None):
        '''
        label: label is a tensor on cpu
        kitti2015_results
        '''
        if pred is not None:
            if ncls ==20:
                for label in reversed(labels):
                    if label.trainId >18 or label.trainId<0:
                        continue 
                    pred[pred == label.trainId] = label.id
        if save:
            image_name = self.images[id].split('/')
            print(image_name[-1])
            pred_image = pred.numpy().astype(np.uint8)
            pred_img = Image.fromarray(pred_image)
            os.makedirs(os.path.join(save_dir,'semantic'),exist_ok=True)
            pred_img.save(os.path.join(save_dir,'test_semantic',"Kitti2015_"+image_name[-1]))

        
class KittiRoad(Dataset):
    #label map {background:0, road: 1, ignore:255}
    ignore_index = 255
    mean = [0.379, 0.4, 0.386]
    std = [0.309, 0.319, 0.329]
    weights = torch.tensor([1.5084, 3.8176])
    def __init__(self, transforms=ToTensor(), dataset_path='data/data_road/', split='train'):
        #types = ['um_lane_', 'umm_road_','um_road_','uu_road_']
        self.labels_folder = 'gt_image_2'
        self.images_folder = 'image_2'
        self.transforms = transforms
        self.dataset_path = dataset_path
        self.split = split
        self.images = []
        self.labels = []
        self.pred_names = []
        if split == 'train':
            self.phase = 'training'
        elif split == 'test':
            self.phase = 'testing'
        self.images_dir = os.path.join(self.dataset_path, self.phase,self.images_folder)
        if split =='train':
            self.labels_dir = os.path.join(self.dataset_path, self.phase, self.labels_folder)
        for img_file in os.listdir(self.images_dir):
            img_name = img_file.split('.')[0]
            self.images.append(os.path.join(self.images_dir, img_name+'.png'))
            temp = img_name.split('_')
            temp.insert(1, 'road')
            label_name = '_'.join(temp)
            if split == 'train':
                self.labels.append(os.path.join(self.labels_dir, label_name+'.png'))
            else:
                self.pred_names.append(label_name+'.png')
        self.images = sorted(self.images)
        if split == 'train':
            self.labels = sorted(self.labels)
        elif split == 'test':
            self.pred_names = sorted(self.pred_names)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        w, h = image.size
        if self.split == 'train':
            label_rgb = Image.open(self.labels[index])
            label_array_rgb = np.array(label_rgb)
            label = np.zeros(label_array_rgb.shape[:2], np.uint8) + 255
            label[label_array_rgb[:,:,2] > 0] = 1 # road
            label[(label_array_rgb[:,:,0] > 0) & (label_array_rgb[:,:,2] == 0)] = 0
            label = Image.fromarray(label)
            if self.transforms:
                image, label = self.transforms(image, label)
            return image, label,0
        else:
            img_name = self.pred_names[index]
            if self.transforms:
                image,_ = self.transforms(image)
            return image,img_name, w,h

    def __len__(self):
        return len(self.images)

# https://github.com/zhechen/PLARD/blob/master/ptsemseg/loader/kitti_road_loader.py
def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

class KITTIRoadLoader(Dataset):
    """KITTI Road Dataset Loader
    http://www.cvlibs.net/datasets/kitti/eval_road.php
    Data is derived from KITTI
    label map {background:0, road: 1, ignore:255}
    """
    mean_rgb = [103.939, 116.779, 123.68] # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(self, root='data/data_road', split="train", is_transform=False, 
                 img_size=(1280, 384), augmentations=None, version='pascal', phase='train'):
        """__init__
        :param root:
        :param split:
        :param is_transform: (not used)
        :param img_size: (not used)
        :param augmentations  (not used)
        """
        self.ignore_val = 255
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 2
        self.img_size = img_size 
        self.mean = np.array(self.mean_rgb)
        self.files = {}
        self.hflip = VisionTrans.RandomHorizontalFlip(p=0)
        if phase == 'train':
            self.images_base = os.path.join(self.root, 'training', 'image_2')
            self.lidar_base = os.path.join(self.root, 'training', 'ADI')
            self.annotations_base = os.path.join(self.root, 'training', 'gt_image_2')
            self.im_files = recursive_glob(rootdir=self.images_base, suffix='.png')
        else:
            self.images_base = os.path.join(self.root, 'testing', 'image_2')
            self.lidar_base = os.path.join(self.root, 'testing', 'ADI')
            self.annotations_base = os.path.join(self.root, 'testing', 'gt_image_2')
            self.split = 'test'

            self.im_files = recursive_glob(rootdir=self.images_base, suffix='.png')
            self.im_files = sorted(self.im_files)

        self.data_size = len(self.im_files)
        self.phase = phase

        print("Found %d %s images" % (self.data_size, self.split))

    def __len__(self):
        """__len__"""
        return self.data_size

    def im_paths(self):
        return self.im_files

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.im_files[index].rstrip()
        im_name_splits = img_path.split(os.sep)[-1].split('.')[0].split('_')
        img_name = img_path.split(os.sep)[-1].split('.')[0]

        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        #lidar = cv2.imread(os.path.join(self.lidar_base, im_name_splits[0] + '_' + im_name_splits[1] + '.png'), cv2.IMREAD_UNCHANGED)
        #lidar = np.array(lidar, dtype=np.uint8)

        if self.phase == 'train':
            lbl_path = os.path.join(self.annotations_base,
                                    im_name_splits[0] + '_road_' + im_name_splits[1] + '.png')

            lbl_tmp = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED) # bgr
            lbl_tmp = np.array(lbl_tmp, dtype=np.uint8)
            # gt color (rgb) : red (255 r, 0 g, 0 b) , pink (255 r, 0 g, 255 b), black (0 r, 0 g, 0 b)
            
            # gt color (bgr): red ( 0 b, 0 g, 255 r) , pink (255 b, 0 g, 255 r), black (0 b, 0 g, 0 r)
            lbl = 255 + np.zeros( (img.shape[0], img.shape[1]), np.uint8) # All pixels are white (white 255) (loss ignore)
            lbl[lbl_tmp[:,:,0] > 0] = 1  # now road pixels are set to one 
            lbl[(lbl_tmp[:,:,2] > 0) & (lbl_tmp[:,:,0] == 0)] = 0 # (valid) and (red) set to 0 (background)
            # label map {background:0, road: 1, ignore:255}
            #img, lidar, lbl = self.transform(img, lidar, lbl)
            img, lbl = self.transform(img, lbl)

            return img, lbl, img_name
        else:
            tr_img = img.copy()
            #tr_lidar = lidar.copy()
            #tr_img, tr_lidar = self.transform(tr_img, tr_lidar)
            tr_img = self.transform(tr_img)
    
            return img, tr_img

    def transform(self, img, lbl=None):
        """transform
        :param img:
        :param lbl:
        """
        img = img.astype(np.float64)
        img -= self.mean

        #lidar = lidar.astype(np.float64) / 128.
        #lidar = lidar - np.mean(lidar[lidar>0]) 

        img = cv2.resize(img, self.img_size)
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        img = self.hflip(img)
        #lidar = cv2.resize(lidar, self.img_size)
        #lidar = lidar[np.newaxis, :, :] 
        #lidar = torch.from_numpy(lidar).float()

        if lbl is not None:
            lbl = cv2.resize(lbl, (int(self.img_size[1]), int(self.img_size[0])), interpolation=cv2.INTER_NEAREST)
            lbl = torch.from_numpy(lbl).long()
            lbl = self.hflip(lbl)
            return img, lbl
        else:
            return img



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
    
    @staticmethod
    def to_orig_size(w,h, pred):
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        pred = cv2.resize(pred, (w,h), 0,0,interpolation=cv2.INTER_NEAREST)
        return pred
    
    @staticmethod
    def save_pred(w,h,pred,name,path):
        result_path = os.path.join(path, name)
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        pred = cv2.resize(pred, (w,h), 0,0,interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(result_path, pred)
    
    @staticmethod
    def save_as_prob(w,h, output,name, path):
        os.makedirs(path, exist_ok=True)
        result_path = os.path.join(path, name[0])
        if torch.is_tensor(output):
            output = torch.softmax(output, dim=1)
            output = output[0][1].cpu().numpy()
            prob = np.floor(255* (output - output.min()) / (output.max() - output.min()))
            prob = np.array(prob).astype(np.uint8)
            #print('prob_shape',prob.shape)
            prob = cv2.resize(prob, (int(w),int(h)),interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(result_path, prob)




        

