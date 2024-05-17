from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import transforms
from PIL import Image
import os
import json
import numpy as np
import pickle
from glob import glob
from joblib import Parallel, delayed
import torch


DEBUG=False

def parallel_load(img_dir, img_list, img_size, verbose=0):
    return Parallel(n_jobs=-1, verbose=verbose)(delayed(
        lambda file: Image.open(os.path.join(img_dir, file)).convert("L").resize(
            (img_size, img_size), resample=Image.BILINEAR))(file) for file in img_list)
    
class dataset_train(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None,mode='train',ar = 0.6):
        super(dataset_train, self).__init__()
        self.root = main_path
        self.datasetA = []
        self.datasetB = []
        self.transform = transform if transform is not None else lambda x: x
        self.anomaly_transform = self.transform

        with open(os.path.join(main_path, "data.json")) as f:
            data_dict = json.load(f)

        print("Loading images")
        train_normal = data_dict["train"]["0"]
        self.datasetB += parallel_load(os.path.join(self.root, "images"), train_normal, img_size)
        self.num_images = len(self.datasetB)
        abnormal_num = int(self.num_images * ar)
        normal_num = self.num_images - abnormal_num
        unlabeled_normal_l = data_dict["train"]["unlabeled"]["0"]
        unlabeled_abnormal_l = data_dict["train"]["unlabeled"]["1"]
        train_unlabeled_l = unlabeled_abnormal_l[:abnormal_num] + unlabeled_normal_l[:normal_num]

        self.datasetA += parallel_load(os.path.join(self.root, "images"), train_unlabeled_l, img_size)
        
        print("Loaded {} normal images, "
                "{} (unlabeled) normal images, "
                "{} (unlabeled) abnormal images.".format(len(train_normal), normal_num, abnormal_num))
        

    def __getitem__(self, index):
        imgA = self.datasetA[index]
        imgB = self.datasetB[index]
        imgA = self.transform(imgA)
        imgB = self.transform(imgB)
        img1 = imgB.clone()
        anomaly_img, mask = self.generate_anomaly(img1, index, core_percent=0.6)
        mask = mask.unsqueeze(dim = 0)

        return imgA,imgB,anomaly_img, mask

    def __len__(self):
        return self.num_images

    def generate_anomaly(self, image, index, core_percent=0.8):
        dims = np.array(np.shape(image)[1:])  # H x W
        core = core_percent * dims  # width of core region
        offset = (1 - core_percent) * dims / 2  # offset to center core

        min_width = np.round(0.05 * dims[1])
        max_width = np.round(0.2 * dims[1])  # make sure it is less than offset

        center_dim1 = np.random.randint(offset[0], offset[0] + core[0])
        center_dim2 = np.random.randint(offset[1], offset[1] + core[1])
        patch_center = np.array([center_dim1, center_dim2])
        patch_width = np.random.randint(min_width, max_width)

        coor_min = patch_center - patch_width
        coor_max = patch_center + patch_width

        # clip coordinates to within image dims
        coor_min = np.clip(coor_min, 0, dims)
        coor_max = np.clip(coor_max, 0, dims)

        alpha = torch.rand(1)  #
        mask = torch.zeros_like(image).squeeze()
        mask[coor_min[0]:coor_max[0], coor_min[1]:coor_max[1]] = alpha
        mask_inv = 1 - mask

        # mix
        anomaly_source_index = np.random.randint(0, len(self.datasetB))
        while anomaly_source_index == index:
            anomaly_source_index = np.random.randint(0, len(self.datasetB))
        anomaly_source = self.datasetB[anomaly_source_index]
        anomaly_source = self.anomaly_transform(anomaly_source)
        image_synthesis = mask_inv * image + mask * anomaly_source

        return image_synthesis, (mask > 0).long()
    
class dataset_test(data.Dataset):
    def __init__(self, main_path, img_size=256, transform=None, mode="train"):
        super(dataset_test, self).__init__()
        assert mode in ["test"]
        self.root = main_path
        self.labels = []
        self.img_id = []
        self.datasetA = []
        self.transform = transform if transform is not None else lambda x: x

        with open(os.path.join(main_path, "data.json")) as f:
            data_dict = json.load(f)
        print("Loading images")

        if mode == "test":
            test_normal = data_dict["test"]["0"]
            test_abnormal = data_dict["test"]["1"]

            test_l = test_normal + test_abnormal
            self.datasetA += parallel_load(os.path.join(self.root, "images"), test_l, img_size)
            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_id += [img_name.split('.')[0] for img_name in test_l]
            print("Loaded {} test normal images, "
                  "{} test abnormal images.".format(len(test_normal), len(test_abnormal)))
            self.num_images = len(self.datasetA)

    def __getitem__(self, index):
        imgA = self.datasetA[index]
        imgA = self.transform(imgA)
        img_id = self.img_id[index]
        label = self.labels[index]

        return imgA,img_id,label
    
    def __len__(self):
        """Return the number of images."""
        return self.num_images

def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]


def get_loader(dataset,bs,img_size,workers=1,mode = 'train'):
    root_path = os.path.join("/data/", "your dataset path")
    if dataset == 'rsna':
        print("Dataset:{}".format(dataset))
        data_path = os.path.join(root_path,"RSNA/")
    elif dataset == 'lag':
        print("Dataset:{}".format(dataset))
        data_path = os.path.join(root_path,"LAG/")
    elif dataset == 'vincxr':
        print("Dataset:{}".format(dataset))
        data_path = os.path.join(root_path,"VinCXR/")
    else:
        raise Exception("Invalid dataset:{}".format(dataset))
    
    transform = transforms.Compose([
                transforms.Resize((img_size,img_size),interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
        ])
    if mode == "train"
        dset = dataset_train(main_path = data_path,img_size=img_size,transform=transform,mode = mode)
    else:
        dset = dataset_test(main_path = data_path,img_size=img_size,transform=transform,mode = mode)

    train_flag =  True if mode== 'train' else False
    dataloader = data.DataLoader(dset,bs,shuffle=train_flag,drop_last=train_flag,num_workers=workers,pin_memory=True)

    return dataloader
