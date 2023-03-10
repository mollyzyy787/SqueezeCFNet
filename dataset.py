import os
import sys
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as Transform
from models.squeezeCFnet import SqueezeCFNet

class FNTDataset(Dataset):
    """
    FathomNet structured dataset for training the squeezeCFNet
    """
    def __init__(self, file, train=True, normalize=True, mean=53.2086, std=26.0667):
        self.json_r = json.load(open(file,'r'))
        self.train = train
        self.normalize = normalize
        self.mean = mean
        self.std = std
        # create index mapping from (species, id) to overall_id
        overall_idx = 0
        self.index_map = dict()
        self.species_list = []
        for species in self.json_r.keys():
            self.species_list.append(species)
            if self.train:
                for i, item in enumerate(self.json_r[species]['train_set']):
                    self.index_map[overall_idx] = (species, i)
                    overall_idx += 1
            else:
                for i, item in enumerate(self.json_r[species]['val_set']):
                    self.index_map[overall_idx] = (species, i)
                    overall_idx += 1


    def __getitem__(self, idx):
        species, item = self.index_map[idx]
        negative_cls = random.choice([cls for cls in self.species_list if cls != species])
        if self.train:
            target_path = self.json_r[species]['train_set'][item]
            nid_range = len(self.json_r[negative_cls]['train_set']) - 1
            nid = random.randint(0,nid_range)
            negative_path = self.json_r[negative_cls]['train_set'][nid]
        else:
            target_path = self.json_r[species]['val_set'][item]
            nid_range = len(self.json_r[negative_cls]['val_set']) - 1
            nid = random.randint(0,nid_range)
            negative_path = self.json_r[negative_cls]['val_set'][nid]


        target = cv2.imread(target_path) # shape is (H, W, C)
        negative = cv2.imread(negative_path)

        target_ = self.convert_format(target) # tensor of shape (1,H,W, dtype is float32)
        negative = self.convert_format(negative) # (1, H, W)

        # choose two modes to transform the target and the search patch respectively
        # 0: random rotation, 1: random horizontal flip, 2: random vertical flip, 3: do nothing
        transform_modes = random.sample(range(4),2)
        target = self.perform_transform(transform_modes[0], target_) # (1, H, W)
        search = self.perform_transform(transform_modes[1], target_) # (1, H, W)

        return target, search, negative #maybe labeled y
        # each of the three patches is of: <class 'torch.Tensor'> torch.Size([1, 200, 200]) torch.float32

    def __len__(self):
        if self.train:
            numTrain = 0
            for species in self.json_r.keys():
                numTrain += len(self.json_r[species]['train_set'])
            return numTrain
        else:
            numVal = 0
            for species in self.json_r.keys():
                numVal += len(self.json_r[species]['val_set'])
            return numVal

    def convert_format(self, cvImg):
        img_transpose = np.transpose(cvImg, (2, 0, 1)).astype(np.float32) # convert shape to (C, H, W)
        img_tensor = torch.from_numpy(img_transpose)
        img_gray = Transform.Grayscale()(img_tensor)
        if self.normalize:
            img_gray = Transform.Normalize(mean=self.mean, std=self.std)(img_gray)
        return img_gray #(1,H,W)

    def perform_transform(self, mode, img):
        if mode == 0:
            rotater = Transform.RandomRotation(degrees=(30,330))
            return rotater(img)
        elif mode == 1:
            return Transform.functional.hflip(img)
        elif mode == 2:
            return Transform.functional.vflip(img)
        else:
            return img

    def get_species_list(self):
        return self.species_list

    def get_path_info(self):
        return self.json_r


def batch_mean_and_sd(loader):

    cnt = 0
    fst_moment = torch.empty(1)
    snd_moment = torch.empty(1)

    for templates, searches, negatives in loader:
        images = templates
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3]) #sum_ is shape 1 for gray images
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)
    return mean,std

if __name__ == '__main__':

    # FathomNet_mini
    json_file_path = '~/DCNN_CF/curate_dataset/data_sample/FathomNet.json' # replace with dataset *json file path
    json_file_path = os.path.expanduser(json_file_path)
    # compute dataset mean and std
    train_dataset = FNTDataset(json_file_path, True, False)
    print("total number of train images: ", len(train_dataset)) #7427 for regular FathomNet, 828 for mini
    val_dataset = FNTDataset(json_file_path, False, False)
    print("total number of val images: ", len(val_dataset)) #818 for regular FathomNet, 87 for mini

    batch_size = 8
    gpu_num = 1
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = 1, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)

    mean, std = batch_mean_and_sd(train_loader)
    print("mean and std: \n", mean, std)

    # for FathomNet_mini: mean=48.89 std=32.44
    # for FathomNet: mean = 42.14 std=32.12
    # for FathomNet_wrap: mean = 53.2086 std= 26.0667


    #target = torch.Tensor(config.y).cuda().unsqueeze(0).unsqueeze(0).repeat(batch_size * gpu_num, 1, 1, 1)  # for training
    sample_result_path = '~/Pictures/iros_figures/training_samples/'
    sample_result_path = os.path.expanduser(sample_result_path)
    for i, (template, search, negative) in enumerate(train_loader):
        print(i)
        if i%100==0:
            template_show = torch.squeeze(template).numpy()[:,:,None]
            search_show = torch.squeeze(search).numpy()[:,:,None]
            negative_show = torch.squeeze(negative).numpy()[:,:,None]
            cv2.imwrite(os.path.join(sample_result_path, str(i).zfill(6)+"_template.jpg"), template_show)
            cv2.imwrite(os.path.join(sample_result_path, str(i).zfill(6)+"_search.jpg"), search_show)
            cv2.imwrite(os.path.join(sample_result_path, str(i).zfill(6)+"_negative.jpg"), negative_show)
