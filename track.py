from os.path import join, isdir
from os import makedirs
import argparse
import json
import numpy as np
import torch
import torchvision.transforms as Transform

import cv2
import time as time
from models.squeezeCFnet_track import SqueezeCFNet
from utils import crop_chw, gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox, convert_format

class TrackerConfig(object):
    def __init__(self, path, use_fire_layer="all", normalize=False, gpu=True):
        self.feature_path = path
        self.use_fire_layer = use_fire_layer
        self.normalize = normalize
        self.gpu = gpu
        # These are the default hyper-params
        self.crop_sz = 200
        self.output_sz = 48

        self.lambda0 = 1e-4
        self.padding = 2
        self.output_sigma_factor = 0.1
        self.interp_factor = 0.01 # (lr for new template, should be big for target that changes fast)
        self.num_scale = 3
        self.scale_step = 1.0275
        self.scale_factor = self.scale_step ** (np.arange(self.num_scale) - self.num_scale / 2)
        self.min_scale_factor = 0.2
        self.max_scale_factor = 5
        self.scale_penalty = 0.9925
        self.scale_penalties = self.scale_penalty ** (np.abs((np.arange(self.num_scale) - self.num_scale / 2)))

        self.net_input_size = [self.crop_sz, self.crop_sz]
        self.output_sigma = self.crop_sz / (1 + self.padding) * self.output_sigma_factor
        self.y = gaussian_shaped_labels(self.output_sigma, [self.output_sz, self.output_sz])
        if self.gpu:
            self.yf = torch.view_as_real(torch.fft.rfft2(torch.Tensor(self.y).view(1, 1, self.output_sz, self.output_sz).cuda(), norm="ortho"))
            self.cos_window = torch.Tensor(np.outer(np.hanning(self.output_sz), np.hanning(self.output_sz))).cuda()
        else:
            self.yf = torch.view_as_real(torch.fft.rfft2(torch.Tensor(self.y).view(1, 1, self.output_sz, self.output_sz), norm="ortho"))
            self.cos_window = torch.Tensor(np.outer(np.hanning(self.output_sz), np.hanning(self.output_sz)))

        self.mean = 42.14 # for img training set FathomNet
        self.std = 32.12
    #mean = 53.2086 # for training set FathomNet_wrap
    #std = 26.0667


class SqueezeCFNetTracker(object):
    def __init__(self, im, init_rect, net_param_path, gpu=True):
        self.gpu = gpu
        self.config = TrackerConfig(path=net_param_path)
        self.net = SqueezeCFNet(self.config)
        self.net.load_param(self.config.feature_path)
        self.net.eval()
        if gpu:
            self.net.cuda()

        # confine results
        target_pos, target_sz = rect1_2_cxy_wh(init_rect) #convert initial bb to pos and sz
        self.min_sz = np.maximum(self.config.min_scale_factor * target_sz, 4)
        self.max_sz = np.minimum(im.shape[:2], self.config.max_scale_factor * target_sz)

        # crop template
        window_sz = target_sz * (1 + self.config.padding)
        bbox = cxy_wh_2_bbox(target_pos, window_sz)
        patch = crop_chw(im, bbox, self.config.crop_sz) #output is numpy array
        patch = np.expand_dims(patch, axis=0).astype(np.float32)

        target = convert_format(patch, self.config.normalize, self.config.mean, self.config.std) #replaced: target = patch - config.net_average_image
        #print(type(target), target.shape)
        self.net.update(target.cuda()) #self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())
        self.target_pos, self.target_sz = target_pos, target_sz
        self.patch_crop = np.zeros((self.config.num_scale, patch.shape[1], patch.shape[2], patch.shape[3]), np.float32)  # buff
        #print(self.config.cos_window)

    def track(self, im):
        for i in range(self.config.num_scale):  # crop multi-scale search region
            window_sz = self.target_sz * (self.config.scale_factor[i] * (1 + self.config.padding))
            bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
            self.patch_crop[i, :] = crop_chw(im, bbox, self.config.crop_sz)

        search = convert_format(self.patch_crop, self.config.normalize, self.config.mean, self.config.std) #search = self.patch_crop - self.config.net_average_image

        if self.gpu:
            [response, encode] = self.net(torch.Tensor(search).cuda())
        else:
            [response, encode] = self.net(torch.Tensor(search))
        peak, idx = torch.max(response.view(self.config.num_scale, -1), 1)
        peak = peak.data.cpu().numpy() * self.config.scale_penalties
        idx = idx.data.cpu().numpy()
        best_scale = np.argmax(peak)
        r_max, c_max = np.unravel_index(idx[best_scale], self.config.net_input_size)

        if r_max > self.config.net_input_size[0] / 2:
            r_max = r_max - self.config.net_input_size[0]
        if c_max > self.config.net_input_size[1] / 2:
            c_max = c_max - self.config.net_input_size[1]
        window_sz = self.target_sz * (self.config.scale_factor[best_scale] * (1 + self.config.padding))

        self.target_pos = self.target_pos + np.array([c_max, r_max]) * window_sz / self.config.net_input_size
        self.target_sz = np.minimum(np.maximum(window_sz / (1 + self.config.padding), self.min_sz), self.max_sz)

        # model update
        window_sz = self.target_sz * (1 + self.config.padding)
        bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
        patch = crop_chw(im, bbox, self.config.crop_sz)
        patch = np.expand_dims(patch, axis=0).astype(np.float32)

        target = convert_format(patch, self.config.normalize, self.config.mean, self.config.std) #target = patch - self.config.net_average_image
        self.net.update(target.cuda(), lr=self.config.interp_factor)

        return cxy_wh_2_rect1(self.target_pos, self.target_sz)  # 1-index

if __name__ == '__main__':
    exit()
