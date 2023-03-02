import cv2
import os
import glob
import time
import torch
import numpy as np
from curate_dataset.parse_annotation import parseManualAnnotation
from models.squeezeCFnet_track import SqueezeCFNet_light
from baseline.kcf import KCF_HOG
from models.DCFnet_track import DCFNetTracker
from track import TrackerConfig
from utils import crop_chw, gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox, convert_format, PSR, APCE

class SqueezeCFNetTracker_light(object):
    def __init__(self, im, init_rect, net_param_path, gpu=True):
        self.gpu = gpu
        self.config = TrackerConfig(path=net_param_path, use_fire_layer="all", normalize=False, gpu=gpu)
        self.net = SqueezeCFNet_light(self.config)
        self.net.load_param(self.config.feature_path)
        self.net.eval()
        if self.gpu:
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
        if self.gpu:
            self.net.update(target.cuda()) #self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())
        else:
            self.net.update(target)
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
            response = self.net(torch.Tensor(search).cuda())
        else:
            response = self.net(torch.Tensor(search))
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
        if self.gpu:
            self.net.update(target.cuda(), lr=self.config.interp_factor)
        else:
            self.net.update(target, lr=self.config.interp_factor)

        return cxy_wh_2_rect1(self.target_pos, self.target_sz)  # 1-index


def test_tracker_speed(img_folder, tracker, tracker_model):
    """
    note: this tracker in the input should already been initialized outside of the function
    """
    annotation_path = glob.glob(os.path.join(img_folder, '*.json'))[0]
    annotation = parseManualAnnotation(annotation_path)
    for obj in annotation[0]:
        if obj["id"] == 0:
            init_rect = obj["bbox"]
    start = time.time()
    frame_count = 0
    for i, img_path in enumerate(sorted(glob.glob(os.path.join(img_folder, '*.jpg')))):
        #print(i)
        img = cv2.imread(img_path)
        if tracker_model == 'hog':
            tracker_bbox = tracker.update(img)
        else:
            tracker_bbox = tracker.track(img)
        frame_count += 1
    elapsed_time = time.time() - start
    print("frame_count: ", frame_count)
    print("elapsed time: ", elapsed_time)
    fps = frame_count/elapsed_time
    return fps


if __name__ == '__main__':
    SqueezeCFnet_param_path = os.path.join(os.getcwd(), 'checkpoints', 'apce_enc_200ep_1e-4_best.pt')
    DCFnet_param_path = os.path.join(os.getcwd(), 'checkpoints', 'model_best_DCFnet_200ep.pt')
    imSeq_dir = '/media/molly/MR_GRAY/DCNNCF_testset/mo15_left'
    annotation_path = glob.glob(os.path.join(imSeq_dir, '*.json'))[0]
    annotation = parseManualAnnotation(annotation_path)
    for obj in annotation[0]:
        if obj["id"] == 0:
            init_rect = obj["bbox"]
    img0_path = os.path.join(imSeq_dir, str(0).zfill(6)+".jpg")
    img0 = cv2.imread(img0_path)
    SCF_tracker = SqueezeCFNetTracker_light(img0, init_rect, SqueezeCFnet_param_path, gpu=False)
    DCF_tracker = DCFNetTracker(img0, init_rect, DCFnet_param_path, gpu=False)
    #hog_tracker = KCF_HOG()
    #hog_tracker.init(img0, init_rect)
    SCF_fps = test_tracker_speed(imSeq_dir, SCF_tracker,'SCF')
    print("SCF: ", SCF_fps)
    DCF_fps = test_tracker_speed(imSeq_dir, DCF_tracker,'DCF')
    print("DCF: ", DCF_fps)
    #hog_fps = test_tracker_speed(imSeq_dir, hog_tracker,'hog')
    #print("hog: ", hog_fps)
