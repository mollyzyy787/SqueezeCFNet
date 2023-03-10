import argparse
import numpy as np
import os
import cv2
import json
import torch
import torch.nn as nn
import glob
import scipy.io

from dataset import FNTDataset
from curate_dataset.parse_annotation import parseManualAnnotation
from baseline.kcf import KCF_HOG
from models.squeezeCFnet_track import FeatSqueezeNet, complex_mul, complex_mulconj
from models.DCFnet_track import TrackerConfig_DCFNet, DCFNet
from track import TrackerConfig
from utils import crop_chw, gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox, convert_format, PSR, APCE

class SqueezeCFNet_reIdTest(nn.Module):
    def __init__(self, config=None, kernel='linear'):
        super().__init__()
        self.feature_net = FeatSqueezeNet()
        self.model_alphaf = []
        self.model_zf = [] 
        self.config = config
        self.use_fire_layer = config.use_fire_layer
        self.kernel = kernel
        self.sigma = 0.5
        self.model_encode = []

    def extract_feature(self, x):
        x_encode, x1_map, x2_map, x3_map = self.feature_net(x)
        if self.use_fire_layer == "1":
            x_map = x1_map
        elif self.use_fire_layer == "2":
            x_map = x2_map
        elif self.use_fire_layer == "3":
            x_map = x3_map
        else:
            x_map = torch.cat((x1_map, x2_map, x3_map), dim=1)
            # add possible channel reduction tricks?
        x_map = x_map * self.config.cos_window
        return x_map, x_encode

    def gaussian_kernel_correlation(self, x, y):
        xf = torch.fft.rfft2(x, norm="ortho")
        yf = torch.fft.rfft2(y, norm="ortho")
        N=xf.shape[-2]*xf.shape[-1]
        xf_size = N*xf.shape[-3]
        xx = (x.flatten(start_dim=1)*x.flatten(start_dim=1)).sum(dim=1)
        yy = (y.flatten(start_dim=1)*y.flatten(start_dim=1)).sum(dim=1)
        xyf = xf*yf.conj()
        xy=torch.fft.irfft2(torch.sum(xyf, dim=1), norm="ortho")
        xy_size = xy.shape[-2]*xy.shape[-1]
        kf = torch.fft.rfft2(torch.exp(-1 / self.sigma ** 2 * (torch.clamp(xx.real[:,None,None]+yy.real[:,None,None]-2*xy.real,min=0)+1e-5)/xf_size), norm="ortho")
        #kf = torch.fft.rfft2(torch.exp(-1 / self.sigma ** 2 * (torch.abs(xx[:,None,None]+yy[:,None,None]-2*xy)+1e-5)/xf_size), norm="ortho")
        return kf[:, None, :, :]

    def forward(self, x): #z is template, x is search (or query), n is negative
        x_map, x_encode = self.extract_feature(x)
        if self.kernel=='gaussian':
            z_map = torch.fft.irfft2(self.model_zf, norm='ortho')
            kxzf = self.gaussian_kernel_correlation(x_map, z_map)
            response = torch.fft.irfft2(kxzf*torch.view_as_complex(self.model_alphaf), norm="ortho")
        else:
            xf = torch.view_as_real(torch.fft.rfft2(x_map, norm="ortho"))
            kxzf = torch.sum(complex_mulconj(xf, self.model_zf), dim=1, keepdim=True)
            response = torch.fft.irfft2(torch.view_as_complex(complex_mul(kxzf, self.model_alphaf)), norm="ortho")
        return [response, x_encode]

    def update(self, z, lr=1.):
        z_map, z_encode = self.extract_feature(z)
        if self.kernel=='gaussian':
            zf = torch.fft.rfft2(z_map, norm="ortho")
            kzzf = self.gaussian_kernel_correlation(z_map, z_map)
            kzzf = torch.view_as_real(kzzf)
        else:
            zf = torch.view_as_real(torch.fft.rfft2(z_map, norm="ortho")) # converts complex tensor to real reprensentation with dim (*,2)
            kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        alphaf = self.config.yf / (kzzf + self.config.lambda0) #yf shape:  torch.Size([1, 1, 48, 25, 2])
        # in re-id testing, templates are saved at command instead of using adaptive updating function
        if lr > 0.99:
            self.model_alphaf = alphaf
            self.model_zf = zf
        else:
            self.model_alphaf = (1 - lr) * self.model_alphaf.data + lr * alphaf.data
            self.model_zf = (1 - lr) * self.model_zf.data + lr * zf.data
        self.model_encode = z_encode

    def load_param(self, path):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict']
            self.load_state_dict(state_dict)
            print("loaded model state_dict")
        else:
            self.feature_net.load_state_dict(checkpoint)

class SqueezeCFNetTracker_reIdTest(object):
    def __init__(self, im, init_rect, net_param_path, gpu=True):
        self.gpu = gpu
        self.config = TrackerConfig(path=net_param_path, use_fire_layer="all", normalize=False)
        self.net = SqueezeCFNet_reIdTest(self.config)
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

        if self.gpu:
            self.net.update(target.cuda()) #self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())
        else:
            self.net.update(target)
        self.target_pos, self.target_sz = target_pos, target_sz
        self.patch_crop = np.zeros((self.config.num_scale, patch.shape[1], patch.shape[2], patch.shape[3]), np.float32)  # buff


    def update(self, img, cand_pos):
        window_sz = self.target_sz *  (1 + self.config.padding)
        bbox  = cxy_wh_2_bbox(cand_pos, window_sz)
        new_template = crop_chw(img, bbox, self.config.crop_sz)
        new_template = convert_format(new_template[None, :], self.config.normalize, self.config.mean, self.config.std)
        if self.gpu:
            self.net.update(torch.Tensor(new_template).cuda())
        else:
            self.net.update(torch.Tensor(new_template))

    def runResponseAnalysis(self, im, cand_pos):
        for i in range(self.config.num_scale):  # crop multi-scale search region
            window_sz = self.target_sz * (self.config.scale_factor[i] * (1 + self.config.padding))
            bbox = cxy_wh_2_bbox(cand_pos, window_sz)
            self.patch_crop[i, :] = crop_chw(im, bbox, self.config.crop_sz)

        search = convert_format(self.patch_crop, self.config.normalize, self.config.mean, self.config.std) #search = self.patch_crop - self.config.net_average_image
        if self.gpu:
            [response, encode] = self.net(torch.Tensor(search).cuda())
        else:
            [response, encode] = self.net(torch.Tensor(search))
        #print("response: ", response.shape) #(1, 3, 48, 48) for gaussian kernel, (3, 1, 48, 48) for linear kernel
        peak, idx = torch.max(response.view(self.config.num_scale, -1), 1) #(3, 48*48)
        peak = peak.data.cpu().numpy() * self.config.scale_penalties
        idx = idx.data.cpu().numpy()
        best_scale = np.argmax(peak)
        r_max, c_max = np.unravel_index(idx[best_scale], self.config.net_input_size)
        response_best_scale = torch.squeeze(response[best_scale,:,:,:]).cpu().detach().numpy()

        if r_max > self.config.net_input_size[0] / 2:
            r_max = r_max - self.config.net_input_size[0]
        if c_max > self.config.net_input_size[1] / 2:
            c_max = c_max - self.config.net_input_size[1]
        window_sz = self.target_sz * (self.config.scale_factor[best_scale] * (1 + self.config.padding))
        pos_diff = np.linalg.norm(np.array([c_max, r_max]) * window_sz / self.config.net_input_size)
        psr = PSR(response_best_scale)
        apce = APCE(response_best_scale)
        return pos_diff, psr, apce, []

    def runRotationAnalysis(self, im, cand_pos):
        rotation_patches = np.zeros((5, self.patch_crop.shape[1], self.patch_crop.shape[2], self.patch_crop.shape[3]), np.float32)  # buff
        window_sz = self.target_sz * (1 + self.config.padding)
        bbox = cxy_wh_2_bbox(cand_pos, window_sz)
        crop_ = crop_chw(im, bbox, self.config.crop_sz)
        crop = np.transpose(crop_, (1,2,0))
        rotation_patches[0,:] = np.transpose(cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),(2,0,1))
        rotation_patches[1,:] = np.transpose(cv2.rotate(crop, cv2.ROTATE_180),(2,0,1))
        rotation_patches[2,:] = np.transpose(cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),(2,0,1))
        rotation_patches[3,:] = np.transpose(cv2.flip(crop, 0), (2,0,1))
        rotation_patches[4,:] = np.transpose(cv2.flip(crop, 1), (2,0,1))

        PSRs = []
        APCEs = []
        search = convert_format(rotation_patches, self.config.normalize, self.config.mean, self.config.std) #search = self.patch_crop - self.config.net_average_image
        if self.gpu:
            [response, encode] = self.net(torch.Tensor(search).cuda())
        else:
            [response, encode] = self.net(torch.Tensor(search))
        for i in range(5):
            psr = PSR(torch.squeeze(response[i,:]).cpu().detach().numpy())
            apce = APCE(torch.squeeze(response[i,:]).cpu().detach().numpy())
            PSRs.append(psr)
            APCEs.append(apce)
        return PSRs, APCEs

class DCFNetTracker_reIdTest(object):
    def __init__(self, im, init_rect, net_param_path, gpu=True):
        self.gpu = gpu
        self.config = TrackerConfig_DCFNet(normalize=False)
        self.net_para_path = net_param_path
        self.net = DCFNet(self.config)
        self.net.load_param(net_param_path)
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
        self.net.update(target.cuda()) #self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())
        self.target_pos, self.target_sz = target_pos, target_sz
        self.patch_crop = np.zeros((self.config.num_scale, patch.shape[1], patch.shape[2], patch.shape[3]), np.float32)  # buff

    def update(self, img, cand_pos):
        window_sz = self.target_sz *  (1 + self.config.padding)
        bbox  = cxy_wh_2_bbox(cand_pos, window_sz)
        new_template = crop_chw(img, bbox, self.config.crop_sz)
        new_template = convert_format(new_template[None, :], self.config.normalize, self.config.mean, self.config.std)
        self.net.update(torch.Tensor(new_template).cuda())

    def runResponseAnalysis(self, im, cand_pos):
        for i in range(self.config.num_scale):  # crop multi-scale search region
            window_sz = self.target_sz * (self.config.scale_factor[i] * (1 + self.config.padding))
            bbox = cxy_wh_2_bbox(cand_pos, window_sz)
            self.patch_crop[i, :] = crop_chw(im, bbox, self.config.crop_sz)

        search = convert_format(self.patch_crop, self.config.normalize, self.config.mean, self.config.std) #search = self.patch_crop - self.config.net_average_image

        if self.gpu:
            response = self.net(torch.Tensor(search).cuda())
        else:
            response = self.net(torch.Tensor(search))
        #print("response: ", response.shape) #(1, 3, 48, 48) for gaussian kernel, (3, 1, 48, 48) for linear kernel
        peak, idx = torch.max(response.view(self.config.num_scale, -1), 1) #(3, 48*48)
        peak = peak.data.cpu().numpy() * self.config.scale_penalties
        idx = idx.data.cpu().numpy()
        best_scale = np.argmax(peak)
        r_max, c_max = np.unravel_index(idx[best_scale], self.config.net_input_size)
        response_best_scale = torch.squeeze(response[best_scale,:,:,:]).cpu().detach().numpy()


        if r_max > self.config.net_input_size[0] / 2:
            r_max = r_max - self.config.net_input_size[0]
        if c_max > self.config.net_input_size[1] / 2:
            c_max = c_max - self.config.net_input_size[1]
        window_sz = self.target_sz * (self.config.scale_factor[best_scale] * (1 + self.config.padding))
        pos_diff = np.linalg.norm(np.array([c_max, r_max]) * window_sz / self.config.net_input_size)
        psr = PSR(response_best_scale)
        apce = APCE(response_best_scale)
        return pos_diff, psr, apce, []

    def runRotationAnalysis(self, im, cand_pos):
        rotation_patches = np.zeros((5, self.patch_crop.shape[1], self.patch_crop.shape[2], self.patch_crop.shape[3]), np.float32)  # buff
        window_sz = self.target_sz * (1 + self.config.padding)
        bbox = cxy_wh_2_bbox(cand_pos, window_sz)
        crop_ = crop_chw(im, bbox, self.config.crop_sz)
        crop = np.transpose(crop_, (1,2,0))
        rotation_patches[0,:] = np.transpose(cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),(2,0,1))
        rotation_patches[1,:] = np.transpose(cv2.rotate(crop, cv2.ROTATE_180),(2,0,1))
        rotation_patches[2,:] = np.transpose(cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),(2,0,1))
        rotation_patches[3,:] = np.transpose(cv2.flip(crop, 0), (2,0,1))
        rotation_patches[4,:] = np.transpose(cv2.flip(crop, 1), (2,0,1))

        PSRs = []
        APCEs = []
        search = convert_format(rotation_patches, self.config.normalize, self.config.mean, self.config.std) #search = self.patch_crop - self.config.net_average_image
        if self.gpu:
            response = self.net(torch.Tensor(search).cuda())
        else:
            response = self.net(torch.Tensor(search))
        for i in range(5):
            psr = PSR(torch.squeeze(response[i,:]).cpu().detach().numpy())
            apce = APCE(torch.squeeze(response[i,:]).cpu().detach().numpy())
            PSRs.append(psr)
            APCEs.append(apce)
        return PSRs, APCEs

def list_mean(list):
    return sum(list)/len(list)

def processTestImSeq(imSeq_dir, net_param_path=None, model='squeezeCF', update=False):
    PSR_p_list = []
    pos_diff_p_list = []
    PSR_n_list = []
    pos_diff_n_list = []
    APCE_p_list = []
    APCE_n_list = []
    score_p_list = []
    score_n_list = []
    if glob.glob(os.path.join(imSeq_dir, '*.json')):
        annotation_path = glob.glob(os.path.join(imSeq_dir, '*.json'))[0]
        annotation = parseManualAnnotation(annotation_path)
    else:
        print("no annotation json file")
        return PSR_p_list, PSR_n_list, pos_diff_p_list, pos_diff_n_list, APCE_p_list, APCE_n_list, []
    for obj in annotation[0]:
        if obj["id"] == 0:
            init_rect = obj["bbox"]
            #save template
    img0_path = os.path.join(imSeq_dir, str(0).zfill(6)+".jpg")
    img0 = cv2.imread(img0_path)
    if model == 'squeezeCF':
        tracker=SqueezeCFNetTracker_reIdTest(img0, init_rect, net_param_path)
    elif model == 'DCFNet':
        tracker = DCFNetTracker_reIdTest(img0, init_rect, net_param_path)
    elif model == 'hog':
        tracker = KCF_HOG()
        tracker.init(img0,init_rect)
    else:
        print("model type not supported")
        return
    correct_dist_count = 0
    correct_apce_count = 0
    correct_psr_count = 0
    frame_count = 0
    for frame_number in sorted(annotation.keys()):
        frame_info = annotation[frame_number]
        img_path = os.path.join(imSeq_dir, str(frame_number).zfill(6)+".jpg")
        img = cv2.imread(img_path)
        min_encode_dist = 1e6
        min_dist_correct = False #whethere or not the minimum encoding distance belongs to target 0
        max_apce = 0
        max_apce_correct = False
        max_psr = 0
        max_psr_correct = False
        for region_info in frame_info:
            cand_cxy_wh = rect1_2_cxy_wh(region_info["bbox"])
            cand_pos = cand_cxy_wh[0]
            if region_info["id"] == 0:
                if update:
                    if model == 'hog':
                        tracker.init(img, region_info["bbox"])
                    else:
                        tracker.update(img, cand_pos)
                pos_diff_p, PSR_p, APCE_p, encode_dist_p = tracker.runResponseAnalysis(img, cand_pos)
                if encode_dist_p:
                    if encode_dist_p < min_encode_dist:
                        min_encode_dist = encode_dist_p
                        min_dist_correct = True
                if APCE_p > max_apce:
                    max_apce = APCE_p
                    max_apce_correct = True
                if PSR_p > max_psr:
                    max_psr = PSR_p
                    max_psr_correct = True
                PSR_p_list.append(PSR_p)
                APCE_p_list.append(APCE_p)
                pos_diff_p_list.append(pos_diff_p)
                score_p = PSR_p*10 - pos_diff_p
                score_p_list.append(score_p)
            else:
                pos_diff_n, PSR_n, APCE_n, encode_dist_n = tracker.runResponseAnalysis(img, cand_pos)
                if encode_dist_n:
                    if encode_dist_n < min_encode_dist:
                        min_encode_dist = encode_dist_n
                        min_dist_correct = False
                if APCE_n > max_apce:
                    max_apce = APCE_n
                    max_apce_correct = False
                if PSR_n > max_psr:
                    max_psr = PSR_n
                    max_psr_correct = False
                PSR_n_list.append(PSR_n)
                APCE_n_list.append(APCE_n)
                pos_diff_n_list.append(pos_diff_n)
                score_n = PSR_n*10 - pos_diff_n
                score_n_list.append(score_n)
        if min_dist_correct:
            correct_dist_count += 1
        if max_apce_correct:
            correct_apce_count += 1
        if max_psr_correct:
            correct_psr_count += 1
        frame_count += 1
    encoding_pred_acc = correct_dist_count/len(PSR_p_list)
    apce_pred_acc = correct_apce_count/len(PSR_p_list)
    psr_pred_acc = correct_psr_count/len(PSR_p_list)
    acc_list = [encoding_pred_acc, apce_pred_acc, psr_pred_acc]

    """
    print("APCE_p mean: ", list_mean(APCE_p_list))
    print("APCE_n mean: ", list_mean(APCE_n_list))
    print("PSR p mean: ", list_mean(PSR_p_list))
    print("PSR n mean: ", list_mean(PSR_n_list))
    print("posdiff p mean: ", list_mean(pos_diff_p_list))
    print("posdiff n mean: ", list_mean(pos_diff_n_list))
    print("mean_scorep: ", list_mean(score_p_list))
    print("mean_scoren: ", list_mean(score_n_list))
    """

    print("encoding prediction accuracy: ", encoding_pred_acc)
    print("APCE prediction accuracy: ", apce_pred_acc)
    print("PSR prediction accuracy: ", psr_pred_acc)
    return PSR_p_list, PSR_n_list, pos_diff_p_list, pos_diff_n_list, APCE_p_list, APCE_n_list, acc_list

def processTrainValDataset(json_file_path, tracker_model='squeezeCF', net_param_path=None, update=False):
    val_dataset = FNTDataset(json_file_path, train=False, normalize=True)
    print("total number of val images: ", len(val_dataset)) #818 for regular FathomNet, 87 for mini
    species_list = val_dataset.get_species_list()
    json_r = val_dataset.get_path_info()
    out = dict()

    #combine train_set and val_set items and sort them by species
    name2list = dict() # key: species name, value: species item paths
    for species in species_list:
        species_item_paths = []
        for i, item in enumerate(json_r[species]['train_set']):
            species_item_paths.append(item)
        for i, item in enumerate(json_r[species]['val_set']):
            species_item_paths.append(item)
        name2list[species] = species_item_paths

    numSpecies = len(species_list)
    name2num = dict()
    for i, (k, v) in enumerate(name2list.items()):
        name2num[k] = i
    print("Species name to species id number: ", name2num)
    avgPSR_conf = np.zeros((numSpecies, numSpecies))
    avgAPCE_conf = np.zeros((numSpecies, numSpecies))
    avgEnc_conf = np.zeros((numSpecies, numSpecies))

    #outer loop: iterate learning each species as template
    for species_learn in name2list.keys():
        item_paths = name2list[species_learn]
        item_id = 0
        pos_diff_list_ego = []
        PSR_list_ego = []
        APCE_list_ego = []
        enc_dist_list_ego = []
        learn_idx = name2num[species_learn]
        for item in item_paths:
            # initialize tracker template
            img = cv2.imread(item)
            img_h, img_w, _ = img.shape
            init_rect = [img_w/3, img_h/3, img_w/3, img_h/3]
            target_pos = [img_w/2, img_h/2]
            if item_id == 0: #initialize tracker at the first item
                if tracker_model=='squeezeCF':
                    tracker=SqueezeCFNetTracker_reIdTest(img, init_rect, net_param_path)
                elif tracker_model == 'DCFNet':
                    tracker = DCFNetTracker_reIdTest(img, init_rect, net_param_path)
                else:
                    tracker = KCF_HOG()
                    tracker.init(img,init_rect)
                for species_test in name2list.keys(): #inner loop
                    print("test_species: ", species_test)
                    if species_test != species_learn:
                        test_idx = name2num[species_test]
                        items2test = name2list[species_test]
                        pos_diff_test = []
                        PSR_test = []
                        APCE_test = []
                        enc_dist_test = []
                        for testItem in items2test:
                            img_test = cv2.imread(testItem)
                            pos_diff, PSR, APCE, enc_dist = tracker.runResponseAnalysis(img_test, target_pos)
                            pos_diff_test.append(pos_diff)
                            PSR_test.append(PSR)
                            APCE_test.append(APCE)
                            enc_dist_test.append(enc_dist)
                        avgPSR_conf[learn_idx][test_idx] = list_mean(PSR_test)
                        avgAPCE_conf[learn_idx][test_idx] = list_mean(APCE_test)
            else:
                if update:
                    if tracker_model == 'hog':
                        tracker.init(img, init_rect)
                    else:
                        tracker.update(img, target_pos)
                pos_diff, PSR, APCE, enc_dist = tracker.runResponseAnalysis(img, target_pos)
                pos_diff_list_ego.append(pos_diff)
                PSR_list_ego.append(PSR)
                APCE_list_ego.append(APCE)
                enc_dist_list_ego.append(enc_dist)
            item_id += 1
        avgPSR_conf[learn_idx][learn_idx] = list_mean(PSR_list_ego)
        avgAPCE_conf[learn_idx][learn_idx] = list_mean(APCE_list_ego)

    print("avgPSR_conf: ", avgPSR_conf)
    print("avgAPCE_conf: ", avgAPCE_conf)
    print("avgEnc_conf: ", avgEnc_conf)
    return avgPSR_conf, avgAPCE_conf, avgEnc_conf

def processRotationTest(imSeq_dir, SqueezeCFnet_param_path, DCFnet_param_path):
    hog_PSRs_total = np.empty((0,5))
    hog_APCEs_total = np.empty((0,5))
    SCF_PSRs_total = np.empty((0,5))
    SCF_APCEs_total = np.empty((0,5))
    DCF_PSRs_total = np.empty((0,5))
    DCF_APCEs_total = np.empty((0,5))
    if glob.glob(os.path.join(imSeq_dir, '*.json')):
        annotation_path = glob.glob(os.path.join(imSeq_dir, '*.json'))[0]
        annotation = parseManualAnnotation(annotation_path)
    else:
        return hog_PSRs_total, hog_APCEs_total, SCF_PSRs_total, SCF_APCEs_total, DCF_PSRs_total, DCF_APCEs_total
    for obj in annotation[0]:
        if obj["id"] == 0:
            init_rect = obj["bbox"]
            #save template
    img0_path = os.path.join(imSeq_dir, str(0).zfill(6)+".jpg")
    img0 = cv2.imread(img0_path)
    SCF_tracker=SqueezeCFNetTracker_reIdTest(img0, init_rect, SqueezeCFnet_param_path)
    DCF_tracker = DCFNetTracker_reIdTest(img0, init_rect, DCFnet_param_path)
    hog_tracker = KCF_HOG()
    hog_tracker.init(img0,init_rect)
    for frame_number in sorted(annotation.keys()):
        frame_info = annotation[frame_number]
        img_path = os.path.join(imSeq_dir, str(frame_number).zfill(6)+".jpg")
        img = cv2.imread(img_path)
        for region_info in frame_info:
            if region_info["id"] == 0:
                cand_cxy_wh = rect1_2_cxy_wh(region_info["bbox"])
                cand_pos = cand_cxy_wh[0]
                hog_tracker.init(img, region_info["bbox"])
                SCF_tracker.update(img, cand_pos)
                DCF_tracker.update(img, cand_pos)
                hog_PSRs, hog_APCEs = hog_tracker.runRotationAnalysis(img, cand_pos)
                SCF_PSRs, SCF_APCEs = SCF_tracker.runRotationAnalysis(img, cand_pos)
                DCF_PSRs, DCF_APCEs = DCF_tracker.runRotationAnalysis(img, cand_pos)
                hog_PSRs_total = np.vstack((hog_PSRs_total, hog_PSRs))
                hog_APCEs_total = np.vstack((hog_APCEs_total, hog_APCEs))
                SCF_PSRs_total = np.vstack((SCF_PSRs_total, SCF_PSRs))
                SCF_APCEs_total = np.vstack((SCF_APCEs_total, SCF_APCEs))
                DCF_PSRs_total = np.vstack((DCF_PSRs_total, DCF_PSRs))
                DCF_APCEs_total = np.vstack((DCF_APCEs_total, DCF_APCEs))
                #print("hog PSRs: ", hog_PSRs)
                #print("hog APCEs: ", hog_APCEs)
                #print("SCF_PSRs: ", SCF_PSRs)
                #print("SCF APCEs: ", SCF_APCEs)
                #print("DCF PSRs: ", DCF_PSRs)
                #print("DCF APCEs: ", DCF_APCEs)
    return hog_PSRs_total, hog_APCEs_total, SCF_PSRs_total, SCF_APCEs_total, DCF_PSRs_total, DCF_APCEs_total





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Testing SqueezeCFNet with baselines in Pytorch 1.12.1')
    parser.add_argument('--seq-root', dest='dataset_root', default='', type=str, help='directory to all image sequence folders')
    parser.add_argument('--json-path', dest='json_file_path', default='', type=str, help='path to dataset *.json file')
    parser.add_argument('--test-mode', dest='test_mode', default=0, type=int, metavar='N',
                        help='testing mode. default 0: test performance on image sequence data; 1: test on FathomNet training data; 2: test rotational invariance on image sequence data')


    args = parser.parse_args()

    SqueezeCFnet_param_path = os.path.join(os.getcwd(), 'checkpoints', 'apce_enc_200ep_1e-4_best.pt') #or replace with other set of trained parameters
    DCFnet_param_path = os.path.join(os.getcwd(), 'checkpoints', 'model_best_DCFnet_200ep.pt')
    dataset_root = args.dataset_root
    json_file_path = args.json_file_path
    seqs = dataset_root + '/*/'
    test_mode = args.test_mode
    test_models = ['squeezeCF', 'DCFNet', 'hog']
    if test_mode == 0: #test re-id on image sequence data
        print("Testing re-id on image sequence data")
        for model in test_models:
            out = dict()
            out["PSR_p_lists"] = dict()
            out["PSR_n_lists"] = dict()
            out["APCE_p_lists"] = dict()
            out["APCE_n_lists"] = dict()
            out["acc_list"] = dict()
            out["pos_diff_p_lists"] = dict()
            FPr_total = []
            FNr_total = []
            PSR_p_total = []
            PSR_n_total = []
            pos_diff_p_total = []
            pos_diff_n_total = []
            for imSeq_dir in glob.glob(seqs):
                seqName = imSeq_dir.split('/')[-2]
                PSR_p_list, PSR_n_list, pos_diff_p_list, pos_diff_n_list, APCE_p_list, APCE_n_list, acc_list \
                    = processTestImSeq(imSeq_dir, SqueezeCFnet_param_path, model=model, update=False)
                if PSR_p_list and PSR_n_list:
                    out["PSR_p_lists"][seqName] = PSR_p_list
                    out["PSR_n_lists"][seqName] = PSR_n_list
                    out["APCE_p_lists"][seqName] = APCE_p_list
                    out["APCE_n_lists"][seqName] = APCE_n_list
                    out["acc_list"][seqName] = acc_list
                    out["pos_diff_p_lists"][seqName] = pos_diff_p_list
            scipy.io.savemat('testImSeq_'+model+'.mat', out)
    elif test_mode == 1: #test on FathomNet training data
        print("Testing re-id on FathomNet training data")
        for model in test_models:
            out = dict()
            avgPSR_conf, avgAPCE_conf, avgEnc_conf = processTrainValDataset(json_file_path,
                                                tracker_model=model, \
                                                net_param_path=SqueezeCFnet_param_path,
                                                update=True)
            out["PSR_conf"] = avgPSR_conf
            out["APCE_conf"] = avgAPCE_conf
            out["avgEnc_conf"] = avgEnc_conf
            scipy.io.savemat('testFathomNet_'+model+'.mat', out)
    else: #test rotational invariance on image sequence data
        print("Testing rotational invariance on image sequence data")
        out = dict()
        hog_PSRs_total = np.empty((0,5))
        hog_APCEs_total = np.empty((0,5))
        SCF_PSRs_total = np.empty((0,5))
        SCF_APCEs_total = np.empty((0,5))
        DCF_PSRs_total = np.empty((0,5))
        DCF_APCEs_total = np.empty((0,5))
        for imSeq_dir in glob.glob(seqs):
            seqName = imSeq_dir.split('/')[-2]
            hog_PSRs_seq, hog_APCEs_seq, SCF_PSRs_seq, SCF_APCEs_seq, DCF_PSRs_seq, DCF_APCEs_seq \
              = processRotationTest(imSeq_dir, SqueezeCFnet_param_path, DCFnet_param_path)
            hog_PSRs_total = np.vstack((hog_PSRs_total, hog_PSRs_seq))
            hog_APCEs_total = np.vstack((hog_APCEs_total, hog_APCEs_seq))
            SCF_PSRs_total = np.vstack((SCF_PSRs_total, SCF_PSRs_seq))
            SCF_APCEs_total = np.vstack((SCF_APCEs_total, SCF_APCEs_seq))
            DCF_PSRs_total = np.vstack((DCF_PSRs_total, DCF_PSRs_seq))
            DCF_APCEs_total = np.vstack((DCF_APCEs_total, DCF_APCEs_seq))
        out["hog_PSR"] = hog_PSRs_total
        out["hog_APCE"] = hog_APCEs_total
        out["SCF_PSR"] = SCF_PSRs_total
        out["SCF_APCE"] = SCF_APCEs_total
        out["DCF_PSR"] = DCF_PSRs_total
        out["DCF_APCE"] = DCF_APCEs_total
        scipy.io.savemat('test_Rotation.mat', out)
