import torch.nn as nn
import torch  # pytorch 0.4.0! fft
import numpy as np
import cv2
from utils import crop_chw, gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox, convert_format, PSR, APCE


def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)

def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g.astype(np.float32)

class TrackerConfig_DCFNet(object):
    def __init__(self, normalize=False, gpu=True):
        self.normalize = normalize
        self.gpu = gpu
        self.crop_sz = 200 #network input size is 200x200
        self.output_sz = 200 #feature map size is 200x200 because DCFnet_track added padding in DCFNetFeature

        self.lambda0 = 1e-4
        self.padding = 2.0
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
            self.yf = torch.view_as_real(torch.fft.rfft2(torch.Tensor(self.y).view(1, 1, self.output_sz, self.output_sz), norm="ortho")).cuda()
            self.cos_window = torch.Tensor(np.outer(np.hanning(self.crop_sz), np.hanning(self.crop_sz))).cuda()  # train without cos window
        else:
            self.yf = torch.view_as_real(torch.fft.rfft2(torch.Tensor(self.y).view(1, 1, self.output_sz, self.output_sz), norm="ortho"))
            self.cos_window = torch.Tensor(np.outer(np.hanning(self.crop_sz), np.hanning(self.crop_sz)))  # train without cos window
        self.mean = 42.14 # for img training set
        self.std = 32.12


class DCFNetFeature(nn.Module):
    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)


class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        self.model_alphaf = []
        self.model_xf = []
        self.config = config

    def forward(self, x):
        x = self.feature(x) * self.config.cos_window
        xf = torch.view_as_real(torch.fft.rfft2(x, norm='ortho'))
        kxzf = torch.sum(complex_mulconj(xf, self.model_zf), dim=1, keepdim=True)
        response =  torch.fft.irfft2(torch.view_as_complex(complex_mul(kxzf, self.model_alphaf)), norm='ortho')
        # r_max = torch.max(response)
        # cv2.imshow('response', response[0, 0].data.cpu().numpy())
        # cv2.waitKey(0)
        return response

    def update(self, z, lr=1.):
        z = self.feature(z) * self.config.cos_window
        zf = torch.view_as_real(torch.fft.rfft2(z, norm='ortho'))
        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        alphaf = self.config.yf / (kzzf + self.config.lambda0)
        if lr > 0.99:
            self.model_alphaf = alphaf
            self.model_zf = zf
        else:
            self.model_alphaf = (1 - lr) * self.model_alphaf.data + lr * alphaf.data
            self.model_zf = (1 - lr) * self.model_zf.data + lr * zf.data

    def load_param(self, path='param.pth'):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict']
            self.load_state_dict(state_dict)
            print("loaded model state_dict")
        else:
            self.feature.load_state_dict(checkpoint)


class DCFNetTracker(object):
    def __init__(self, im, init_rect, net_param_path, gpu=True):
        self.gpu = gpu
        config=TrackerConfig_DCFNet(normalize=True, gpu=gpu)
        self.config = config
        self.net = DCFNet(config)
        self.net.load_param(net_param_path)
        self.net.eval()
        if gpu:
            self.net.cuda()

        # confine results
        target_pos, target_sz = rect1_2_cxy_wh(init_rect)
        self.min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
        self.max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

        # crop template
        window_sz = target_sz * (1 + config.padding)
        bbox = cxy_wh_2_bbox(target_pos, window_sz)
        patch = crop_chw(im, bbox, self.config.crop_sz)
        patch = np.expand_dims(patch, axis=0).astype(np.float32)

        target = convert_format(patch, self.config.normalize, self.config.mean, self.config.std) #replaced: target = patch - config.net_average_image
        if gpu:
            self.net.update(torch.Tensor(target).cuda())
        else:
            self.net.update(torch.Tensor(target))
        self.target_pos, self.target_sz = target_pos, target_sz
        self.patch_crop = np.zeros((config.num_scale, patch.shape[1], patch.shape[2], patch.shape[3]), np.float32)  # buff

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
        target = convert_format(patch, self.config.normalize, self.config.mean, self.config.std) #replaced: target = patch - config.net_average_image
        if self.gpu:
            self.net.update(torch.Tensor(target).cuda(), lr=self.config.interp_factor)
        else:
            self.net.update(torch.Tensor(target), lr=self.config.interp_factor)
        return cxy_wh_2_rect1(self.target_pos, self.target_sz)  # 1-index

    """
    The following methods are used for reId tests only, not tracking
    """
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
