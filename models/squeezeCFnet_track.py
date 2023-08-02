import cv2
from torchvision.models import squeezenet
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from utils import crop_chw, gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox, convert_format, PSR, APCE

def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)

def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)

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

class FeatSqueezeNet(nn.Module):
    def __init__(self, version: str = "1", dropout: float = 0.5) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        self.version = version
        self.prefire = nn.Sequential(
                nn.Conv2d(1, 96, kernel_size=7, stride=2), #(in_channels, out_channels)
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                )
        self.fire1 = squeezenet.Fire(96, 16, 64, 64)
        self.fire2 = squeezenet.Fire(128, 16, 64, 64)
        self.fire3 = squeezenet.Fire(128, 32, 128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire4 = squeezenet.Fire(256, 32, 128, 128)
        self.fire5 = squeezenet.Fire(256, 48, 192, 192)
        self.fire6 = squeezenet.Fire(384, 48, 192, 192)
        self.fire7 = squeezenet.Fire(384, 64, 256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire8 = squeezenet.Fire(512, 64, 256, 256)
        if version == "1": #1.7M params
            # Final convolution is initialized differently from the rest
            final_conv = nn.Conv2d(512, 1000, kernel_size=1)
            self.narrowing = nn.Sequential(
                nn.Dropout(p=dropout),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.encoding = nn.Sequential(
                nn.Linear(1000, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
        elif version == "2": #8M params (concentrated in the final layers)
            final_conv = nn.Conv2d(512, 512, kernel_size = 3)
            self.encoding = nn.Sequential(
                nn.Dropout(p=dropout),
                final_conv,
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=7, stride=2, ceil_mode=True),
                nn.Flatten(),
                nn.Linear(4608,1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
        else:
            raise ValueError(f"Unsupported version {version}: 1 or 2 expected")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prefire(x)
        fire1_feat = x = self.fire1(x) #128 channels
        fire2_feat = x = self.fire2(x) #128 channels
        fire3_feat = x = self.fire3(x) #256 channels
        x = self.pool2(x)
        x = self.fire7(self.fire6(self.fire5(self.fire4(x))))
        x = self.pool3(x)
        x = self.fire8(x)
        if self.version == "1":
            x = self.narrowing(x)
            x = torch.squeeze(x)
            x = self.encoding(x)
        else:
            x = self.encoding(x)

        return [x, fire1_feat, fire2_feat, fire3_feat]

class SqueezeCFNet(nn.Module):
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

class SqueezeCFNetTracker(object):
    def __init__(self, im, init_rect, net_param_path, gpu=True):
        self.gpu = gpu
        self.config = TrackerConfig(path=net_param_path, use_fire_layer="all", normalize=False)
        self.net = SqueezeCFNet(self.config)
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
        r_max, c_max = np.unravel_index(idx[best_scale], [self.config.output_sz, self.config.output_sz])

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

    """
    The following methods are used for re-id tests only
    """
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
        r_max, c_max = np.unravel_index(idx[best_scale], [self.config.output_sz, self.config.output_sz])
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

class FeatSqueezeNet_light(nn.Module):
    def __init__(self, version: str = "1", dropout: float = 0.5) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        self.version = version
        self.prefire = nn.Sequential(
                nn.Conv2d(1, 96, kernel_size=7, stride=2), #(in_channels, out_channels)
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                )
        self.fire1 = squeezenet.Fire(96, 16, 64, 64)
        self.fire2 = squeezenet.Fire(128, 16, 64, 64)
        self.fire3 = squeezenet.Fire(128, 32, 128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire4 = squeezenet.Fire(256, 32, 128, 128)
        self.fire5 = squeezenet.Fire(256, 48, 192, 192)
        self.fire6 = squeezenet.Fire(384, 48, 192, 192)
        self.fire7 = squeezenet.Fire(384, 64, 256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire8 = squeezenet.Fire(512, 64, 256, 256)
        if version == "1": #1.7M params
            # Final convolution is initialized differently from the rest
            final_conv = nn.Conv2d(512, 1000, kernel_size=1)
            self.narrowing = nn.Sequential(
                nn.Dropout(p=dropout),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.encoding = nn.Sequential(
                nn.Linear(1000, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
        elif version == "2": #8M params (concentrated in the final layers)
            final_conv = nn.Conv2d(512, 512, kernel_size = 3)
            self.encoding = nn.Sequential(
                nn.Dropout(p=dropout),
                final_conv,
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=7, stride=2, ceil_mode=True),
                nn.Flatten(),
                nn.Linear(4608,1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
        else:
            raise ValueError(f"Unsupported version {version}: 1 or 2 expected")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prefire(x)
        fire1_feat = x = self.fire1(x) #128 channels
        fire2_feat = x = self.fire2(x) #128 channels
        fire3_feat = x = self.fire3(x) #256 channels
        return [fire1_feat, fire2_feat, fire3_feat]

class SqueezeCFNet_light(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.feature_net = FeatSqueezeNet_light()
        self.model_alphaf = []
        self.model_zf = []
        self.config = config
        self.use_fire_layer = config.use_fire_layer

    def forward(self, x): #z is template, x is search (or query), n is negative
        x1_map, x2_map, x3_map = self.feature_net(x)
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
        xf = torch.view_as_real(torch.fft.rfft2(x_map, norm="ortho"))
        kxzf = torch.sum(complex_mulconj(xf, self.model_zf), dim=1, keepdim=True)
        response = torch.fft.irfft2(torch.view_as_complex(complex_mul(kxzf, self.model_alphaf)), norm="ortho")
        return response

    def update(self, z, lr=1.):
        z1_map, z2_map, z3_map = self.feature_net(z)
        if self.use_fire_layer == "1":
            z_map = z1_map
        elif self.use_fire_layer == "2":
            z_map = z2_map
        elif self.use_fire_layer == "3":
            z_map = z3_map
        else:
            z_map = torch.cat((z1_map, z2_map, z3_map), dim=1)
        z_map = z_map * self.config.cos_window
        zf = torch.view_as_real(torch.fft.rfft2(z_map, norm="ortho")) # converts complex tensor to real reprensentation with dim (*,2)
        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        alphaf = self.config.yf / (kzzf + self.config.lambda0)

        if lr > 0.99:
            self.model_alphaf = alphaf
            self.model_zf = zf
        else:
            self.model_alphaf = (1 - lr) * self.model_alphaf.data + lr * alphaf.data
            self.model_zf = (1 - lr) * self.model_zf.data + lr * zf.data

    def load_param(self, path):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict']
            self.load_state_dict(state_dict)
            print("loaded model state_dict")
        else:
            self.feature_net.load_state_dict(checkpoint)

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
        r_max, c_max = np.unravel_index(idx[best_scale], [self.config.output_sz, self.config.output_sz])

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


if __name__ == '__main__':

    # network test
    net = FeatSqueezeNet()
    net.eval()
