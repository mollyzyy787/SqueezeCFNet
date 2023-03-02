from torchvision.models import squeezenet
import torch
import torch.nn as nn
import torch.nn.init as init

def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)

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
    def __init__(self, config=None):
        super().__init__()
        self.feature_net = FeatSqueezeNet()
        self.model_alphaf = []
        self.model_zf = []
        self.config = config
        self.use_fire_layer = config.use_fire_layer

    def forward(self, x): #z is template, x is search (or query), n is negative
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
        xf = torch.view_as_real(torch.fft.rfft2(x_map, norm="ortho"))
        kxzf = torch.sum(complex_mulconj(xf, self.model_zf), dim=1, keepdim=True)
        response = torch.fft.irfft2(torch.view_as_complex(complex_mul(kxzf, self.model_alphaf)), norm="ortho")
        return [response, x_encode]

    def update(self, z, lr=1.):
        z_encode, z1_map, z2_map, z3_map = self.feature_net(z)
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



if __name__ == '__main__':

    # network test
    net = FeatSqueezeNet()
    net.eval()
