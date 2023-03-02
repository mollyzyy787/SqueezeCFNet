import numpy as np
import cv2
from .libs.features import fhog
from .fft_tools import fft2,ifft2

def extract_hog_feature(img, cell_size=4):
    fhog_feature=fhog(img.astype(np.float32),cell_size,num_orients=9,clip=0.2)[:,:,:-1]
    return fhog_feature

def cos_window(sz):
    """
    width, height = sz
    j = np.arange(0, width)
    i = np.arange(0, height)
    J, I = np.meshgrid(j, i)
    cos_window = np.sin(np.pi * J / width) * np.sin(np.pi * I / height)
    """

    cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(np.hanning(int(sz[0]))[np.newaxis, :])
    return cos_window

"""
max val at the top left loc
"""
def gaussian2d_rolled_labels(sz,sigma):
    w,h=sz
    xs, ys = np.meshgrid(np.arange(w)-w//2, np.arange(h)-h//2)
    dist = (xs**2+ys**2) / (sigma**2)
    labels = np.exp(-0.5*dist)
    labels = np.roll(labels, -int(np.floor(sz[0] / 2)), axis=1)
    labels=np.roll(labels,-int(np.floor(sz[1]/2)),axis=0)
    return labels

def PSR(response):
    response_map=response.copy()
    max_loc=np.unravel_index(np.argmax(response_map, axis=None),response_map.shape)
    y,x=max_loc
    F_max = np.max(response_map)
    response_map[y-5:y+6,x-5:x+6]=0.
    mean=np.mean(response_map[response_map>0])
    std=np.std(response_map[response_map>0])
    psr=(F_max-mean)/(std+1e-5)
    return psr

def APCE(response_map):
    Fmax=np.max(response_map)
    Fmin=np.min(response_map)
    apce=(Fmax-Fmin)**2/(np.mean((response_map-Fmin)**2))
    return apce

class BaseCF:
    def __init__(self):
        raise NotImplementedError

    def init(self,first_frame,bbox):
        raise NotImplementedError

    def update(self,current_frame):
        raise NotImplementedError

# from: https://github.com/fengyang95/pyCFTrackers/blob/master/cftracker/kcf.py
class KCF_HOG(BaseCF):
    def __init__(self, padding=1.5, features='hog', kernel='gaussian'):
        super(KCF_HOG).__init__()
        self.padding = padding
        self.lambda_ = 1e-4
        self.features = features
        self.w2c=None
        # hog feature parameters
        self.interp_factor = 0.02
        self.sigma = 0.5
        self.cell_size=4
        self.output_sigma_factor=0.1
        self.kernel=kernel

    def init(self,first_frame,bbox):
        assert len(first_frame.shape)==3 and first_frame.shape[2]==3
        bbox = np.array(bbox).astype(np.int64)
        x0, y0, w, h = tuple(bbox)
        self.crop_size = (int(np.floor(w * (1 + self.padding))), int(np.floor(h * (1 + self.padding))))# for vis
        self._center = (np.floor(x0 + w / 2),np.floor(y0 + h / 2))
        self.w, self.h = w, h
        self.window_size=(int(np.floor(w*(1+self.padding)))//self.cell_size,int(np.floor(h*(1+self.padding)))//self.cell_size)
        self._window = cos_window(self.window_size)

        s=np.sqrt(w*h)*self.output_sigma_factor/self.cell_size
        self.yf = fft2(gaussian2d_rolled_labels(self.window_size, s))

        # extract hog features
        x=self._crop(first_frame,self._center,(w,h))
        x=cv2.resize(x,(self.window_size[0]*self.cell_size,self.window_size[1]*self.cell_size))
        x=extract_hog_feature(x, cell_size=self.cell_size)

        self.xf = fft2(self._get_windowed(x, self._window))
        self.init_response_center = (0,0)
        self.alphaf = self._training(self.xf,self.yf)

    def runResponseAnalysis(self,current_frame, cand_pos):
        assert len(current_frame.shape) == 3 and current_frame.shape[2] == 3
        # update template hog features
        z = self._crop(current_frame, cand_pos, (self.w, self.h))
        z = cv2.resize(z, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
        z = extract_hog_feature(z, cell_size=self.cell_size)

        zf = fft2(self._get_windowed(z, self._window))
        responses = self._detection(self.alphaf, self.xf, zf, kernel=self.kernel)

        curr =np.unravel_index(np.argmax(responses, axis=None),responses.shape)

        if curr[0]+1>self.window_size[1]/2:
            dy=curr[0]-self.window_size[1]
        else:
            dy=curr[0]
        if curr[1]+1>self.window_size[0]/2:
            dx=curr[1]-self.window_size[0]
        else:
            dx=curr[1]
        dy,dx=dy*self.cell_size,dx*self.cell_size
        pos_diff = np.linalg.norm(np.array([dy, dx]))
        psr = PSR(responses)
        apce = APCE(responses)
        return pos_diff, psr, apce, []

    def runRotationAnalysis(self, im, cand_pos):
        crop = self._crop(im, cand_pos, (self.w, self.h))
        crop_dim = (crop.shape[1], crop.shape[0])
        test_crop = cv2.resize(cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE), crop_dim)
        rotation_patches = np.zeros((5, crop.shape[0], crop.shape[1], crop.shape[2]), np.float32)  # buff
        rotation_patches[0,:] = cv2.resize(cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE), crop_dim)
        rotation_patches[1,:] = cv2.rotate(crop, cv2.ROTATE_180)
        rotation_patches[2,:] = cv2.resize(cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE), crop_dim)
        rotation_patches[3,:] = cv2.flip(crop, 0)
        rotation_patches[4,:] = cv2.flip(crop, 1)
        PSRs = []
        APCEs = []
        for i in range(5):
            hog_feat = extract_hog_feature(rotation_patches[i,:])
            hog_feat_f = fft2(self._get_windowed(hog_feat, self._window))
            response = self._detection(self.alphaf, self.xf, hog_feat_f, kernel=self.kernel)
            psr = PSR(response)
            apce = APCE(response)
            PSRs.append(psr)
            APCEs.append(apce)
        return PSRs, APCEs

    def update(self,current_frame,vis=False):
        assert len(current_frame.shape) == 3 and current_frame.shape[2] == 3

        # update template hog features
        z = self._crop(current_frame, self._center, (self.w, self.h))
        z = cv2.resize(z, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
        z = extract_hog_feature(z, cell_size=self.cell_size)

        zf = fft2(self._get_windowed(z, self._window))
        responses = self._detection(self.alphaf, self.xf, zf, kernel=self.kernel)
        if vis is True:
            self.score=responses
            self.score = np.roll(self.score, int(np.floor(self.score.shape[0] / 2)), axis=0)
            self.score = np.roll(self.score, int(np.floor(self.score.shape[1] / 2)), axis=1)

        curr =np.unravel_index(np.argmax(responses, axis=None),responses.shape)

        if curr[0]+1>self.window_size[1]/2:
            dy=curr[0]-self.window_size[1]
        else:
            dy=curr[0]
        if curr[1]+1>self.window_size[0]/2:
            dx=curr[1]-self.window_size[0]
        else:
            dx=curr[1]
        dy,dx=dy*self.cell_size,dx*self.cell_size
        x_c, y_c = self._center
        x_c+= dx
        y_c+= dy
        self._center = (np.floor(x_c), np.floor(y_c))

        # extract hog at the updated location
        new_x = self._crop(current_frame, self._center, (self.w, self.h))
        new_x = cv2.resize(new_x, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
        new_x= extract_hog_feature(new_x, cell_size=self.cell_size)

        new_xf = fft2(self._get_windowed(new_x, self._window))
        self.alphaf = self.interp_factor * self._training(new_xf, self.yf, kernel=self.kernel) + (1 - self.interp_factor) * self.alphaf
        self.xf = self.interp_factor * new_xf + (1 - self.interp_factor) * self.xf
        return [(self._center[0] - self.w / 2), (self._center[1] - self.h / 2), self.w, self.h]

    def _kernel_correlation(self, xf, yf, kernel='gaussian'):
        if kernel== 'gaussian':
            N=xf.shape[0]*xf.shape[1]
            xx=(np.dot(xf.flatten().conj().T,xf.flatten())/N)
            yy=(np.dot(yf.flatten().conj().T,yf.flatten())/N)
            xyf=xf*np.conj(yf)
            xy=np.sum(np.real(ifft2(xyf)),axis=2)
            kf = fft2(np.exp(-1 / self.sigma ** 2 * np.clip(xx+yy-2*xy,a_min=0,a_max=None) / np.size(xf)))
        elif kernel== 'linear':
            kf= np.sum(xf*np.conj(yf),axis=2)/np.size(xf)
        else:
            raise NotImplementedError
        return kf

    def _training(self, xf, yf, kernel='gaussian'):
        kf = self._kernel_correlation(xf, xf, kernel)
        alphaf = yf/(kf+self.lambda_)
        return alphaf

    def _detection(self, alphaf, xf, zf, kernel='gaussian'):
        kzf = self._kernel_correlation(zf, xf, kernel)
        responses = np.real(ifft2(alphaf * kzf))
        return responses

    def _crop(self,img,center,target_sz):
        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
        w,h=target_sz
        """
        # the same as matlab code
        w=int(np.floor((1+self.padding)*w))
        h=int(np.floor((1+self.padding)*h))
        xs=(np.floor(center[0])+np.arange(w)-np.floor(w/2)).astype(np.int64)
        ys=(np.floor(center[1])+np.arange(h)-np.floor(h/2)).astype(np.int64)
        xs[xs<0]=0
        ys[ys<0]=0
        xs[xs>=img.shape[1]]=img.shape[1]-1
        ys[ys>=img.shape[0]]=img.shape[0]-1
        cropped=img[ys,:][:,xs]
        """
        cropped=cv2.getRectSubPix(img,(int(np.floor((1+self.padding)*w)),int(np.floor((1+self.padding)*h))),center)
        return cropped

    def _get_windowed(self,img,cos_window):
        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
        windowed = cos_window[:,:,None] * img
        return windowed
