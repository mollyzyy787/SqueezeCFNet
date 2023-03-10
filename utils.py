import numpy as np
import cv2
import torchvision.transforms as Transform
import torch
import matplotlib.pyplot as plt

def cxy_wh_2_rect1(pos, sz):
    return np.array([pos[0]-sz[0]/2+1, pos[1]-sz[1]/2+1, sz[0], sz[1]])  # 1-index


def rect1_2_cxy_wh(rect):
    """
    Input is rectangle of [xmin, ymin, width, height] format
    Output is [cx, cy], [width, height]
    """
    return np.array([rect[0]+rect[2]/2-1, rect[1]+rect[3]/2-1]), np.array([rect[2], rect[3]])  # 0-index


def cxy_wh_2_bbox(cxy, wh):
    """
    Output format is [xmin, ymin, xmax, ymax]
    """
    return np.array([cxy[0]-wh[0]/2, cxy[1]-wh[1]/2, cxy[0]+wh[0]/2, cxy[1]+wh[1]/2])  # 0-index


def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g


def crop_chw(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float32)
    #crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_WRAP)
    return np.transpose(crop, (2, 0, 1))


def convert_format(img, normalize=False, mean=0., std=1.):
    """
    converts numpy image to torch tensor, normalized if needed
    """
    img = img.astype(np.float32)
    img_tensor = torch.from_numpy(img)
    img_gray = Transform.Grayscale()(img_tensor)
    if normalize:
        img_gray = Transform.Normalize(mean=mean, std=std)(img_gray)
    return img_gray #(1,1,H,W)


def compute_ious(gts, preds_bb):
    """
        gts and preds_bb both nx4 numpy array
        ious is nx1 (one value per row)
        NEED TESTING
    """
    length=min(len(gts),len(preds))
    gts=gts[:length,:]
    preds=preds[:length,:]
    intersect_tl_x = np.max((gts[:, 0], preds[:, 0]), axis=0)
    intersect_tl_y = np.max((gts[:, 1], preds[:, 1]), axis=0)
    intersect_br_x = np.min((gts[:, 0] + gts[:, 2], preds[:, 0] + preds[:, 2]), axis=0)
    intersect_br_y = np.min((gts[:, 1] + gts[:, 3], preds[:, 1] + preds[:, 3]), axis=0)
    intersect_w = intersect_br_x - intersect_tl_x
    intersect_w[intersect_w < 0] = 0
    intersect_h = intersect_br_y - intersect_tl_y
    intersect_h[intersect_h < 0] = 0
    intersect_areas = intersect_h * intersect_w
    union_areas = gts[:, 2] * gts[:, 3] + preds[:, 2] * preds[:, 3] - intersect_areas
    present_idx = np.where(union_areas!=0)
    ious[np.where(union_areas==0)] = -1 #exclude the frames where target is absent
    ious[present_idx] = intersect_areas[present_idx]/union_areas[present_idx]
    return ious

def unravel_index(
    indices: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = torch.div(indices, dim, rounding_mode='floor')

    return coord.flip(-1)


"""
the following utils functions are from:
    https://github.com/fengyang95/pyCFTrackers
"""

def APCE(response_map):
    Fmax=np.max(response_map)
    Fmin=np.min(response_map)
    apce=(Fmax-Fmin)**2/(np.mean((response_map-Fmin)**2))
    return apce

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

def to_color_map(score,sz):
    score = cv2.resize(score, sz)
    score -= score.min()
    score = score / score.max()
    score = (score * 255).astype(np.uint8)
    # score = 255 - score
    score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
    return score

def get_img_list(img_dir):
    frame_list = []
    for frame in sorted(os.listdir(img_dir)):
        if os.path.splitext(frame)[1] == '.jpg':
            frame_list.append(os.path.join(img_dir, frame))
    return frame_list


if __name__ == '__main__':
    a = gaussian_shaped_labels(10, [5,5])
    print(a)
