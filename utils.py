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

def intersectBool(box1, box2):
    intersect_tl_x = np.max(box1[0], box2[0])
    intersect_tl_y = np.max(box1[1], box1[1])
    intersect_br_x = np.min(box1[0] + box1[2], box2[0] + box2[2])
    intersect_br_y = np.min(box1[1] + box1[3], box2[1] + box2[3])
    intersect_w = intersect_br_x - intersect_tl_x
    intersect_w[intersect_w < 0] = 0
    intersect_h = intersect_br_y - intersect_tl_y
    intersect_h[intersect_h < 0] = 0
    bool_intersect = (intersect_w*intersect_h > 0)
    return bool_intersect

"""
To DO:
write function that computes TNR, TPR from gts, preds and iou_threshold and tracker score threshold
"""
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

def compute_tpr_tnr(gts, preds, score_thresh=0, iou_thresh=0.5):
    """
    input:
        gts is nx4 np array. if target absent at row i, sum(gts[i,:])=0
        preds is nx5 np array, first 4 columns is bbox, last column is score (PSR or APCE)
        score_thresh is the minimum PSR or APCE to make positive prediction
    output:
        tpr and tnr are both scalar
    """
    # count number of true positive
    # coutn number of true negative
    # initialize counts
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # NEED to complete
    tpr = 0
    tnr = 0
    return tpr, tnr


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

def calAUC(value_list):
    length=len(value_list)
    delta=1./(length-1)
    area=0.
    for i in range(1,length):
        area+=(delta*((value_list[i]+value_list[i-1])/2))
    return area

def get_img_list(img_dir):
    frame_list = []
    for frame in sorted(os.listdir(img_dir)):
        if os.path.splitext(frame)[1] == '.jpg':
            frame_list.append(os.path.join(img_dir, frame))
    return frame_list

def plot_precision(gts,preds,save_path):
    # x,y,w,h
    threshes,precisions=get_thresh_precision_pair(gts,preds)
    idx20 = [i for i, x in enumerate(threshes) if x == 20][0]
    plt.plot(threshes,precisions,label=str(precisions[idx20])[:5])
    plt.title('Precision Plots')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def get_thresh_precision_pair(gts,preds):
    # gts and preds are numpy ndarrays
    # N x 4 (where N is the number of entries of frame rectangle result)
    length=min(len(gts),len(preds))
    gts=gts[:length,:]
    preds=preds[:length,:]
    gt_centers_x = (gts[:, 0]+gts[:,2]/2)
    gt_centers_y = (gts[:, 1]+gts[:,3]/2)
    preds_centers_x = (preds[:, 0]+preds[:,2]/2)
    preds_centers_y = (preds[:, 1]+preds[:,3]/2)
    dists = np.sqrt((gt_centers_x - preds_centers_x) ** 2 + (gt_centers_y - preds_centers_y) ** 2)
    threshes = []
    precisions = []
    for thresh in np.linspace(0, 50, 101):
        true_len = len(np.where(dists < thresh)[0])
        precision = true_len / len(dists)
        threshes.append(thresh)
        precisions.append(precision)
    return threshes,precisions


def plot_success(gts,preds,save_path):
    threshes, successes=get_thresh_success_pair(gts, preds)
    plt.plot(threshes,successes,label=str(calAUC(successes))[:5])
    plt.title('Success Plot')
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def get_thresh_success_pair(gts, preds):
    # gts and preds are numpy ndarrays
    # N x 4 (where N is the number of entries of frame rectangle result)
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
    ious = intersect_areas / (gts[:, 2] * gts[:, 3] + preds[:, 2] * preds[:, 3] - intersect_areas)
    threshes = []
    successes = []
    for thresh in np.linspace(0, 1, 101):
        success_len = len(np.where(ious > thresh)[0])
        success = success_len / len(ious)
        threshes.append(thresh)
        successes.append(success)
    return threshes,successes


"""
The following utils function are from:

"""
def geometric_mean(*args):
    with np.errstate(divide='ignore'):
        # log(zero) leads to -inf
        # log(negative) leads to nan
        # log(nan) leads to nan
        # nan + anything is nan
        # -inf + (any finite value) is -inf
        # exp(-inf) is 0
        return np.exp(np.mean(np.log(args))).tolist()

def max_geometric_mean_line(x1, y1, x2, y2):
    # Obtained using Matlab symbolic toolbox.
    # >> syms x1 x2 y1 y2 th
    # >> x = (1-th)*x1 + th*x2
    # >> y = (1-th)*y1 + th*y2
    # >> f = x * y
    # >> coeffs(f, th)
    # [ x1*y1, - y1*(x1 - x2) - x1*(y1 - y2), (x1 - x2)*(y1 - y2)]
    a = (x1 - x2) * (y1 - y2)
    b = - y1 * (x1 - x2) - x1 * (y1 - y2)
    # Maximize the quadratic on [0, 1].
    # Check endpoints.
    candidates = [0.0, 1.0]
    if a < 0:
        # Concave quadratic. Check if peak is in bounds.
        th_star = -b / (2 * a)
        if 0 <= th_star <= 1:
            candidates.append(th_star)
    g = lambda x, y: math.sqrt(x * y)
    h = lambda th: g((1 - th) * x1 + th * x2, (1 - th) * y1 + th * y2)
    return max([h(th) for th in candidates])



if __name__ == '__main__':
    a = gaussian_shaped_labels(10, [5,5])
    print(a)
