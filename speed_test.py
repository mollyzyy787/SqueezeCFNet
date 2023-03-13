import cv2
import os
import glob
import time
import torch
import numpy as np
from curate_dataset.parse_annotation import parseManualAnnotation
from models.squeezeCFnet_track import SqueezeCFNetTracker_light
from baseline.kcf import KCF_HOG
from models.DCFnet_track import DCFNetTracker
from utils import crop_chw, gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox, convert_format, PSR, APCE

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
    imSeq_dir = 'Your_imSeq_dir' # Replace with image sequence directory
    annotation_path = glob.glob(os.path.join(imSeq_dir, '*.json'))[0]
    annotation = parseManualAnnotation(annotation_path)
    for obj in annotation[0]:
        if obj["id"] == 0:
            init_rect = obj["bbox"]
    img0_path = os.path.join(imSeq_dir, str(0).zfill(6)+".jpg")
    img0 = cv2.imread(img0_path)
    SCF_tracker = SqueezeCFNetTracker_light(img0, init_rect, SqueezeCFnet_param_path, gpu=False)
    DCF_tracker = DCFNetTracker(img0, init_rect, DCFnet_param_path, gpu=False)
    hog_tracker = KCF_HOG()
    hog_tracker.init(img0, init_rect)
    SCF_fps = test_tracker_speed(imSeq_dir, SCF_tracker,'SCF')
    print("SCF: ", SCF_fps)
    DCF_fps = test_tracker_speed(imSeq_dir, DCF_tracker,'DCF')
    print("DCF: ", DCF_fps)
    hog_fps = test_tracker_speed(imSeq_dir, hog_tracker,'hog')
    print("hog: ", hog_fps)
