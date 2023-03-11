import cv2
import json
import os
import torch
from track import TrackerConfig,SqueezeCFNetTracker
from curate_dataset.parse_annotation import parseManualAnnotation
import glob
from test import SqueezeCFNetTracker_reIdTest, DCFNetTracker_reIdTest
from baseline.kcf import KCF_HOG
from utils import crop_chw, gaussian_shaped_labels, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox, convert_format, PSR, APCE

def processImSeq(img_folder, net_param_path, result_path, saveVid=False):
    """
    all img folder in testset includes image sequences and an annotation.json file
    which contains the info from manual labelling
    """
    annotation_path = glob.glob(os.path.join(img_folder, '*.json'))[0]
    annotation = parseManualAnnotation(annotation_path)
    # find the sequence name, to save the result in a folder under the same name
    seqName = img_folder.split('/')[-1]
    if not seqName:
        seqName = img_folder.split('/')[-2]
    tracker_result = dict()
    tracker_result_folder = os.path.join(result_path, seqName)
    if not os.path.exists(tracker_result_folder):
        os.makedirs(tracker_result_folder)
    tracker_result_file = os.path.join(tracker_result_folder, 'tracking_result.json')
    for obj in annotation[0]:
        if obj["id"] == 0:
            init_rect = obj["bbox"]
    for i, img_path in enumerate(sorted(glob.glob(os.path.join(img_folder, '*.jpg')))):
        img = cv2.imread(img_path)
        if i == 0:
            tracker=SqueezeCFNetTracker(img, init_rect, net_param_path)
            tracker_bbox = init_rect
            frame_height, frame_width, _ = img.shape
            if saveVid:
                vidFilePath = os.path.join(tracker_result_folder, "vid_demo.avi")
                vidFile = cv2.VideoWriter(vidFilePath,
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, (frame_width, frame_height))
        else:
            tracker_bbox = tracker.track(img)
            tracker_result[i] = tracker_bbox.tolist()
            print(i, ": ", tracker_bbox)
        if saveVid:
            x0,y0,bbw,bbh = tracker_bbox
            # Blue color in BGR
            color = (255, 0, 0)
            thickness = 2
            img = cv2.rectangle(img, (int(x0),int(y0)), (int(x0+bbw),int(y0+bbh)),
                                color, thickness)
            vidFile.write(img)
    vidFile.release()
    json.dump(tracker_result, open(tracker_result_file, 'w'), indent=2)

def analyzeImSeq(imSeq_dir, result_path, DCFNet_param_path, SqueezeCFnet_param_path):
    """
    all img folder in testset includes image sequences and an annotation.json file
    which contains the info from manual labelling
    """
    annotation_path = glob.glob(os.path.join(imSeq_dir, '*.json'))[0]
    annotation = parseManualAnnotation(annotation_path)
    # find the sequence name, to save the result in a folder under the same name
    seqName = imSeq_dir.split('/')[-1]
    if not seqName:
        seqName = imSeq_dir.split('/')[-2]
    tracker_result = dict()
    tracker_result_folder = os.path.join(result_path, seqName)
    if not os.path.exists(tracker_result_folder):
        os.makedirs(tracker_result_folder)
    for obj in annotation[0]:
        if obj["id"] == 0:
            init_rect = obj["bbox"]
    img0_path = os.path.join(imSeq_dir, str(0).zfill(6)+".jpg")
    img0 = cv2.imread(img0_path)
    SqueezeCFNet_tracker = SqueezeCFNetTracker_reIdTest(img0, init_rect, SqueezeCFnet_param_path)
    DCFNet_tracker = DCFNetTracker_reIdTest(img0, init_rect, DCFNet_param_path)
    hog_tracker = KCF_HOG()
    hog_tracker.init(img0,init_rect)
    thickness = 2
    color0 = (0,204,0)
    color1 = (51, 51, 255)
    hog_color = (255, 128,0)
    SCF_color = (153, 51, 255)
    DCF_color = (0,128,255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    for frame_number in sorted(annotation.keys()):
        frame_info = annotation[frame_number]
        img_path = os.path.join(imSeq_dir, str(frame_number).zfill(6)+".jpg")
        print(img_path)
        img = cv2.imread(img_path)
        img_out = img
        for region_info in frame_info:
            cand_cxy_wh = rect1_2_cxy_wh(region_info["bbox"])
            cand_pos = cand_cxy_wh[0]
            if region_info["id"] == 0:
                _, _, APCE_SCF,_ = SqueezeCFNet_tracker.runResponseAnalysis(img, cand_pos)
                _, _, APCE_DCF,_ = DCFNet_tracker.runResponseAnalysis(img, cand_pos)
                _, _, APCE_HOG,_ = hog_tracker.runResponseAnalysis(img, cand_pos)
                x0,y0,bbox_w,bbox_h = region_info["bbox"]
                start_point = (x0, y0)
                end_point=(x0+bbox_w, y0+bbox_h)
                img_out = cv2.rectangle(img_out, start_point, end_point, color0, thickness)
                img_out = cv2.putText(img_out, 'HoG: '+"{:.2f}".format(APCE_HOG), (x0,y0-10),font,
                            fontScale, hog_color, 1, cv2.LINE_AA)
                img_out = cv2.putText(img_out, 'SCF: '+"{:.2f}".format(APCE_SCF), (x0,y0-30),font,
                            fontScale, SCF_color, 1, cv2.LINE_AA)
                img_out = cv2.putText(img_out, 'DCF: '+"{:.2f}".format(APCE_DCF), (x0,y0-50),font,
                            fontScale, DCF_color, 1, cv2.LINE_AA)
            else:
                _, _, APCE_SCF,_ = SqueezeCFNet_tracker.runResponseAnalysis(img, cand_pos)
                _, _, APCE_DCF,_ = DCFNet_tracker.runResponseAnalysis(img, cand_pos)
                _, _, APCE_HOG,_ = hog_tracker.runResponseAnalysis(img, cand_pos)
                x0,y0,bbox_w,bbox_h = region_info["bbox"]
                start_point = (x0, y0)
                end_point=(x0+bbox_w, y0+bbox_h)
                img_out = cv2.rectangle(img_out, start_point, end_point, color1, thickness)
                img_out = cv2.putText(img_out, 'HoG: '+"{:.2f}".format(APCE_HOG), (x0,y0-10),font,
                            fontScale, hog_color, 1, cv2.LINE_AA)
                img_out = cv2.putText(img_out, 'SCF: '+"{:.2f}".format(APCE_SCF), (x0,y0-30),font,
                            fontScale, SCF_color, 1, cv2.LINE_AA)
                img_out = cv2.putText(img_out, 'DCF: '+"{:.2f}".format(APCE_DCF), (x0,y0-50),font,
                            fontScale, DCF_color, 1, cv2.LINE_AA)
        img_save_path = os.path.join(tracker_result_folder, str(frame_number).zfill(6)+".jpg")
        cv2.imwrite(img_save_path, img_out)
    return

if __name__ == '__main__':
    SqueezeCFnet_param_path = os.path.join(os.getcwd(), 'checkpoints', 'apce_enc_200ep_1e-4_best.pt')
    DCFnet_param_path = os.path.join(os.getcwd(), 'checkpoints', 'model_best_DCFnet_200ep.pt')
    result_path = 'Your_result_path'
    imSeq_dir = 'Your_imSeq_dir'
    """
    processImSeq('/media/molly/MR_GRAY/DCNNCF_testset/mo20_left', net_param_path,
                result_path, True)
    """
    analyzeImSeq(imSeq_dir, result_path, DCFnet_param_path, SqueezeCFnet_param_path)
