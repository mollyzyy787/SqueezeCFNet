import cv2
import numpy as np
import os
import sys
import glob

def saveSeq(vid_name, vid_path, save_root, src='left'):
    if src=='left':
        saveDir = os.path.join(save_root, vid_name+'_left')
    elif src=='right':
        saveDir = os.path.join(save_root, vid_name+'_right')
    else:
        saveDir = os.path.join(save_root, vid_name+'_mono')
    if not os.path.isdir(saveDir):
        os.mkdir(saveDir)

    # read the video file
    cap = cv2.VideoCapture(vid_path)
    # start the loop
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        cv2.imwrite(os.path.join(saveDir, str(count).zfill(6) + '.jpg'), frame)
        # increment the frame count
        count += 1
    return count


if __name__ == "__main__":
    def getPathDict(vidNameList, root, src='left'):
        vidDict = dict()
        if src=='left':
            for vid_name in vidNameList:
                vid_path = os.path.join(root, vid_name) + '_left.avi'
                vidDict[vid_name] = vid_path
        elif src=='right':
            for vid_name in vidNameList:
                vid_path = os.path.join(root, vid_name) + '_right.avi'
                vidDict[vid_name] = vid_path
        else:
            for vid_name in vidNameList:
                vid_path = glob.glob(os.path.join(root, vid_name) + '.*')[0]
                print(vid_path)
                vidDict[vid_name] = vid_path
        return vidDict

    MO_list = ['mo4', 'mo5', 'mo7', 'mo8','mo9', 'mo11', 'mo12', 'mo13', 'mo14',
                'mo15', 'mo16', 'mo19', 'mo20', 'mo21', 'mo22']
    MO_root = '/media/molly/MR_GRAY/LCM_dataset/stereo/MO'
    MO_vids = getPathDict(MO_list, MO_root)

    OOV_list = ['oov3', 'oov4']
    OOV_root = '/media/molly/MR_GRAY/LCM_dataset/stereo/OOV'
    OOV_vids = getPathDict(OOV_list, OOV_root)

    change_list = ['change4', 'change_mo1', 'change_mo2']
    change_root = '/media/molly/MR_GRAY/LCM_dataset/stereo/Change'
    change_vids = getPathDict(change_list, change_root)

    mono_list = ['mo_m4', 'mo_m5', 'mo_m6', 'mo_m8']
    mono_root = '/media/molly/MR_GRAY/LCM_dataset/mono'
    mono_vids = getPathDict(mono_list, mono_root, src='mono')

    #all_vids = MO_vids
    #all_vids.update(OOV_vids)
    #all_vids.update(change_vids)
    all_vids = mono_vids

    save_root = '/media/molly/MR_GRAY/DCNNCF_testset'

    total_frame_count = 0
    for vid_name, vid_path in all_vids.items():
        print("name: ", vid_name)
        print("     path: ", vid_path)
        vid_count = saveSeq(vid_name, vid_path, save_root, src='mono')
        print("     vid_coutn = ", vid_count)
        total_frame_count += vid_count

    print("total_frame_count: ", total_frame_count)
