import argparse
import cv2
import glob
import numpy as np
import os
from parse_annotation import parseVOC

parse = argparse.ArgumentParser(description='Generate training data (cropped) for SqueezeCFNet')
parse.add_argument('-o', '--output_size', dest='output_size', default=200, type=int, help='crop output size')
parse.add_argument('-p', '--padding', dest='padding', default=2, type=float, help='crop padding size')
args = parse.parse_args()

def crop_region(image, bbox, out_sz, padding):
    """
    returns img patch for the training dataset
    it is resized and grayed, centered at the training object
    the target bbox should be in the format of [xmin, ymin, xmax, ymax]
    """
    target_pos = [(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2]
    target_sz = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])
    window_sz = target_sz * (1+padding)
    crop_bbox = cxy_wh_2_bbox(target_pos, window_sz)
    crop_bbox = [float(x) for x in crop_bbox]
    a = (out_sz-1) / (crop_bbox[2]-crop_bbox[0]) #output size/bboxw
    b = (out_sz-1) / (crop_bbox[3]-crop_bbox[1]) #output size/bboxh
    c = -a * crop_bbox[0]
    d = -b * crop_bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float32) #shift the image to center the target and resize
    #crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_REPLICATE)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    #crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_WRAP)
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return crop_gray

def cxy_wh_2_bbox(cxy, wh):
    return np.array([cxy[0] - wh[0] / 2, cxy[1] - wh[1] / 2, cxy[0] + wh[0] / 2, cxy[1] + wh[1] / 2])  # 0-index

def contour_properties(cnt):
    """
    return properties of a single contour
    """
    area= cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    circularity = (4*np.pi*area)/perimeter**2
    x1,y1,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(max(w,h))/min(w,h)
    rect_area = w*h
    extent = float(area)/rect_area
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = float(area)/hull_area
    cont_props = [aspect_ratio, circularity, extent, convexity]
    return cont_props

def blob_analysis(img, compute_mask=True):
    """
    returns blob properties and maksed img of the blob (subtracting the background)
    preferrably the img is a patch containing only one query blob.
    """
    _,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY) #Threshold
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #RETR_EXTERNAL flag means ouly return the most outer contour
    largest_contour = max(contours, key= cv2.contourArea)
    props = contour_properties(largest_contour)
    if compute_mask:
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [largest_contour],
                    0, (255, 255, 255), -1)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        return props, masked_img, mask
    else:
        return props


if __name__ == '__main__':
    print(args)
    """
    # generate image patches for mini dataset
    folder_list = glob.glob(r"Your_raw_data_dir/*/")
    dataset_root = "Your_dataset_root"
    out_sz = args.output_size
    padding = args.padding
    selected_species = ["solmissus", "lampocteis_cruentiventer", "mitrocoma",
    "cydippida", "calycophorae", "beroe","bathochordaeus", "atolla","aegina",
    "poeobius","prayidae"]
    for folder_path in folder_list:
        species_name = folder_path.split("/")[-2].lower()
        if species_name in selected_species:
            species_dir = os.path.join(dataset_root, species_name)
            if not os.path.exists(species_dir):
                os.makedirs(species_dir)
                print(species_dir)
            xml_paths = folder_path + "*.xml"
            i=1
            for xml_path in glob.glob(xml_paths):
                if i <= 100:
                    bb,img_path,name = parseVOC(xml_path)
                    uid = os.path.splitext(xml_path)[0].split("/")[-1]
                    img = cv2.imread(img_path)
                    crop_gray = crop_region(img,bb,out_sz,padding)
                    #props = blob_analysis(crop_gray, False)
                    #print(props)
                    crop_gray_path = os.path.join(species_dir, str(i).zfill(4) + '.jpg')
                    cv2.imwrite(crop_gray_path, crop_gray)
                    print(i)
                    i += 1
    """

    # generate image patches for regular sized FathomNet dataset
    folder_list = glob.glob(r"Your_raw_data_dir/*/") # Replace the directory
    dataset_root = "Your_dataset_root" # Replace the directory
    out_sz = args.output_size
    padding = args.padding
    selected_species = ["solmissus", "lampocteis_cruentiventer", "mitrocoma",
    "cydippida", "calycophorae", "beroe","bathochordaeus", "atolla","aegina",
    "poeobius","prayidae"]
    for folder_path in folder_list:
        species_name = folder_path.split("/")[-2].lower()
        if species_name in selected_species:
            species_dir = os.path.join(dataset_root, species_name)
            if not os.path.exists(species_dir):
                os.makedirs(species_dir)
                print(species_dir)
            xml_paths = folder_path + "*.xml"
            i=1
            for xml_path in glob.glob(xml_paths):
                bb,img_path,name = parseVOC(xml_path)
                if sum(bb) == 0:
                    continue
                uid = os.path.splitext(xml_path)[0].split("/")[-1]
                img = cv2.imread(img_path)
                if img is None:
                    print("failed to read image, skipping")
                    continue
                crop_gray = crop_region(img,bb,out_sz,padding)
                #props = blob_analysis(crop_gray, False)
                #print(props)
                crop_gray_path = os.path.join(species_dir, str(i).zfill(4) + '.jpg')
                cv2.imwrite(crop_gray_path, crop_gray)
                print(i)
                i += 1
