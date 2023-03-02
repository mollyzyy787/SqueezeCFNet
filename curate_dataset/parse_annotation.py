import cv2
import xml
import os
import xmltodict
import json

def parseVOC(xml_path):
    """
    parse annotation *.xml file in pascal-voc format
    """
    with open(xml_path) as file:
        file_data = file.read()
        dict_data = xmltodict.parse(file_data)
        annot = dict_data['annotation']
        img_path = annot['path']
        main_class = annot['folder'].lower()
        xmin = ymin = xmax = ymax = 0
        if isinstance(annot['object'],list):
            exact_match = False
            for obj in annot['object']:
                if obj['name'].lower()==main_class:
                    exact_match = True
                    xmin = int(obj['bndbox']['xmin'])
                    ymin = int(obj['bndbox']['ymin'])
                    xmax = int(obj['bndbox']['xmax'])
                    ymax = int(obj['bndbox']['ymax'])
            if not exact_match:
                for obj in annot['object']:
                    if main_class in obj['name'].lower():
                        xmin = int(obj['bndbox']['xmin'])
                        ymin = int(obj['bndbox']['ymin'])
                        xmax = int(obj['bndbox']['xmax'])
                        ymax = int(obj['bndbox']['ymax'])
                        print("use inexact match: ", obj['name'])
        else:
            xmin = int(annot['object']['bndbox']['xmin'])
            ymin = int(annot['object']['bndbox']['ymin'])
            xmax = int(annot['object']['bndbox']['xmax'])
            ymax = int(annot['object']['bndbox']['ymax'])
        bndbox = [xmin, ymin, xmax, ymax]
    return bndbox, img_path, main_class

def parseManualAnnotation(annot_path):
    dir = os.path.dirname(annot_path)
    print(dir)
    f = open(annot_path)
    annotation = json.load(f)
    output_annot = dict()
    for key in sorted(annotation.keys()):
        item = annotation[key]
        img_file_path = os.path.join(dir, item["filename"])
        frame_number = int(os.path.splitext(item["filename"])[0])
        frame_info = []
        for region in item["regions"]:
            x = region["shape_attributes"]["x"]
            y = region["shape_attributes"]["y"]
            w = region["shape_attributes"]["width"]
            h = region["shape_attributes"]["height"]
            region_info = dict()
            region_info["bbox"] = [int(x), int(y), int(w), int(h)]
            region_info["id"] = int(region["region_attributes"]["target_id"])
            frame_info.append(region_info)
        output_annot[frame_number] = frame_info
    return output_annot

if __name__ == '__main__':
    # parsing sample annotation tests
    xml_path = "/home/molly/DCNN_CF/data_sample/FathomNet_sample.xml"
    bb, path, name = parseVOC(xml_path)
    print(path)
    print(name)
    img = cv2.imread(path)
    xmin, ymin, xmax, ymax = bb
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0,0),4)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
