import glob
import os

"""
This script renames all img files in each class folder into consecutive integer in order
"""
def main():
    dataset_root = "/media/molly/MR_GRAY/DCNNCF_trainingset/FathomNet_mini"
    class_folders = dataset_root + "/*/"
    for folder_path in glob.glob(class_folders):
        imgs_path = folder_path + "*.jpg"
        i=1
        img_paths = sorted(glob.glob(imgs_path, recursive=True))
        for img_path in img_paths:
            correct_path = folder_path+str(i).zfill(4)+".jpg"
            if img_path != correct_path:
                os.rename(img_path, correct_path)
            i+=1


if __name__ == "__main__":
    main()
