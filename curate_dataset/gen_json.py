import glob
import json
import numpy as np
import os


def create_FathomNet_json(dataset_root, dataset_name, val=0.1):
    selected_species = ["solmissus", "lampocteis_cruentiventer", "mitrocoma",
    "cydippida", "calycophorae", "beroe","bathochordaeus", "atolla","aegina",
    "poeobius","prayidae", "bathochordaeus_filter"]
    dataset = dict()
    folders_list = dataset_root + "/*/"
    for folder_path in glob.glob(folders_list):
        species_name = folder_path.split("/")[-2].lower()
        if species_name in selected_species:
            dataset[species_name] = dict()
            dataset[species_name]['train_set'] = []
            dataset[species_name]['val_set'] = []
            item_list = []
            for img_path in os.listdir(folder_path):
                img_full_path = os.path.join(folder_path, img_path)
                item_list.append(img_full_path)
            numTotal=len(item_list)
            numVal = int(numTotal*val)
            shuffle = np.random.choice(numTotal, numTotal, False) # shuffle the idx and split train and val
            train_ids = shuffle[:numTotal-numVal]
            for train_id in train_ids:
                dataset[species_name]['train_set'].append(item_list[train_id])
            val_ids = shuffle[numTotal-numVal:]
            for val_id in val_ids:
                dataset[species_name]['val_set'].append(item_list[val_id])
    json.dump(dataset, open('data_sample/'+dataset_name+'.json', 'w'), indent=2)
    return dataset

def create_Vid_json(dataset_root, dataset_name, val=0.1):
    dataset = dict()
    return dataset

if __name__ == '__main__':
    dataset_root = "/media/molly/MR_GRAY/DCNNCF_trainingset/FathomNet_wrap" # replace the directory
    dataset = create_FathomNet_json(dataset_root, "FathomNet_wrap")
