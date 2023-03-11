# SqueezeCFNet
An architecture for learning the feature representations for target re-identification
in long-term DCF Tracking <br>

**Documentation to be updated

## Environment
See `requirement.txt`

## Models

#### KCF with HOG
`KCF_HOG()` in `baseline/kcf.py` is the only version, used in testing and demo
#### DCFNet
`DCFNet()` in `models/DCFnet.py` is for training, which takes (template, search) as input for the `forward` method <br>
`DCFNet()` in `models/DCFnet_track.py` is for tracking and re-id testing in `test.py`, which only takes (search) as input for the `forward` method, and updates the template in the `update` method. <br>
`DCFNetTracker()` in `models/DCFnet_track.py` can be used for continous tracking with the `track` method <br>
`DCFNetTracker_reIdTest()` in `test.py` is used for re-id test with special methods `runResponseAnalysis`, `runRotationAnalysis`<br>
#### SqueezeCFNet
`SqueezeCFNet()` in `models/squeezeCFnet.py` is for training, which takes (template, search, negative) as input for the `forward` method <br>
`SqueezeCFNet()` in `models/squeezeCFnet_track.py` is for tracking and re-id testing in `test.py`, which only takes (search) as input for the `forward` method, and updates the template in the `update` method. <br>
`SqueezeCFNet_light()` in `models/squeezeCFnet_track.py` is for tracking and speed testing in `speed_test.py`, which skips the encoding stage and only process the shallow part of the network in forward pass.
`SqueezeCFNetTracker()` in `models/track.py` can be used for continous tracking with the `track` method <br>
`SqueezeCFNetTracker_reIdTest()` in `test.py` is used for re-id test with special methods `runResponseAnalysis`, `runRotationAnalysis` <br>

##  Step 1: Curate dataset
Raw training and validation data are downloaded from [FathomNet](https://www.mbari.org/data/fathomnet/) using the [fathomnet-py](https://fathomnet-py.readthedocs.io/en/latest/) API. Examples of the raw FathomNet data are `curate_dataset/data_sample/FathomNet_sample.*` <br>
Then run `curate_dataset/gen_patch.py` to generate training and validation image patches. Replace `folder_list` directory to the root directory of raw FathomNet data, and `dataset_root` directory with a new directory for generated training and validation image patches. <br>
Then run `curate_dataset/gen_json.py` (Replace `dataset_root` with the directory of the generated image patches) to generate the dataset json file that links to all the image patches. Some examples of the json files are `curate_dataset/data_sample/FathomNet*.json`

## Step 2: Train
#### Train SqueezeCFNet
```console
$ python train.py --dataset <path-to-dataset *.json file> [options]
```
#### Train DCFNet
```console
$ python train_DCFNet.py --dataset <path-to-dataset *.json file> [options]
```

## Step 3: Test and Demo
### Testing on re-identification performance
```console
$ python test.py --seq-root <root directory to image sequence folders> --json-path <path to dataset *.json file> --test-mode <0:re-id on image sequence, 1:re-id on FathomNet training set, 2:re-id on transformation>
```
#### Test image sequence folder structure
The image sequences for testing need to be of the following structure. Each image sequence comes from a continous tracked video. The anntoation is done at every 50 frames using the [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/). An example of the json annotation file can be found at `curate_dataset/data_sample/annotation.json`. <br>
```
├── seq-root
│   ├── seq1
│   │   ├── *.jpg
│   │   ├── str(frame_number).zfill(6).jpg
│   │   ├── *.jpg
│   │   ├── annotation.json
│   ...
│   ├── seqN
│   │   ├── *.jpg
│   │   ├── str(frame_number).zfill(6).jpg
│   │   ├── *.jpg
│   │   ├── annotation.json
```
### Demo



### Acknowledgement
The KCF with HOG feature tracker in `baseline` referenced [pyTrackers](https://github.com/fengyang95/pyCFTrackers) under the MIT license. <br>
The DCFNet baseline in `models` referenced [DCFNet](https://github.com/foolwood/DCFNet_pytorch) under the MIT license.
