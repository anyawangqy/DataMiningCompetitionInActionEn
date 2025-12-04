# Sartorius - Cell Instance Segmentation

This project is based on the project kaggle-sartorius-cell-instance-segmentation-solution by tasj(https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution), with modifications.

Competition address:
https://www.kaggle.com/c/sartorius-cell-instance-segmentation

## Environment Setup
Build the docker image

```
bash .dev_scripts/build.sh
```

Set environment variables

```
export DATA_DIR="/path/to/data"
export CODE_DIR="/path/to/this/repo"
```


Start the docker container
```
bash .dev_scripts/start.sh all
```

## Data Preparation


1. Download the competition dataset.
2. Download the LIVECell dataset (https://github.com/sartorius-research/LIVECell) .
3. Extract all data in the following format:

```
├── LIVECell_dataset_2021
│   ├── images
│   ├── livecell_coco_train.json
│   ├── livecell_coco_val.json
│   └── livecell_coco_test.json
├── train
├── train_semi_supervised
└── train.csv
```

Start the docker container and execute the following commands:

```
mkdir /data/checkpoints/
python tools/prepare_livecell.py
python tools/prepare_kaggle.py
```


The result should look like the following:
```
├── LIVECell_dataset_2021
│   ├── images
│   ├── train_8class.json
│   ├── val_8class.json
│   ├── test_8class.json
│   ├── livecell_coco_train.json
│   ├── livecell_coco_val.json
│   └── livecell_coco_test.json
├── train
├── train_semi_supervised
├── checkpoints
├── train.csv
├── dtrainval.json
├── dtrain_g0.json
└── dval_g0.json
```

## Training


Download the COCO pre-trained YOLOX-x model weights from https://github.com/Megvii-BaseDetection/YOLOX.
Convert the weight format:

```
python tools/convert_official_yolox.py /path/to/yolox_x.pth /path/to/data/checkpoints/yolox_x_coco.pth
```

Execute the following commands in the docker container to start training:

```
# Pre-train the detector using the LIVECell dataset
python tools/det/train.py configs/det/yolox_x_livecell.py

# Perform inference on the LIVECell validation set using the detector
python tools/det/test.py configs/det/yolox_x_livecell.py work_dirs/yolox_x_livecell/epoch_30.pth --out work_dirs/yolox_x_livecell/val_preds.pkl --eval bbox

# Fine-tune the detector on the competition data
python tools/det/train.py configs/det/yolox_x_kaggle.py --load-from work_dirs/yolox_x_livecell/epoch_15.pth

# Perform inference on the competition data validation set using the detector
python tools/det/test.py configs/det/yolox_x_kaggle.py work_dirs/yolox_x_kaggle/epoch_30.pth --out work_dirs/yolox_x_kaggle/val_preds.pkl --eval bbox

# Pre-train the segmenter using the LIVECell dataset
python tools/seg/train.py configs/seg/upernet_swin-t_livecell.py

# Fine-tune the segmenter on the competition data
python tools/seg/train.py configs/seg/upernet_swin-t_kaggle.py --load-from work_dirs/upernet_swin-t_livecell/epoch_1.pth

# Predict segmentation masks for the competition data validation set
python tools/seg/test.py configs/seg/upernet_swin-t_kaggle.py work_dirs/upernet_swin-t_kaggle/epoch_10.pth --out work_dirs/upernet_swin-t_kaggle/val_results.pkl --eval dummy
```
