# Sartorius - Cell Instance Segmentation
本项目是基于[tasj](https://github.com/tascj) 的项目[kaggle-sartorius-cell-instance-segmentation-solution
](https://github.com/tascj/kaggle-sartorius-cell-instance-segmentation-solution) 修改的

比赛地址：
https://www.kaggle.com/c/sartorius-cell-instance-segmentation

## 环境搭建

建立docker镜像
```
bash .dev_scripts/build.sh
```

设置环境变量

```
export DATA_DIR="/path/to/data"
export CODE_DIR="/path/to/this/repo"
```

开启docker容器
```
bash .dev_scripts/start.sh all
```

## 数据准备

1. 下载比赛数据集
2. 下载[LIVECell](https://github.com/sartorius-research/LIVECell) 数据集
3. 按如下格式解压全部数据

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

启动docker容器并执行如下指令

```
mkdir /data/checkpoints/
python tools/prepare_livecell.py
python tools/prepare_kaggle.py
```

结果应如下所示

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

## 训练

下载COCO预训练的YOLOX-x模型权重 https://github.com/Megvii-BaseDetection/YOLOX

转换权重格式

```
python tools/convert_official_yolox.py /path/to/yolox_x.pth /path/to/data/checkpoints/yolox_x_coco.pth
```

在docker容器中执行如下指令开始训练

```
# 使用LIVECell数据集预训练检测器
python tools/det/train.py configs/det/yolox_x_livecell.py

# 使用检测器在LIVECell验证集上推理
python tools/det/test.py configs/det/yolox_x_livecell.py work_dirs/yolox_x_livecell/epoch_30.pth --out work_dirs/yolox_x_livecell/val_preds.pkl --eval bbox

# 在比赛数据上微调检测器
python tools/det/train.py configs/det/yolox_x_kaggle.py --load-from work_dirs/yolox_x_livecell/epoch_15.pth

# 使用检测器在比赛数据验证集上推理
python tools/det/test.py configs/det/yolox_x_kaggle.py work_dirs/yolox_x_kaggle/epoch_30.pth --out work_dirs/yolox_x_kaggle/val_preds.pkl --eval bbox

# 使用LIVECell数据集预训练分割器
python tools/seg/train.py configs/seg/upernet_swin-t_livecell.py

# 在比赛数据上微调分割器
python tools/seg/train.py configs/seg/upernet_swin-t_kaggle.py --load-from work_dirs/upernet_swin-t_livecell/epoch_1.pth

# 预测比赛数据验证集的分割掩膜
python tools/seg/test.py configs/seg/upernet_swin-t_kaggle.py work_dirs/upernet_swin-t_kaggle/epoch_10.pth --out work_dirs/upernet_swin-t_kaggle/val_results.pkl --eval dummy
```
