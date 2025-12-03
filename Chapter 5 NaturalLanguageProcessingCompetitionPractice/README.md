# SereTOD2022

https://github.com/SereTOD/SereTOD2022/tree/main/Track1

## Data download

1. 官方训练数据集（包括有标签数据和无标签数据）、验证数据集、测试数据集均通过以下链接下载，并保存于input文件夹下。
```
https://www.kaggle.com/datasets/yankuoaaagmailcom/seretod2022
```

## Data preparation

1. 纠正训练数据集
```
python input/correct_train.py
```
2. 验证数据集格式转化
```
python input/dev_conver.py
```
3. 训练数据集和验证数据集合并、分折
```
python input/split_fold_train_dev.py
```

## 预训练(mlm)

1. 预训练数据准备
```
python lm/create_lm_data_v2.py
```
2. 开始预训练
```
#使用DeBERTa模型做预训练
python lm/deberta_base_lm_512_v2.py
#使用roformer模型做预训练
python lm/deberta_base_lm_512_v1.py

# RoBERTa、MacBERT、NEZHA模型预训练需要修改deberta_base_lm_512_v2.py文件的模型类别及输出路径。
RoBERTa预训练过程修改参数：MODEL_DIR:"hfl/chinese-roberta-wwm-ext";TrainConfig.output_dir:'../pretrain_model/roberta'
MacBERT预训练过程修改参数：MODEL_DIR:"hfl/chinese-macbert-base";TrainConfig.output_dir:'../pretrain_model/macbert'
NEZHA预训练过程修改参数：MODEL_DIR:"sijunhe/nezha-base-wwm";TrainConfig.output_dir:'../pretrain_model/nezha'
```

最后预训练模型均保存于pretrain_model文件夹下。
```
├── pretrain_model
│   ├── deberta
│   ├── roberta
│   ├── roformer
│   ├── macbert
│   └── nezha
```


## 训练

1.实体抽取

```
# roformer、macbert、deberta、nezha模型需执行train_ee.py训练脚本，在训练前需修改预训练模型路径。
(roformer训练过程修改参数:CFG.model:"../../pretrain_model/roformer";OUTPUT_DIR:'../../output_model/ee/roformer/'
deberta训练过程修改参数:CFG.model:"../../pretrain_model/deberta";OUTPUT_DIR:'../../output_model/ee/deberta/'
macbert训练过程修改参数:CFG.model:"../../pretrain_model/macbert";OUTPUT_DIR:'../../output_model/ee/macbert/'
nezha训练过程修改参数:CFG.model:"../../pretrain_model/nezha";OUTPUT_DIR:'../../output_model/ee/nezha/')
python train/entity-extraction/train_ee.py

# roberta模型需执行train_effcient.py训练脚本。
python entity-extraction/train_efficient.py
```

2.实体共指解析

```
cd train/entity-coreference
CUDA_VISIBLE_DEVICES=$1 python main.py config.yaml 
```

3.槽位提取

```
# roformer、macbert、nezha、roberta模型需执行train_ef.py训练脚本，在训练前需修改预训练模型路径。
(roformer训练过程修改参数:CFG.model:"../../pretrain_model/roformer";OUTPUT_DIR:'../../output_model/sf/roformer/'
roberta训练过程修改参数:CFG.model:"../../pretrain_model/roberta";OUTPUT_DIR:'../../output_model/sf/roberta/'
macbert训练过程修改参数:CFG.model:"../../pretrain_model/macbert";OUTPUT_DIR:'../../output_model/sf/macbert/'
nezha训练过程修改参数:CFG.model:"../../pretrain_model/nezha";OUTPUT_DIR:'../../output_model/sf/nezha/')
python train/slot-filling/train_ef.py
```

4.实体槽位对齐

```
cd train/entity-slot-alignment
CUDA_VISIBLE_DEVICES=$1 python main.py config.yaml 
```

## 预测提交


```
cd submissions 
sh get_submissions.sh

```
