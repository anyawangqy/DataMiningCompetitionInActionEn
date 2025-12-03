# SereTOD2022

https://github.com/SereTOD/SereTOD2022/tree/main/Track1

## Data download

1. The official training dataset (including labeled and unlabeled data), validation dataset, and test dataset are all available for download at the following links and are saved in the `input` folder.

```
https://www.kaggle.com/datasets/yankuoaaagmailcom/seretod2022
```

## Data preparation

1. Corrected Training Dataset
```
python input/correct_train.py
```
2. Validation Dataset Format Conversion
```
python input/dev_conver.py
```
3. Merging and Splitting the Training and Validation Datasets
```
python input/split_fold_train_dev.py
```

## Pretraining(mlm)

1. Pretraining Data Preparation
```
python lm/create_lm_data_v2.py
```
2. Start pretraining
```
#Use the DeBERTa model for pretraining.
python lm/deberta_base_lm_512_v2.py
#Use the roformer model for pretraining.
python lm/deberta_base_lm_512_v1.py


# When pretraining RoBERTa, MacBERT, or NEZHA models, modify the model class and output path in deberta_base_lm_512_v2.py.
During RoBERTa pretraining, set/configure the MODEL_DIR parameter:"hfl/chinese-roberta-wwm-ext";TrainConfig.output_dir:'../pretrain_model/roberta'
During MacBERT pretraining, set/configure the MODEL_DIR parameter:"hfl/chinese-macbert-base";TrainConfig.output_dir:'../pretrain_model/macbert'
During NEZHA   pretraining, set/configure the MODEL_DIR parameter:"sijunhe/nezha-base-wwm";TrainConfig.output_dir:'../pretrain_model/nezha'
```

The final pretrained models are all saved in the pretrain_model folder.
```
├── pretrain_model
│   ├── deberta
│   ├── roberta
│   ├── roformer
│   ├── macbert
│   └── nezha
```


## train

1.Entity extraction

```
# roformer、macbert、deberta、nezha models need to run the train_ee.py script，Before training, modify the pretrained model path.
(During roformer training, modify the parameter CFG.model:"../../pretrain_model/roformer";OUTPUT_DIR:'../../output_model/ee/roformer/'
During  deberta  training, modify the parameter CFG.model:"../../pretrain_model/deberta";OUTPUT_DIR:'../../output_model/ee/deberta/'
During  macbert  training, modify the parameter CFG.model: "../../pretrain_model/macbert";OUTPUT_DIR:'../../output_model/ee/macbert/'
During  nezha    training, modify the parameter CFG.model:"../../pretrain_model/nezha";OUTPUT_DIR:'../../output_model/ee/nezha/')
python train/entity-extraction/train_ee.py

# the roberta  model need to run the train_effcient.py script.
python entity-extraction/train_efficient.py
```

2.Entity coreference resolution

```
cd train/entity-coreference
CUDA_VISIBLE_DEVICES=$1 python main.py config.yaml 
```

3.Slot extraction

```
# roformer、macbert、nezha、roberta  models need to run the train_ef.py script，Before training, modify the pretrained model path.
(During roformer training, modify the parameter CFG.model:"../../pretrain_model/roformer";OUTPUT_DIR:'../../output_model/sf/roformer/'
During  roberta training, modify the parameter  CFG.model:"../../pretrain_model/roberta";OUTPUT_DIR:'../../output_model/sf/roberta/'
During  macbert training, modify the parameter  CFG.model:"../../pretrain_model/macbert";OUTPUT_DIR:'../../output_model/sf/macbert/'
During  nezha training, modify the parameter  CFG.model:"../../pretrain_model/nezha";OUTPUT_DIR:'../../output_model/sf/nezha/')
python train/slot-filling/train_ef.py
```

4.Entity-slot alignment

```
cd train/entity-slot-alignment
CUDA_VISIBLE_DEVICES=$1 python main.py config.yaml 
```

## Prediction Submission


```
cd submissions 
sh get_submissions.sh

```
