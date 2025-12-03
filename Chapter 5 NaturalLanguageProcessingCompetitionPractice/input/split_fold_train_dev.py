import json
import numpy as np
import pandas as pd
import random
random.seed(1994)
from sklearn.model_selection import KFold,StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
input_file1 = 'data_label_v1.2.json'
input_file2 = 'dev_datav1.1.json'


data1 = json.load(open(input_file1,'r', encoding='utf-8'))
data2 = json.load(open(input_file2,'r', encoding='utf-8'))
data_all = data1+data2
random.shuffle(data_all)
kf = KFold(n_splits=4,random_state=42,shuffle=True)
for n,(train_index, test_index) in enumerate(kf.split(data_all)):
    for i in test_index:
        data_all[i]['fold'] = n
json.dump(data_all, open('data_label_kfold_v1.1.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


#
#
# data = json.load(open(input_file,'r', encoding='utf-8'))
# id_type_num={"id":[]}
# type2id_path = '/home/user0731/yankuo/SereTOD/baseline0802/baseline/data/type2id.json'
# type2id = json.load(open(type2id_path))
# type_num = {label: ([0]*len(data)) for label, id in type2id.items() if label !="NA"}
# id_type_num.update(type_num)
#
# for i,item in enumerate(data):
#     id_type_num['id'].append(item['id'])
#     for turn in item["content"]:
#         for ent in turn["info"]["ents"]:
#             if ent['type'] == '5G套餐':
#                 continue
#             id_type_num[ent['type']][i] += 1
#
# df = pd.DataFrame(id_type_num)
# mskf = MultilabelStratifiedKFold(n_splits=4, shuffle=True, random_state=42)
# labels = [c for c in df.columns if c != "id"]
# df_labels = df[labels]
# df["kfold"] = -1
# for fold, (trn_, val_) in enumerate(mskf.split(df, df_labels)):
#     print(len(trn_), len(val_))
#     df.loc[val_, "kfold"] = fold
#
# for fold,per_data in zip(df.kfold.values,data):
#     per_data['fold'] = int(fold)
# json.dump(data, open('data_label_fold_group.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
