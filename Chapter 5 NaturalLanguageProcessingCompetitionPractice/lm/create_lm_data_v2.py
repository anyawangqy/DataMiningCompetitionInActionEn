import numpy as np
import pandas as pd
import re
import gc
import json
print('Start to create language mask dataset...')
with open('../input/data_label_v1.1.json',mode='r',encoding='utf-8') as fin:
    df=json.load(fin)
with open('../input/data_unlabel.json',mode='r',encoding='utf-8') as fin:
    df2=json.load(fin)
with open('../input/test_data1 for track1.json',mode='r',encoding='utf-8') as fin:
    df3=json.load(fin)
with open('../input/dev_data.json',mode='r',encoding='utf-8') as fin:
    df4=json.load(fin)
# df.extend(df2)
print(f'label size:{len(df)}')
print(f'unlabel size:{len(df2)}')
print(f'test unlabel size:{len(df3)}')
print(f'test dev size:{len(df4)}')
all_texts = []
for line in df:
    str_list = []
    for i, content in enumerate(line['content']):
        # print(f'content:{i}')
        for k in content.keys():
            # print(f'key={k}')
            if k == 'info':
                child = content[k]
                for m in child:
                    # print(m)
                    m = m.replace('ents', ' ')
                    m = m.replace('triples', ' ')
                    str_list.append(m)
            else:
                str_list.append(content[k])
    # print(str_list)
    all_texts.append(' '.join(str_list))
    # break

for line in df2:
    # print(type(line))
    str_list = []
    for i, content in enumerate(line):
        for k in content.keys():
            str_list.append(content[k])
    all_texts.append(' '.join(str_list))


for line in df3:
    str_list = []
    for i, content in enumerate(line['content']):
        # print(f'content:{i}')
        for k in content.keys():
            # print(f'key={k}')
            if k == 'info':
                child = content[k]
                for m in child:
                    # print(m)
                    m = m.replace('ents', ' ')
                    m = m.replace('triples', ' ')
                    str_list.append(m)
            else:
                str_list.append(content[k])
    # print(str_list)
    all_texts.append(' '.join(str_list))

for line in df4:
    str_list = []
    for i, content in enumerate(line['content']):
        # print(f'content:{i}')
        for k in content.keys():
            # print(f'key={k}')
            if k == 'info':
                child = content[k]
                for m in child:
                    # print(m)
                    m = m.replace('ents', ' ')
                    m = m.replace('triples', ' ')
                    str_list.append(m)
            else:
                str_list.append(content[k])
    # print(str_list)
    all_texts.append(' '.join(str_list))

train_df=pd.DataFrame(all_texts,columns=['text'])

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext=cleantext.replace('\n',' ')
    return cleantext

train_df['text']=train_df['text'].apply(cleanhtml)



del df,df2,df3
gc.collect()
train_df.to_csv('df_lm_v2.csv',index=False)
print('Done!')