import json
from tqdm import tqdm
import re
count = 0
data=json.load(open('dev_data.json', 'r', encoding='utf-8'))
for item in tqdm(data):
    for turn in item['content']:
        is_traing = True
        text_in_turn_list = []
        for key in list(turn.keys())[:2]:
            text_in_turn_list.append(turn[key])
        pos = []
        for triple in turn['info']['triples']:
            value = triple['value']
            sub_list = []
            for index,text in enumerate(text_in_turn_list):
                # print(item['id'],value,'/t',text)
                try:
                    sub_index = re.finditer(value, text)
                except Exception as e:
                    if_find = text.find(value)
                    if if_find != -1:
                        sub_index = [(if_find,if_find+len(value))]
                    else:
                        sub_index = []
                if isinstance(sub_index,list):
                    per_sub = [[i,index+1] for i in sub_index]
                    print(1111)
                else:
                    per_sub = [[i.span(),index+1] for i in sub_index]
                sub_list += per_sub
            if len(sub_list) != 1:
                is_traing = False
                count += 1
                break
            else:
                pos.append(sub_list[0])
        if is_traing ==True:
            turn['is_traing'] = 'true'
            for i,triple in enumerate(turn['info']['triples']):
                triple['pos'] = [pos[i][1],pos[i][0][0],pos[i][0][1]]
        else:
            turn['is_traing'] = 'false'
print(count)
#
#
json.dump(data, open('dev_datav1.1.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


            

