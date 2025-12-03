import json
from tqdm import tqdm
data=json.load(open('data_label_v1.1.json', 'r', encoding='utf-8'))

for item in tqdm(data, desc="Parsing result"):
    for turn in item["content"]:
        text_in_turn_list = []
        for key in list(turn.keys())[:2]:
            text_in_turn_list.append(turn[key])

        for triple in turn["info"]["triples"]:
            if triple['prop'] == "欠费":
                triple["ent-id"]='NA'
                triple["ent-name"]='NA'
                triple['prop']='账户余额（欠费）'
            if triple['prop'] == "用户要求":
                triple["ent-id"]='NA'
                triple["ent-name"]='NA'
                triple['prop']='用户需求'
            if triple['prop'] == "剩余话费" or triple['prop'] == "话费余额":
                triple["ent-id"]='NA'
                triple["ent-name"]='NA'
                triple['prop']='账户余额'
            index = triple['pos'][0]
            start = triple['pos'][1]
            end = triple['pos'][2]
            value = triple['value']
            if end > len(text_in_turn_list[index-1]) or (value.replace("_",' ') != text_in_turn_list[index-1][start:end].replace("_",' ')):
                if index == 1:
                    triple['pos'] = [2,start,end]
                else:
                    triple['pos'] = [1, start, end]


json.dump(data, open('data_label_v1.2.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


            

