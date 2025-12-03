#!/usr/bin/env python
# coding: utf-8
from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
from torch.optim import lr_scheduler
import random
from roformer import RoFormerModel
import os
import numpy as np
import pandas as pd
import json
import torch.nn as nn
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

type2id_path = 'type2id.json'
type2id = json.load(open(type2id_path))
type2id = {label: idx for idx, label in enumerate(type2id.keys())}
id2type = {v: k for k, v in type2id.items()}


# CFG
class CFG:
    test_file = '../data/test_with_labels.json'
    apex = True
    num_workers = 8
    ner_maxlength = 20
    max_len = 256
    type2id = type2id
    id2type = id2type
    ENT_CLS_NUM = len(id2type)
    seed = 42


class CFG1:
    path = "../../output_model/ee/roformer"
    weight_names = ['ent_model0.pth', 'ent_model1.pth', 'ent_model2.pth', 'ent_model3.pth']
    # weight_names = ['ent_model3.pth']
    model = "../../pretrain_model/roformer"
    batch_size = 32
    max_len = 384
    tokenizer = None
    model_func = RoFormerModel


CFG1.tokenizer = AutoTokenizer.from_pretrained(CFG1.model)

class CFG2:
    path = "../../output_model/ee/macbert"
    weight_names = ['ent_model0.pth', 'ent_model1.pth', 'ent_model2.pth', 'ent_model3.pth']
    # weight_names = ['ent_model3.pth']
    model = "../../pretrain_model/macbert"
    batch_size = 32
    max_len = 256
    tokenizer = None
    model_func = AutoModel


CFG2.tokenizer = AutoTokenizer.from_pretrained(CFG2.model)

class CFG3:
    path = "../../output_model/ee/nezha"
    weight_names = ['ent_model0.pth', 'ent_model1.pth', 'ent_model2.pth', 'ent_model3.pth']
    # weight_names = ['ent_model3.pth']
    model = "../../pretrain_model/nezha"
    batch_size = 32
    max_len = 256
    tokenizer = None
    model_func = AutoModel


CFG3.tokenizer = AutoTokenizer.from_pretrained(CFG3.model)

class CFG4:
    path = "../../output_model/ee/roberta"
    weight_names = ['ent_model0.pth', 'ent_model1.pth', 'ent_model2.pth', 'ent_model3.pth']
    # weight_names = ['ent_model3.pth']
    model = "../../pretrain_model/roberta"
    batch_size = 32
    max_len = 256
    tokenizer = None
    model_func = AutoModel


CFG4.tokenizer = AutoTokenizer.from_pretrained(CFG4.model)

class CFG5:
    path = "../../output_model/ee/deberta"
    weight_names = ['ent_model0.pth', 'ent_model1.pth', 'ent_model2.pth', 'ent_model3.pth']
    # weight_names = ['ent_model3.pth']
    model = "../../pretrain_model/deberta"
    batch_size = 32
    max_len = 280
    tokenizer = None
    model_func = AutoModel


CFG5.tokenizer = AutoTokenizer.from_pretrained(CFG5.model)



# Utils
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)


# # Data Loading
def compute_offset_in_turn(ent, turn, text_in_turn, original_pos):
    speaker1 = list(turn.keys())[0]
    # speaker2 = list(turn.keys())[1]
    # start, end = None, None
    if original_pos[0] == 1:
        start, end = original_pos[1:]
    else:
        start = original_pos[1] + len(turn[speaker1])
        end = original_pos[2] + len(turn[speaker1])
    text1 = ("".join(text_in_turn[start:end])).replace("_", " ")
    text2 = ent["name"].replace("_", " ")
    assert text1 == text2
    while text_in_turn[start] == ' ' or text_in_turn[start] == '_':
        start += 1
    while text_in_turn[end - 1] == ' ' or text_in_turn[end - 1] == '_':
        end -= 1
    type_label = type2id[ent['type']]
    return start, end, type_label


def load_data(path):
    data = json.load(open(path))
    processed_data = []
    for item in tqdm(data, desc="Reading"):
        context_ = []  # 缓存所有的对话文本
        for turn in item["content"]:
            text_in_turn_list = []
            for key in list(turn.keys())[:2]:
                text_in_turn_list.append(turn[key])
            text_in_turn = list("".join(text_in_turn_list))
            context_ += text_in_turn

        for turn in item["content"]:
            text_in_turn_list = []
            for key in list(turn.keys())[:2]:
                text_in_turn_list.append(turn[key])
            text_in_turn = list("".join(text_in_turn_list))
            processed_data.append([text_in_turn])
            processed_data[-1].append(context_)

            for ent in turn["info"]["ents"]:
                if ent['type'] == '5G套餐':
                    continue
                for position in ent["pos"]:
                    offset = compute_offset_in_turn(ent, turn, text_in_turn, position)
                    processed_data[-1].append(offset)
    return processed_data


# Dataset
class EntDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        if self.is_train:
            text = item[0]
            text_context = item[1]
            text_mapping = self.tokenizer(text, max_length=self.max_len, truncation=True, is_split_into_words=True)
            word_ids = text_mapping.word_ids()
            token_length = len(word_ids) - 1
            text_all = text + text_context
            encoder_txt = self.tokenizer(text_all, max_length=self.max_len, truncation=True, is_split_into_words=True)
            input_ids = encoder_txt["input_ids"]
            token_type_ids = encoder_txt["token_type_ids"]
            attention_mask = encoder_txt["attention_mask"]

            return text, input_ids, token_type_ids, attention_mask, token_length, word_ids
        else:
            # TODO 测试
            pass

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate(self, examples):
        batch_input_ids, batch_attention_mask, batch_segment_ids, batch_token_length, batch_word_ids = \
            [], [], [], [], []
        for item in examples:
            raw_text, input_ids, token_type_ids, attention_mask, token_length, word_ids = self.encoder(item)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_token_length.append([1] * token_length)
            batch_word_ids.append(word_ids)
        batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segment_ids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attention_mask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()

        max_length = batch_input_ids.shape[1]
        batch_token_length = torch.tensor(self.sequence_padding(batch_token_length, length=max_length)).float()

        return batch_input_ids, batch_attention_mask, batch_segment_ids, batch_token_length, batch_word_ids

    def __getitem__(self, index):
        item = self.data[index]
        return item


# MODEL1
class GlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        self.device = None

        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids, token_length):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = token_length.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        return logits / self.inner_dim ** 0.5
#MODEL2
class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """

    def __init__(
            self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs, position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)
class EffiGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True,fp16=False):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super(EffiGlobalPointer, self).__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.RoPE = RoPE
        self.fp16=fp16
        if fp16:
            self.epsilon=65504
        else:
            self.epsilon =1e12

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(self.hidden_size,
                                 self.ent_type_size * 2)  # 原版的dense2是(inner_dim * 2, ent_type_size * 2)
        self.drop = nn.Dropout(p=0.2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.dropout5 = nn.Dropout(0.5)

    def sequence_masking(self, x, mask, value='-inf', axis=None):

        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -self.epsilon
            elif value == 'inf':
                value = self.epsilon
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * self.epsilon
        return logits

    def forward(self, input_ids, attention_mask, token_type_ids,token_lengh):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs.last_hidden_state
        # batch_size = last_hidden_state.size()[0]
        # seq_len = last_hidden_state.size()[1]
        # outputs = self.dense_1(last_hidden_state)
        out = self.drop(last_hidden_state)
        logits1 = self.dense_1(self.dropout1(out))
        logits2 = self.dense_1(self.dropout2(out))
        logits3 = self.dense_1(self.dropout3(out))
        logits4 = self.dense_1(self.dropout4(out))
        logits5 = self.dense_1(self.dropout5(out))
        outputs = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        qw, kw = outputs[..., ::2], outputs[..., 1::2]  # 从0,1开始间隔为2
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim ** 0.5
        bias = torch.einsum('bnh->bhn', self.dense_2(last_hidden_state)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # logits[:, None] 增加一个维度
        # print(token_lengh.size())
        # pad_mask = token_lengh.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = token_lengh.unsqueeze(1).unsqueeze(1).expand(batch_size, seq_len)
        # print(pad_mask.size(),attention_mask.size())
        # logits = self.add_mask_tril(logits, mask=attention_mask)
        logits = self.add_mask_tril(logits, mask=token_lengh)
        return logits


# Helper function
def get_pred(logits, batch_tag, batch_size, batch_word_ids, final_prob):
    # batch_size, ent_type_size, seq_len, seq_len
    pred_prob = logits.sigmoid()
    sample_num = pred_prob.shape[0]
    max_len = min(pred_prob.shape[2], CFG.max_len)
    start_index = batch_tag * batch_size
    for i in range(max_len):
        span = min(CFG.ner_maxlength, max_len - i)
        final_prob[start_index:start_index + sample_num, :, i, :span] += \
            pred_prob[:, :, i, i:i + span].cpu().detach().numpy()
    return final_prob


def inference_fn(test_loader, model, batch_size, device, final_prob):
    # single model
    model.eval()
    tk0 = tqdm(test_loader, total=len(test_loader))
    for batch_index, batch in enumerate(tk0):
        input_ids, attention_mask, segment_ids, token_length, batch_word_ids = batch
        input_ids, attention_mask, segment_ids, token_length = \
            input_ids.to(device), attention_mask.to(device), segment_ids.to(device), token_length.to(device)
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                logits = model(input_ids, attention_mask, segment_ids, token_length)
                final_prob = get_pred(logits, batch_index, batch_size, batch_word_ids, final_prob)
    return final_prob


# 初始化概率矩阵
data = json.load(open(CFG.test_file))
sample_num = 0
for item in tqdm(data, desc="Reading"):
    sample_num += len(item["content"])

cfg_all = [CFG1, CFG2, CFG3, CFG4, CFG5]

final_prob = np.zeros((sample_num, CFG.ENT_CLS_NUM, CFG.max_len, CFG.ner_maxlength), dtype=np.float16)

model_count = 0
for idx, cfg in enumerate(cfg_all):
    ner_evl = EntDataset(load_data(CFG.test_file), tokenizer=cfg.tokenizer, max_len=cfg.max_len)
    ner_loader_evl = DataLoader(
        ner_evl, batch_size=cfg.batch_size, collate_fn=ner_evl.collate, shuffle=False,
        num_workers=CFG.num_workers)
    for weight_names in cfg.weight_names:
        encoder = cfg.model_func.from_pretrained(cfg.model)
        if idx == 0:
            model = EffiGlobalPointer(encoder, CFG.ENT_CLS_NUM, 64, fp16=True).to(device)
        else:
            model = GlobalPointer(encoder, CFG.ENT_CLS_NUM, 64).to(device)
        state = torch.load(os.path.join(cfg.path, weight_names), map_location=device)
        model.load_state_dict(state)
        final_prob = inference_fn(ner_loader_evl, model, cfg.batch_size, device, final_prob)
        del encoder, model, state
        gc.collect()
        torch.cuda.empty_cache()
        model_count += 1

final_prob /= model_count

pred_token_result = []
for sample in final_prob:
    sample_pred = []
    if (sample > 0.5).sum == 0:
        pred_token_result.append(sample_pred)
    else:
        for label_id, start, end in zip(*np.where(sample > 0.5)):
            if start != 0:
                prob = sample[label_id][start][end]
                sample_pred.append((label_id, start, start + end, prob))
        sample_pred.sort(key=lambda x: (x[1], x[2]), reverse=False)
        pred_token_result.append(sample_pred)


def convert_pos(one_turn, position):
    speaker1 = list(turn.keys())[0]
    offset = None
    if position[0] >= len(one_turn[speaker1]):
        offset = [
            2, position[0]-len(one_turn[speaker1]), position[1]-len(one_turn[speaker1])
        ]
    else:
        offset = [
            1, *position
        ]
    return [offset]


# 获得预测entity
data = json.load(open(CFG.test_file))
idx = 0
for item in tqdm(data):
    for turn in item["content"]:
        ent_list = []
        text_in_turn_list = []
        for key in list(turn.keys())[:2]:
            text_in_turn_list.append(turn[key])
        text_in_turn = list("".join(text_in_turn_list))
        text_mapping = CFG1.tokenizer(text_in_turn, is_split_into_words=True)
        word_ids = text_mapping.word_ids()

        temp_pred = []
        for pred_sample in pred_token_result[idx]:
            start_char = word_ids[pred_sample[1]]
            end_char = word_ids[pred_sample[2]]
            type_ner = CFG.id2type[pred_sample[0]]
            prob = pred_sample[3]
            # if start_char<len(text_in_turn_list[0]) and end_char >=len(text_in_turn_list[0]):
            #     continue
            if start_char is not None and end_char is not None:
                if start_char < len(text_in_turn_list[0]) and end_char >= len(text_in_turn_list[0]):
                    continue
                if len(temp_pred) != 0 and start_char <= temp_pred[-1][1]:
                    if temp_pred[-1][-1] < prob:
                        temp_pred.pop()
                    else:
                        continue

                temp_pred.append([start_char, end_char, type_ner, prob])

        for start_char, end_char, type_ner, _ in temp_pred:

            ent_name = ''.join(text_in_turn[start_char:end_char + 1])
            ent_list.append(
                {
                    "pos": convert_pos(turn, [start_char, end_char + 1]),
                    "type": type_ner,
                    "name": ent_name
                }
            )

        turn["info"]["ents"] = ent_list
        idx += 1
assert idx == len(pred_token_result), print(idx, len(pred_token_result))
json.dump(data, open("../data/test_with_entity_mentions.json", "w"), indent=4, ensure_ascii=False)
