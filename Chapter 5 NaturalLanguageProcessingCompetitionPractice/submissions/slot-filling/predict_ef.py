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

role2id_path = 'role2id.json'
role2id = json.load(open(role2id_path))
role2id = {label: idx for idx, label in enumerate(role2id.keys())}
id2role = {v: k for k, v in role2id.items()}


# CFG
class CFG:
    test_file = '../data/test_with_entity_coref.json'
    apex = True
    num_workers = 8
    ner_maxlength = 60
    max_len = 256
    role2id = role2id
    id2role = id2role
    ENT_CLS_NUM = len(id2role)
    seed = 42


class CFG1:
    path = "../../output_model/sf/roformer"
    weight_names = ['ent_model0.pth', 'ent_model1.pth', 'ent_model2.pth', 'ent_model3.pth']
    # weight_names = ['ent_model3.pth']
    model = "../../pretrain_model/roformer"
    batch_size = 32
    max_len = 256
    tokenizer = None
    model_func = RoFormerModel


CFG1.tokenizer = AutoTokenizer.from_pretrained(CFG1.model)

class CFG2:
    path = "../../output_model/sf/roberta"
    weight_names = ['ent_model0.pth', 'ent_model1.pth', 'ent_model2.pth', 'ent_model3.pth']
    # weight_names = ['ent_model3.pth']
    model = "../../pretrain_model/roberta"
    batch_size = 32
    max_len = 256
    tokenizer = None
    model_func = AutoModel
#
#
CFG2.tokenizer = AutoTokenizer.from_pretrained(CFG2.model)
#
class CFG3:
    path = "../../output_model/sf/macbert"
    weight_names = ['ent_model0.pth', 'ent_model1.pth', 'ent_model2.pth', 'ent_model3.pth']
    # weight_names = ['ent_model3.pth']
    model = "../../pretrain_model/macbert"
    batch_size = 32
    max_len = 256
    tokenizer = None
    model_func = AutoModel


CFG3.tokenizer = AutoTokenizer.from_pretrained(CFG3.model)
#
class CFG4:
    path = "../../output_model/sf/nezha"
    weight_names = ['ent_model0.pth', 'ent_model1.pth', 'ent_model2.pth', 'ent_model3.pth']
    # weight_names = ['ent_model3.pth']
    model = "../../pretrain_model/nezha"
    batch_size = 32
    max_len = 256
    tokenizer = None
    model_func = AutoModel


CFG4.tokenizer = AutoTokenizer.from_pretrained(CFG4.model)

# Utils
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=CFG.seed)


# # Data Loading
def compute_offset_in_turn(triple, turn, text_in_turn, original_pos, item):
    speaker1 = list(turn.keys())[0]
    # speaker2 = list(turn.keys())[1]
    # start, end = None, None
    if original_pos[0] == 1:
        start, end = original_pos[1:]
    else:
        start = original_pos[1] + len(turn[speaker1])
        end = original_pos[2] + len(turn[speaker1])
    text1 = ("".join(text_in_turn[start:end])).replace("_", " ")
    text2 = triple["value"].replace("_", " ")
    assert text1 == text2
    if text_in_turn[start] == ' ' or text_in_turn[start] == '_':
        start += 1
    if text_in_turn[end - 1] == ' ' or text_in_turn[end - 1] == '_':
        end -= 1
    type_lable = role2id[triple['prop']]
    return (start, end, type_lable)


def load_data(path):
    D = []
    data = json.load(open(path))
    for item in tqdm(data, desc="Reading"):
        first_line = []  # 缓存所有的对话文本
        for turn in item["content"]:
            text_in_turn_list = []
            for key in list(turn.keys())[:2]:
                text_in_turn_list.append(turn[key])
            text_in_turn = list("".join(text_in_turn_list))
            first_line += text_in_turn

        for turn in item["content"]:
            text_in_turn_list = []
            for key in list(turn.keys())[:2]:
                text_in_turn_list.append(turn[key])
            text_in_turn = list("".join(text_in_turn_list))
            D.append([text_in_turn])
            D[-1].append(first_line)

            for triple in turn["info"]["triples"]:
                if triple['prop'] == '故障问题':
                    continue
                if 'pos' not in triple:
                    continue
                offset = compute_offset_in_turn(triple, turn, text_in_turn, triple["pos"], item)
                D[-1].append(offset)
    return D


# # Dataset

# ====================================================
# Dataset
# ====================================================
class EntDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, istrain=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.istrain = istrain

    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        if self.istrain:
            text = item[0]
            text_context = item[1]
            text_mapping = self.tokenizer(text, max_length=self.max_len, truncation=True, is_split_into_words=True)
            word_ids = text_mapping.word_ids()
            token_lengh = len(word_ids) - 1
            text_all = text + text_context
            encoder_txt = self.tokenizer(text_all, max_length=self.max_len, truncation=True, is_split_into_words=True)
            input_ids = encoder_txt["input_ids"]
            token_type_ids = encoder_txt["token_type_ids"]
            attention_mask = encoder_txt["attention_mask"]

            return text, input_ids, token_type_ids, attention_mask, token_lengh, word_ids
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
        batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids, batch_token_lengh = [], [], [], [], []
        batch_nomark_num = 0
        for item in examples:
            raw_text, input_ids, token_type_ids, attention_mask, token_lengh, word_ids = self.encoder(item)

            labels = np.zeros((len(role2id), self.max_len, self.max_len))
            for start, end, label in item[2:]:
                if start in word_ids and (end - 1) in word_ids:
                    start_index = word_ids.index(start)
                    end_index = len(word_ids) - 1 - list(reversed(word_ids)).index(end - 1)
                    labels[label, start_index, end_index] = 1
                else:
                    batch_nomark_num += 1
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
            batch_token_lengh.append([1] * token_lengh)
        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()

        max_lengh = batch_inputids.shape[1]
        batch_token_lengh = torch.tensor(self.sequence_padding(batch_token_lengh, length=max_lengh)).float()

        return batch_inputids, batch_attentionmask, batch_segmentids, batch_labels, batch_token_lengh, batch_nomark_num

    def __getitem__(self, index):
        item = self.data[index]
        return item


# # Model
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

    def forward(self, input_ids, attention_mask, token_type_ids, token_lengh):
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
        pad_mask = token_lengh.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        return logits / self.inner_dim ** 0.5


# # Helpler functions

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    # return (0.4*neg_loss + 0.6*pos_loss).mean()
    return (neg_loss + pos_loss).mean()


def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_evaluate_xyz(self, y_pred, y_true, batch_index):
        y_pred = y_pred.sigmoid()
        pred = []
        true = []
        y_pred_index = (y_pred > 0.5).nonzero()
        y_true_index = (y_true > 0).nonzero()

        for index_pred in y_pred_index:
            index_pred_list = index_pred.tolist()
            index_pred_list.append(batch_index)
            pred.append(tuple(index_pred_list))

        for index_true in y_true_index:
            index_true_list = index_true.tolist()
            index_true_list.append(batch_index)
            true.append(tuple(index_true_list))

        return pred, true


def train_fn(train_loader, model, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch + 1}", ncols=100)):
        input_ids, attention_mask, segment_ids, labels, token_lengh, _ = batch
        input_ids, attention_mask, segment_ids, labels, token_lengh = input_ids.to(device), attention_mask.to(device), \
                                                                      segment_ids.to(device), labels.to(
            device), token_lengh.to(device)
        batch_size = input_ids.shape[0]
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            logits = model(input_ids, attention_mask, segment_ids, token_lengh)
        loss = loss_fun(logits, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if CFG.batch_scheduler:
                scheduler.step()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  .format(epoch + 1, step, len(train_loader),
                          loss=losses,
                          grad_norm=grad_norm))
    return losses.avg


def valid_fn(valid_loader, model, metrics, device):
    model.eval()
    no_mark_num = 0
    pred_all = []
    true_all = []
    for step, batch in enumerate(tqdm(valid_loader, desc="Valing")):
        input_ids, attention_mask, segment_ids, labels, token_lengh, batch_nomark_num = batch
        input_ids, attention_mask, segment_ids, labels, token_lengh = input_ids.to(device), attention_mask.to(device), \
                                                                      segment_ids.to(device), labels.to(
            device), token_lengh.to(device)
        # batch_size = input_ids.shape[0]
        with torch.no_grad():
            logits = model(input_ids, attention_mask, segment_ids, token_lengh)
        pred, true = metrics.get_evaluate_xyz(logits, labels, step)
        pred_all += pred
        true_all += true
        no_mark_num += batch_nomark_num
    return pred_all, true_all, no_mark_num


# Helper function
def get_pred(logits, batch_tag, batch_size, final_prob):
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
        input_ids, attention_mask, segment_ids, _, token_length, _ = batch
        input_ids, attention_mask, segment_ids, token_length = \
            input_ids.to(device), attention_mask.to(device), segment_ids.to(device), token_length.to(device)
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                logits = model(input_ids, attention_mask, segment_ids, token_length)
                final_prob = get_pred(logits, batch_index, batch_size, final_prob)
    return final_prob


# 初始化概率矩阵
data = json.load(open(CFG.test_file))
sample_num = 0
for item in tqdm(data, desc="Reading"):
    sample_num += len(item["content"])

cfg_all = [CFG1,CFG2,CFG3,CFG4]

final_prob = np.zeros((sample_num, CFG.ENT_CLS_NUM, CFG.max_len, CFG.ner_maxlength), dtype=np.float16)

model_count = 0
for idx, cfg in enumerate(cfg_all):
    ner_evl = EntDataset(load_data(CFG.test_file), tokenizer=cfg.tokenizer, max_len=cfg.max_len)
    ner_loader_evl = DataLoader(
        ner_evl, batch_size=cfg.batch_size, collate_fn=ner_evl.collate, shuffle=False,
        num_workers=CFG.num_workers)
    for weight_names in cfg.weight_names:
        encoder = cfg.model_func.from_pretrained(cfg.model)
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
                sample_pred.append((label_id, start, start + end))
        pred_token_result.append(sample_pred)


def convert_pos(one_turn, position):
    speaker1 = list(turn.keys())[0]
    offset = None
    if position[0] >= len(one_turn[speaker1]):
        offset = [
            2, position[0] - len(one_turn[speaker1]), position[1] - len(one_turn[speaker1])
        ]
    else:
        offset = [
            1, *position
        ]
    return offset


# 获得预测entity
data = json.load(open(CFG.test_file))
idx = 0
for item in tqdm(data):
    for turn in item["content"]:
        prop_list = []
        text_in_turn_list = []
        for key in list(turn.keys())[:2]:
            text_in_turn_list.append(turn[key])
        text_in_turn = list("".join(text_in_turn_list))
        text_mapping = CFG1.tokenizer(text_in_turn, is_split_into_words=True)
        word_ids = text_mapping.word_ids()

        for pred_sample in pred_token_result[idx]:
            start_char = word_ids[pred_sample[1]]
            end_char = word_ids[pred_sample[2]]
            if start_char<len(text_in_turn_list[0]) and end_char >=len(text_in_turn_list[0]):
                continue
            type_prop = CFG.id2role[pred_sample[0]]
            value_name = ''.join(text_in_turn[start_char:end_char + 1])
            prop_list.append(
                {
                    "pos": convert_pos(turn, [start_char, end_char + 1]),
                    "prop": type_prop,
                    "value": value_name
                }
            )
        turn["info"]["triples"] = prop_list
        idx += 1
assert idx == len(pred_token_result), print(idx, len(pred_token_result))
json.dump(data, open("../data/test_with_triples.json", "w"), indent=4, ensure_ascii=False)
