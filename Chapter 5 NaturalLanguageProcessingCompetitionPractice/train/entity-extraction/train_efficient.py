#!/usr/bin/env python
# coding: utf-8
from transformers import BertTokenizerFast, BertModel,AutoTokenizer,AutoModel
from torch.utils.data import DataLoader,Dataset
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
from efficient_model import EffiGlobalPointer
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_DIR = '../../input/'
OUTPUT_DIR = '../../output_model/roberta/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

type2id_path = INPUT_DIR +'type2id.json'
type2id = json.load(open(type2id_path))
type2id = {label: (id-1) for label, id in type2id.items() if label !="NA"}
id2type = {}
for k, v in type2id.items(): id2type[v] = k

# # CFG
class CFG:
    apex=True
    print_freq=100
    num_workers=8
    model="../../pretrain_model/roberta"
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=5
    learning_rate=2e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    train_batch_size=32
    eval_batch_sizie=32
    max_len=256
    ENT_CLS_NUM=9
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    n_fold=4
    trn_fold=[0,1,2,3]


# # tokenizer

# ====================================================
# tokenizer
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
CFG.tokenizer = tokenizer

# # Utils


def get_score(pred_all, true_all,no_mark_num,epoch):
    # 存储实体
    type_entity_posion = {}
    for i in id2type:
        type_entity_posion[i] = {'pred': [], 'true': []}
    #

    R = set(pred_all)
    T = set(true_all)
    total_common = len(R & T)
    total_pred = len(R)
    total_true = len(T)
    total_true += no_mark_num
    f1, precision, recall = 2 * total_common / (total_pred + total_true), total_common / (total_pred+(1e-8)), total_common / total_true
    LOGGER.info("EPOCH：{}\tEVAL_F1:{:.4f}\tPrecision:{:.4f}\tRecall:{:.4f}\t".format(epoch+1, f1, precision, recall))
    LOGGER.info('未标记数量:{},total_common:{},total_pred:{},total_true:{}'.format(no_mark_num, total_common, total_pred,total_true))

    # 打印每个类别f1
    for pred_value in pred_all:
        type_entity_posion[pred_value[1]]['pred'].append(pred_value)
    for true_value in true_all:
        type_entity_posion[true_value[1]]['true'].append(true_value)

    for key, value in type_entity_posion.items():
        R = set(value['pred'])
        T = set(value['true'])
        per_common = len(R & T)
        per_pred = len(R)
        per_true = len(T)
        per_f1, per_precision, per_recall = 2 * per_common / (per_pred + per_true), \
                                            per_common / (per_pred + (1e-8)), per_common / per_true
        # LOGGER.info(id2type[key], "\tF1:{}\tP:{}\tR:{}\t".format(per_f1, per_precision, per_recall))
        LOGGER.info("{}\tF1:{:.3f}".format(id2type[key],per_f1))
    return total_common,total_pred,total_true,f1

def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger
LOGGER = get_logger()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed=42)


# # Data Loading
def compute_offset_in_turn(ent,turn,text_in_turn,original_pos):
    speaker1 = list(turn.keys())[0]
    # speaker2 = list(turn.keys())[1]
    # start, end = None, None
    if original_pos[0] == 1:
        start, end = original_pos[1:]
    else:
        start = original_pos[1] + len(turn[speaker1])
        end = original_pos[2] + len(turn[speaker1])
    text1 = ("".join(text_in_turn[start:end])).replace("_"," ")
    text2 = ent["name"].replace("_"," ")
    assert text1 == text2
    if text_in_turn[start] ==' ' or text_in_turn[start] =='_':
        start += 1
    if text_in_turn[end-1] ==' ' or text_in_turn[end-1] =='_':
        end -= 1
    type_lable = type2id[ent['type']]
    return (start,end,type_lable)

def load_data(data,is_traing=True):
    D = []
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

            for ent in turn["info"]["ents"]:
                if ent['type'] == '5G套餐':
                    continue
                for position in ent["pos"]:
                    offset = compute_offset_in_turn(ent,turn,text_in_turn,position)
                    D[-1].append(offset)
    return D



# # Dataset

# ====================================================
# Dataset
# ====================================================
class EntDataset(Dataset):
    def __init__(self, data, tokenizer,max_len,istrain=True):
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
            text_mapping = self.tokenizer(text,max_length=self.max_len, truncation=True,is_split_into_words=True)
            word_ids = text_mapping.word_ids()
            token_lengh = len(word_ids)-1
            text_all = text+text_context
            encoder_txt = self.tokenizer(text_all, max_length=self.max_len, truncation=True,is_split_into_words=True)
            input_ids = encoder_txt["input_ids"]
            token_type_ids = encoder_txt["token_type_ids"]
            attention_mask = encoder_txt["attention_mask"]

            return text, input_ids, token_type_ids, attention_mask,token_lengh,word_ids
        else:
            #TODO 测试
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
        batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids,batch_token_lengh = [], [], [], [],[]
        batch_nomark_num = 0
        for item in examples:
            raw_text, input_ids, token_type_ids, attention_mask,token_lengh,word_ids = self.encoder(item)

            labels = np.zeros((len(type2id), self.max_len, self.max_len))
            for start, end, label in item[2:]:
                if start in word_ids and (end-1) in word_ids:
                    start_index = word_ids.index(start)
                    end_index = len(word_ids)-1-list(reversed(word_ids)).index(end-1)
                    labels[label, start_index, end_index] = 1
                else:
                    batch_nomark_num+=1
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
            batch_token_lengh.append([1]*token_lengh)
        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()

        max_lengh = batch_inputids.shape[1]
        batch_token_lengh = torch.tensor(self.sequence_padding(batch_token_lengh,length=max_lengh)).float()




        return batch_inputids, batch_attentionmask, batch_segmentids, batch_labels,batch_token_lengh,batch_nomark_num

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
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (0.4*neg_loss + 0.6*pos_loss).mean()
    # return (neg_loss + pos_loss).mean()

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
    def get_evaluate_xyz(self, y_pred,y_true,batch_index):
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

        return pred,true


def train_fn(train_loader, model, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    for step, batch in enumerate(tqdm(train_loader,desc=f"Training epoch {epoch+1}",ncols=100)):
        input_ids, attention_mask, segment_ids, labels, token_lengh, _ = batch
        input_ids, attention_mask, segment_ids, labels, token_lengh = input_ids.to(device), attention_mask.to(device),\
                                                                      segment_ids.to(device), labels.to(device), token_lengh.to(device)
        batch_size = input_ids.shape[0]
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            logits = model(input_ids, attention_mask, segment_ids,token_lengh)
        loss = loss_fun(logits, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(),batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if CFG.batch_scheduler:
                scheduler.step()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  .format(epoch+1, step, len(train_loader),
                          loss=losses,
                          grad_norm=grad_norm))
    return losses.avg


def valid_fn(valid_loader,model,metrics,device):
    model.eval()
    no_mark_num = 0
    pred_all = []
    true_all = []
    for step, batch in enumerate(tqdm(valid_loader,desc="Valing")):
        input_ids, attention_mask, segment_ids, labels, token_lengh, batch_nomark_num = batch
        input_ids, attention_mask, segment_ids, labels, token_lengh = input_ids.to(device), attention_mask.to(device), \
                                                                      segment_ids.to(device), labels.to(device), token_lengh.to(device)
        # batch_size = input_ids.shape[0]
        with torch.no_grad():
            logits = model(input_ids, attention_mask, segment_ids,token_lengh)
        pred, true = metrics.get_evaluate_xyz(logits, labels, step)
        pred_all += pred
        true_all += true
        no_mark_num += batch_nomark_num
    return pred_all, true_all,no_mark_num

# ====================================================
# train loop
# ====================================================
def train_loop(folds,fold):
    
    LOGGER.info(f"========== fold: {fold} training ==========")
    # ====================================================
    # loader
    # ====================================================
    train_folds = []
    valid_folds = []
    for value in folds:
        if value['fold'] != fold:
            train_folds.append(value)
        else:
            valid_folds.append(value)
    ner_train = EntDataset(load_data(train_folds), tokenizer=CFG.tokenizer, max_len=CFG.max_len)
    ner_loader_train = DataLoader(ner_train, batch_size=CFG.train_batch_size, collate_fn=ner_train.collate, shuffle=True,num_workers=CFG.num_workers)
    ner_evl = EntDataset(load_data(valid_folds), tokenizer=tokenizer,max_len=CFG.max_len)
    ner_loader_evl = DataLoader(ner_evl, batch_size=CFG.eval_batch_sizie, collate_fn=ner_evl.collate, shuffle=False,num_workers=CFG.num_workers)

    # ====================================================
    #  GP MODEL & optimizer & scheduler
    # ====================================================
    if 'roformer' in CFG.model:
        encoder = RoFormerModel.from_pretrained(CFG.model)
    else:
        encoder = AutoModel.from_pretrained(CFG.model)
    # model = GlobalPointer(encoder, CFG.ENT_CLS_NUM, 64).to(device)  # 9个实体类型
    model = EffiGlobalPointer(encoder, CFG.ENT_CLS_NUM, 64, fp16=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
    T_max = 500
    min_lr = 1e-6
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr)
    # ====================================================
    # loop
    # ====================================================
    metrics = MetricsCalculator()
    max_f_val = 0.0

    for epoch in range(CFG.epochs):
        # train
        avg_loss = train_fn(ner_loader_train, model,optimizer,epoch,scheduler,device)
        LOGGER.info(f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}')
        # eval
        pred_all,true_all,no_mark_num = valid_fn(ner_loader_evl,model,metrics,device)
        # scoring
        total_common,total_pred,total_true,score = get_score(pred_all, true_all,no_mark_num,epoch)

        if max_f_val < score:
            max_f_val = score
            span_num = (total_common,total_pred,total_true)
            # LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save(model.state_dict(),OUTPUT_DIR+f"ent_model{fold}.pth")
    torch.cuda.empty_cache()
    gc.collect()
    
    return span_num,max_f_val

if __name__ == '__main__':
    
    def get_result(final_result):
        pred_num,true_num,common_num = 0,0,0
        for value in final_result:
            pred_num += value[1]
            true_num += value[2]
            common_num += value[0]
            f1, precision, recall = 2 * common_num / (pred_num + true_num), common_num / (pred_num+(1e-8)), common_num / true_num
        LOGGER.info("平均分数\tEVAL_F1:{:.4f}\tPrecision:{:.4f}\tRecall:{:.4f}\t".format(f1, precision, recall))

    train = json.load(open(INPUT_DIR+'data_label_fold_id.json'))
    final_result = []
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            span_num,max_f_val = train_loop(train,fold)
            LOGGER.info(f"========== fold: {fold} result ==========")
            LOGGER.info(f'Score: {max_f_val:.4f}')
            LOGGER.info(f'\n')
            final_result.append(span_num)
    LOGGER.info(f"========== CV ==========")
    get_result(final_result)

