# Copyright 2022 SereTOD Challenge Organizers
# Authors: Hao Peng (peng-h21@mails.tsinghua.edu.cn)
# Apache 2.0
"""
'micro_f1': 0.8758727413101164
0.8765
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"
os.environ['TOKENIZERS_PARALLELISM'] = "True"
from pathlib import Path
import pdb
import sys
sys.path.append("../")
import json
import torch
import random 

import numpy as np

from tqdm import tqdm 
from torch.optim import Adam, AdamW
# from transformers import set_seed
from transformers.integrations import TensorBoardCallback
from transformers import EarlyStoppingCallback
from transformers import get_linear_schedule_with_warmup

from arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from backbone import get_backbone
from data_processor import (
    get_dataloader
)
from coref_model import get_model
from coref_metric import (
    compute_bcubed,
    get_pred_clusters
)
from coref_utils import (
    to_cuda, to_var
)
from dump_result import dump_result


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# argument parser
parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
    model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# set seed
set_seed(training_args)

# output dir
model_name_or_path = model_args.model_name_or_path.split("/")[-1]
output_dir = Path(
    os.path.join(os.path.join(os.path.join(training_args.output_dir, training_args.task_name), model_args.paradigm),
                 model_name_or_path))
# output_dir.mkdir(exist_ok=True, parents=True)
training_args.output_dir = str(output_dir)

# local rank
# training_args.local_rank = int(os.environ["LOCAL_RANK"])

# prepare labels
training_args.label_name = ["label_groups"]

# markers 
data_args.markers = ["<entity>", "</entity>"]

print(data_args, model_args, training_args)


# writter
earlystoppingCallBack = EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience,
                                              early_stopping_threshold=training_args.early_stopping_threshold)

# model 
backbone, tokenizer, config = get_backbone(model_args.model_type, model_args.model_name_or_path,
                                           model_args.model_name_or_path, data_args.markers,
                                           new_tokens=data_args.markers)
model = get_model(model_args, backbone)
model.cuda()

# dataloader
train_dataloader = get_dataloader(data_args, training_args, tokenizer, data_args.train_file, True)
eval_dataloader = get_dataloader(data_args, training_args, tokenizer, data_args.validation_file, False)


# optimizer
bert_optimizer = AdamW([p for p in model.encoder.model.parameters() if p.requires_grad], lr=training_args.learning_rate)
optimizer = Adam([p for p in model.scorer.parameters() if p.requires_grad], lr=1e-3)
num_training_steps = len(train_dataloader) * training_args.num_train_epochs
scheduler = get_linear_schedule_with_warmup(
    bert_optimizer, num_warmup_steps=int(num_training_steps*0.1), num_training_steps=num_training_steps
)

model_list = [
    '../../output_model/ec/best',
]


def inference_fn(model, dataloader):
    model.eval()
    all_probs = []
    all_example_ids = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            for k in data:
                if isinstance(data[k], torch.Tensor):
                    data[k] = to_cuda(data[k])
            outputs = model(**data)
            all_probs.extend([prob.detach().cpu() for prob in outputs["logits"]])
            all_example_ids.extend(data["doc_id"])
    return all_probs, all_example_ids


final_prob = None
final_all_example_ids = None
for model_path in model_list:
    model.load_state_dict(torch.load(model_path)['model'])
    test_dataloader = get_dataloader(
        data_args, training_args, tokenizer, data_args.test_file, shuffle=False,
        is_testing=True
    )
    all_probs, all_example_ids = inference_fn(model, test_dataloader)
    if not final_prob:
        final_prob = all_probs
        final_all_example_ids = all_example_ids
    else:
        for idx in range(len(all_probs)):
            final_prob[idx] += all_probs[idx]
        assert final_all_example_ids == all_example_ids

final_prob = [i / len(model_list) for i in final_prob]

pred_clusters = get_pred_clusters(final_prob)
assert len(pred_clusters) == len(final_all_example_ids)
dump_results = [
    {
        "doc_id": final_all_example_ids[i], "clusters": pred_clusters[i]
    } for i in range(len(pred_clusters))
]
json.dump(dump_results, open("output/pred_clusters.json", "w"), indent=4, ensure_ascii=False)

dump_result(data_args.test_file, dump_results)



