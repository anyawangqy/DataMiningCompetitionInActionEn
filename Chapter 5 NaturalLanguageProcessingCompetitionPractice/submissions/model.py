# Copyright 2022 SereTOD Challenge Organizers
# Authors: Hao Peng (peng-h21@mails.tsinghua.edu.cn)
# Apache 2.0
import pdb
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from crf import CRF

class MyFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, weight=None, size_average=True,ignore_index=-100,device='cuda'):
        super(MyFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.size_average = size_average
        self.elipson = 0.000001
        self.ignore_index=ignore_index
        self.device=device

    def forward(self, input, target):
        """
        cal culates loss
        logits: [batch_size* seq_length , labels_length ]
        labels: [batch_size * seq_length,]
        """

        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

def get_model(model_args, backbone):
    if model_args.paradigm == "token_classification":
        return ModelForTokenClassification(model_args, backbone)
    elif model_args.paradigm == "sequence_labeling":
        return ModelForSequenceLabeling(model_args, backbone)
    else:
        raise ValueError("No such paradigm")


def select_cls(hidden_states: torch.Tensor) -> torch.Tensor:
    """Select CLS token as for textual information.
    
    Args:
        hidden_states: Hidden states encoded by backbone. shape: [batch_size, max_seq_length, hidden_size]

    Returns:
        hidden_state: Aggregated information. shape: [batch_size, hidden_size]
    """
    return hidden_states[:, 0, :]


class ClassificationHead(nn.Module):
    def __init__(self, config):
        super(ClassificationHead, self).__init__()
        scale = 2 if config.aggregation=="dm" else 1
        self.classifier = nn.Linear(config.hidden_size*scale, config.num_labels)
        self.config=config
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.dropout5 = nn.Dropout(0.5)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Classify hidden_state to label distribution.
        
        Args:
            hidden_state: Aggregated textual information. shape: [batch_size, ..., hidden_size]
        
        Returns:
            logits: Raw, unnormalized scores for each label. shape: [batch_size, ..., num_labels]
        """
        # logits = self.classifier(hidden_state)
        logits1 = self.classifier(self.dropout1(hidden_state))
        logits2 = self.classifier(self.dropout2(hidden_state))
        logits3 = self.classifier(self.dropout3(hidden_state))
        logits4 = self.classifier(self.dropout4(hidden_state))
        logits5 = self.classifier(self.dropout5(hidden_state))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        return logits

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class ModelForTokenClassification(nn.Module):
    """Bert model for token classification."""

    def __init__(self, config, backbone):
        super(ModelForTokenClassification, self).__init__()
        self.backbone = backbone 
        # self.aggregation = DynamicPooling(config)
        self.aggregation = select_cls
        self.cls_head = ClassificationHead(config)
        # self.pooler = MeanPooling()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
        ) -> Dict[str, torch.Tensor]:
        # backbone encode 
        outputs = self.backbone(input_ids=input_ids, \
                                attention_mask=attention_mask, \
                                token_type_ids=token_type_ids,
                                return_dict=True)
        hidden_states = outputs.last_hidden_state
        # hidden_states = self.pooler(outputs.last_hidden_state, attention_mask)
        # aggregation 
        # hidden_state = self.aggregation.select_cls(hidden_states)
        hidden_state = self.aggregation(hidden_states)
        # classification
        logits = self.cls_head(hidden_state)
        # compute loss 
        loss = None 
        if labels is not None:
            # loss_fn = nn.CrossEntropyLoss()
            loss_fn =MyFocalLoss(gamma=0.5, alpha=1)#0.5
            loss = loss_fn(logits, labels)
        return dict(loss=loss, logits=logits)


class ModelForSequenceLabeling(nn.Module):
    """Bert model for token classification."""

    def __init__(self, config, backbone):
        super(ModelForSequenceLabeling, self).__init__()
        self.backbone = backbone 
        # self.crf = CRF(config.num_labels, batch_first=True)
        # self.pooler = MeanPooling()
        self.cls_head = ClassificationHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # backbone encode 
        outputs = self.backbone(input_ids=input_ids, \
                                attention_mask=attention_mask, \
                                token_type_ids=token_type_ids,
                                return_dict=True)   
        hidden_states = outputs.last_hidden_state
        # hidden_states = self.pooler(outputs.last_hidden_state, attention_mask)
        # print(hidden_states.size())
        # classification
        logits = self.cls_head(hidden_states) # [batch_size, seq_length, num_labels]
        # compute loss 
        loss = None 
        if labels is not None:
            # loss_fn = nn.CrossEntropyLoss()
            loss_fn = MyFocalLoss(gamma=0.5, alpha=1)#0.5
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            # CRF
            # mask = labels != -100
            # mask[:, 0] = 1
            # labels[:, 0] = 0
            # labels = labels * mask.to(torch.long)
            # loss = -self.crf(emissions=logits, 
            #                 tags=labels,
            #                 mask=mask,
            #                 reduction = "token_mean")
        else:
            # preds = self.crf.decode(emissions=logits, mask=mask)
            # logits = torch.LongTensor(preds)
            pass 

        return dict(loss=loss, logits=logits)