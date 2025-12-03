# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from transformers import AutoModel

from mmaction.models.builder import RECOGNIZERS, build_loss
from mmaction.models.recognizers.base import BaseRecognizer
from mmaction.core import top_k_accuracy


@RECOGNIZERS.register_module()
class VideoTextContrastRecognizer(BaseRecognizer):
    """3D recognizer model framework."""
    def __init__(self, num_class=0, text_encoder_path='roberta', feature_dim=1024, **kwargs):
        super().__init__(**kwargs)
        self.con_loss = build_loss(dict(type='VideoTextContrastLoss'))
        self.text_encoder = AutoModel.from_pretrained(text_encoder_path)
        self.video_pool = nn.AdaptiveAvgPool3d(1)
        self.transform_video = nn.Linear(1024, 1024)
        self.transform_text = nn.Linear(768, 1024)
        if num_class != 0:
            self.is_classification = True
            self.classifier = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_class)
            )
            self.cls_loss = build_loss(dict(type='CrossEntropyLoss'))
        else:
            self.is_classification = False


    def forward_train(self, imgs, text, attention_mask, label=None, **kwargs):
        """Defines the computation performed at every call when training."""
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        vf = self.extract_feat(imgs)
        tf = self.text_encoder(text).pooler_output

        vf = self.video_pool(vf)
        vf = vf.reshape((-1, 1024))
        transform_vf = self.transform_video(vf)
        transform_tf = self.transform_text(tf)

        sim_matrix = torch.cosine_similarity(transform_vf.unsqueeze(1), transform_tf.unsqueeze(0), dim=2)

        if self.is_classification:
            cls_score = self.classifier(vf)
        else:
            cls_score = None

        return self.loss(sim_matrix, cls_score, label)

    def forward(self, imgs, text, attention_mask, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(imgs, **kwargs)
        if return_loss:
            return self.forward_train(imgs, text, attention_mask, label, **kwargs)

        return self.forward_test(imgs, text, attention_mask, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self(**data_batch, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        aux_info = {}
        for item in self.aux_info:
            aux_info[item] = data_batch[item]

        losses = self(**data_batch, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs


    def _do_test(self, imgs, text, attention_mask):
        imgs = imgs.reshape((-1,) + imgs.shape[2:])

        vf = self.extract_feat(imgs)
        tf = self.text_encoder(text).pooler_output

        vf = self.video_pool(vf)
        vf = vf.reshape((-1, 1024))
        transform_vf = self.transform_video(vf)
        transform_tf = self.transform_text(tf)

        sim_matrix = torch.cosine_similarity(transform_vf.unsqueeze(1), transform_tf.unsqueeze(0), dim=2)

        if self.is_classification:
            cls_score = self.classifier(vf)
        else:
            cls_score = None

        return sim_matrix, cls_score


    def forward_test(self, imgs, text, attention_mask):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs, text, attention_mask).cpu().numpy()

    def forward_gradcam(self, imgs):
        pass

    def loss(self, sim_matrix, cls_score=None, labels=None):
        losses = dict()
        loss_con = self.con_loss(sim_matrix)
        if isinstance(loss_con, dict):
            losses.update(loss_con)
        else:
            losses['loss_con'] = loss_con
        if self.is_classification:
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       (1,5))
            for k, a in zip((1, 5), top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_score.device)

            loss_cls = self.cls_loss(cls_score, labels.squeeze())
            if isinstance(loss_cls, dict):
                losses.update(loss_cls)
            else:
                losses['loss_cls'] = loss_cls
        return losses




