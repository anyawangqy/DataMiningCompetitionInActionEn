# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmaction.datasets.base import BaseDataset
from mmaction.datasets.builder import DATASETS
from transformers import AutoTokenizer
import numpy as np

@DATASETS.register_module()
class VideoTextDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, namelist_file, titlelist_file, tokenizer_path, title_max_len, pipeline, start_index=0, **kwargs):
        self.namelist_file = namelist_file
        self.titlelist_file = titlelist_file
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.title_max_len = title_max_len
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)



    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        title_dict = {}
        name_list = []
        with open(self.namelist_file, 'r') as fin_file:
            for line in fin_file:
                name = line.strip()
                name_list.append(name)
        title_list = []
        with open(self.titlelist_file, 'r') as fin_title:
            for line in fin_title:
                title = line.strip()
                title_list.append(title)
        assert len(title_list) == len(name_list)
        for i, name in enumerate(name_list):
            title_dict[name] = title_list[i]

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split
                    label = int(label)
                name = osp.basename(filename)[:osp.basename(filename).rfind('.avi')]
                title = title_dict[name]
                title = self.tokenizer(title, max_length=self.title_max_len, truncation=True)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, label=label,
                                        text=self.sequence_padding(title['input_ids'], length=self.title_max_len),
                                        attention_mask=self.sequence_padding(title['attention_mask'], length=self.title_max_len)))
        return video_infos

    def sequence_padding(self, inputs, length=None, value=0):
        padding_length = length - len(inputs)
        return inputs + [value] * padding_length