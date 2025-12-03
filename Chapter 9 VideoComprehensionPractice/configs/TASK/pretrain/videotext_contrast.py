_base_ = [
'../../_base_/default_runtime.py'
]

transformer_path = 'modules/transformer_model/roberta-base'

# model settings
model = dict(
    type='VideoTextContrastRecognizer',
    num_class=2015,
    text_encoder_path=transformer_path,
    backbone=dict(
        type='SwinTransformer3D',
        patch_size=(4,4,4),
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(8, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True),
)

# dataset settings
dataset_type = 'VideoTextDataset'
data_root = 'data/ucf101/videos'
data_root_val = 'data/ucf101/videos'
ann_file_train = 'data/ucf101/ucf101_train_split_1_videos.txt'
namelist_file_train = 'data/ucf101/ucf101_train_split_1_namelist.txt'
titlelist_file_train = 'data/ucf101/ucf101_train_split_1_titlelist.txt'
ann_file_val = 'data/ucf101/ucf101_val_split_1_videos.txt'
namelist_file_val = 'data/ucf101/ucf101_val_split_1_namelist.txt'
titlelist_file_val = 'data/ucf101/ucf101_val_split_1_titlelist.txt'
ann_file_test = 'data/ucf101/ucf101_val_split_1_videos.txt'
namelist_file_test = 'data/ucf101/ucf101_val_split_1_namelist.txt'
titlelist_file_test = 'data/ucf101/ucf101_val_split_1_titlelist.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=6, num_clips=1),
    dict(type='DecordDecode'),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label', 'text', 'attention_mask'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'text', 'attention_mask'])
]

val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=6,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label', 'text', 'attention_mask'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'text', 'attention_mask'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=6,
        num_clips=2,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label', 'text', 'attention_mask'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'text', 'attention_mask'])
]


data = dict(
    videos_per_gpu=4,
    workers_per_gpu=0,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=2
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=2
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        namelist_file=namelist_file_train,
        titlelist_file=titlelist_file_train,
        tokenizer_path=transformer_path,
        title_max_len=128,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        namelist_file=namelist_file_val,
        titlelist_file=titlelist_file_val,
        tokenizer_path=transformer_path,
        title_max_len=128,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        namelist_file=namelist_file_test,
        titlelist_file=titlelist_file_test,
        tokenizer_path=transformer_path,
        title_max_len=128,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2
)
total_epochs = 10

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/pretrain'
load_from = 'checkpoints/swin_base_patch244_window877_kinetics600_22k.pth'
find_unused_parameters = False


# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))

gpu_ids = range(0,1)

