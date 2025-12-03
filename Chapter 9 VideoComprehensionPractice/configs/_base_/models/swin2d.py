# Swin-L
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='SwinTransformer',
        num_classes=1000,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        drop_path_rate=0.2),

    cls_head=dict(
        type='AvgHead',
        num_classes=240),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips=None))