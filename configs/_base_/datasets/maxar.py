crop_size=(512, 512)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
            dict(type='LoadTifFromFile'),
            dict(type='LoadTifAnnotations'),
            dict(type='RandomCrop', crop_size=crop_size),
            dict(type='Normalize', **img_norm_cfg), 
            dict(type='ImageToTensor', keys=['img', 'gt_semantic_seg']),
            dict(type='Collect', 
                 keys=['img', 'gt_semantic_seg'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg')
                ),
            ]
val_pipeline = [
            dict(type='LoadTifFromFile'),
            dict(
                    type='MultiScaleFlipAug', 
                    img_scale=crop_size, 
                    flip=False,
                    transforms=[
                        dict(type='CenterCrop', crop_size=(1024, 1024)),
                        dict(type='Normalize', **img_norm_cfg), 
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', 
                            keys=['img'],
                            meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'flip',)
                        ),
                    ]
                )
            ]
test_pipeline = [
            dict(type='LoadTifFromFile'),
            dict(
                    type='MultiScaleFlipAug', 
                    img_scale=crop_size, 
                    flip=False,
                    transforms=[
                        dict(type='Normalize', **img_norm_cfg), 
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', 
                            keys=['img'],
                            meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'flip',)
                        ),
                    ]
                )
            ]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=16,
    train=dict(
        type='TifDataset',
        data_root='./',
        img_dir = 'data/maxar-open-data/',
        img_suffix = '.tif',
        ann_dir = 'data/outputs/04_05/train/',
        seg_map_suffix = '.tif',
        split = 'train',
        pipeline=train_pipeline,
    ),
    val=dict(
        type='TifDataset',
        data_root='./',
        img_dir = 'data/maxar-open-data/',
        img_suffix = '.tif',
        ann_dir = 'data/outputs/04_05/val/',
        seg_map_suffix = '.tif',
        split = 'val',
        pipeline=val_pipeline,
    ),
    test=dict(
        type='TifDataset',
        data_root='./',
        img_dir = 'data/maxar-open-data/',
        img_suffix = '.tif',
        ann_dir = 'data/outputs/04_05/test/',
        seg_map_suffix = '.tif',
        split = 'test',
        pipeline=test_pipeline,
    )
)