img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size=(512,512)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='TifDataset',
        data_root='./',
        img_dir = 'data/maxar-open-data/',
        img_suffix = '.tif',
        ann_dir = 'data/outputs/19_4/',
        seg_map_suffix = '.tif',
        split = 'train',
        pipeline=[
            dict(type='LoadTifFromFile'),
            dict(type='LoadTifAnnotations'),
            dict(type='RandomCrop', crop_size=crop_size),
            dict(type='Normalize', **img_norm_cfg), 
            dict(type='ImageToTensor', keys=['img', 'gt_semantic_seg']),
            dict(type='Collect', 
                 keys=['img', 'gt_semantic_seg'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'ann_info')
                ),
            ]
    ),
    val=dict(
        type='TifDataset',
        data_root='./',
        img_dir = 'data/maxar-open-data/',
        img_suffix = '.tif',
        ann_dir = 'data/outputs/19_4/',
        seg_map_suffix = '.tif',
        split = 'val',
        pipeline=[
            dict(type='LoadTifFromFile'),
            dict(type='LoadTifAnnotations'),
            dict(type='RandomCrop', crop_size=crop_size),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='SwapChannels'),
            dict(type='Collect', 
                 keys=['img', 'gt_semantic_seg'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'ann_info')
                ),
            ]
    ),
    test=dict(
        type='TifDataset',
        data_root='./',
        img_dir = 'data/maxar-open-data/',
        img_suffix = '.tif',
        ann_dir = 'data/outputs/19_4/',
        seg_map_suffix = '.tif',
        split = 'test',
        pipeline=[
            dict(type='LoadTifFromFile'),
            dict(type='LoadTifAnnotations'),
            dict(type='RandomCrop', crop_size=crop_size),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='SwapChannels'),
            dict(type='Collect', 
                 keys=['img', 'gt_semantic_seg'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'ann_info')
                ),
            ]
    )
)