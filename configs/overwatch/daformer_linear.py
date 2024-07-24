_base_ = [
    '../_base_/default_runtime.py',
    "../_base_/models/daformer_sepaspp_mitb3.py",
    "../_base_/datasets/uda_maxar_to_maxar_notrees.py",
    "../_base_/uda/dacs.py",
    '../_base_/schedules/adamw.py',
]
# Random Seed
seed = 0
optimizer_config = None
optimizer = dict(
    lr=1e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)))
)

lr_config = dict(policy='fixed')

runner = dict(type='IterBasedRunner', max_iters=100000)

# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=1000000, max_keep_ckpts=20)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')

model = dict(
    decode_head=dict(
        num_classes=3,
        loss_decode=dict(
            class_weight=[ # inverse of class frequency
                1/(0.964458+0.011841),
                1/0.013446,
                1/0.010255,
                ]
            ),
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384),)
)