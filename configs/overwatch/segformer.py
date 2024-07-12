_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/segformer_b3.py', 
    '../_base_/datasets/maxar_notrees.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)))
)

runner = dict(type='IterBasedRunner', max_iters=30000)

# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=20)
evaluation = dict(interval=20, metric='mIoU', save_best='mIoU')


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
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(512, 512),)
)