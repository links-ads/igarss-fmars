_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/segformer_b3.py', 
    '../_base_/datasets/maxar.py',
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
evaluation = dict(interval=5000, metric='mIoU', save_best='mIoU')