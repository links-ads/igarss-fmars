_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/segformer_b5.py', 
    '../_base_/datasets/maxar_ij.py',
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

# lr_config = dict(policy='fixed')

runner = dict(type='IterBasedRunner', max_iters=100000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=1, save_optimizer=True)
evaluation = dict(interval=5000, metric='mIoU', save_best='mIoU')