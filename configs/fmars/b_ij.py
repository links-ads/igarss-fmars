_base_ = [
    '../_base_/models/segformer_b0.py', 
    '../_base_/default_runtime.py', 
    '../_base_/schedules/adamw.py',
    '../_base_/datasets/maxar_ij.py',
]

lr_config = dict(policy='fixed')

# Logging Configuration
checkpoint_config = dict(by_epoch=True, interval=1, max_keep_ckpts=1)
runner = dict(type='EpochBasedRunner', max_epochs=100)
evaluation = dict(interval=1, metric='mIoU')