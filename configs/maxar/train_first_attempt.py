_base_ = [
    '../_base_/models/segformer_b5.py', 
    '../_base_/default_runtime.py', 
    '../_base_/schedules/adamw.py',
    '../_base_/datasets/maxar.py',
]

lr_config = dict(policy='fixed')

# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=100)
