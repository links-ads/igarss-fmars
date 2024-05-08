_base_ = [
    '../_base_/models/segformer_b5.py', 
    '../_base_/default_runtime.py', 
    '../_base_/schedules/adamw.py',
    '../_base_/datasets/maxar.py',
]


total_iters = 1000
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)

# Logging Configuration
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=1)
runner = dict(type='IterBasedRunner', max_iters=40000)
