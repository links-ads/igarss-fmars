_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/uda_maxar_to_maxar.py',
    '../_base_/uda/dacs.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0
# Modifications to Basic UDA
uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120
    )

optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=50000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=10000000, max_keep_ckpts=1, save_optimizer=True)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')