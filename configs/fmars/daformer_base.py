_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/daformer_sepaspp_mitb5.py",
    "../_base_/datasets/uda_maxar_to_maxar.py",
    "../_base_/uda/dacs.py",
    "../_base_/schedules/adamw.py",
    "../_base_/schedules/poly10warm.py",
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
            norm=dict(decay_mult=0.0),
        )
    ),
)

data = dict(samples_per_gpu=2, workers_per_gpu=1)
runner = dict(type="IterBasedRunner", max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=4)
evaluation = dict(interval=1000, metric="mIoU")


name = "daformer_basic"
