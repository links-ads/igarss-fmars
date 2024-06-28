_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/daformer_sepaspp_mitb5.py",
    "../_base_/datasets/uda_maxar_to_maxar_entropy.py",
    "../_base_/uda/dacs.py",
    "../_base_/schedules/adamw.py",
    "../_base_/schedules/poly10warm.py",
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
            norm=dict(decay_mult=0.0),
        )
    ),
)

data = dict(samples_per_gpu=4, workers_per_gpu=16)
runner = dict(type="IterBasedRunner", max_iters=30000)

# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=20)
evaluation = dict(interval=5000, metric="mIoU", save_best="mIoU")


name = "daformer_basic"
