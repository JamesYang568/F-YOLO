_base_ = './yolov6_s_syncbn_fast_8xb32-400e_coco.py'

# ======================= Frequently modified parameters =====================
# -----train val related-----
# Base learning rate for optim_wrapper
max_epochs = 100  # Maximum training epochs
num_last_epochs = 10  # Last epoch number to switch training pipeline TODO 15
save_epoch_intervals = 1

train_batch_size_per_gpu = 32  # impl 48
train_num_workers = 8

n_fuzzy = 3
# ============================== Unmodified in most cases ===================
model = dict(
    backbone=dict(frozen_stages=4),
    neck=dict(freeze_all=True),
    bbox_head=dict(
        head_module=dict(
            type='YOLOv6HeadModuleFuzzy',
            num_classes=_base_.num_classes,
            in_channels=[128, 256, 512],
            widen_factor=_base_.widen_factor,
            n_fuzzy=n_fuzzy,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[8, 16, 32])
    )
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers
)

val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers
)
test_dataloader = val_dataloader

base_lr = _base_.base_lr / 4
optim_wrapper = dict(optimizer=dict(lr=base_lr))
_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

default_hooks = dict(
    param_scheduler=dict(
        warmup_epochs=5,
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(interval=save_epoch_intervals, max_keep_ckpts=2, save_best='auto'),
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - num_last_epochs,
        switch_pipeline=_base_.train_pipeline_stage2)
]

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])

# load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth'  # noqa
load_from = 'ckpt/yolov6_s_backbone&neck.pth'

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])

find_unused_parameters = True
