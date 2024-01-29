_base_ = './yolov6_s_syncbn_fast_8xb32-400e_coco.py'

# ======================= Frequently modified parameters =====================

data_root = '../../datasets/VisDrone2019/'
class_name = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
              "awning-tricycle", "bus", "motor")
num_classes = len(class_name)
metainfo = dict(classes=class_name,
                palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
                         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)])

# -----train val related-----
# Base learning rate for optim_wrapper
max_epochs = 100  # Maximum training epochs  TODO 130 for better results.
num_last_epochs = 10  # todo Last epoch number to switch training pipeline
save_epoch_intervals = 1  # 5
train_batch_size_per_gpu = 32
train_num_workers = 8

n_fuzzy = 3

deepen_factor = 0.33
widen_factor = 0.5

# ============================== Unmodified in most cases ===================
model = dict(
    backbone=dict(
        # frozen_stages=0, no frozzing
        type='YOLOv6EfficientRepExt',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        deep_enhance_cfg=dict(type='GlobalFusionCalibration',
                              in_channels=512,
                              dilation=[1, 2, 2, 2],
                              reduction=8,
                              norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                              act_cfg=dict(type='ReLU', inplace=True)
                              ),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True)),
    bbox_head=dict(
        head_module=dict(
            type='YOLOv6HeadModuleFuzzy',
            num_classes=num_classes,
            in_channels=[128, 256, 512],
            widen_factor=_base_.widen_factor,
            n_fuzzy=n_fuzzy,
            squeeze_fuzzy=16,
            adaptive_fuzzy=True,
            residual_fuzzy=True,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[8, 16, 32],
        )
    ),
    train_cfg=dict(
        initial_assigner=dict(num_classes=num_classes),
        assigner=dict(num_classes=num_classes))
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
    )
)

val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')  # 位置改变了
test_evaluator = val_evaluator

base_lr = _base_.base_lr / 8
optim_wrapper = dict(optimizer=dict(lr=base_lr))
_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

default_hooks = dict(
    param_scheduler=dict(
        warmup_epochs=5,  # todo 3 for better results.
        warmup_mim_iter=0,
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(interval=save_epoch_intervals,
                    max_keep_ckpts=3,
                    save_best='auto'),
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

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth'  # noqa

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])

find_unused_parameters = True  # 如果冻结了部分参数，会导致多卡训练时发生部分梯度无法同步回传的问题（因为没有为其grad=False），