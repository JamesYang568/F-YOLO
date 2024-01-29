_base_ = './yolox_s_fast_8xb8-300e_coco.py'

data_root = '../../datasets/VisDrone2019/'
class_name = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
              "awning-tricycle", "bus", "motor")
num_classes = len(class_name)
metainfo = dict(classes=class_name,
                palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
                         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)])

max_epochs = 40
train_batch_size_per_gpu = 16
train_num_workers = 4
num_last_epochs = 5

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_fast_8xb8-300e_coco/yolox_s_fast_8xb8-300e_coco_20230213_142600-2b224d8b.pth' # noqa
model = dict(
    # backbone=dict(frozen_stages=4),  # todo
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    )

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
    ))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
    ))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

base_lr = _base_.base_lr / 4
optim_wrapper = dict(optimizer=dict(lr=base_lr))
# _base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

_base_.custom_hooks[0].num_last_epochs = num_last_epochs

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    logger=dict(type='LoggerHook', interval=50))

param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 15 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]


train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
