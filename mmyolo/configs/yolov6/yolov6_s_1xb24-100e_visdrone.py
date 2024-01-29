_base_ = './yolov6_s_syncbn_fast_8xb32-400e_coco.py'

data_root = '../../datasets/VisDrone2019/'
class_name = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
              "awning-tricycle", "bus", "motor")
num_classes = len(class_name)
metainfo = dict(classes=class_name,
                palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
                         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)])

max_epochs = 100
train_batch_size_per_gpu = 24
train_num_workers = 8

num_last_epochs = 15

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco/yolov6_s_syncbn_fast_8xb32-400e_coco_20221102_203035-932e1d91.pth'  # noqa

model = dict(
    # backbone=dict(frozen_stages=4),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(
        initial_assigner=dict(num_classes=num_classes),
        assigner=dict(num_classes=num_classes)))

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
_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

_base_.custom_hooks[1].switch_epoch = max_epochs - num_last_epochs

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_epochs=3),
    logger=dict(type='LoggerHook', interval=50))

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])  # noqa
