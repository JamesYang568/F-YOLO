_base_ = 'yolov8_s_syncbn_fast_8xb16-500e_coco.py'

data_root = '../../../datasets/BDD100K/'
class_name = ("pedestrian", "rider", "car", "truck", "bus", "train", " motorcycle", "bicycle",
              "traffic light", "traffic sign")
num_classes = len(class_name)
metainfo = dict(classes=class_name,
                palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
                         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)])

close_mosaic_epochs = 5

max_epochs = 100
train_batch_size_per_gpu = 32
train_num_workers = 8

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'  # noqa

model = dict(
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

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
        metainfo=metainfo,
        data_root=data_root,
    )
)

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.custom_hooks[1].switch_epoch = max_epochs - close_mosaic_epochs

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=100),
    logger=dict(type='LoggerHook', interval=50))
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
find_unused_parameters = True
