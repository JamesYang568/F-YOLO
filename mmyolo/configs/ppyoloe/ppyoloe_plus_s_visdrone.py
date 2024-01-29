# Compared to other same scale models, this configuration consumes too much
# GPU memory and is not validated for now
_base_ = 'ppyoloe_plus_s_fast_8xb8-80e_coco.py'

data_root = '../../datasets/VisDrone2019/'
class_name = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
              "awning-tricycle", "bus", "motor")
num_classes = len(class_name)
metainfo = dict(classes=class_name,
                palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
                         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)])


num_last_epochs = 5

max_epochs = 40
train_batch_size_per_gpu = 16
train_num_workers = 4

load_from = 'https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco/ppyoloe_plus_s_fast_8xb8-80e_coco_20230101_154052-9fee7619.pth'  # noqa

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
        metainfo=metainfo
    ))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
    ))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/instances_val2017.json')
test_evaluator = val_evaluator

default_hooks = dict(
    param_scheduler=dict(
        warmup_min_iter=50,
        warmup_epochs=3,
        total_epochs=int(max_epochs * 1.2)
    ),
    checkpoint=dict(interval=5, max_keep_ckpts=2, save_best='auto'),
    logger=dict(type='LoggerHook', interval=50))
train_cfg = dict(max_epochs=max_epochs, val_interval=5//2)
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
