_base_ = ['../_base_/default_runtime.py', '../_base_/det_p5_tta.py']

# ========================Frequently modified parameters======================
# -----data related-----
data_root = '../../datasets/coco/'  # Root path of data
# Path of train annotation file
train_ann_file = 'annotations/instances_train2017.json'
train_data_prefix = 'train2017/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/instances_val2017.json'
val_data_prefix = 'val2017/'  # Prefix of val image path

num_classes = 80  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 16
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
base_lr = 0.01
max_epochs = 500  # Maximum training epochs
# Disable mosaic augmentation for final 10 epochs (stage 2)
close_mosaic_epochs = 10

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (640, 640)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# Config of batch shapes. Only on val.
# We tested YOLOv8-m will get 0.02 higher than not using it.
batch_shapes_cfg = None  # batch shapes 配置
# You can turn on `batch_shapes_cfg` by uncommenting the following lines.
# batch_shapes_cfg = dict(
#     type='BatchShapePolicy',   # 确保同一个 batch 内的图像 pad 像素最少，不要求整个验证过程中所有 batch 的图像尺度一样
#     batch_size=val_batch_size_per_gpu,
#     img_size=img_scale[0],
#     # The image scale of padding should be divided by pad_size_divisor
#     size_divisor=32,
#     # Additional paddings for pixel scale
#     extra_pad_ratio=0.5)

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5
# Strides of multi-scale prior box
strides = [8, 16, 32]
# The output channel of the last stage
last_stage_out_channels = 1024
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
# YOLOv5RandomAffine aspect ratio of width and height thres to filter bboxes
max_aspect_ratio = 100
tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics
# TODO: Automatically scale loss_weight based on number of detection layers
loss_cls_weight = 0.5
loss_bbox_weight = 7.5
# Since the dfloss is implemented differently in the official
# and mmdet, we're going to divide loss_weight by 4.
loss_dfl_weight = 1.5 / 4
lr_factor = 0.01  # Learning rate scaling factor
weight_decay = 0.0005
# Save model checkpoint and validation intervals in stage 1
save_epoch_intervals = 1
# validation intervals in stage 2（这里所说的stage是指训练分成两个阶段）
val_interval_stage2 = 1
# The maximum checkpoints to keep.
max_keep_ckpts = 2
# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=tal_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-9)),
    test_cfg=model_test_cfg)

albu_train_transforms = [  # 覆写albu transform的配置
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

pre_transform = [  # 数据读取流程
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),  # 从文件路径里加载图像，文件读取后端的配置，默认从硬盘读取
    dict(type='LoadAnnotations', with_bbox=True)  # 对于当前图像，加载它的注释信息；是否使用标注框(bounding box)，目标检测需要设置为 True
]

last_transform = [
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,  # 传入要进行的操作
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),

    dict(type='YOLOv5HSVRandomAug'),  # HSV通道随机增强
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',  # 将数据转换为检测器输入格式的流程  非常重要！！
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline = [  # mmcv的数据通道保证随着定义顺序，总配置可以被覆写
    *pre_transform,  # 加载数据
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,  # 空区域填充像素值
        pre_transform=pre_transform),  # 之前创建的 pre_transform 训练数据读取流程
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),  # 图像缩放系数的范围
        max_aspect_ratio=max_aspect_ratio,
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),  # 从输入图像的高度和宽度两侧调整输出形状的距离
        border_val=(114, 114, 114)),  # 边界区域填充像素值
    *last_transform  # 先进行马赛克&仿射变换之后才进行数据的增强
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border_val=(114, 114, 114)), *last_transform
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,  # 如果设置为 True，dataloader 在迭代完一轮之后不会关闭数据读取的子进程，可以加速训练
    pin_memory=True,  # 开启锁页内存，节省 CPU 内存拷贝时间
    sampler=dict(type='DefaultSampler', shuffle=True),  # 默认的采样器，同时支持分布式和非分布式训练
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),  # 图像和标注的过滤配置，这个很有用
        pipeline=train_pipeline))  # pipeline是最需要关注的内容，在定义数据集的同时给出transform

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),  # 读图片，先不加载标注!
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),  # 因为是测试，所以需要保持长宽比！
    dict(
        type='LetterResize',  # 满足多种步幅要求的图像大小缩放
        scale=img_scale,
        allow_scale_up=False,  # 不允许尺度改变
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),  # 对于当前图像，加载它的注释信息
    dict(  # 转化为检测格式
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,  # 是否丢弃最后未能组成一个批次的数据，注意测试的时候绝对不能丢弃
    sampler=dict(type='DefaultSampler', shuffle=False),  # 验证和测试时不打乱数据顺序
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,  # 开启测试模式，避免数据集过滤图像和标注
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader

# 配置参数调度器（Parameter Scheduler）来调整优化器的超参数（例如学习率和动量）。用户可以组合多个调度器来创建所需的参数调整策略。
param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',   # 优化器封装的类型。可以切换至 AmpOptimWrapper 来启用混合精度训练
    clip_grad=dict(max_norm=10.0),  # 注意这里梯度裁剪了
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,  # 开启Nesterov momentum
        batch_size_per_gpu=train_batch_size_per_gpu),  # 该选项实现了自动权重衰减系数缩放（这是构造器constructor的属性）
    constructor='YOLOv5OptimizerConstructor')  # 这里没有使用默认的optim构造器，而是使用覆写的YOLOv5 optim构造器
# OptimizerConstructor 的作用就是给不同层设置不同的模型优化超参


default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',   # MMYOLO 中默认采用 Hook 方式进行优化器超参数的调节！！！
        scheduler_type='linear',
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,  # 何时保存ckpt，注意通常保存ckpt的时候进行val！
        save_best='auto',  # 自动保存最好的
        max_keep_ckpts=max_keep_ckpts))  # 最多保存几个ckpt

custom_hooks = [
    dict(
        type='EMAHook',   # 实现权重 EMA(指数移动平均) 更新的 Hook
        ema_type='ExpMomentumEMA',   # YOLO 中使用的带动量 EMA
        momentum=0.0001,
        update_buffers=True,  # 是否计算模型的参数和缓冲的 running averages
        strict_load=False,
        priority=49),  # 优先级略高于 NORMAL(50)
    dict(
        type='mmdet.PipelineSwitchHook',  # 实现两阶段训练
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),  # 用于评估检测任务时，选取的Proposal数量，用于计算召回率和精确率
    ann_file=data_root + val_ann_file,
    metric='bbox')  # 需要计算的评价指标，`bbox` 用于检测
test_evaluator = val_evaluator

# MMEngine 的 Runner 使用 Loop 来控制训练，验证和测试过程
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,  # 多少个epoch进行一次val
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),  # 动态测评，在val_interval的基础上加
                        val_interval_stage2)])  # 例[(280,1)]：到 280 epoch 开始切换为间隔 1 的评估方式
# 对于测试和验证只需要跑一次并且一定是跑完，也不要数据增强，所以直接定义
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
