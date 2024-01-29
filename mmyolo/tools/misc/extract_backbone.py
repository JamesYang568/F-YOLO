# Jiaxiong Yang reserved.
import torch
import os
import json

# 这个脚本用于将完整的yolo模型中的backbone部分提取出来，用于后续的模型微调，所有标记的位置都需要手动更改！
# 需要将ckpt放在checkpoint文件夹内，生成的文件也会出现在该文件夹内
from mmyolo.models import YOLOv6EfficientRep, YOLOv7Backbone, YOLOv8CSPDarknet, YOLOXCSPDarknet

check = False  # 是否要检测提取的是否正确    如果要检查，需要根据主干网络修改下面的内容 标记
model_name = 'yolov6_s'  # 模型名称  标记
info = torch.load(os.path.join('checkpoints', model_name + '.pth'), map_location='cpu')  # 加载已有模型  标记

meta_info = info['meta']  # 模型快照包含元信息和下面的state_dict

meta_json = json.dumps(meta_info)
with open(os.path.join('checkpoints', model_name + '_meta.json'), 'w') as f:  # 保存元信息
    f.write(meta_json)

with open(os.path.join('checkpoints', 'cfg.py'), 'w') as f:  # 保存这个模型训练的配置文件，为py
    f.write(meta_info['cfg'])

epoch = meta_info['epoch']
iter = meta_info['iter']
total_batchsize = 118287 / (iter // epoch)
print('total_batchsize: {}'.format(round(total_batchsize)))

from checkpoints.cfg import train_batch_size_per_gpu as bs_pre_gpu

print('total gpu: {}'.format(round(total_batchsize / bs_pre_gpu)))

state_dict = info['state_dict']
# 如果state_dict中包含module.backbone，则放到新的dict backbone中
from collections import OrderedDict

backbone = OrderedDict()
for k, v in state_dict.items():
    if 'backbone' in k:
        # backbone[k.replace('backbone.', '')] = v  # 要将前缀backbone.删除掉   检查 标记
        backbone[k] = v  # 直接在mmyolo里面增加，那就还需要backbone.   检查 标记
    elif 'neck' in k:
        # backbone[k.replace('neck.', '')] = v
        backbone[k] = v

if check:
    # 下面的部分是针对不同模型给出不同的backbone，需要手动修改 用于检查是否提取正确
    deepen_factor = 0.33
    widen_factor = 0.5
    mm = YOLOv6EfficientRep(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True))
    # mm = YOLOv7Backbone(arch='X',
    #                     norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
    #                     act_cfg=dict(type='SiLU', inplace=True))
    # mm = YOLOv8CSPDarknet(
    #     arch='P5',
    #     last_stage_out_channels=1024,
    #     deepen_factor=deepen_factor,
    #     widen_factor=widen_factor,
    #     norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
    #     act_cfg=dict(type='SiLU', inplace=True))
    # mm = YOLOXCSPDarknet(
    #     deepen_factor=deepen_factor,
    #     widen_factor=widen_factor,
    #     out_indices=(2, 3, 4),
    #     spp_kernal_sizes=(5, 9, 13),
    #     norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
    #     act_cfg=dict(type='SiLU', inplace=True),
    # )
    mm.load_state_dict(backbone, strict=True)  # 不需要返回值
backbone = {'state_dict': backbone}
torch.save(backbone, os.path.join('checkpoints', model_name + '_backbone&neck.pth'))
