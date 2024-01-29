# Jiaxiong Yang reserved.
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule, MaxPool2d,
                      build_norm_layer)
from typing import Optional, Sequence, Tuple, Union
from mmyolo.registry import MODELS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module(['ASPP', 'CFE'])
class CFE(BaseModule):
    """Pyramid Feature Attention Network for Saliency detection
    Context Feature Enhancement

    Args:
        in_channels: 输入的通道数
        out_channels: 输出的通道数，通常和in_channels一样，对concat的多尺度特征进行一次卷积降维
        dilation_rate (list): default [3,5,7]
        out_kernel: 最后一个卷积的kernel_size。 default=1
        conv_cfg: default None
        norm_cfg: 如果使用则进行正则和激活，否则设置为None
        act_cfg: 如果使用则进行正则和激活，否则设置为None
        init_cfg: default None
    """

    def __init__(self, in_channels, out_channels,
                 dilation_rate: list = [3, 5, 7],
                 out_kernel=1,
                 conv_cfg: ConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super(CFE, self).__init__(init_cfg=init_cfg)
        self.conv1x1 = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            padding=0,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )
        self.dilation_rate = dilation_rate
        self.dilation_convs3x3 = nn.ModuleList(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg
            ) for dilation in dilation_rate
        )
        self.out_conv = ConvModule(
            in_channels=in_channels * (1 + len(dilation_rate)),
            out_channels=out_channels,
            kernel_size=out_kernel,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg
        ) if in_channels == out_channels else nn.Identity()

    def forward(self, x):
        conv_out = [conv_dilation(x) for conv_dilation in self.dilation_convs3x3]
        conv_out.insert(0, self.conv1x1(x))
        conv_out = torch.concat(conv_out, dim=1)
        conv_out = self.out_conv(conv_out)
        return conv_out


class CEM(BaseModule):
    """A context- and level-aware feature pyramid network for object detection with attention mechanism

    context enhancement module

    Args:
        in_channels: 输入的通道数
        out_channels: 输出的通道数，通常和in_channels一样，对concat的多尺度特征进行一次卷积降维
        dilation_rate (list): default [3,5,7]
        out_kernel: 最后一个卷积的kernel_size。 default=1
        conv_cfg: default None
        norm_cfg: 如果使用则进行正则和激活，否则设置为None
        act_cfg: 如果使用则进行正则和激活，否则设置为None
        init_cfg: default None
    """

    def __init__(self, in_channels, out_channels,
                 dilation_rate: list = [3, 6, 9],
                 out_kernel=1,
                 conv_cfg: ConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg=None):
        super(CEM, self).__init__(init_cfg=init_cfg)
        self.dilation_rate = dilation_rate
        self.dilation_convs3x3 = nn.ModuleList(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg
            ) for dilation in dilation_rate
        )
        self.fuse = ConvModule(
            in_channels=in_channels * len(dilation_rate),
            out_channels=len(dilation_rate),
            kernel_size=1,
            act_cfg=dict(type='Sigmoid'),
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg
        )
        self.out_conv = ConvModule(
            in_channels=in_channels * len(dilation_rate),
            out_channels=out_channels,
            kernel_size=out_kernel,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg
        ) if in_channels == out_channels else nn.Identity()

    def forward(self, x):
        dilation_out = [conv_dilation(x) for conv_dilation in self.dilation_convs3x3]
        conv_out = torch.concat(dilation_out, dim=1)
        conv_out = self.fuse(conv_out)
        conv_out = torch.split(conv_out, len(self.dilation_rate), dim=1)
        enhance = None
        for d_conv, weight in zip(dilation_out, conv_out):
            enhance += d_conv * weight
        enhance = self.out_conv(enhance)
        return enhance


@MODELS.register_module()
class DCEMSerial(BaseModule):
    """Context Enhance Module using Dense-Serial connections

    by 'Attention-guided Context Feature Pyramid Network for Object Detection'

    Args:
        in_channels: 输入的通道数
        plane: 空洞卷积的输出通道数
        out_channels: 输出的通道数，通常和in_channels一样
        dilation_rate (list): default [3,6,9]
        conv_cfg: default None  论文中使用的是可变形卷积
        norm_cfg: 如果使用则进行正则和激活，否则设置为None
        act_cfg: 如果使用则进行正则和激活，否则设置为None
        init_cfg: default None
    """

    def __init__(self, in_channels, plane, out_channels,
                 dilation_rate: list = [3, 6, 9],
                 conv_cfg: ConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg=None):
        super(DCEMSerial, self).__init__(init_cfg=init_cfg)
        self.dilation_rate = dilation_rate
        self.dilation_convs3x3 = nn.ModuleList(
            ConvModule(
                in_channels=in_channels + i * plane,
                out_channels=plane,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg
            ) for i, dilation in enumerate(dilation_rate)
        )
        self.out_conv = ConvModule(
            in_channels=in_channels + len(dilation_rate) * plane,
            out_channels=out_channels,
            kernel_size=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            conv_cfg=conv_cfg
        )

    def forward(self, x):
        feat = [x]
        for conv in self.dilation_convs3x3:
            x = conv(feat[0])
            feat.append(x)
            feat = [torch.concat(feat, dim=1)]
        out = self.out_conv(feat[0])
        return out

@MODELS.register_module()
class GhostModule(BaseModule):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, dilation=1,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 init_cfg=None):
        super(GhostModule, self).__init__(init_cfg=init_cfg)
        self.oup = oup
        self.plane = oup // ratio
        new_channels = self.plane * (ratio - 1)
        self.primary_conv = ConvModule(
            in_channels=inp,
            out_channels=self.plane,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )
        self.cheap_operation = ConvModule(
            in_channels=self.plane,
            out_channels=new_channels,
            kernel_size=dw_size,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=self.plane,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]
