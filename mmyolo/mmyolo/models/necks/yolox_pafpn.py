# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.backbones.csp_darknet import CSPLayer
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from .base_yolo_neck import BaseYOLONeck


@MODELS.register_module()
class YOLOXPAFPN(BaseYOLONeck):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        freeze_all(bool): Whether to freeze the model. Defaults to False.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 use_depthwise: bool = False,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        self.num_csp_blocks = round(num_csp_blocks * deepen_factor)
        self.use_depthwise = use_depthwise

        super().__init__(
            in_channels=[
                int(channel * widen_factor) for channel in in_channels
            ],
            out_channels=int(out_channels * widen_factor),
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if idx == 2:
            layer = ConvModule(  # 将通道数和上一层的对齐，之后拼接就是当前通道数*2
                self.in_channels[idx],
                self.in_channels[idx - 1],
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            layer = nn.Identity()

        return layer

    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(scale_factor=2, mode='nearest')

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule  # 使用卷积进行下采样
        return conv(
            self.in_channels[idx],
            self.in_channels[idx],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.  top指的是深层，从深层到浅层逐步拼接升采样

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        if idx == 1:
            return CSPLayer(  # 用的mmdet的
                self.in_channels[idx - 1] * 2,  # 这里进行了降维
                self.in_channels[idx - 1],
                num_blocks=self.num_csp_blocks,
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        elif idx == 2:
            return nn.Sequential(
                CSPLayer(
                    self.in_channels[idx - 1] * 2,  # 这里进行了降维
                    self.in_channels[idx - 1],
                    num_blocks=self.num_csp_blocks,
                    add_identity=False,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    self.in_channels[idx - 1],  # 这里进行了降维  把降维的工作放在了CSP后面，从而完全串行
                    self.in_channels[idx - 2],
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayer(
            self.in_channels[idx] * 2,  # 这里实际上是没有维度改变的
            self.in_channels[idx + 1],
            num_blocks=self.num_csp_blocks,
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """
        return ConvModule(
            self.in_channels[idx],
            self.out_channels,
            1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
