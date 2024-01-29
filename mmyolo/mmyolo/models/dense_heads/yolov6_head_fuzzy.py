# Jiaxion Yang reserved.
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptMultiConfig)
from mmengine.model import BaseModule, bias_init_with_prob
from torch import Tensor

from mmyolo.registry import MODELS


@MODELS.register_module()
class YOLOv6HeadModuleFuzzy(BaseModule):
    """YOLOv6Head head module used in `YOLOv6.

    <https://arxiv.org/pdf/2209.02976>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors: (int): The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
            None, otherwise False. Defaults to "auto".
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 n_fuzzy: int = 2,
                 squeeze_fuzzy: int = 4,
                 adaptive_fuzzy: bool = True,
                 residual_fuzzy: bool = True,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.n_fuzzy = n_fuzzy
        self.squeeze_fuzzy = squeeze_fuzzy
        self.adaptive_fuzzy = adaptive_fuzzy
        self.residual_fuzzy = residual_fuzzy
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        if isinstance(in_channels, int):
            self.in_channels = [int(in_channels * widen_factor)
                                ] * self.num_levels
        else:
            self.in_channels = [int(i * widen_factor) for i in in_channels]

        self._init_layers()

    def _init_layers(self):
        """initialize conv layers"""
        # Init decouple head
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        for i in range(self.num_levels):
            self.stems.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=1,
                    stride=1,
                    padding=1 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
            )
            self.cls_convs.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=3 // 2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    FuzzyAttention(self.in_channels[i],
                                   self.n_fuzzy,
                                   self.squeeze_fuzzy,
                                   self.adaptive_fuzzy,
                                   self.residual_fuzzy)
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=3 // 2,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    FuzzyAttention(self.in_channels[i],
                                   self.n_fuzzy,
                                   self.squeeze_fuzzy,
                                   self.adaptive_fuzzy,
                                   self.residual_fuzzy)
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=self.num_base_priors * self.num_classes,
                    kernel_size=1))
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=self.num_base_priors * 4,
                    kernel_size=1))

            self.fuzzychannel = FuzzyChannelAtt(fusion_branch_cfg=self.in_channels,
                                                adaptive=self.adaptive_fuzzy,
                                                residual=self.residual_fuzzy,
                                                squeeze=self.squeeze_fuzzy)

    def init_weights(self):
        super().init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv in self.cls_preds:
            conv.bias.data.fill_(bias_init)
            conv.weight.data.fill_(0.)

        for conv in self.reg_preds:
            conv.bias.data.fill_(1.0)
            conv.weight.data.fill_(0.)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions.
        """
        assert len(x) == self.num_levels
        x = self.fuzzychannel(x)  # todo
        return multi_apply(self.forward_single, x, self.stems, self.cls_convs,
                           self.cls_preds, self.reg_convs, self.reg_preds)

    def forward_single(self, x: Tensor, stem: nn.Module, cls_conv: nn.Module,
                       cls_pred: nn.Module, reg_conv: nn.Module,
                       reg_pred: nn.Module) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level."""
        y = stem(x)
        cls_x = y
        reg_x = y
        cls_feat = cls_conv(cls_x)
        reg_feat = reg_conv(reg_x)

        cls_score = cls_pred(cls_feat)
        bbox_pred = reg_pred(reg_feat)

        return cls_score, bbox_pred


class FuzzyChannelAtt_inblock(nn.Module):
    def __init__(self, channel, n, squeeze=4, adaptive=False, residual=True):
        """特征图通过n个隶属度函数，拼接，得到通道的融合信息
        每个隶属度函数对应一个sigma和mu，之后类似SE生成一个多融合的通道注意力权重

        :param channel 特征图的大小
        :param n 隶属度函数的个数
        :param adaptive: 是否采用自适应池化  池化更快一点点
        :return 一个输出长度为info_channels的通道权重向量
        """
        super(FuzzyChannelAtt_inblock, self).__init__()
        self.plane = n * channel
        self.feedforward = nn.Sequential(
            nn.Conv2d(self.plane, self.plane // squeeze, 1),
            nn.BatchNorm2d(self.plane // squeeze),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.plane // squeeze, channel, 1),
            nn.Sigmoid()
        )
        self.mu = nn.Parameter(torch.randn((n, 1, 1, 1)))  # n mu，channel shared
        self.sigma = nn.Parameter(torch.randn((n, 1, 1, 1)))  # n sigma，channel shared
        self.adaptive = adaptive
        self.residual = residual

    def forward(self, x):
        data = x
        # expand to save n relation-maps result.
        x = x.unsqueeze(1)
        tmp = -((x - self.mu) / self.sigma) ** 2  # opt: b,n,c,h,w
        if self.adaptive:
            tmp = F.adaptive_avg_pool2d(torch.exp(tmp), (1, 1))
        else:
            tmp = torch.mean(torch.exp(tmp), dim=(3, 4), keepdim=True)  # sum or mean
        # b,n,c,1,1 -> b,n*c,1,1
        tmp = tmp.view(x.size(0), -1, 1, 1)
        fNeural = self.feedforward(tmp)
        if self.residual:
            return fNeural * data + data
        else:
            return fNeural * data


class FuzzyChannelAtt_inblock_channel(nn.Module):
    def __init__(self, channel, m, squeeze=4, adaptive=True, residual=True):
        """特征图通过n个隶属度函数，拼接，得到通道的融合信息
        每个隶属度函数对应一个sigma和mu，之后类似SE生成一个多融合的通道注意力权重

        :param channel 特征图的大小
        :param m 隶属度函数的个数
        :param adaptive: 是否采用自适应池化  池化更快一点点
        :return 一个输出长度为info_channels的通道权重向量
        """
        super(FuzzyChannelAtt_inblock_channel, self).__init__()
        self.plane = m * channel
        self.feedforward = nn.Sequential(
            nn.Conv2d(self.plane, self.plane // squeeze, 1, bias=False),
            nn.BatchNorm2d(self.plane // squeeze),
            nn.SiLU(inplace=True),  # or relu
            nn.Conv2d(self.plane // squeeze, channel, 1),
            nn.Sigmoid()
        )
        self.mu = nn.Parameter(torch.randn((m, channel, 1, 1)))  # n mu, channel independent
        self.sigma = nn.Parameter(torch.randn((m, channel, 1, 1)))  # n sigma, channel independent
        self.adaptive = adaptive
        self.residual = residual

    def forward(self, x):
        data = x
        # 扩增一维，用于保存n个不同的隶属度函数
        x = x.unsqueeze(1)
        tmp = -((x - self.mu) / self.sigma) ** 2  # opt: b,n,c,h,w
        if self.adaptive:
            tmp = F.adaptive_avg_pool2d(torch.exp(tmp), (1, 1))
        else:
            tmp = torch.mean(torch.exp(tmp), dim=(3, 4), keepdim=True)
        # b,n,c,1,1 -> b,n*c,1,1
        tmp = tmp.view(x.size(0), -1, 1, 1)
        fNeural = self.feedforward(tmp)
        if self.residual:
            return fNeural * data + data
        else:
            return fNeural * data


class FuzzySpatialAtt_inblock(nn.Module):
    def __init__(self, n, residual=True):
        """特征图通过各自的隶属度函数,拼接，得到空间的融合信息
        每一个隶属度函数只有一个sigma和mu，之后类似空间注意力形成一个多融合的空间注意力权重
        """
        super(FuzzySpatialAtt_inblock, self).__init__()
        self.mu = nn.Parameter(torch.randn((n, 1, 1, 1)))  # n mu，channel shared
        self.sigma = nn.Parameter(torch.randn((n, 1, 1, 1)))  # n sigma，channel shared
        self.conv = nn.Sequential(
            nn.Conv2d(n * 1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.residual = residual

    def forward(self, x):
        data = x
        x = x.unsqueeze(1)
        tmp = -((x - self.mu) / self.sigma) ** 2
        tmp = torch.mean(torch.exp(tmp), dim=2)
        fNeural = self.conv(tmp)
        if self.residual:
            return fNeural * data + data
        else:
            return fNeural * data


class FuzzySpatialAtt_inblock_channel(nn.Module):
    def __init__(self, channel, m, residual=True):
        """特征图通过各自的隶属度函数,拼接，得到空间的融合信息
        每一个隶属度函数只有一个sigma和mu，之后类似空间注意力形成一个多融合的空间注意力权重
        """
        super(FuzzySpatialAtt_inblock_channel, self).__init__()
        self.mu = nn.Parameter(torch.randn((m, channel, 1, 1)))  # n mu, channel independent
        self.sigma = nn.Parameter(torch.randn((m, channel, 1, 1)))  # n sigma, channel independent
        self.conv = nn.Sequential(
            nn.Conv2d(m * 1, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.residual = residual

    def forward(self, x):
        data = x
        x = x.unsqueeze(1)
        tmp = -((x - self.mu) / self.sigma) ** 2
        tmp = torch.mean(torch.exp(tmp), dim=2)  # opt: b,n,h,w
        fNeural = self.conv(tmp)
        if self.residual:
            return fNeural * data + data
        else:
            return fNeural * data


class FuzzyAttention(BaseModule):
    def __init__(self, channel, n, squeeze=4, adaptive=True, residual=True,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.ca = FuzzyChannelAtt_inblock_channel(channel, n, squeeze=squeeze, adaptive=adaptive, residual=residual)
        self.sa = FuzzySpatialAtt_inblock_channel(channel, n, residual=residual)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class FuzzyChannelAtt(nn.Module):
    def __init__(self, fusion_branch_cfg: Union[int, Sequence], info_channels=None, squeeze=4, adaptive=True,
                 residual=True):
        """不同的特征图通过各自的隶属度函数进行融合，再拼接，得到通道的融合信息
        每一个特征图只有一个sigma和mu，之后类似SE生成一个多融合的通道注意力权重

        :param fusion_branch_cfg: [c,c',...] 每一个特征图的通道数
        :param info_channels: 融合信息的通道数，权重向量输出长度
        :param adaptive: 是否采用自适应池化 todo adptive pool 或者torch.sum(tmp, dim=(2, 3))
        :return 一个输出长度为info_channels的通道权重向量
        """
        super(FuzzyChannelAtt, self).__init__()
        self.fusion_branch_cfg = fusion_branch_cfg
        self.residual = residual
        self.n = len(fusion_branch_cfg)
        self.c = sum(fusion_branch_cfg)
        info_channels = self.c if info_channels is None else info_channels
        self.feedforward = nn.Sequential(  # excitation router
            nn.Conv2d(self.c, self.c // squeeze * self.n, 1, bias=False),
            nn.BatchNorm2d(self.c // squeeze * self.n),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.c // squeeze * self.n, info_channels, 1),
            nn.Sigmoid()
        )
        self.mu = nn.Parameter(torch.randn(self.n))  # n mu
        self.sigma = nn.Parameter(torch.randn(self.n))  # n sigma
        self.adaptive = adaptive

    def forward(self, inputs):
        aggr_info = []
        for i, x in enumerate(inputs):
            tmp = -((x - self.mu[i]) / self.sigma[i]) ** 2
            if self.adaptive:
                tmp = F.adaptive_avg_pool2d(torch.exp(tmp), (1, 1))
            else:
                tmp = torch.mean(torch.exp(tmp), dim=(2, 3), keepdim=True)
            # b,c',1,1
            aggr_info.append(tmp)
        fNeural = self.feedforward(torch.concat(aggr_info, dim=1))  # b,csum,1,1
        priv = 0
        outputs = []
        for i, x in enumerate(inputs):
            tmp = fNeural[:, priv: priv + self.fusion_branch_cfg[i], ] * x
            if self.residual:
                outputs.append(tmp + x)
            else:
                outputs.append(tmp)
            priv += self.fusion_branch_cfg[i]
        return outputs


@DeprecationWarning
class FuzzySpatialAtt(nn.Module):
    def __init__(self, fusion_branch_cfg: Union[int, Sequence], simple=True,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True)
                 ):
        """不同的特征图通过各自的隶属度函数进行融合，再拼接，得到空间的融合信息
        每一个特征图只有一个sigma和mu，之后类似空间注意力形成一个多融合的空间注意力权重

        :param fusion_branch_cfg: [c, c', ...] 每一个特征图的通道数
        :param simple: 使用简单的上下采样和复杂的上下采样
        :return 一个输出大小为1*h*w的空间权重图
        """
        super(FuzzySpatialAtt, self).__init__()
        self.n = len(fusion_branch_cfg)
        self.mu = nn.Parameter(torch.randn(self.n))
        self.sigma = nn.Parameter(torch.randn(self.n))
        self.conv = nn.Sequential(
            nn.Conv2d(self.n * 1, self.n, 3, padding=1, groups=self.n),
            nn.BatchNorm2d(self.n),
            nn.Sigmoid()
        )

        if simple:
            # down sample
            self.downsample0 = nn.MaxPool2d(kernel_size=2)
            # up sample
            self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')
            self.downsample2 = nn.MaxPool2d(kernel_size=2)
            self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            # down sample
            self.downsample0 = ConvModule(
                in_channels=int(fusion_branch_cfg[0]),
                out_channels=int(fusion_branch_cfg[0]),
                kernel_size=3,
                stride=2,
                padding=3 // 2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            # up sample
            self.upsample0 = nn.ConvTranspose2d(
                in_channels=int(fusion_branch_cfg[0]),
                out_channels=int(fusion_branch_cfg[0]),
                kernel_size=2,
                stride=2,
                bias=True)
            self.downsample2 = ConvModule(
                in_channels=int(fusion_branch_cfg[-1]),
                out_channels=int(fusion_branch_cfg[-1]),
                kernel_size=3,
                stride=2,
                padding=3 // 2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            # up sample
            self.upsample2 = nn.ConvTranspose2d(
                in_channels=int(fusion_branch_cfg[-1]),
                out_channels=int(fusion_branch_cfg[-1]),
                kernel_size=2,
                stride=2,
                bias=True)

    def forward(self, inputs):
        # inputs is a list containing the feature maps with same height and width.
        # ipt[*80*64-40*40*128-20*20*256]
        tensors = [self.downsample0(inputs[0]), inputs[1], self.upsample2(inputs[2])]
        aggr_info = []
        for i, x in enumerate(tensors):
            tmp = -((x - self.mu[i]) / self.sigma[i]) ** 2
            tmp = torch.mean(torch.exp(tmp), dim=1, keepdim=True)
            aggr_info.append(tmp)
        fNeural = self.conv(torch.concat(aggr_info, dim=1))  # concat, c = len(inputs)
        outputs = []
        for i, x in enumerate(tensors):
            # i-channel of fNeural
            outputs.append(fNeural[:, i, ].unsqueeze(1) * x + x)
        outputs = [self.upsample0(outputs[0]), outputs[1], self.downsample2(outputs[2])]
        return outputs
