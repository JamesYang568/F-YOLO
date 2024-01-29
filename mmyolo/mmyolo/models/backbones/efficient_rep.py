# Jiaxiong Yang reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import BaseModule

from mmyolo.models.layers.yolo_bricks import SPPFBottleneck
from mmyolo.registry import MODELS
from ..layers import BepC3StageBlock, RepStageBlock
from ..layers.my_bricks import GhostModule
from ..utils import make_round
from .base_backbone import BaseBackbone


@MODELS.register_module()
class YOLOv6EfficientRep(BaseBackbone):
    """EfficientRep backbone used in YOLOv6.
    Args:
        arch (str): Architecture of BaseDarknet, from {P5, P6}.
            Defaults to P5.
        plugins (list[dict]): List of plugins for stages, each dict contains:
            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels (int): Number of input image channels. Defaults to 3.
        out_indices (Tuple[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        block_cfg (dict): Config dict for the block used to build each
            layer. Defaults to dict(type='RepVGGBlock').
        init_cfg (Union[dict, list[dict]], optional): Initialization config
            dict. Defaults to None.
    Example:
        >>> from mmyolo.models import YOLOv6EfficientRep
        >>> import torch
        >>> model = YOLOv6EfficientRep()
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, use_spp
    arch_settings = {
        'P5': [[64, 128, 6, False], [128, 256, 12, False],
               [256, 512, 18, False], [512, 1024, 6, True]]
    }

    def __init__(self,
                 arch: str = 'P5',
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 norm_eval: bool = False,
                 block_cfg: ConfigType = dict(type='RepVGGBlock'),
                 init_cfg: OptMultiConfig = None):
        self.block_cfg = block_cfg
        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""

        block_cfg = self.block_cfg.copy()
        block_cfg.update(
            dict(
                in_channels=self.input_channels,
                out_channels=int(self.arch_setting[0][0] * self.widen_factor),
                kernel_size=3,
                stride=2,
            ))
        return MODELS.build(block_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, use_spp = setting

        in_channels = int(in_channels * self.widen_factor)
        out_channels = int(out_channels * self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)

        rep_stage_block = RepStageBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            block_cfg=self.block_cfg,
        )

        block_cfg = self.block_cfg.copy()
        block_cfg.update(
            dict(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2))
        stage = []

        ef_block = nn.Sequential(MODELS.build(block_cfg), rep_stage_block)

        stage.append(ef_block)

        if use_spp:
            spp = SPPFBottleneck(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        return stage

    def init_weights(self):
        if self.init_cfg is None:
            """Initialize the parameters."""
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()


@MODELS.register_module()
class YOLOv6CSPBep(YOLOv6EfficientRep):
    """CSPBep backbone used in YOLOv6.
    Args:
        arch (str): Architecture of BaseDarknet, from {P5, P6}.
            Defaults to P5.
        plugins (list[dict]): List of plugins for stages, each dict contains:
            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels (int): Number of input image channels. Defaults to 3.
        out_indices (Tuple[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        block_cfg (dict): Config dict for the block used to build each
            layer. Defaults to dict(type='RepVGGBlock').
        block_act_cfg (dict): Config dict for activation layer used in each
            stage. Defaults to dict(type='SiLU', inplace=True).
        init_cfg (Union[dict, list[dict]], optional): Initialization config
            dict. Defaults to None.
    Example:
        >>> from mmyolo.models import YOLOv6CSPBep
        >>> import torch
        >>> model = YOLOv6CSPBep()
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, use_spp
    arch_settings = {
        'P5': [[64, 128, 6, False], [128, 256, 12, False],
               [256, 512, 18, False], [512, 1024, 6, True]]
    }

    def __init__(self,
                 arch: str = 'P5',
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 hidden_ratio: float = 0.5,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 block_cfg: ConfigType = dict(type='ConvWrapper'),
                 init_cfg: OptMultiConfig = None):
        self.hidden_ratio = hidden_ratio
        super().__init__(
            arch=arch,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            block_cfg=block_cfg,
            init_cfg=init_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, use_spp = setting
        in_channels = int(in_channels * self.widen_factor)
        out_channels = int(out_channels * self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)

        rep_stage_block = BepC3StageBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            hidden_ratio=self.hidden_ratio,
            block_cfg=self.block_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        block_cfg = self.block_cfg.copy()
        block_cfg.update(
            dict(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2))
        stage = []

        ef_block = nn.Sequential(MODELS.build(block_cfg), rep_stage_block)

        stage.append(ef_block)

        if use_spp:
            spp = SPPFBottleneck(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        return stage


@MODELS.register_module()
class YOLOv6EfficientRepExt(YOLOv6EfficientRep):
    # From left to right:
    # in_channels, out_channels, num_blocks, use_spp
    arch_settings = {
        'P5': [[64, 128, 6, False], [128, 256, 12, False],
               [256, 512, 18, False], [512, 1024, 6, False]]  # todo
    }

    def __init__(self,
                 arch: str = 'P5',
                 arch_settings=None,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 norm_eval: bool = False,
                 block_cfg: ConfigType = dict(type='RepVGGBlock'),
                 deep_enhance_cfg: ConfigType = None,
                 init_cfg: OptMultiConfig = None):
        if arch_settings is not None:  # todo important
            self.arch_settings = arch_settings
        self.deep_enhance_cfg = deep_enhance_cfg
        super().__init__(
            arch,
            plugins,
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            block_cfg=block_cfg,
            init_cfg=init_cfg)
        if self.deep_enhance_cfg is not None and not self.arch_settings[arch][-1][-1]:  # extra block in the tail
            enhance = MODELS.build(self.deep_enhance_cfg)
            self.add_module(f'additional', enhance)
            self.layers.append(f'additional')

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, use_spp_like = setting
        in_channels = int(in_channels * self.widen_factor)
        out_channels = int(out_channels * self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)

        rep_stage_block = RepStageBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            block_cfg=self.block_cfg,
        )

        block_cfg = self.block_cfg.copy()
        block_cfg.update(
            dict(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2))
        stage = []

        ef_block = nn.Sequential(MODELS.build(block_cfg), rep_stage_block)

        stage.append(ef_block)

        if use_spp_like and self.deep_enhance_cfg is None:
            spp = SPPFBottleneck(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        elif use_spp_like and self.deep_enhance_cfg is not None:
            enhance = MODELS.build(self.deep_enhance_cfg)
            stage.append(enhance)
        else:
            pass
        return stage


@DeprecationWarning
@MODELS.register_module()
class Crossover_Ghost_serial(BaseModule):
    def __init__(self, in_channels, reduction=1, layer_num=3, dilation: list = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True)):
        super(Crossover_Ghost_serial, self).__init__()
        self.plane = in_channels // reduction
        self.reduction = reduction
        if reduction != 1:
            self.condense = GhostModule(in_channels, self.plane, act_cfg=act_cfg, norm_cfg=norm_cfg)
            self.expand = ConvModule(self.plane, in_channels, kernel_size=3,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dilation = [1] * layer_num if dilation is None else dilation
        assert len(self.dilation) == layer_num, "dilation should have the same length with branches"
        self.conv = nn.ModuleList(
            [GhostModule(self.plane, self.plane, dilation=d, act_cfg=act_cfg, norm_cfg=norm_cfg) for d in self.dilation]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.reduction != 1:
            x = self.condense(x)
        t = torch.ones((x.size(0), 1, x.size(2), x.size(3)))
        t = t.cuda() if torch.cuda.is_available() else t  # todo 如果计算模型参数请注释这句话
        for conv in self.conv:
            x = conv(x) * t
            t = self.sigmoid(torch.mean(x, dim=1, keepdim=True))
        if self.reduction != 1:
            x = self.expand(x)
        return x


@MODELS.register_module()
class GlobalFusionCalibration(BaseModule):
    """改进Crossover_Ghost_serial"""

    def __init__(self, in_channels, dilation: list = None, reduction=4,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True)):
        super(GlobalFusionCalibration, self).__init__()
        if dilation is None:
            dilation = [1, 2, 2, 2]
        self.convs = nn.ModuleList(
            [GhostModule(in_channels, in_channels,
                         ratio=reduction,
                         dilation=d,
                         act_cfg=act_cfg,
                         norm_cfg=norm_cfg)
             for d in dilation]
        )
        self.integrate = ConvModule(in_channels * len(dilation), in_channels, kernel_size=1, padding=0,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        t = torch.ones((x.size(0), 1, x.size(2), x.size(3)))
        t = t.cuda() if torch.cuda.is_available() else t  # todo 如果计算模型参数请注释这句话
        outs = []
        for conv in self.convs:
            x = conv(x) * t
            outs.append(x)
            t = self.sigmoid(torch.mean(x, dim=1, keepdim=True))
        output = self.integrate(torch.cat(outs, dim=1))
        return output


@DeprecationWarning
@MODELS.register_module()
class Crossover_Ghost_parallel(BaseModule):
    def __init__(self, in_channels, branches=4, dilation: list = None,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True)):
        super(Crossover_Ghost_parallel, self).__init__()
        self.plane = in_channels // branches
        self.condense = GhostModule(in_channels, self.plane, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.dilation = [1] * branches if dilation is None else dilation
        assert len(self.dilation) == branches, "dilation should have the same length with branches"
        self.conv = nn.ModuleList(
            [GhostModule(self.plane, self.plane, dilation=d, act_cfg=act_cfg, norm_cfg=norm_cfg) for d in self.dilation]
        )
        self.integrate = ConvModule(self.plane * branches, in_channels, kernel_size=3,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)  # 四个拼接成1个
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        trace = []
        x = self.condense(x)  # c/branches
        t = torch.ones((x.size(0), 1, x.size(2), x.size(3)))
        t = t.cuda() if torch.cuda.is_available() else t  # todo 如果计算模型参数请注释这句话
        for conv in self.conv:
            a = conv(x) * t
            trace.append(a)
            t = self.sigmoid(torch.mean(a, dim=1, keepdim=True))
        x = torch.cat(trace, dim=1)
        return self.integrate(x)


@DeprecationWarning
@MODELS.register_module()
class Pileup_Ghost(BaseModule):
    def __init__(self, in_channels, courses=3, reduction=1, dilation: list = None, add=True,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True)
                 ):
        super(Pileup_Ghost, self).__init__()
        self.plane = in_channels // reduction
        self.reduction = reduction
        self.add = add
        if reduction != 1:
            self.condense = GhostModule(in_channels, self.plane, act_cfg=act_cfg, norm_cfg=norm_cfg)
            self.expand = ConvModule(self.plane, in_channels, kernel_size=3,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dilation = [1] * courses if dilation is None else dilation
        assert len(self.dilation) == courses, "dilation should have the same length with branches"
        if not add:
            assert reduction == courses, "courses need to be same as reduction"  # 降维必须要和分支数一致，拼接才能对齐
            t = self.plane
            convs = []
            for d in self.dilation:
                convs.append(GhostModule(t, self.plane, dilation=d, act_cfg=act_cfg, norm_cfg=norm_cfg))  # 每一次拼接通道数都会增加
                t += self.plane
            self.convs = nn.ModuleList(convs)
            self.expand = ConvModule(self.plane * (courses + 1), in_channels, kernel_size=3,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.convs = nn.ModuleList(
                [GhostModule(self.plane, self.plane, dilation=d, act_cfg=act_cfg, norm_cfg=norm_cfg) for d in
                 self.dilation]
            )

    def forward(self, x):
        if self.reduction != 1:
            x = self.condense(x)
        for conv in self.convs:
            if self.add:
                x = conv(x) + x
            else:
                t = conv(x)
                x = torch.concat([t, x], dim=1)
        if self.reduction != 1:
            x = self.expand(x)
        return x
