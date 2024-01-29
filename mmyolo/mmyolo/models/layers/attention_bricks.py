# Jiaxiong Yang reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType
from mmengine.model import BaseModule
from torch.nn import init
from torchvision.models.convnext import LayerNorm2d
import mmdet.models.layers.se_layer as se_layer
from mmyolo.models.utils import make_divisible
from mmyolo.registry import MODELS

from mmcv.cnn.bricks.non_local import NonLocal2d

__all__ = ['NonLocal2d', 'SELayer',  # reload
           'BottleNeckAttention', 'DilationSpatialAttention', 'FullyAttentionalBlock', 'GlobalContext', 'SPALayer',
           'PSABlock', 'CnAM',
           'EfficientMSA', 'MultiScaleAttention',  # from fightingcv_attention
           'GatherExcite', 'SKAttention'  # from timm
           ]


@MODELS.register_module()
class SELayer(se_layer.SELayer):
    """See SELayer in mmdetection
    """

    def __init__(self,
                 in_channels: int,  # keep same name
                 ratio: int = 16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid')),
                 init_cfg=None) -> None:
        super(SELayer, self).__init__(in_channels, ratio, conv_cfg, act_cfg, init_cfg)


@MODELS.register_module()
class SPALayer(BaseModule):
    """SPATIAL PYRAMID ATTENTION\n
     Spatial Pyramid Attention Network for Enhanced Image Recognition
     https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9102906

    """

    def __init__(self, in_channels, channel=None, reduction=16):
        """
        Args:
            channel: check if channel is same as in_channels to conduct filter
        """
        super(SPALayer, self).__init__()
        if channel is None:
            channel = in_channels
        if in_channels != channel:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, channel, kernel_size=1, stride=1, bias=False),  # noqa
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            )
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Linear(channel * 21, channel * 21 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 21 // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x) if hasattr(self, 'conv1') else x  # 检查是否需要维度对齐
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x).view(b, c)
        y2 = self.avg_pool2(x).view(b, 4 * c)  # from 2*2 -> 4*1
        y3 = self.avg_pool4(x).view(b, 16 * c)  # from 4*4 -> 16*1
        y = torch.cat((y1, y2, y3), 1)  # (1+4+16=21)
        y = self.fc(y)
        b, out_channel = y.size()
        y = y.view(b, out_channel, 1, 1)
        return x * y


@MODELS.register_module()
class GlobalContext(BaseModule):
    """`GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond`
     https://arxiv.org/abs/1904.11492 \n
     Origin from Timm
    """

    def __init__(self, in_channels, use_attn=True, fuse_mode='add', rd_ratio=1. / 8, act_layer=nn.ReLU):
        super(GlobalContext, self).__init__()
        reduce_channels = make_divisible(in_channels, rd_ratio, divisor=1)

        self.conv_attn = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True) if use_attn else None  # noqa

        self.mlp = ConvMlp(in_channels, reduce_channels, act_layer=act_layer, norm_layer=LayerNorm2d)

        self.fuse_mode = fuse_mode
        if fuse_mode == 'add':
            self.gate = nn.Sigmoid()
        elif fuse_mode == 'scale':
            self.gate = nn.Identity()
        else:
            raise Exception('没有这个mode')

        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(self.conv_attn.weight, mode='fan_in', nonlinearity='relu')
        if self.fuse_mode == 'add':
            nn.init.zeros_(self.mlp.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn  # 将注意力加载上去
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)

        mlp_x = self.gate(self.mlp(context))

        if self.fuse_mode == 'scale':
            x = x * mlp_x
        elif self.fuse_mode == 'add':
            x = x + mlp_x

        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """

    def __init__(
            self,
            in_channels,
            hidden_channels=None,
            out_channels=None,
            act_layer=nn.ReLU,
            norm_layer=None,
            drop=0.,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels

        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)  # noqa
        self.norm = norm_layer(hidden_channels) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True)  # noqa

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


@MODELS.register_module()
class GatherExcite(BaseModule):
    """GENet from timm"""

    def __init__(self, in_channels, feat_size=None, extra_params=False, extent=0, use_mlp=True, rd_ratio=1. / 16):
        super().__init__(init_cfg=None)
        try:
            from timm.layers.gather_excite import GatherExcite
            self.model = GatherExcite(in_channels, feat_size, extra_params, extent, use_mlp, rd_ratio)
        except ImportError as e:
            print(e)
            pass  # module doesn't exist, deal with it.

    def forward(self, x):
        return self.model(x)


@MODELS.register_module()
class SKAttention(BaseModule):
    """Selective Kernel Convolution Module  \n
    https://arxiv.org/abs/1903.06586
    """

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 stride=1,
                 dilation=1,
                 rd_ratio=1. / 16,
                 keep_3x3=True,
                 split_input=True,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True)):
        super(SKAttention, self).__init__(init_cfg=None)
        # 使用xxxbuild方法都是生成对应的实例对象，和所谓的类名传递不同
        # 这个方法是ConvModule给出的，其实现了细粒度的划分
        # from mmcv.cnn.bricks.norm import build_norm_layer
        # from mmcv.cnn.bricks.activation import build_activation_layer
        # bn = build_norm_layer(norm_cfg, in_channels)
        # ac = build_activation_layer(act_cfg)
        # 这个方法是MMEngine给出的，需要关注变量
        # bn = MODELS.build(norm_cfg) or None
        # ac = MODELS.build(act_cfg) or None
        if norm_cfg['type'] == 'BN':
            bn = nn.BatchNorm2d
        elif norm_cfg['type'] == 'LN':
            bn = nn.LayerNorm
        else:
            bn = nn.SyncBatchNorm

        if act_cfg['type'] == 'ReLU':
            ac = nn.ReLU
        elif act_cfg['type'] == 'SiLU':
            ac = nn.SiLU
        else:
            ac = nn.GELU

        try:
            from timm.layers.selective_kernel import SelectiveKernel
            self.model = SelectiveKernel(in_channels, out_channels,
                                         stride=stride, dilation=dilation, rd_ratio=rd_ratio,
                                         keep_3x3=keep_3x3, split_input=split_input,
                                         norm_layer=bn, act_layer=ac)
        except ImportError as e:
            print(e)
            pass  # module doesn't exist, deal with it.

    def forward(self, x):
        return self.model(x)


class SelfAttentionLikeModule(BaseModule):
    """针对transformer类型的注意力（针对[b,n,c]维度）进行外层封装
    """

    def __init__(self, init_cfg=None):
        super(SelfAttentionLikeModule, self).__init__(init_cfg=init_cfg)
        self.model = None

    def forward(self, x, attention_mask=None, attention_weights=None):  # self attn
        b, _, h, w = x.size()
        x = x.reshape(b, -1, h * w).permute(0, 2, 1).contiguous()
        out = self.model(x, x, x, attention_mask, attention_weights)
        return out.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()


@MODELS.register_module(['EMSA', 'EfficientMSA'])
class EfficientMSA(SelfAttentionLikeModule):
    """Efficiint Multi-head Self-Attention \n
    https://arxiv.org/abs/2105.13677
    \n fightingcv_attention implementation
    """

    def __init__(self, in_channels, d_k=None, d_v=None, head=8,
                 dropout=.1, H=7, W=7, ratio=3, apply_transform=True):
        super(EfficientMSA, self).__init__(init_cfg=None)
        d_model = in_channels  # 注意这里实际上是这个名字
        d_k = d_k if d_k is not None else in_channels
        d_v = d_v if d_v is not None else in_channels
        try:
            from fightingcv_attention.attention.EMSA import EMSA
            self.model = EMSA(d_model, d_k, d_v, head, dropout, H, W, ratio, apply_transform)
        except ImportError as e:
            print(e)
            pass  # module doesn't exist, deal with it.


@MODELS.register_module()
class MultiScaleAttention(SelfAttentionLikeModule):
    """Parallel Multi-Scale Attention for Sequence to Sequence Learning \n
    https://arxiv.org/abs/1911.09483
    """

    def __init__(self, in_channels, d_k=None, d_v=None,
                 head=8, dropout=.1):
        super(MultiScaleAttention, self).__init__(init_cfg=None)
        d_model = in_channels  # 注意这里实际上是这个名字
        d_k = d_k if d_k is not None else in_channels
        d_v = d_v if d_v is not None else in_channels
        try:
            from fightingcv_attention.attention.MUSEAttention import MUSEAttention
            self.model = MUSEAttention(d_model, d_k, d_v, head, dropout)
        except ImportError as e:
            print(e)
            pass  # module doesn't exist, deal with it.


@MODELS.register_module()
class DilationSpatialAttention(BaseModule):
    """the Spatial Attention in BAM: Bottleneck Attention Module
    """

    def __init__(self, in_channel, re_channel=None,
                 rd_ratio=1. / 16,
                 num_layers=3,
                 dilation=2,
                 sigmoid=True,  # 对于BottleNeckAttention是最后一起sigmoid，需要设置为False，如果单独使用就需要开启sigmoid
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True)):
        super(DilationSpatialAttention, self).__init__()
        self.is_sigmoid = sigmoid
        self.in_channel = in_channel
        self.reduction = rd_ratio
        self.rd_channel = make_divisible(in_channel, rd_ratio, divisor=1) if re_channel is None else re_channel
        self.intro = ConvModule(in_channel, self.rd_channel, kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.dilation_convs = nn.Sequential(
            *[ConvModule(self.rd_channel, self.rd_channel,
                         kernel_size=3, dilation=dilation, padding=2,  #
                         norm_cfg=norm_cfg, act_cfg=act_cfg)
              for _ in range(num_layers)]
        )
        self.ending = ConvModule(self.rd_channel, 1, kernel_size=1, norm_cfg=None, act_cfg=None)

    def forward(self, in_x):
        x = self.intro(in_x)
        x = self.dilation_convs(x)
        x = self.ending(x)
        x = x.expand_as(in_x)
        if self.is_sigmoid:
            x = F.sigmoid(x)
        return x


@MODELS.register_module()
class BottleNeckAttention(BaseModule):
    """BAM: Bottleneck Attention Module \n
    https://arxiv.org/pdf/1807.06514.pdf
    """

    def __init__(self, in_channels,
                 rd_ratio=1. / 16,
                 num_layers=3,
                 dilation=2,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super(BottleNeckAttention, self).__init__(init_cfg=init_cfg)
        self.reduction = rd_ratio
        self.rd_channel = make_divisible(in_channels, rd_ratio, divisor=1)

        self.channels = [in_channels] + [self.rd_channel] * (num_layers - 1) + [in_channels]  # [c,r,r,c]
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            *[
                nn.Sequential(
                    nn.Linear(self.channels[i], self.channels[i + 1]),
                    # MODELS.build(norm_cfg, features=self.channels[i + 1]),  无法传入features
                    # nn.BatchNorm1d(self.channels[i + 1]),
                    MODELS.build(act_cfg)
                ) for i in range(num_layers - 1)  # cr,rr
            ],
            nn.Linear(self.channels[-2], self.channels[-1])  # rc
        )

        self.spatial_attn = DilationSpatialAttention(in_channels, self.rd_channel,
                                                     # 对于BottleNeckAttn来说需要能够完全控制spatial_attn
                                                     rd_ratio, num_layers, dilation, sigmoid=False,
                                                     norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):  # todo 是否可以直接这样初始化，越过BaseModel
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        ca = self.channel_attn(x).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        sa = self.spatial_attn(x)
        weight = self.sigmoid(ca + sa)
        out = (1 + weight) * x
        return out


@MODELS.register_module()
class PSABlock(BaseModule):
    """EPSANet: An Efficient Pyramid Split Attention Block on Convolutional Neural Network  \n
    https://arxiv.org/pdf/2105.14447.pdf
    """

    def __init__(self, in_channels, rd_ratio=1. / 16, split=4, share_se=False,  # share_se 是否共用SE操作
                 norm_cfg: ConfigType = None,
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super(PSABlock, self).__init__(init_cfg=init_cfg)
        self.split = split
        self.share_se = share_se
        self.rd_channel = make_divisible(in_channels, rd_ratio)

        self.multi_scale_conv = nn.ModuleList([  # [3, 5, 7, 9]
            ConvModule(in_channels // split, in_channels // split, kernel_size=2 * (i + 1) + 1, padding=i + 1,
                       act_cfg=act_cfg, norm_cfg=norm_cfg)
            for i in range(split)
        ])

        if share_se:
            split = 1
        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(in_channels // self.split, self.rd_channel // self.split, kernel_size=1, bias=False,
                           norm_cfg=None, act_cfg=act_cfg),
                ConvModule(self.rd_channel // self.split, in_channels // self.split, kernel_size=1, bias=False,
                           norm_cfg=None, act_cfg=None),
                nn.Sigmoid()
            ) for _ in range(split)
        ])

        self.softmax = nn.Softmax(dim=1)  # dim=0是对每一维度相同的位置进行softmax运算；dim=1是对列；dim=2或-1是对行；

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, self.split, c // self.split, h, w)  # bs,s,ci,h,w

        multi_scale_feat = [conv(x[:, i, ...]) for i, conv in enumerate(self.multi_scale_conv)]  # 获取划分通道的多尺度特征（空间）

        from itertools import zip_longest
        se_feat = [se(feat) for feat, se in zip_longest(multi_scale_feat, self.se_blocks, fillvalue=self.se_blocks[0])]
        se_feat = torch.stack(se_feat, dim=1)  # 得到每个不同尺度上的通道注意力
        # se_feat = torch.concat(se_feat, dim=1)
        se_feat = se_feat.expand_as(x)  # todo why

        att_weight = self.softmax(se_feat)  # softmax对多尺度通道注意力向量进行特征重新校准，得到新的多尺度通道交互后的注意力权重

        # multi_scale_feat = torch.concat(multi_scale_feat, dim=1)
        multi_scale_feat = torch.stack(multi_scale_feat, dim=1)
        out = multi_scale_feat * att_weight
        out = out.view(b, -1, h, w)
        return out


@MODELS.register_module()
class FullyAttentionalBlock(BaseModule):
    """Fully Attentional Network for Semantic Segmentation \n
    https://arxiv.org/abs/2112.04108 \n
    改进Nonlocal，降低计算量，同时考虑通道和空间， https://blog.csdn.net/m0_61899108/article/details/126086217
    自始至终W和H都是没有拼接的，也就是论文中的C,S,H+W是没有出现的，实现上C,-1,H 和C,-1,W分别
    """

    def __init__(self, in_channels, gamma=0.,
                 is_concat=False,
                 reduction_ratio=1. / 4,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 init_cfg=None):
        self.is_concat = is_concat
        super(FullyAttentionalBlock, self).__init__(init_cfg)
        if reduction_ratio is not None:  # 首先要进行降维
            self.duct = True
            plane = make_divisible(in_channels, reduction_ratio)
            self.reduce_conv = nn.Sequential(
                ConvModule(in_channels, plane, 3, bias=False, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                ConvModule(plane, plane, 3, bias=False, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            )
        else:
            self.duct = False
            plane = in_channels

        self.conv1 = nn.Linear(plane, plane)  # keep channel
        self.conv2 = nn.Linear(plane, plane)

        if is_concat:
            self.conv = ConvModule(plane * 2, in_channels, 3, stride=1, padding=1, bias=False,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.conv = ConvModule(plane, in_channels, 3, stride=1, padding=1, bias=False,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.softmax = nn.Softmax(dim=-1)
        # torch.zeros(1)
        self.gamma = nn.Parameter(torch.tensor([gamma]))  # 可学习参数scale，初始化为0， 初始化为1,0.5会不会好一点？？

    def forward(self, x):
        x = self.reduce_conv(x) if self.duct else x
        b, _, h, w = x.size()

        feat_h = x.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # h方向进行attn  -1=c
        feat_w = x.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)  # w方向进行attn

        # pool的作用就是抽象语义
        encode_h = self.conv1(F.avg_pool2d(x, [1, w]).view(b, -1, h).permute(0, 2, 1).contiguous())  # b,h,-1
        encode_w = self.conv2(F.avg_pool2d(x, [h, 1]).view(b, -1, w).permute(0, 2, 1).contiguous())  # b,w,-1

        # 第一维重复w次，其他两维度不变  [b,h,-1] -> [b*w,h,-1]
        energy_h = torch.matmul(feat_h, encode_h.repeat(w, 1, 1))  # [b*w,h,-1] * [bw,-1,h]  -> [bw,-1,-1]
        full_relation_h = self.softmax(energy_h)  # [b*w, -1, -1]  得到attn weights 实际上是affinity的过程
        # [bw,-1,-1] * [bw,-1,h] -> [bw,-1,h]  ->  view拆解乘法，permute交换
        full_aug_h = torch.bmm(full_relation_h, feat_h).view(b, w, -1, h).permute(0, 2, 3, 1)  # b,c,h,w

        energy_w = torch.matmul(feat_w, encode_w.repeat(h, 1, 1))  # [b*h,w,-1] * [bh,-1,w] -> [bh,-1,-1]
        full_relation_w = self.softmax(energy_w)  # [b*w, -1, -1]  得到attn weights  实际上是affinity的过程
        full_aug_w = torch.bmm(full_relation_w, feat_w).view(b, h, -1, w).permute(0, 2, 1, 3)  # b,c,h,w

        if self.is_concat:  # 论文中是Concat了
            out = self.gamma * torch.concat((full_aug_h, full_aug_w), dim=1) + x
        else:  # 注意这里可以相加的原因是w和h相同
            out = self.gamma * (full_aug_h + full_aug_w) + x
        out = self.conv(out)
        return out


@MODELS.register_module()
class CnAM(BaseModule):
    def __init__(self, lr_channels, hr_channels, gamma=0.,
                 reduction_ratio=4,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super(CnAM, self).__init__(init_cfg=init_cfg)
        self.downsample = ConvModule(
            in_channels=lr_channels,
            out_channels=lr_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.conv_in_NC = ConvModule(
            in_channels=lr_channels,
            out_channels=lr_channels // reduction_ratio,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.conv_in_CN = ConvModule(
            in_channels=lr_channels,
            out_channels=lr_channels // reduction_ratio,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )
        self.conv_straight = ConvModule(
            in_channels=hr_channels,
            out_channels=hr_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, lr, hr):
        lr = self.downsample(lr)
        b, _, h, w = lr.shape
        hr = self.conv_straight(hr)
        NC = self.conv_in_NC(lr).view(b, -1, h * w).permute(0, 2, 1)  # b n c
        CN = self.conv_in_CN(lr).view(b, -1, h * w)  # b c n
        # todo 在浅层使用hw*hw实在是太大了，所以在这里进行了下采样
        context = torch.bmm(NC, CN).view(b, -1, h, w)  # b n h w
        avg_out = torch.mean(context, dim=1, keepdim=True)  # b 1 h w
        attention_weight = F.sigmoid(avg_out)
        out = hr * attention_weight
        return out
