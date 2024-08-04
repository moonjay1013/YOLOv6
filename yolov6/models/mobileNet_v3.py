# 2022-10-12
from typing import Callable, List, Optional, Any, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial

# from yolov6.layers.common import SimSPPF


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # make sure that round down does not go down by more than 10%
    if new_ch < 0.8 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer(inplace=True))


class SqueezeExcitation(nn.Module):
    """
    SE通道注意力模块
    """

    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))  # (64, 1, 1)
        scale = self.fc1(scale)  # (16, 1, 1)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)  # (64, 1, 1)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class InvertedResidualConfig:
    def __init__(self,
                 input_c: int,
                 kernel: int,
                 expanded_c: int,
                 out_c: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 width_multi: float):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError('illegal stride value')

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(
                cnf.input_c,
                cnf.expanded_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            ))

        # depth wise
        layers.append(ConvBNActivation(
            cnf.expanded_c,
            cnf.expanded_c,
            kernel_size=cnf.kernel,
            stride=cnf.stride,
            groups=cnf.expanded_c,
            norm_layer=norm_layer,
            activation_layer=activation_layer
        ))

        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))

        layers.append(ConvBNActivation(
            cnf.expanded_c,
            cnf.out_c,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.Identity
        ))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_stride = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return result


class MobileNetV3(nn.Module):
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int,
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError('This inverted_residual_setting should not be empty')
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(inverted_residual_setting, List) for s in inverted_residual_setting])):
            raise ValueError('This inverted_residual_setting should be List[InvertedResidualConfig]')

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        first_conv_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3,
                                       first_conv_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        layers.append(ConvBNActivation(last_channel,
                                       last_channel,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        # building last serval layers
        # layers.append(SimSPPF(last_channel, last_channel))  # new: append SimSPPF

        self.features = nn.Sequential(*layers)

        # detection don't need
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Sequential(nn.Linear(last_conv_output_c, last_channel),
        #                                 nn.Hardswish(inplace=True),
        #                                 nn.Dropout(p=0.2, inplace=True),
        #                                 nn.Linear(last_channel, num_classes))

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> tuple[Any]:
        outputs = []

        # large
        # x = self.features[:5](x)
        # outputs.append(x)
        # x = self.features[5:8](x)
        # outputs.append(x)
        # x = self.features[8:-1](x)
        # outputs.append(x)

        # small
        x = self.features[:3](x)
        outputs.append(x)
        x = self.features[3:5](x)
        outputs.append(x)
        x = self.features[5:-1](x)
        outputs.append(x)

        # x = self.avg_pool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)

        return tuple(outputs)

    def forward(self, x: Tensor) -> tuple[Any]:
        return self._forward_impl(x)


def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    reduced_tail（bool）:If True， reduces the channel counts of all feature layers between C4 and C5 by 2.
                        It is used to reduce the channel redundancy in the backbone for Detection
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(32, 3, 32, 32, False, 'RE', 1),
        bneck_conf(32, 3, 64, 32, False, 'RE', 2),  # C1
        bneck_conf(32, 3, 128, 64, False, 'RE', 1),
        bneck_conf(64, 5, 200, 128, True, 'RE', 2),  # C2   80x80
        bneck_conf(128, 5, 200, 128, True, 'RE', 1),
        bneck_conf(128, 5, 240, 128, True, 'RE', 1),
        bneck_conf(128, 3, 400, 256, False, 'HS', 2),  # C3  40x40
        bneck_conf(256, 3, 480, 256, False, 'HS', 1),
        bneck_conf(256, 3, 480, 256, False, 'HS', 1),
        bneck_conf(256, 3, 480, 256, False, 'HS', 1),
        bneck_conf(256, 3, 672, 512, True, 'HS', 1),
        bneck_conf(512, 3, 672, 512, True, 'HS', 1),
        bneck_conf(512, 5, 672, 512 // reduce_divider, True, 'HS', 2),  # C4  20x20
        bneck_conf(512 // reduce_divider, 5, 960 // reduce_divider, 512 // reduce_divider, True, 'HS', 1),
        bneck_conf(512 // reduce_divider, 5, 960 // reduce_divider, 512 // reduce_divider, True, 'HS', 1),
    ]

    last_channel = adjust_channels(512 // reduce_divider)  # C5
    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    """
    reduced_tail（bool）:If True， reduces the channel counts of all feature layers between C4 and C5 by 2.
                        It is used to reduce the channel redundancy in the backbone for Detection
    """
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        # bneck_conf(64, 3, 64, 64, True, 'RE', 2),  # C1
        # bneck_conf(64, 3, 128, 128, False, 'RE', 2),  # C2
        # bneck_conf(128, 3, 240, 128, False, 'RE', 1),
        # bneck_conf(128, 5, 400, 256, True, 'HS', 2),  # C3
        # bneck_conf(256, 5, 400, 256, True, 'HS', 1),
        # bneck_conf(256, 5, 400, 256, True, 'HS', 1),
        # bneck_conf(256, 5, 480, 256, True, 'HS', 1),
        # bneck_conf(256, 5, 480, 256, True, 'HS', 1),
        # bneck_conf(256, 5, 672, 512 // reduce_divider, True, 'HS', 2),  # C4
        # bneck_conf(512 // reduce_divider, 5, 672 // reduce_divider, 512 // reduce_divider, True, 'HS', 1),
        # bneck_conf(512 // reduce_divider, 5, 672 // reduce_divider, 512 // reduce_divider, True, 'HS', 1),

        # v6_s
        # bneck_conf(32, 3, 32, 32, True, 'RE', 2),  # C1
        # bneck_conf(32, 3, 128, 128, False, 'RE', 2),  # C2
        # bneck_conf(128, 3, 200, 128, False, 'RE', 1),
        # bneck_conf(128, 5, 320, 256, True, 'HS', 2),  # C3
        # bneck_conf(256, 5, 320, 256, True, 'HS', 1),
        # bneck_conf(256, 5, 400, 256, True, 'HS', 1),
        # bneck_conf(256, 5, 400, 256, True, 'HS', 1),
        # bneck_conf(256, 5, 400, 256, True, 'HS', 1),
        # bneck_conf(256, 5, 608, 512 // reduce_divider, True, 'HS', 2),  # C4
        # bneck_conf(512 // reduce_divider, 5, 608 // reduce_divider, 512 // reduce_divider, True, 'HS', 1),
        # bneck_conf(512 // reduce_divider, 5, 608 // reduce_divider, 512 // reduce_divider, True, 'HS', 1),

        # v6_t
        bneck_conf(24, 3, 48, 24, True, 'RE', 2),  # C1
        bneck_conf(24, 3, 192, 96, False, 'RE', 2),  # C2
        bneck_conf(96, 3, 192, 96, False, 'RE', 1),
        bneck_conf(96, 5, 384, 192, True, 'HS', 2),  # C3
        bneck_conf(192, 5, 384, 192, True, 'HS', 1),
        bneck_conf(192, 5, 384, 192, True, 'HS', 1),
        bneck_conf(192, 5, 384, 192, True, 'HS', 1),
        bneck_conf(192, 5, 768, 384, True, 'HS', 1),
        bneck_conf(384, 5, 768, 384 // reduce_divider, True, 'HS', 2),  # C4
        bneck_conf(384 // reduce_divider, 5, 768 // reduce_divider, 384 // reduce_divider, True, 'HS', 1),
        bneck_conf(384 // reduce_divider, 5, 768 // reduce_divider, 384 // reduce_divider, True, 'HS', 1),

        # v6_t_few_channel
        # bneck_conf(24, 3, 48, 24, True, 'RE', 2),  # C1
        # bneck_conf(24, 3, 128, 96, False, 'RE', 2),  # C2
        # bneck_conf(96, 3, 128, 96, False, 'RE', 1),
        # bneck_conf(96, 5, 240, 192, True, 'HS', 2),  # C3
        # bneck_conf(192, 5, 240, 192, True, 'HS', 1),
        # bneck_conf(192, 5, 240, 192, True, 'HS', 1),
        # bneck_conf(192, 5, 240, 192, True, 'HS', 1),
        # bneck_conf(192, 5, 400, 384, True, 'HS', 1),
        # bneck_conf(384, 5, 400, 384 // reduce_divider, True, 'HS', 2),  # C4
        # bneck_conf(384 // reduce_divider, 5, 400 // reduce_divider, 384 // reduce_divider, True, 'HS', 1),
        # bneck_conf(384 // reduce_divider, 5, 400 // reduce_divider, 384 // reduce_divider, True, 'HS', 1),
    ]

    last_channel = adjust_channels(384 // reduce_divider)  # C5
    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


if __name__ == '__main__':
    input_x = torch.randn((1, 3, 640, 640))

    model = mobilenet_v3_small(num_classes=80, reduced_tail=False)

    print(model)

    out_tup = model(input_x)
    print(out_tup[0].shape)
    print(out_tup[1].shape)
    print(out_tup[2].shape)
