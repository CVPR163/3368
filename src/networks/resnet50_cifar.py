import torch
from torch import Tensor
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from collections import OrderedDict

__all__ = ['ResNetBottleneck', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet50_cifar', 'resnet101_cifar',
           'resnet152', 'resnext50_32x4d_cifar', 'resnext101_32x8d_cifar',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d_cifar': 'https://download.pytorch.org/models/resnext50_32x4d_cifar-7cdf4587.pth',
    'resnext101_32x8d_cifar': 'https://download.pytorch.org/models/resnext101_32x8d_cifar-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNetBottleneck V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv1x1(inplanes, width)
        # self.bn1 = norm_layer(width)
        self.conv1_layers = [
            # The 1-st passage for Conv1.
            ("conv1_new_1_1", nn.Conv2d(inplanes, width, 1, stride=1, padding=0, bias=False, groups=4)),
            ("bn1_new_1_1", norm_layer(width)),
            # The 2-nd passage for Conv1.
            ("conv1_new_2_1", nn.Conv2d(inplanes, width, 1, stride=1, padding=0, bias=False, groups=4)),
            ("bn1_new_2_1", norm_layer(width)),
        ]
        self.conv1 = nn.Sequential(OrderedDict(self.conv1_layers))

        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        # self.bn2 = norm_layer(width)
        self.conv2_layers = [
            # The 1-st passage for Conv1.
            ("conv2_new_1_1", nn.Conv2d(width, width, 1, stride=1, padding=0, bias=False)),
            ("bn2_new_1_1", norm_layer(width)),
            ("conv2_new_1_2", conv3x3(width, width, stride, groups=width, dilation=dilation)),
            ("bn2_new_1_2", norm_layer(width)),
            ("conv2_new_1_3", nn.Conv2d(width, width, 1, stride=1, padding=0, bias=False)),
            ("bn2_new_1_3", norm_layer(width)),
            # The 2-nd passage for Conv1.
            ("conv2_new_2_1", nn.Conv2d(width, width, 1, stride=1, padding=0, bias=False)),
            ("bn2_new_2_1", norm_layer(width)),
            ("conv2_new_2_2", conv3x3(width, width, stride, groups=width, dilation=dilation)),
            ("bn2_new_2_2", norm_layer(width)),
            ("conv2_new_2_3", nn.Conv2d(width, width, 1, stride=1, padding=0, bias=False)),
            ("bn2_new_2_3", norm_layer(width)),
        ]
        self.conv2 = nn.Sequential(OrderedDict(self.conv2_layers))

        # self.conv3 = conv1x1(width, planes * self.expansion)
        # self.bn3 = norm_layer(planes * self.expansion)
        self.conv3_layers = [
            # The 1-st passage for Conv1.
            # ("conv3_new_1_1", conv1x1(width, planes * self.expansion)),
            ("conv3_new_1_1", nn.Conv2d(width, planes * self.expansion, 1, stride=1, padding=0, bias=False, groups=4)),
            ("bn3_new_1_1", norm_layer(planes * self.expansion)),
            # The 2-nd passage for Conv1.
            # ("conv3_new_2_1", conv1x1(width, planes * self.expansion)),
            ("conv3_new_2_1", nn.Conv2d(width, planes * self.expansion, 1, stride=1, padding=0, bias=False, groups=4)),
            ("bn3_new_2_1", norm_layer(planes * self.expansion)),
        ]
        self.conv3 = nn.Sequential(OrderedDict(self.conv3_layers))

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # identity = x
        x1, x2 = x
        residual_1 = x1
        residual_2 = x2

        # Conv1
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        x1 = self.conv1[0:2](x1)
        x2 = self.conv1[2:4](x2)
        out = torch.cat((x1, x2), dim=0)
        out = self.relu(out)

        # Conv2
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        x1, x2 = torch.split(out, split_size_or_sections=[x1.shape[0], x2.shape[0]], dim=0)
        x1 = self.conv2[0:6](x1)
        x2 = self.conv2[6:12](x2)
        out = torch.cat((x1, x2), dim=0)
        out = self.relu(out)

        # Conv2
        # out = self.conv3(out)
        # out = self.bn3(out)
        x1, x2 = torch.split(out, split_size_or_sections=[x1.shape[0], x2.shape[0]], dim=0)
        x1 = self.conv3[0:2](x1)
        x2 = self.conv3[2:4](x2)
        out = torch.cat((x1, x2), dim=0)

        # if self.downsample is not None:
        #     identity = self.downsample(x)
        if self.downsample is not None:
            residual_1 = self.downsample[0:2](residual_1)
            residual_2 = self.downsample[2:4](residual_2)

        # out += identity
        # out = self.relu(out)
        x1 += residual_1
        x2 += residual_2

        x1 = self.relu(x1)
        x2 = self.relu(x2)
        
        out = (x1, x2)
        return out


class ResNetBottleneck(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        self.conv_stem_layers = [
            # The 1-st passage for Conv1.
            ("conv0_new_1_1", nn.Conv2d(3, self.inplanes, 1, stride=1, padding=0, bias=False)),
            ("bn0_new_1_1", norm_layer(self.inplanes)),
            ("conv0_new_1_2", nn.Conv2d(self.inplanes, self.inplanes, 3, stride=1, padding=1, bias=False, groups=self.inplanes)),
            ("bn0_new_1_2", norm_layer(self.inplanes)),
            ("conv0_new_1_3", nn.Conv2d(self.inplanes, self.inplanes, 1, stride=1, padding=0, bias=False)),
            ("bn0_new_1_3", norm_layer(self.inplanes)),
            # The 2-nd passage for Conv1.
            ("conv0_new_2_1", nn.Conv2d(3, self.inplanes, 1, stride=1, padding=0, bias=False)),
            ("bn0_new_2_1", norm_layer(self.inplanes)),
            ("conv0_new_2_2", nn.Conv2d(self.inplanes, self.inplanes, 3, stride=1, padding=1, bias=False, groups=self.inplanes)),
            ("bn0_new_2_2", norm_layer(self.inplanes)),
            ("conv0_new_2_3", nn.Conv2d(self.inplanes, self.inplanes, 1, stride=1, padding=0, bias=False)),
            ("bn0_new_2_3", norm_layer(self.inplanes)),
        ]
        self.conv1 = nn.Sequential(OrderedDict(self.conv_stem_layers))

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.last_dim = self.fc.in_features
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

        self.w_avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.h_avgpool = nn.AdaptiveAvgPool2d((None ,1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride),
            #     norm_layer(planes * block.expansion),
            # )
            downsample_layers = [
                # The 1-st passage for downsample.
                ("conv_d_new_1_1_d", conv1x1(self.inplanes, planes * block.expansion, stride)),
                ("bn_d_new_1_1_d", norm_layer(planes * block.expansion)),
                # The 2-nd passage for downsample.
                ("conv_d_new_2_1_d", conv1x1(self.inplanes, planes * block.expansion, stride)),
                ("bn_d_new_2_1_d", norm_layer(planes * block.expansion)),
            ]
            downsample = nn.Sequential(OrderedDict(downsample_layers))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        outs1_p = []
        outs2_p = []
        outs1_w = []
        outs2_w = []
        outs1_h = []
        outs2_h = []
        
        # Conv1
        x1, x2 = x
        x1 = self.conv1[0:6](x1)
        x2 = self.conv1[6:12](x2)
        x = torch.cat((x1, x2), dim=0)
        
        x1_w = self.w_avgpool(x1)
        x2_w = self.w_avgpool(x2)
        x1_h = self.h_avgpool(x1)
        x2_h = self.h_avgpool(x2)
        outs1_p.append(x1)
        outs2_p.append(x2)
        outs1_w.append(x1_w)
        outs2_w.append(x2_w)
        outs1_h.append(x1_h)
        outs2_h.append(x2_h)
        
        x = self.relu(x)
        
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x1, x2 = torch.split(x, split_size_or_sections=[x1.shape[0], x2.shape[0]], dim=0)
        x = (x1, x2)

        # import pdb
        # pdb.set_trace()

        x = self.layer1(x)
        x1, x2 = x

        # import pdb
        # pdb.set_trace()
        
        x1_w = self.w_avgpool(x1)
        x2_w = self.w_avgpool(x2)
        x1_h = self.h_avgpool(x1)
        x2_h = self.h_avgpool(x2)
        outs1_p.append(x1)
        outs2_p.append(x2)
        outs1_w.append(x1_w)
        outs2_w.append(x2_w)
        outs1_h.append(x1_h)
        outs2_h.append(x2_h)

        x = self.layer2(x)

        x1, x2 = x
        x1_w = self.w_avgpool(x1)
        x2_w = self.w_avgpool(x2)
        x1_h = self.h_avgpool(x1)
        x2_h = self.h_avgpool(x2)
        outs1_p.append(x1)
        outs2_p.append(x2)
        outs1_w.append(x1_w)
        outs2_w.append(x2_w)
        outs1_h.append(x1_h)
        outs2_h.append(x2_h)

        x = self.layer3(x)

        x1, x2 = x
        x1_w = self.w_avgpool(x1)
        x2_w = self.w_avgpool(x2)
        x1_h = self.h_avgpool(x1)
        x2_h = self.h_avgpool(x2)
        outs1_p.append(x1)
        outs2_p.append(x2)
        outs1_w.append(x1_w)
        outs2_w.append(x2_w)
        outs1_h.append(x1_h)
        outs2_h.append(x2_h)

        x = self.layer4(x)

        x1, x2 = x
        x1_w = self.w_avgpool(x1)
        x2_w = self.w_avgpool(x2)
        x1_h = self.h_avgpool(x1)
        x2_h = self.h_avgpool(x2)
        outs1_p.append(x1)
        outs2_p.append(x2)
        outs1_w.append(x1_w)
        outs2_w.append(x2_w)
        outs1_h.append(x1_h)
        outs2_h.append(x2_h)
        
        x1, x2 = x
        x = torch.cat((x1, x2), dim=0)
        # import pdb
        # pdb.set_trace()
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, ((outs1_p, outs1_w, outs1_h), (outs2_p, outs2_w, outs2_h))

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNetBottleneck:
    model = ResNetBottleneck(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetBottleneck:
    r"""ResNetBottleneck-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetBottleneck:
    r"""ResNetBottleneck-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


# def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetBottleneck:
#     r"""ResNetBottleneck-50 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)

def resnet50_cifar(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetBottleneck:
    r"""ResNetBottleneck-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50_cifar', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetBottleneck:
    r"""ResNetBottleneck-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

# def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetBottleneck:
#     r"""ResNetBottleneck-101 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
#                    **kwargs)

def resnet101_cifar(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetBottleneck:
    r"""ResNetBottleneck-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101_cifar', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetBottleneck:
    r"""ResNetBottleneck-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d_cifar(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetBottleneck:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d_cifar', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d_cifar(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetBottleneck:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d_cifar', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetBottleneck:
    r"""Wide ResNetBottleneck-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNetBottleneck except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNetBottleneck-50 has 2048-512-2048
    channels, and in Wide ResNetBottleneck-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetBottleneck:
    r"""Wide ResNetBottleneck-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNetBottleneck except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNetBottleneck-50 has 2048-512-2048
    channels, and in Wide ResNetBottleneck-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

def log_model_info(model, verbose=False):
    """Logs model info"""
    if verbose:
        logger.info(f"Classification Model:\n{model}")
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))
    # print("tuned percent:%.3f"%(model_grad_params/model_total_params*100))

if __name__ == "__main__":
    # models ok
    import torch
    import torchvision.models as models
    from ptflops import get_model_complexity_info

    # resnet18_ = models.resnet18(pretrained=False)
    # model = models.resnet101(pretrained=False, num_classes=100)
    model = resnet50_cifar(pretrained=False, num_classes=100)
    # print(model)

    x = torch.randn(2, 3, 224, 224)
    # print(x.shape)
    x_s = (x[0::2,:,:,:], x[1::2,:,:,:])
    # print(x_s[0].shape, x_s[1].shape)

    # 
    # x = model(x)
    out1, out2 = model(x_s)
    # print(out1.shape, out2.shape)
    
    # log_model_info(model)
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
    #                                            print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    log_model_info(model)

    # import pdb
    # pdb.set_trace()