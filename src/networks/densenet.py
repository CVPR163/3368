import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torchvision.models.utils import load_state_dict_from_url
from torch import Tensor
from typing import Any, List, Tuple


__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False
    ):
        super(_DenseLayer, self).__init__()

        self.norm1: nn.BatchNorm2d
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))

        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))

        # self.conv1: nn.Conv2d
        # self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
        #                                    growth_rate, kernel_size=1, stride=1,
        #                                    bias=False))
        self.conv1: nn.Conv2d
        self.conv1_layers = [
            # The i-st passage for Conv1.
            ("conv1_1_1", nn.Conv2d(num_input_features, bn_size * growth_rate, 1, stride=1, padding=0, bias=False, groups=4)),
            # ("bn1_1_1", nn.BatchNorm2d(bn_size * growth_rate)),
        ]
        self.add_module('conv1', nn.Sequential(OrderedDict(self.conv1_layers)))

        self.norm2: nn.BatchNorm2d
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))

        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))

        # self.conv2: nn.Conv2d
        # self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        #                                    kernel_size=3, stride=1, padding=1,
        #                                    bias=False))
        self.conv2: nn.Conv2d
        self.conv2_layers = [
            # The i-st passage for Conv2.
            ("conv2_1_1", nn.Conv2d(bn_size * growth_rate, bn_size * growth_rate, 1, stride=1, padding=0, bias=False)),
            ("bn2_1_1", nn.BatchNorm2d(bn_size * growth_rate)),
            ("conv2_1_2", nn.Conv2d(bn_size * growth_rate, growth_rate, 3, stride=1, padding=1, bias=False, groups=growth_rate)),
            ("bn2_1_2", nn.BatchNorm2d(growth_rate)),
            ("conv2_1_3", nn.Conv2d(growth_rate, growth_rate, 1, stride=1, padding=0, bias=False)),
            # ("bn2_1_3", nn.BatchNorm2d(growth_rate)),
        ]
        self.add_module('conv2', nn.Sequential(OrderedDict(self.conv2_layers)))

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False
    ):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False
    ):

        super(DenseNet, self).__init__()

        # First convolution
        self.features_1 = nn.Sequential(OrderedDict([
            # ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
            #                     padding=3, bias=False)),
            # ('norm0', nn.BatchNorm2d(num_init_features)),
            ("conv0_new_1_1", nn.Conv2d(3, num_init_features, 1, stride=1, padding=0, bias=False)),
            ("bn0_new_1_1", nn.BatchNorm2d(num_init_features)),
            ("conv0_new_1_2", nn.Conv2d(num_init_features, num_init_features, 7, stride=2, padding=3, bias=False, groups=num_init_features)),
            ("bn0_new_1_2", nn.BatchNorm2d(num_init_features)),
            ("conv0_new_1_3", nn.Conv2d(num_init_features, num_init_features, 1, stride=1, padding=0, bias=False)),
            ("bn0_new_1_3", nn.BatchNorm2d(num_init_features)),
            ('relu0_1', nn.ReLU(inplace=True)),
            ('pool0_1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.features_2 = nn.Sequential(OrderedDict([
            # ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
            #                     padding=3, bias=False)),
            # ('norm0', nn.BatchNorm2d(num_init_features)),
            ("conv0_new_2_1", nn.Conv2d(3, num_init_features, 1, stride=1, padding=0, bias=False)),
            ("bn0_new_2_1", nn.BatchNorm2d(num_init_features)),
            ("conv0_new_2_2", nn.Conv2d(num_init_features, num_init_features, 7, stride=2, padding=3, bias=False, groups=num_init_features)),
            ("bn0_new_2_2", nn.BatchNorm2d(num_init_features)),
            ("conv0_new_2_3", nn.Conv2d(num_init_features, num_init_features, 1, stride=1, padding=0, bias=False)),
            ("bn0_new_2_3", nn.BatchNorm2d(num_init_features)),
            ('relu0_2', nn.ReLU(inplace=True)),
            ('pool0_2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block_1 = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            block_2 = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features_1.add_module('denseblock%d' % (i + 1), block_1)
            self.features_2.add_module('denseblock%d' % (i + 1), block_2)

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans_1 = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                trans_2 = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features_1.add_module('transition%d' % (i + 1), trans_1)
                self.features_2.add_module('transition%d' % (i + 1), trans_2)
                num_features = num_features // 2

        # Final batch norm
        self.features_1.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features_2.add_module('norm5', nn.BatchNorm2d(num_features))

        self.w_avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.h_avgpool = nn.AdaptiveAvgPool2d((None ,1))

        # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)
        self.fc = nn.Linear(num_features, num_classes)

        self.last_dim = self.fc.in_features
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def get_features(self, x, pod_w, pod_h, branch_model):
        # for k, v in branch_model.named_parameters():
        for item in branch_model._modules.items():
            each_layer = getattr(branch_model, item[0])
            x = each_layer(x)
            # import pdb
            # pdb.set_trace()
            if "denseblock" in item[0] or 'relu0' in item[0]:
                x_w = self.w_avgpool(x)
                x_h = self.h_avgpool(x)
                pod_w.append(x_w)
                pod_h.append(x_h)

        return x, (pod_w, pod_h)

    def forward(self, x):
        x1, x2 = x

        outs1_w = []
        outs1_h = []

        outs2_w = []
        outs2_h = []

        x1, outs1 = self.get_features(x1, outs1_w, outs1_h, self.features_1)
        x2, outs2 = self.get_features(x2, outs2_w, outs2_h, self.features_2)

        out_1 = F.relu(x1, inplace=True)
        out_2 = F.relu(x2, inplace=True)

        out_1 = F.adaptive_avg_pool2d(out_1, (1, 1))
        out_2 = F.adaptive_avg_pool2d(out_2, (1, 1))

        out = torch.cat((out_1, out_2), dim=0)
        out = torch.flatten(out, 1)

        # out = self.classifier(out)
        out = self.fc(out)
        return out, (outs1, outs2)


def _load_state_dict(model: nn.Module, model_url: str, progress: bool):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)



def densenet161(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)



def densenet169(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)



def densenet201(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)

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
    
    # model = models.resnet34(pretrained=False)

    # model = models.densenet121(num_classes=100)
    # model = models.densenet169(num_classes=100)
    model = models.densenet201(num_classes=100)

    # model = densenet121(num_classes=100)
    # model = densenet169(num_classes=100)
    # model = densenet201(num_classes=100)

    model_dict = model.state_dict()
    model_dict_1 = {k: v for k, v in model_dict.items() if 'features_1' in k}
    model_dict_2 = {k: v for k, v in model_dict.items() if 'features_2' in k}
    model_dict_3 = {k: v for k, v in model_dict.items() if 'features_1' not in k and 'features_2' not in k}

    # import pdb
    # pdb.set_trace()

    # print(model_dict_3.keys())
    # print(model)

    x = torch.randn(4, 3, 32, 32)
    print(x.shape)
    # x = (x[0::3,:,:,:], x[0::3,:,:,:])

    x = model(x)
    # x, outs1 = model(x)
    print(model)
    import pdb
    pdb.set_trace()

    outs1_1, outs1_2 = outs1
    
    for i in range(len(outs1_1)):
        for j in range(len(outs1_1[i])):
            print('outs1_1_{}_{}'.format(i, outs1_1[i][j].shape))
            print('outs1_2_{}_{}'.format(i, outs1_2[i][j].shape))

    log_model_info(model)



    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
    #                                            print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    


