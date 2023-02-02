import torch
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, image_size=32):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.last_dim = self.fc.in_features
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'
        self.image_size = image_size
        # ------------------------------------------------------------------------------------------------
        # Add by NieX.
        self.xER = nn.Parameter(torch.randn([20, 3, self.image_size, self.image_size]))
        # ------------------------------------------------------------------------------------------------

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        import pdb
        pdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_cifar(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


if __name__ == '__main__':
    # from ptflops import get_model_complexity_info
    # # model = resnet50_reuse(pretrained=False, num_classes=10)
    # model = resnet50(pretrained=False, num_classes=10)

    # model_dict = model.state_dict()

    # pretrained_dict = torch.load('/home1/niexing/projects/mmdetection/pretrain/resnet50-19c8e357.pth')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    # for k, v in model.named_parameters():
    #     if 'fc' not in k:
    #         v.requires_grad = False

    # for k, v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))

    # # with torch.cuda.device(0):
    # #   macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
    # #                                            print_per_layer_stat=True, verbose=True)
    # #   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # #   print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # import pdb;pdb.set_trace()

    ##########################################################################################
    # 1. ResNet50_reuse
    ##########################################################################################
    import argparse
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')
    parser.add_argument('--image_size', type=int, default=32, help='image size')
    args, extra_args = parser.parse_known_args()

    from ptflops import get_model_complexity_info
    import importlib
    from ptflops import get_model_complexity_info

    # model = resnet50_reuse(pretrained=False, num_classes=10)
    model = resnet18_cifar(pretrained=False, image_size=args.image_size)

    # import pdb
    # pdb.set_trace()
    
    model_dict = model.state_dict()    
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # import pdb
    # pdb.set_trace()

    # pretrained_dict = torch.load('/home1/niexing/projects/mmdetection/pretrain/resnet50-19c8e357.pth')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    # pretrained_dict_reuse_only = torch.load('/home1/niexing/projects/mmdetection/pretrain/resnet50_reuse_only.pth')
    # model_dict.update(pretrained_dict_reuse_only)
    # model.load_state_dict(model_dict)

    # ---------------------------------------------------------------------------
    # adapter + stem微调
    # ---------------------------------------------------------------------------
    # for k, v in model.named_parameters():
    #     if '_new'             not in k and \
    #        'fc'               not in k and \
    #        'conv1.weight'         != k and \
    #        'bn1.weight'           != k and \
    #        'bn1.bias'             != k :
    #         v.requires_grad = False
    # ---------------------------------------------------------------------------