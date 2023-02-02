import torch.nn as nn
import math
from collections import OrderedDict
import torch

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv1_layers = [
            # The 1-st passage for Conv1.
            ("conv1_new_1_1", nn.Conv2d(inplanes, planes, 1, stride=1, padding=0, bias=False)),
            ("bn1_new_1_1", nn.BatchNorm2d(planes)),
            ("conv1_new_1_2", nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False,
                                groups=planes)),
            ("bn1_new_1_2", nn.BatchNorm2d(planes)),
            ("conv1_new_1_3", nn.Conv2d(planes, planes, 1, stride=1, padding=0, bias=False)),
            ("bn1_new_1_3", nn.BatchNorm2d(planes)),
            # The 2-nd passage for Conv1.
            ("conv1_new_2_1", nn.Conv2d(inplanes, planes, 1, stride=1, padding=0, bias=False)),
            ("bn1_new_2_1", nn.BatchNorm2d(planes)),
            ("conv1_new_2_2", nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False,
                                groups=planes)),
            ("bn1_new_2_2", nn.BatchNorm2d(planes)),
            ("conv1_new_2_3", nn.Conv2d(planes, planes, 1, stride=1, padding=0, bias=False)),
            ("bn1_new_2_3", nn.BatchNorm2d(planes)),
        ]
        self.conv1 = nn.Sequential(OrderedDict(self.conv1_layers))
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv2_layers = [
            # The 1-st passage for Conv2.
            ("conv2_new_1_1", nn.Conv2d(planes, planes, 1, stride=1, padding=0, bias=False)),
            ("bn2_new_1_1", nn.BatchNorm2d(planes)),
            ("conv2_new_1_2", nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False,
                                groups=planes)),
            ("bn2_new_1_2", nn.BatchNorm2d(planes)),
            ("conv2_new_1_3", nn.Conv2d(planes, planes, 1, stride=1, padding=0, bias=False)),
            ("bn2_new_1_3", nn.BatchNorm2d(planes)),
            # The 2-nd passage for Conv2.
            ("conv2_new_2_1", nn.Conv2d(planes, planes, 1, stride=1, padding=0, bias=False)),
            ("bn2_new_2_1", nn.BatchNorm2d(planes)),
            ("conv2_new_2_2", nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False,
                                groups=planes)),
            ("bn2_new_2_2", nn.BatchNorm2d(planes)),
            ("conv2_new_2_3", nn.Conv2d(planes, planes, 1, stride=1, padding=0, bias=False)),
            ("bn2_new_2_3", nn.BatchNorm2d(planes))
        ]
        self.conv2 = nn.Sequential(OrderedDict(self.conv2_layers))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x1, x2 = x
        residual_1 = x1
        residual_2 = x2

        # Conv1
        x1 = self.conv1[0:6](x1)
        x2 = self.conv1[6:12](x2)
        out_conv1 = torch.cat((x1, x2), dim=0)
        out_conv1 = self.relu(out_conv1)

        # Conv2
        x1, x2 = torch.split(out_conv1, split_size_or_sections=[x1.shape[0], x2.shape[0]], dim=0)
        out_conv2_branch1 = self.conv2[0:6](x1)
        out_conv2_branch2 = self.conv2[6:12](x2)

        if self.downsample is not None:
            residual_1 = self.downsample[0:2](residual_1)
            residual_2 = self.downsample[2:4](residual_2)

        out_conv2_branch1 += residual_1
        out_conv2_branch2 += residual_2

        out_conv2_branch1 = self.relu(out_conv2_branch1)
        out_conv2_branch2 = self.relu(out_conv2_branch2)
        
        out = (out_conv2_branch1, out_conv2_branch2)
        return out

    # def forward(self, x):
        # residual = x
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        # out += residual
        # out = self.relu(out)

        # return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.input_channel = 3
        self.output_channel = 64
        self.conv_stem_layers = [
            # The 1-st passage for Conv1.
            ("conv1_new_1_1", nn.Conv2d(self.input_channel, self.output_channel, 1, stride=1, padding=0, bias=False)),
            ("bn1_new_1_1", nn.BatchNorm2d(self.output_channel)),
            ("conv1_new_1_2", nn.Conv2d(self.output_channel, self.output_channel, 7, stride=2, padding=3, bias=False,
                                groups=self.output_channel)),
            ("bn1_new_1_2", nn.BatchNorm2d(self.output_channel)),
            ("conv1_new_1_3", nn.Conv2d(self.output_channel, self.output_channel, 1, stride=1, padding=0, bias=False)),
            ("bn1_new_1_3", nn.BatchNorm2d(self.output_channel)),
            # The 2-nd passage for Conv1.
            ("conv1_new_2_1", nn.Conv2d(self.input_channel, self.output_channel, 1, stride=1, padding=0, bias=False)),
            ("bn1_new_2_1", nn.BatchNorm2d(self.output_channel)),
            ("conv1_new_2_2", nn.Conv2d(self.output_channel, self.output_channel, 7, stride=2, padding=3, bias=False,
                                groups=self.output_channel)),
            ("bn1_new_2_2", nn.BatchNorm2d(self.output_channel)),
            ("conv1_new_2_3", nn.Conv2d(self.output_channel, self.output_channel, 1, stride=1, padding=0, bias=False)),
            ("bn1_new_2_3", nn.BatchNorm2d(self.output_channel)),
        ]
        self.conv1 = nn.Sequential(OrderedDict(self.conv_stem_layers))
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.w_avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.h_avgpool = nn.AdaptiveAvgPool2d((None ,1))

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.last_dim = self.fc.in_features
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(planes * block.expansion),
            # )
            downsample_layers = [
                # The 1-st passage for downsample.
                ("conv1_new_1_1_d", nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)),
                ("bn1_new_1_1_d", nn.BatchNorm2d(planes * block.expansion)),
                # The 2-nd passage for downsample.
                ("conv1_new_2_1_d", nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)),
                ("bn1_new_2_1_d", nn.BatchNorm2d(planes * block.expansion)),
            ]
            downsample = nn.Sequential(OrderedDict(downsample_layers))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
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
        x = self.maxpool(x)
        
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x1, x2 = torch.split(x, split_size_or_sections=[x1.shape[0], x2.shape[0]], dim=0)
        x = (x1, x2)

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


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

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

    model = resnet18_cifar(pretrained=False)
    # print(model)

    input = torch.randn(2, 3, 224, 224)
    print(input.shape)
    x = (input[0::2,:,:,:], input[1::2,:,:,:])
    # x = {0:input[0::2,:,:,:], 1:input[1::2,:,:,:]}
    x, outs1 = model(x)
    import pdb
    pdb.set_trace()
    print(x.shape, outs.shape)
    
    log_model_info(model)
    # log_model_info(resnet18_)
    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
    #                                            print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    import pdb
    pdb.set_trace()