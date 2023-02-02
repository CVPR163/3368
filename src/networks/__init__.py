from torchvision import models

from .resnet18 import resnet18, resnet34
from .resnet18_cifar import resnet18_cifar, resnet34_cifar

from .resnet50 import resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
from .resnet50_cifar import resnet50_cifar, resnet101_cifar, resnext50_32x4d_cifar, resnext101_32x8d_cifar

from .resnet18_cifar_plus import resnet18_cifar_plus
from .resnet18_cifar_plus_asy import resnet18_cifar_plus_asy

from .resnet18_cifar_conv1_s import resnet18_cifar_conv1_s
from .resnet18_cifar_group import resnet18_cifar_group
from .resnet18_cifar_group_s import resnet18_cifar_group_s


from .resnet18_cifar_2C import resnet18_cifar_2C
from .resnet18_cifar_1p5C import resnet18_cifar_1p5C
from .resnet18_cifar_1p25C import resnet18_cifar_1p25C
from .resnet18_cifar_05C import resnet18_cifar_05C

# available torchvision models
tvmodels = ['alexnet',
            'googlenet',
            'inception_v3',
            'mobilenet_v2',
            'resnet152',
            'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
            'squeezenet1_0', 'squeezenet1_1',
            'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
            'wide_resnet50_2', 'wide_resnet101_2'
            ]

allmodels = tvmodels + ['resnet18', 
                        'resnet34',
                        'resnet50', 
                        'resnet101', 
                        'resnet18_cifar', 
                        'resnet34_cifar',
                        'resnet50_cifar', 
                        'resnet101_cifar', 
                        'resnet18_cifar_plus', 
                        'resnet18_cifar_plus_asy', 
                        'resnet18_cifar_conv1_s',
                        'resnet18_cifar_group',
                        'resnet18_cifar_group_s',
                        'resnext50_32x4d', 
                        'resnext101_32x8d',
                        'resnext50_32x4d_cifar', 
                        'resnext101_32x8d_cifar',
                        'resnet18_cifar_2C', 
                        'resnet18_cifar_1p5C',
                        'resnet18_cifar_1p25C', 
                        'resnet18_cifar_05C'
                        ]

def set_tvmodel_head_var(model):
    if type(model) == models.AlexNet:
        model.head_var = 'classifier'
    elif type(model) == models.DenseNet:
        model.head_var = 'classifier'
    elif type(model) == models.Inception3:
        model.head_var = 'fc'
    elif type(model) == models.ResNet:
        model.head_var = 'fc'
    elif type(model) == models.VGG:
        model.head_var = 'classifier'
    elif type(model) == models.GoogLeNet:
        model.head_var = 'fc'
    elif type(model) == models.MobileNetV2:
        model.head_var = 'classifier'
    elif type(model) == models.ShuffleNetV2:
        model.head_var = 'fc'
    elif type(model) == models.SqueezeNet:
        model.head_var = 'classifier'
    else:
        raise ModuleNotFoundError