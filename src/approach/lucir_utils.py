import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter


# Sec 3.2: This class implements the cosine normalizing linear layer module using Eq. 4
class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out_s = self.sigma * out
        else:
            out_s = out
        if self.training:
            return {'wsigma': out_s, 'wosigma': out}
        else:
            return out_s

# This class implements a ResNet Basic Block without the final ReLu in the forward
class BasicBlockNoRelu(nn.Module):
    expansion = 1

    def __init__(self, conv1, relu, conv2, downsample):
        super(BasicBlockNoRelu, self).__init__()
        # self.conv1 = conv1
        # self.bn1 = bn1
        self.conv1 = conv1
        self.relu = relu
        # self.conv2 = conv2
        # self.bn2 = bn2
        self.conv2 = conv2
        self.downsample = downsample

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

        # Removed final ReLU
        # out_conv2_branch1 = self.relu(out_conv2_branch1)
        # out_conv2_branch2 = self.relu(out_conv2_branch2)

        out = (out_conv2_branch1, out_conv2_branch2)
        return out

# ----------------------------------------------------------------------------------------------------
# Add by NieX for ResNet50.
class BottleneckNoRelu(nn.Module):
    expansion = 4

    # def __init__(self, conv1, bn1, relu, conv2, bn2, conv3, bn3, downsample):
    def __init__(self, conv1, relu, conv2, conv3, downsample):
        super(BottleneckNoRelu, self).__init__()
        self.conv1 = conv1
        # self.bn1 = bn1
        self.conv2 = conv2
        # self.bn2 = bn2
        self.conv3 = conv3
        # self.bn3 = bn3
        self.relu = relu
        self.downsample = downsample

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

        # Removed final ReLU
        # x1 = self.relu(x1)
        # x2 = self.relu(x2)
        
        out = (x1, x2)
        return out
# ----------------------------------------------------------------------------------------------------

# # This class implements a ResNet Basic Block without the final ReLu in the forward
# class BasicBlockNoRelu(nn.Module):
#     expansion = 1

#     def __init__(self, conv1, bn1, relu, conv2, bn2, downsample):
#         super(BasicBlockNoRelu, self).__init__()
#         self.conv1 = conv1
#         self.bn1 = bn1
#         self.relu = relu
#         self.conv2 = conv2
#         self.bn2 = bn2
#         self.downsample = downsample

#     def forward(self, x):
#         residual = x
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out += residual
#         # Removed final ReLU
#         return out

# class BottleneckNoRelu(nn.Module):
#     expansion = 4

#     def __init__(self, conv1, bn1, relu, conv2, bn2, conv3, bn3, downsample):
#         super(BottleneckNoRelu, self).__init__()
#         self.conv1 = conv1
#         self.bn1 = bn1
#         self.conv2 = conv2
#         self.bn2 = bn2
#         self.conv3 = conv3
#         self.bn3 = bn3
#         self.relu = relu
#         self.downsample = downsample

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         # Removed final ReLU
#         return out