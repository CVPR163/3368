import math
import numpy as np
import torch
import torch.nn as nn
from .utils import *
from torch.distributions import Normal, Independent
from torch import distributed as dist
from torch.autograd.function import Function

# function credit to https://github.com/facebookresearch/barlowtwins/blob/main/main.py
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class DecorrelateLossClass(nn.Module):

    def __init__(self, reject_threshold=1, ddp=False):
        super(DecorrelateLossClass, self).__init__()
        self.eps = 1e-8
        self.reject_threshold = reject_threshold
        self.ddp = ddp

    def forward(self, x, y):
        _, C = x.shape
        if self.ddp:
            # if DDP
            # first gather all x and labels from the world
            x = torch.cat(GatherLayer.apply(x), dim=0)
            y = global_gather(y)

        loss = 0.0
        uniq_l, uniq_c = y.unique(return_counts=True)
        n_count = 0
        for i, label in enumerate(uniq_l):
            if uniq_c[i] <= self.reject_threshold:
                continue
            x_label = x[y==label, :]
            x_label = x_label - x_label.mean(dim=0, keepdim=True)
            x_label = x_label / torch.sqrt(self.eps + x_label.var(dim=0, keepdim=True))

            N = x_label.shape[0]
            corr_mat = torch.matmul(x_label.t(), x_label)

            # Notice that here the implementation is a little bit different
            # from the paper as we extract only the off-diagonal terms for regularization.
            # Mathematically, these two are the same thing since diagonal terms are all constant 1.
            # However, we find that this implementation is more numerically stable.
            loss += (off_diagonal(corr_mat).pow(2)).mean()

            n_count += N

        if n_count == 0:
            # there is no effective class to compute correlation matrix
            return 0
        else:
            loss = loss / n_count
            return loss

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None

# import math
# import numpy as np
# import torch
# import torch.nn as nn
# from .utils import *
# from torch.distributions import Normal, Independent
# from torch import distributed as dist

# # function credit to https://github.com/facebookresearch/barlowtwins/blob/main/main.py
# def off_diagonal(x):
#     # return a flattened view of the off-diagonal elements of a square matrix
#     n, m = x.shape
#     assert n == m
#     return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# class DecorrelateLossClass(nn.Module):

#     def __init__(self, reject_threshold=1, ddp=False):
#         super(DecorrelateLossClass, self).__init__()
#         self.eps = 1e-8
#         self.reject_threshold = reject_threshold
#         self.ddp = ddp

#     def forward(self, x, y):
#         _, C = x.shape
#         if self.ddp:
#             # if DDP
#             # first gather all x and labels from the world
#             x = torch.cat(GatherLayer.apply(x), dim=0)
#             y = global_gather(y)

#         loss = 0.0
#         uniq_l, uniq_c = y.unique(return_counts=True)
#         n_count = 0
#         for i, label in enumerate(uniq_l):
#             if uniq_c[i] <= self.reject_threshold:
#                 continue
#             x_label = x[y==label, :]
#             x_label = x_label - x_label.mean(dim=0, keepdim=True)
#             x_label = x_label / torch.sqrt(self.eps + x_label.var(dim=0, keepdim=True))

#             N = x_label.shape[0]
#             corr_mat = torch.matmul(x_label.t(), x_label)

#             # Notice that here the implementation is a little bit different
#             # from the paper as we extract only the off-diagonal terms for regularization.
#             # Mathematically, these two are the same thing since diagonal terms are all constant 1.
#             # However, we find that this implementation is more numerically stable.
#             loss += (off_diagonal(corr_mat).pow(2)).mean()

#             n_count += N

#         if n_count == 0:
#             # there is no effective class to compute correlation matrix
#             return 0
#         else:
#             loss = loss / n_count
#             return loss
