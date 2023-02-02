import copy
import math
import torch
import warnings
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
import numpy as np

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset

from torch.nn.parallel import DistributedDataParallel as DDP
# from .lucir_utils import CosineLinear, BasicBlockNoRelu, BottleneckNoRelu

import time

from .utils import *
import heapq

# from .Distill_loss_detach import Distill_Loss_detach

from collections import OrderedDict

class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, l_alpha, l_beta, l_gamma, buffer_size, minibatch_size_1, minibatch_size_2, network, nepochs=160, lr=0.1, decay_mile_stone=[80,120], lr_decay=0.1, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, ddp=False, local_rank=0, logger=None, exemplars_dataset=None,
                 lamb=5., lamb_mr=1., dist=0.5, K=2, pca_flag=True, lamb_pca=5., K_pca=5):
        super(Appr, self).__init__(model, device, nepochs, lr, decay_mile_stone, lr_decay, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, ddp, local_rank,
                                   logger, exemplars_dataset)
        self.trn_datasets = []
        self.val_datasets = []

        self.pca_flag = pca_flag
        self.lamb_pca = lamb_pca
        self.K_pca = K_pca

        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: Replay is expected to use exemplars. Check documentation.")

        # ----------------------------------------------------------------------------------------------
        self.buffer_size = buffer_size
        self.minibatch_size_1 = minibatch_size_1
        self.minibatch_size_2 = minibatch_size_2
        self.l_alpha = l_alpha
        self.l_beta = l_beta
        self.l_gamma = l_gamma
        # self.buffer = Buffer(self.buffer_size, self.device)
        self.network = network
        # ----------------------------------------------------------------------------------------------

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--pca-flag', action='store_true', help='whether to use pce loss')
        parser.add_argument('--lamb-pca', default=5., type=float, required=False,
                            help='Trade-off for the PCA loss (default=%(default)s)')
        parser.add_argument('--K-pca', default=5, type=int, required=False,
                            help='Number of "old classes chosen'
                                 'for PCA loss (default=%(default)s)')
        return parser.parse_known_args(args)
    # def _get_optimizer(self):
    #     """Returns the optimizer"""
    #     if self.ddp:
    #         model = self.model.module
    #     else:
    #         model = self.model
    #     if self.less_forget:
    #         # Don't update heads when Less-Forgetting constraint is activated (from original code)
    #         params = list(model.model.parameters()) + list(model.heads[-1].parameters())
    #     else:
    #         params = model.parameters()
    #     return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if self.ddp:
            model = self.model.module
        else:
            model = self.model
        params = model.parameters()

        optimizer = torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        print(optimizer.param_groups[0]['lr'])
        return optimizer

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        if self.ddp:
            model = self.model.module
        else:
            model = self.model

        '''
        if t == 0:
            # Sec. 4.1: "the ReLU in the penultimate layer is removed to allow the features to take both positive and
            # negative values"
            if model.model.__class__.__name__ == 'ResNetCifar':
                old_block = model.model.layer3[-1]
                model.model.layer3[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                               old_block.conv2, old_block.bn2, old_block.downsample)
            elif model.model.__class__.__name__ == 'ResNet':
                old_block = model.model.layer4[-1]
                model.model.layer4[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                               old_block.conv2, old_block.bn2, old_block.downsample)
            elif model.model.__class__.__name__ == 'ResNetBottleneck':
                old_block = model.model.layer4[-1]
                model.model.layer4[-1] = BottleneckNoRelu(old_block.conv1, old_block.bn1,
                                                          old_block.relu, old_block.conv2, old_block.bn2,
                                                          old_block.conv3, old_block.bn3, old_block.downsample)
            else:
                warnings.warn("Warning: ReLU not removed from last block.")

        # Changes the new head to a CosineLinear
        model.heads[-1] = CosineLinear(model.heads[-1].in_features, model.heads[-1].out_features)
        '''

        model.to(self.device)

        # yujun: debug to make sure this one is ok
        if self.ddp:
            # ------------------------------------------------------------------------------------------------------------------------
            # Add for DDP by NieX.
            self.model = DDP(self.model.module, device_ids=[self.local_rank], broadcast_buffers=False, find_unused_parameters=True)
            # ------------------------------------------------------------------------------------------------------------------------
    
        # if ddp option is activated, need to re-wrap the ddp model
        # yujun: debug to make sure this one is ok
        # The original code has an option called "imprint weights" that seems to initialize the new head.
        # However, this is not mentioned in the paper and doesn't seem to make a significant difference.
        super().pre_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        if t == 0:
            dset = trn_loader.dataset
            trn_loader = torch.utils.data.DataLoader(dset,
                    batch_size=trn_loader.batch_size,
                    sampler=trn_loader.sampler,
                    num_workers=trn_loader.num_workers,
                    pin_memory=trn_loader.pin_memory)

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            dset = trn_loader.dataset + self.exemplars_dataset
            if self.ddp:
                trn_sampler = torch.utils.data.DistributedSampler(dset, shuffle=True)
                trn_loader = torch.utils.data.DataLoader(dset,
                                                         batch_size=trn_loader.batch_size,
                                                         sampler=trn_sampler,
                                                         num_workers=trn_loader.num_workers,
                                                         pin_memory=trn_loader.pin_memory)
            else:
                trn_loader = torch.utils.data.DataLoader(dset,
                                                         batch_size=trn_loader.batch_size,
                                                         shuffle=True,
                                                         num_workers=trn_loader.num_workers,
                                                         pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

    def post_train_process(self, t, trn_loader, val_loader):
        # select exemplars
        if len(self.exemplars_dataset) > 0 and t > 0:
            dset = trn_loader.dataset + self.exemplars_dataset
        else:
            dset = trn_loader.dataset
        trn_loader = torch.utils.data.DataLoader(dset,
            batch_size=trn_loader.batch_size, shuffle=False, num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform, self.ddp)

        """Runs after training all the epochs of the task (after the train session)"""
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()

        # ------------------------------------------------------------------------------------------------------
        # Add for dark branch by NieX.
        model_dict =  self.model.state_dict()
        model_dict_1 = {k: v for k, v in model_dict.items() if 'new_1' in k}
        model_dict_2 = {k: v for k, v in model_dict.items() if 'new_2' in k}
        # model_dict_3 = {k: v for k, v in model_dict.items() if 'new_1' not in k and 'new_2' not in k}

        model_dict_1_new = OrderedDict()
        for k, v in model_dict_1.items():
            k_new = k.replace('new_1', 'new_2')
            model_dict_1_new[k_new] = v

        model_dict_2.update(model_dict_1_new)

        self.model.load_state_dict(model_dict_2, strict=False)

        # import pdb
        # pdb.set_trace()

        # print(model_dict_2.keys())
        # print(len(model_dict_2.keys()))
        # print(model_dict_2.keys())
        # print(model_dict_1_new.keys())
        # print(len(model_dict_2.keys()))

        # state_dict = {k:v for k,v in model_dict_2.items() if k in model_dict_1.keys()}
        # model_dict_2_new = copy.deepcopy(model_dict_1_new)
        # model_dict_2.update(model_dict_1_new)
        # self.model.load_state_dict(model_dict_2)

        # model_dict_1['model.layer4.1.conv2.bn2_new_1_3.weight']
        # model_dict_2['model.layer4.1.conv2.bn2_new_2_3.weight']
        # print(model_dict_1['model.layer4.1.conv2.bn2_new_1_3.weight']==model_dict_2['model.layer4.1.conv2.bn2_new_2_3.weight'])

        # print(model_dict_1['model.layer3.1.conv2.bn2_new_1_3.weight']==model_dict_2_new['model.layer3.1.conv2.bn2_new_2_3.weight'])
        # print(type(model_dict_2_new))
        # print(type(model_dict_1_new))
        # print(type(model_dict_2))
        # print(type(model_dict_1))
        # print(type(model_dict))
        # print(model_dict_2_new==model_dict_2)

        # ------------------------------------------------------------------------------------------------------

        # Make the old model return outputs without the sigma (eta in paper) factor
        if self.ddp:
            for h in self.ref_model.module.heads:
                h.train()
            self.ref_model.module.freeze_all()
        else:
            for h in self.ref_model.heads:
                h.train()
            self.ref_model.freeze_all()

        # balanced finetuning
        # balanced finetune has not supported ddp yet !!!!
        
    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        # if self.fix_bn and t > 0:
        #     self.model.freeze_bn()
        for images, targets in trn_loader:
            images, targets = images.to(self.device), targets.to(self.device)

            # --------------------------------------------------------------------------------------------------------------
            # Add for Decom by NieX.
            ref_outputs = None
            ref_features = None

            # Add for DenseNet in 10_22.
            if self.network == 'densenet121' \
                or self.network == 'densenet169' \
                or self.network == 'densenet201' \
                or self.network == 'densenet121_cifar' \
                or self.network == 'densenet169_cifar' \
                or self.network == 'densenet201_cifar' \
                or self.network == 'densenet121_downcha' \
                or self.network == 'densenet169_downcha' \
                or self.network == 'densenet201_downcha' \
                or self.network == 'densenet121_cifar_downcha' \
                or self.network == 'densenet169_cifar_downcha' \
                or self.network == 'densenet201_cifar_downcha':
                from .Distill_loss_detach_dense import Distill_Loss_detach
            else:
                from .Distill_loss_detach import Distill_Loss_detach

            # Distill_Loss_detach
            outputs, _, pod_features = self.model((images, images), return_features=True)
            loss = self.l_gamma * Distill_Loss_detach()(pod_features)

            if t == 0:
                targets = torch.cat((targets, targets), dim=0)
                outputs, _, _ = self.model((images, images), return_features=True)
                # outputs = self.model((images, images))
            else:
                buf_outputs = []
                
                # old_class_num = sum([h.out_features for h in self.model.heads[:-1]])
                # new_class_num = self.model.heads[-1].out_features
                if self.ddp:
                    old_class_num = sum([h.out_features for h in self.model.module.heads[:-1]])
                    new_class_num = self.model.module.heads[-1].out_features
                else:
                    old_class_num = sum([h.out_features for h in self.model.heads[:-1]])
                    new_class_num = self.model.heads[-1].out_features

                bz = trn_loader.batch_size
                old_num = int(bz*old_class_num/(old_class_num+new_class_num))
                new_num = bz-old_num
                old_indx = np.random.choice(len(self.exemplars_dataset),size=old_num, replace=False)
                # new_indx = np.random.choice(images.shape[0], size=new_num, replace=False)
                new_indx = np.random.choice(images.shape[0], size=min(images.shape[0], new_num), replace=False)

                balanced_images = []
                balanced_targets = []

                for oi in old_indx:
                    balanced_images.append(self.exemplars_dataset[oi][0].to(self.device))
                    balanced_targets.append(self.exemplars_dataset[oi][1])
                for ni in new_indx:
                    balanced_images.append(images[ni])
                    balanced_targets.append(targets[ni].cpu().numpy())
                
                balanced_images = torch.stack(balanced_images, dim=0)
                balanced_targets = torch.tensor(balanced_targets).to(self.device)

                outputs, _, _ = self.model((images, balanced_images), return_features=True)
                # outputs, _ = self.model((images, balanced_images))
                for i in range(len(outputs)):
                    # buf_outputs.append(outputs[i]['wsigma'][images.shape[0]:])
                    buf_outputs.append(outputs[i][images.shape[0]:])
                    # outputs[i]['wsigma'] = outputs[i]['wsigma'][0:images.shape[0]]
                    # outputs[i]['wosigma'] = outputs[i]['wosigma'][0:images.shape[0]]
                    outputs[i] = outputs[i][0:images.shape[0]]

                buf_outputs = torch.cat([o for o in buf_outputs], dim=1)

                # loss += nn.CrossEntropyLoss()(buf_outputs, balanced_targets)
                loss = loss + self.l_beta*nn.CrossEntropyLoss()(buf_outputs, balanced_targets)

            # loss += self.criterion(t, outputs, targets, ref_outputs, features, ref_features)
            # loss += self.criterion(t, outputs, targets)
            loss = loss + self.l_alpha*self.criterion(t, outputs, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
    # def train_epoch(self, t, trn_loader):
    #     """Runs a single epoch"""
    #     self.model.train()
    #     # if self.fix_bn and t > 0:
    #     #     self.model.freeze_bn()
    #     import time
    #     for images, targets in trn_loader:
            
    #         images, targets = images.to(self.device), targets.to(self.device)
            
    #         # Forward current model
    #         outputs, _ = self.model(images, return_features=True)
    #         # Forward previous model

    #         loss = self.criterion(t, outputs, targets)
    #         # loss = self.criterion(t, outputs, targets, features)
    #         # Backward
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()

    def criterion(self, t, outputs, targets, output_old=None, features=None, features_old=None):
        """Returns the loss value"""

        if type(outputs[0])==dict:
            outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)
        else:
            outputs = torch.cat([o for o in outputs], dim=1)

        return torch.nn.functional.cross_entropy(outputs, targets)
        
    @staticmethod
    def warmup_luci_loss(outputs, targets):
        if type(outputs) == dict:
            # needed during train
            return torch.nn.functional.cross_entropy(outputs['wosigma'], targets)
        else:
            # needed during eval()
            return torch.nn.functional.cross_entropy(outputs, targets)
