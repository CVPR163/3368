import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from .lucir_utils import CosineLinear, BasicBlockNoRelu, BottleneckNoRelu

import moxing as mox

class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, l_alpha, l_beta, l_gamma, buffer_size, minibatch_size_1, minibatch_size_2, network, nepochs=160, lr=0.1, decay_mile_stone=[80,120], lr_decay=0.1, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, ddp=False, local_rank=0, logger=None, exemplars_dataset=None,
                 ##########################################################################################
                 # Add ksi
                 ##########################################################################################
                 lamb=5., lamb_mr=1., dist=0.5, K=2, ksi=5e-3):
        super(Appr, self).__init__(model, device, nepochs, lr, decay_mile_stone, lr_decay, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, ddp, local_rank,
                                   logger, exemplars_dataset)
        self.trn_datasets = []
        self.val_datasets = []

        ##########################################################################################
        self.ksi = ksi
        ##########################################################################################

        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        assert (have_exemplars == 0), 'Warning: Joint does not use exemplars. Comment this line to force it.'

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        if self.ddp:
            model = self.model.module
        else:
            model = self.model

        if t == 0:
            # Sec. 4.1: "the ReLU in the penultimate layer is removed to allow the features to take both positive and
            # negative values"
            if model.model.__class__.__name__ == 'ResNetCifar':
                old_block = model.model.layer3[-1]
                model.model.layer3[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                               old_block.conv2, old_block.bn2, old_block.downsample)
            elif model.model.__class__.__name__ == 'ResNet':
                old_block = model.model.layer4[-1]
                # model.model.layer4[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                #                                                old_block.conv2, old_block.bn2, old_block.downsample)
                model.model.layer4[-1] = BasicBlockNoRelu(old_block.conv1, old_block.relu,
                                                               old_block.conv2, old_block.downsample)
            elif model.model.__class__.__name__ == 'ResNetBottleneck':
                old_block = model.model.layer4[-1]
                # ------------------------------------------------------------------------------------------------------
                # Add for other networks by NieX.
                model.model.layer4[-1] = BottleneckNoRelu(old_block.conv1,
                                                          old_block.relu, old_block.conv2,
                                                          old_block.conv3, old_block.downsample)
                # Add for DenseNet in 10_22.
            elif model.model.__class__.__name__ == 'DenseNet':
                pass
                # ------------------------------------------------------------------------------------------------------
            else:
                warnings.warn("Warning: ReLU not removed from last block.")
        # Changes the new head to a CosineLinear
        model.heads[-1] = CosineLinear(model.heads[-1].in_features, model.heads[-1].out_features)
        model.to(self.device)

        # if ddp option is activated, need to re-wrap the ddp model
        # yujun: debug to make sure this one is ok
        if self.ddp:
            self.model = DDP(self.model.module, device_ids=[self.local_rank])
        # The original code has an option called "imprint weights" that seems to initialize the new head.
        # However, this is not mentioned in the paper and doesn't seem to make a significant difference.
        super().pre_train_process(t, trn_loader)

    def post_train_process(self, t, trn_loader, val_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add new datasets to existing cumulative ones
        self.trn_datasets.append(trn_loader.dataset)
        self.val_datasets.append(val_loader.dataset)
        trn_dset = JointDataset(self.trn_datasets)
        val_dset = JointDataset(self.val_datasets)
        trn_loader = DataLoader(trn_dset,
                                batch_size=trn_loader.batch_size,
                                shuffle=True,
                                num_workers=trn_loader.num_workers,
                                pin_memory=trn_loader.pin_memory)
        val_loader = DataLoader(val_dset,
                                batch_size=val_loader.batch_size,
                                shuffle=False,
                                num_workers=val_loader.num_workers,
                                pin_memory=val_loader.pin_memory)
        # continue training as usual
        super().train_loop(t, trn_loader, val_loader)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        ##########################################################################################
        mox.file.make_dirs('/cache/pretrain_model')
        print('Copy pretrained model to cache ...')
        model_path = '/cache/pretrain_model'

        # mox.file.copy_parallel('s3://bucket-6643/niexing/output/C_C1z_Bic_Decom_Distill_det_br1_C3_1st128in_exe_A1G1/cifar100_icarl_bic_nc_first_50_ntask_6/models', '/cache/task1_model')
        import os
        mox.file.copy_parallel('s3://bucket-6643/niexing/output/C_C1z_Cwd_Decom_Distill_det_br1_R18_11_1st128in_exe_A8G1/cifar100_icarl_lucir_wo_reservoir_cwd_nc_first_50_ntask_11/models', model_path)
        pretrained_model = torch.load(os.path.join(model_path, 'task10.ckpt'))
        self.model.load_state_dict(pretrained_model, strict=False)

        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                # import pdb
                # pdb.set_trace()
                noise = torch.zeros(parameter.shape)
                from torch import nn
                noise = nn.init.normal_(tensor=noise, mean=0.0, std=self.ksi)
                parameter += noise.to(self.device)
            # Add for freezing model.
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad = False
            ##########################################################################################
            for images, targets in trn_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                # Forward current model
                outputs = self.model(images)
                loss = self.criterion(t, outputs, targets)
                ##########################################################################################
                # Backward
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
        ##########################################################################################

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if type(outputs[0])==dict:
            outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)
        else:
            outputs = torch.cat([o for o in outputs], dim=1)
        return torch.nn.functional.cross_entropy(outputs, targets)

class JointDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset accumulates each task dataset incrementally"""

    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum([len(d) for d in self.datasets])

    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        for d in self.datasets:
            if len(d) <= index:
                index -= len(d)
            else:
                x, y = d[index]
                return x, y
