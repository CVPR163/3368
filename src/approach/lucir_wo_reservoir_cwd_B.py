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

from .buffer import Buffer
import time

from .Distill_loss_detach import Distill_Loss_detach

# ----------------------- auxiliary loss specifics -----------------------------
from .aux_loss import DecorrelateLossClass
from .utils import reduce_tensor_mean, reduce_tensor_sum, global_gather
# ------------------------------------------------------------------------------

from collections import OrderedDict



class Appr(Inc_Learning_Appr):
    """Class implementing the Learning a Unified Classifier Incrementally via Rebalancing (LUCI) approach
    described in http://dahua.me/publications/dhl19_increclass.pdf
    Original code available at https://github.com/hshustc/CVPR19_Incremental_Learning
    """

    # Sec. 4.1: "we used the method proposed in [29] based on herd selection" and "first one stores a constant number of
    # samples for each old class (e.g. R_per=20) (...) we adopt the first strategy"
    def __init__(self, model, device, l_alpha, l_beta, l_gamma, buffer_size, minibatch_size_1, minibatch_size_2, network, nepochs=160, lr=0.1, decay_mile_stone=[80,120], lr_decay=0.1, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, ddp=False, local_rank=0, logger=None, exemplars_dataset=None,
                 lamb=5., lamb_mr=1., dist=0.5, K=2, scale=1.0,
                 remove_less_forget=False, remove_margin_ranking=False, remove_adapt_lamda=False,
                 # # auxiliary loss specifics
                 aux_coef=0.1, reject_threshold=1, first_task_lr=0.1, first_task_bz=128, bal_ft=False):
        super(Appr, self).__init__(model, device, nepochs, lr, decay_mile_stone, lr_decay, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, ddp, local_rank,
                                   logger, exemplars_dataset)
        self.lamb = lamb
        self.lamb_mr = lamb_mr
        self.dist = dist
        self.K = K
        self.less_forget = not remove_less_forget
        self.margin_ranking = not remove_margin_ranking
        self.adapt_lamda = not remove_adapt_lamda

        self.lamda = self.lamb
        self.ref_model = None

        self.warmup_loss = self.warmup_luci_loss
        self.first_task_lr = first_task_lr
        self.first_task_bz = first_task_bz
        self.bal_ft = bal_ft

        # ---------------- auxiliary loss specifics -------------------------------------
        self.scale = scale
        # -------------------------------------------------------------------------------

        # LUCIR is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: LUCIR is expected to use exemplars. Check documentation.")

        # ---------------- auxiliary loss specifics -------------------
        self.aux_coef = aux_coef
        self.reject_threshold = reject_threshold

        out_size = self.model.module.out_size if self.ddp else self.model.out_size

        self.aux_loss = DecorrelateLossClass(reject_threshold=self.reject_threshold, ddp=ddp)
        # -------------------------------------------------------------

        # flag variable that identify whether we're learning the first task
        self.first_task = True

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
        # Sec. 4.1: "lambda base is set to 5 for CIFAR100 and 10 for ImageNet"
        parser.add_argument('--lamb', default=5., type=float, required=False,
                            help='Trade-off for distillation loss (default=%(default)s)')
        # Loss weight for the Inter-Class separation loss constraint, set to 1 in the original code
        parser.add_argument('--lamb-mr', default=1., type=float, required=False,
                            help='Trade-off for the MR loss (default=%(default)s)')
        # Sec 4.1: "m is set to 0.5 for all experiments"
        parser.add_argument('--dist', default=.5, type=float, required=False,
                            help='Margin threshold for the MR loss (default=%(default)s)')
        # Sec 4.1: "K is set to 2"
        parser.add_argument('--K', default=2, type=int, required=False,
                            help='Number of "new class embeddings chosen as hard negatives '
                                 'for MR loss (default=%(default)s)')

        # Flags for ablating the approach
        parser.add_argument('--remove-less-forget', action='store_true', required=False,
                            help='Deactivate Less-Forget loss constraint(default=%(default)s)')
        parser.add_argument('--remove-margin-ranking', action='store_true', required=False,
                            help='Deactivate Inter-Class separation loss constraint (default=%(default)s)')
        parser.add_argument('--remove-adapt-lamda', action='store_true', required=False,
                            help='Deactivate adapting lambda according to the number of classes (default=%(default)s)')
        parser.add_argument('--first-task-lr', default=0.1, type=float)
        parser.add_argument('--first-task-bz', default=128, type=int)
        parser.add_argument('--bal-ft', action='store_true', help='whether to do class bal ft')
        # ------------------ auxiliary loss specifics -------------------
        parser.add_argument('--aux-coef', default=0.1, type=float, required=False,
                            help='coefficient for auxiliary loss')
        parser.add_argument('--reject-threshold', default=1, type=int, required=True,
                            help='rejection threshold for calculating correlation')
        # ---------------------------------------------------------------
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
        if self.less_forget:
            # Don't update heads when Less-Forgetting constraint is activated (from original code)
            params = list(model.model.parameters()) + list(model.heads[-1].parameters())
        else:
            params = model.parameters()

        if self.first_task:
            self.first_task = False
            optimizer = torch.optim.SGD(params, lr=self.first_task_lr, weight_decay=self.wd, momentum=self.momentum)
        else:
            optimizer = torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        print(optimizer.param_groups[0]['lr'])
        return optimizer

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        # ------------------------------------------------------------------------------------------------------
        # Add for other networks by NieX.
        if self.network == 'resnet18_cifar' \
            or self.network == 'resnet18_cifar_conv1' \
            or self.network == 'resnet18_cifar_conv1_s' \
            or self.network == 'resnet18_cifar_group' \
            or self.network == 'resnet18_cifar_group_s' \
            or self.network == 'resnet18' \
            or self.network == 'resnet34_cifar' \
            or self.network == 'resnet34' \
            or self.network == 'resnet50_cifar' \
            or self.network == 'resnet50' \
            or self.network == 'resnet101_cifar' \
            or self.network == 'resnet101':
            from .lucir_utils import CosineLinear, BasicBlockNoRelu, BottleneckNoRelu
        # ------------------------------------------------------------------------------------------------------
        elif self.network == 'resnet18_cifar_plus':
            from .lucir_utils_plus import CosineLinear, BasicBlockNoRelu, BottleneckNoRelu
        elif self.network == 'resnet18_cifar_plus_asy':
            from .lucir_utils_plus_asy import CosineLinear, BasicBlockNoRelu, BottleneckNoRelu
        elif self.network == 'resnet18_cifar_plus_res':
            from .lucir_utils_plus_res import CosineLinear, BasicBlockNoRelu, BottleneckNoRelu
        else:
            print('args.network input error!')
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
                # ------------------------------------------------------------------------------------------------------
            else:
                warnings.warn("Warning: ReLU not removed from last block.")
        # Changes the new head to a CosineLinear
        model.heads[-1] = CosineLinear(model.heads[-1].in_features, model.heads[-1].out_features)
        model.to(self.device)
        if t > 0:
            # Share sigma (Eta in paper) between all the heads
            model.heads[-1].sigma = model.heads[-2].sigma
            # Fix previous heads when Less-Forgetting constraint is activated (from original code)
            if self.less_forget:
                for h in model.heads[:-1]:
                    for param in h.parameters():
                        param.requires_grad = False
                model.heads[-1].sigma.requires_grad = True
            # Eq. 7: Adaptive lambda
            if self.adapt_lamda:
                self.lamda = self.lamb * math.sqrt(sum([h.out_features for h in model.heads[:-1]])
                                                   / model.heads[-1].out_features)
                if self.local_rank == 0:
                    print('lambda value after adaptation: ', self.lamda)

        # if ddp option is activated, need to re-wrap the ddp model
        # yujun: debug to make sure this one is ok
        if self.ddp:
            self.model = DDP(self.model.module, device_ids=[self.local_rank], broadcast_buffers=False, find_unused_parameters=True)
        # The original code has an option called "imprint weights" that seems to initialize the new head.
        # However, this is not mentioned in the paper and doesn't seem to make a significant difference.
        super().pre_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        if t == 0:
            dset = trn_loader.dataset
            trn_loader = torch.utils.data.DataLoader(dset,
                    batch_size=self.first_task_bz,
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
        # import pdb
        # pdb.set_trace()

        super().train_loop(t, trn_loader, val_loader)

        #######################
        # if self.ddp:
        #     self.model.module.set_state_dict(best_model)
        # else:
        #     self.model.set_state_dict(best_model)
        #######################

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
        if t > 0 and self.bal_ft:
            balanced_loader = torch.utils.data.DataLoader(self.exemplars_dataset,
                batch_size=trn_loader.batch_size, shuffle=True, num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory)
            params = [h.weight for h in self.model.heads]
            optim = torch.optim.SGD(params, lr=0.01, weight_decay=self.wd, momentum=0.9)
            # self.model.model.eval()
            # self.model.heads.train()
            self.model.train()
            print('start classifier balance ft')
            for _ in range(20):
                for images, targets in balanced_loader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = self.model(images)
                    outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)
                    loss = nn.CrossEntropyLoss(None)(outputs, targets)

                    optim.zero_grad()
                    loss.backward()
                    optim.step()
            print('balanced ft complete')

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images, targets = images.to(self.device), targets.to(self.device)

            # --------------------------------------------------------------------------------------------------------------
            # Add for Decom by NieX.
            ref_outputs = None
            ref_features = None

            # Distill_Loss_detach
            outputs, _, pod_features = self.model((images, images), return_features=True)
            loss = self.l_gamma * Distill_Loss_detach()(pod_features)

            if t == 0:
                targets = torch.cat((targets, targets), dim=0)
                outputs, features, _ = self.model((images, images), return_features=True)
                if t > 0:
                    ref_outputs, ref_features, _ = self.ref_model((images, images), return_features=True)
                features = F.normalize(features, p=2, dim=-1)
                loss_sc = self.aux_loss(features, targets)
                loss = loss + self.criterion(t, outputs, targets, ref_outputs, features, ref_features)
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

                outputs, features, _ = self.model((images, balanced_images), return_features=True)
                for i in range(len(outputs)):
                    buf_outputs.append(outputs[i]['wsigma'][images.shape[0]:])

                    outputs[i]['wsigma'] = outputs[i]['wsigma'][0:images.shape[0]]
                    outputs[i]['wosigma'] = outputs[i]['wosigma'][0:images.shape[0]]

                buf_outputs = torch.cat([o for o in buf_outputs], dim=1)

                # loss += nn.CrossEntropyLoss()(buf_outputs, balanced_targets)
                loss = loss + self.l_beta*nn.CrossEntropyLoss()(buf_outputs, balanced_targets)

                features = features[0:images.shape[0]]
                if t > 0:
                    ref_outputs, ref_features, _ = self.ref_model((images, images), return_features=True)
                    for i in range(len(ref_outputs)):
                        ref_outputs[i]['wsigma'] = ref_outputs[i]['wsigma'][0:images.shape[0]]
                        ref_outputs[i]['wosigma'] = ref_outputs[i]['wosigma'][0:images.shape[0]]
                    ref_features = ref_features[0:images.shape[0]]

                loss_sc = 0.0
                loss = loss + self.l_alpha*self.criterion(t, outputs, targets, ref_outputs, features, ref_features)

            loss = loss + self.aux_coef*loss_sc


            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # def train_epoch(self, t, trn_loader):
    #     """Runs a single epoch"""

    #     # ------------------------------------------------------------------------------------------
    #     # Add for Decom by NieX.
    #     # self.transform = transforms.Compose(
    #     #         [transforms.RandomCrop(32, padding=4),
    #     #          transforms.RandomHorizontalFlip(),
    #     #          transforms.ToTensor(),
    #     #          transforms.Normalize((0.5071, 0.4867, 0.4408),
    #     #                               (0.2675, 0.2565, 0.2761))])
    #     # ------------------------------------------------------------------------------------------

    #     self.model.train()
    #     if self.fix_bn and t > 0:
    #         self.model.freeze_bn()
    #     for images, targets in trn_loader:
    #         images, targets = images.to(self.device), targets.to(self.device)

    #         # --------------------------------------------------------------------------------------------------------------
    #         # Add for Decom by NieX.
    #         ref_outputs = None
    #         ref_features = None

    #         # Distill_Loss_detach
    #         outputs, _, pod_features = self.model((images, images), return_features=True)
    #         loss = self.l_gamma * Distill_Loss_detach()(pod_features)

    #         if self.buffer.is_empty() or t == 0:
    #             targets = torch.cat((targets, targets), dim=0)
    #             outputs, features, _ = self.model((images, images), return_features=True)
    #             if t > 0:
    #                 ref_outputs, ref_features, _ = self.ref_model((images, images), return_features=True)

    #         else:
    #             buf_outputs = []
    #             buf_inputs, buf_labels = self.buffer.get_data(self.minibatch_size, transform=None)
    #             outputs, features, _ = self.model((images, buf_inputs), return_features=True)

    #             for i in range(len(outputs)):
    #                 # if i == 0:
    #                 #     buf_outputs = torch.split(outputs[i]['wsigma'], split_size_or_sections=[images.shape[0], buf_inputs.shape[0]], dim=0)[1]
    #                 # else:
    #                 #     buf_outputs = torch.cat((buf_outputs, torch.split(outputs[i]['wsigma'], split_size_or_sections=[images.shape[0], buf_inputs.shape[0]], dim=0)[1]), dim=0)
    #                 # buf_outputs = torch.split(outputs[i]['wsigma'], split_size_or_sections=[images.shape[0], buf_inputs.shape[0]], dim=0)[1]

    #                 # buf_outputs.append(torch.split(outputs[i]['wsigma'], split_size_or_sections=[images.shape[0], buf_inputs.shape[0]], dim=0)[1])
    #                 buf_outputs.append(outputs[i]['wsigma'][images.shape[0]:])

    #                 # import pdb
    #                 # pdb.set_trace()

    #                 outputs[i]['wsigma'] = outputs[i]['wsigma'][0:images.shape[0]]
    #                 outputs[i]['wosigma'] = outputs[i]['wosigma'][0:images.shape[0]]

    #             buf_outputs = torch.cat([o for o in buf_outputs], dim=1)

    #             # loss += self.args.beta * self.loss(buf_outputs, buf_labels)

    #             # import pdb
    #             # pdb.set_trace()

    #             # loss += nn.CrossEntropyLoss()(buf_outputs, buf_labels)
    #             loss = loss + self.l_beta*nn.CrossEntropyLoss()(buf_outputs, buf_labels)

    #             # features, _ = torch.split(features_all, split_size_or_sections=[images.shape[0], buf_inputs.shape[0]], dim=0)
    #             features = features[0:images.shape[0]]
    #             if t > 0:
    #                 ref_outputs, ref_features, _ = self.ref_model((images, images), return_features=True)
    #                 for i in range(len(ref_outputs)):
    #                     # import pdb
    #                     # pdb.set_trace()
    #                     ref_outputs[i]['wsigma'] = ref_outputs[i]['wsigma'][0:images.shape[0]]
    #                     ref_outputs[i]['wosigma'] = ref_outputs[i]['wosigma'][0:images.shape[0]]
    #                 # ref_features, _ = torch.split(ref_features, split_size_or_sections=[images.shape[0], buf_inputs.shape[0]], dim=0)
    #                 ref_features = ref_features[0:images.shape[0]]

    #         # add data
    #         self.buffer.add_data(examples=images, labels=targets)


    #         # --------------------------------------------------------------------------------------------------------------

    #         # --------------------------------------------------------------------------------------------------------------
    #         # if t > 0:
    #         #     ref_outputs, ref_features = self.ref_model(images, return_features=True)

    #         # if t > 0:
    #         #     images_all = (images, images)
    #         #     ref_outputs, ref_features = self.ref_model(images_all, return_features=True)

    #         # --------------------------------------------------------------------------------------------------------------

    #         # loss += self.criterion(t, outputs, targets, ref_outputs, features, ref_features)
    #         loss = loss + self.l_alpha*self.criterion(t, outputs, targets, ref_outputs, features, ref_features)

    #         # import pdb
    #         # pdb.set_trace()

    #         # Backward
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()

    def del_tensor_ele_n(arr, index):
        """
        arr: input tensor
        index: 
        n: from index，the number of removing items
        """
        arr1 = arr[0:index]
        arr2 = arr[index+n:]
        return torch.cat((arr1,arr2),dim=0)

    def criterion(self, t, outputs, targets, ref_outputs=None, features=None, ref_features=None):
        """Returns the loss value"""
        if ref_outputs is None or ref_features is None:

            if type(outputs[0]) == dict:
                outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)
            else:
                outputs = torch.cat(outputs, dim=1)

            # import pdb
            # pdb.set_trace()

            # Eq. 1: regular cross entropy
            loss = nn.CrossEntropyLoss(None)(outputs, targets)
        else:
            if self.less_forget:
                # Eq. 6: Less-Forgetting constraint
                # import pdb
                # pdb.set_trace()
                loss_dist = nn.CosineEmbeddingLoss()(features, ref_features.detach(),
                                                     torch.ones(targets.shape[0]).to(self.device)) * self.lamda
            else:
                # Scores before scale, [-1, 1]
                ref_outputs = torch.cat([ro['wosigma'] for ro in ref_outputs], dim=1).detach()
                old_scores = torch.cat([o['wosigma'] for o in outputs[:-1]], dim=1)
                num_old_classes = ref_outputs.shape[1]

                # Eq. 5: Modified distillation loss for cosine normalization
                loss_dist = nn.MSELoss()(old_scores, ref_outputs) * self.lamda * num_old_classes

            loss_mr = torch.zeros(1).to(self.device)
            if self.margin_ranking:
                # Scores before scale, [-1, 1]
                outputs_wos = torch.cat([o['wosigma'] for o in outputs], dim=1)
                num_old_classes = outputs_wos.shape[1] - outputs[-1]['wosigma'].shape[1]

                # Sec 3.4: "We select those new classes that yield highest responses to x (...)"
                # The index of hard samples, i.e., samples from old classes
                hard_index = targets < num_old_classes
                hard_num = hard_index.sum()

                if hard_num > 0:
                    # Get "ground truth" scores
                    gt_scores = outputs_wos.gather(1, targets.unsqueeze(1))[hard_index]
                    gt_scores = gt_scores.repeat(1, self.K)

                    # Get top-K scores on novel classes
                    max_novel_scores = outputs_wos[hard_index, num_old_classes:].topk(self.K, dim=1)[0]

                    assert (gt_scores.size() == max_novel_scores.size())
                    assert (gt_scores.size(0) == hard_num)
                    # Eq. 8: margin ranking loss
                    loss_mr = nn.MarginRankingLoss(margin=self.dist)(gt_scores.view(-1, 1),
                                                                     max_novel_scores.view(-1, 1),
                                                                     torch.ones(hard_num * self.K).to(self.device).view(-1, 1))
                    loss_mr *= self.lamb_mr

            # Eq. 1: regular cross entropy
            loss_ce = nn.CrossEntropyLoss()(torch.cat([o['wsigma'] for o in outputs], dim=1), targets)
            
            # Eq. 9: integrated objective
            # ============================================================================================
            # To do.
            if loss_mr == 0:
                loss = loss_dist + loss_ce
            else:
                loss = loss_dist + loss_ce + loss_mr
            # loss = loss_dist + loss_ce + loss_mr
            # ============================================================================================
        return loss

    @staticmethod
    def warmup_luci_loss(outputs, targets):
        if type(outputs) == dict:
            # needed during train
            return torch.nn.functional.cross_entropy(outputs['wosigma'], targets)
        else:
            # needed during eval()
            return torch.nn.functional.cross_entropy(outputs, targets)
