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


from .utils import *
import heapq
from .aux_loss import CenterLoss

from sklearn.cluster import KMeans

# from .Distill_loss_detach import Distill_Loss_detach

import time

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
                 lamb=5., lamb_mr=1., dist=0.5, K=2,
                 remove_less_forget=False, remove_margin_ranking=False, remove_adapt_lamda=False,
                 first_task_lr=0.1, first_task_bz=128, bal_ft=False):
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
        self.device = device

        self.lamda = self.lamb
        self.ref_model = None

        self.warmup_loss = self.warmup_luci_loss
        self.first_task_lr = first_task_lr
        self.first_task_bz = first_task_bz
        self.bal_ft = bal_ft

        # LUCIR is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: LUCIR is expected to use exemplars. Check documentation.")

        # flag variable that identify whether we're learning the first task
        self.first_task = True

        # ----------------------------------------------------------------------------------------------
        # self.buffer_size = buffer_size
        # self.minibatch_size_1 = minibatch_size_1
        # self.minibatch_size_2 = minibatch_size_2
        # self.buffer = Buffer(self.buffer_size, self.device)
        self.l_alpha = l_alpha
        self.l_beta = l_beta
        self.l_gamma = l_gamma
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
        return parser.parse_known_args(args)

    # def _gen_weights(self):
    #     self.model.heads[-1].add_imprinted_classes(self._n_classes, self._task_size, self.inc_dataset)
        # if self._weight_generation:
        #     utils.add_new_weights(
        #         self._network, self._weight_generation if self._task != 0 else "basic",
        #         self._n_classes, self._task_size, self.inc_dataset
        #     )

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
        # Add for DenseNet in 10_22.
        elif self.network == 'densenet121' \
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
            from .lucir_utils import CosineLinear
        # ------------------------------------------------------------------------------------------------------
        elif self.network == 'resnet18_cifar_plus':
            from .lucir_utils_plus import CosineLinear, BasicBlockNoRelu, BottleneckNoRelu
        elif self.network == 'resnet18_cifar_plus_asy':
            from .lucir_utils_plus_asy import CosineLinear, BasicBlockNoRelu, BottleneckNoRelu
        elif self.network == 'resnet18_cifar_plus_res':
            from .lucir_utils_plus_res import CosineLinear, BasicBlockNoRelu, BottleneckNoRelu
        else:
            from .lucir_utils import CosineLinear, BasicBlockNoRelu, BottleneckNoRelu
            # print('args.network input error!')
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
        # Changes the new head to a MultiCosineLinear

        model.heads[-1] = MultiCosineLinear(model.heads[-1].in_features, model.heads[-1].out_features,)
        # model.heads[-1] = CosineLinear(model.heads[-1].in_features, model.heads[-1].out_features)
        model.to(self.device)
        # if t > 0:
        #     # Share sigma (Eta in paper) between all the heads
        #     model.heads[-1].sigma = model.heads[-2].sigma
        #     # Fix previous heads when Less-Forgetting constraint is activated (from original code)
        #     if self.less_forget:
        #         for h in model.heads[:-1]:
        #             for param in h.parameters():
        #                 param.requires_grad = False
        #         model.heads[-1].sigma.requires_grad = True
        #     # Eq. 7: Adaptive lambda
        #     if self.adapt_lamda:
        #         self.lamda = self.lamb * math.sqrt(sum([h.out_features for h in model.heads[:-1]])
        #                                            / model.heads[-1].out_features)
        #         if self.local_rank == 0:
        #             print('lambda value after adaptation: ', self.lamda)

        # if ddp option is activated, need to re-wrap the ddp model
        # yujun: debug to make sure this one is ok
        if self.ddp:
            # ------------------------------------------------------------------------------------------------------------------------
            # Add for DDP by NieX.
            self.model = DDP(self.model.module, device_ids=[self.local_rank], broadcast_buffers=False, find_unused_parameters=True)
            # ------------------------------------------------------------------------------------------------------------------------
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
                outputs, features, _ = self.model((images, images), return_features=True)
                ##############################################################################################
                # Add by NieX for pod_loss.
                features_mid = None
                ref_features_mid = None
                ##############################################################################################
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
                    # buf_outputs.append(outputs[i]['wsigma'][images.shape[0]:])
                    buf_outputs.append(outputs[i][images.shape[0]:])
                    # outputs[i]['wsigma'] = outputs[i]['wsigma'][0:images.shape[0]]
                    # outputs[i]['wosigma'] = outputs[i]['wosigma'][0:images.shape[0]]
                    outputs[i] = outputs[i][0:images.shape[0]]

                buf_outputs = torch.cat([o for o in buf_outputs], dim=1)

                # loss += nn.CrossEntropyLoss()(buf_outputs, balanced_targets)
                loss = loss + self.l_beta*nn.CrossEntropyLoss()(buf_outputs, balanced_targets)

                features = features[0:images.shape[0]]
                ##############################################################################################
                # Add by NieX for pod_loss.
                ref_outputs, ref_features, ref_features_mid = self.ref_model((images, images), return_features=True)
                ##############################################################################################
                for i in range(len(ref_outputs)):
                    # ref_outputs[i]['wsigma'] = ref_outputs[i]['wsigma'][0:images.shape[0]]
                    # ref_outputs[i]['wosigma'] = ref_outputs[i]['wosigma'][0:images.shape[0]]
                    ref_outputs[i] = ref_outputs[i][0:images.shape[0]]        
                ref_features = ref_features[0:images.shape[0]]
                ##############################################################################################
                # Add by NieX for pod_loss.
                features_mid = pod_features[0:images.shape[0]]
                ref_features_mid = ref_features_mid[0:images.shape[0]]
                ##############################################################################################

            loss = loss + self.l_alpha*self.criterion(t, outputs, targets, ref_outputs, features, ref_features, features_mid, ref_features_mid)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def criterion(self, t, outputs, targets, ref_outputs=None, features=None, ref_features=None, features_mid=None, ref_features_mid=None):
        loss = 0.
        outputs = torch.cat(outputs, axis=1)
        # print(outputs.shape)
        # nca 分类loss
        loss += nca(
                outputs,
                targets,
                # memory_flags=memory_flags,
                # class_weights=self._class_weights
            )

        # ce_loss 二选一
        # loss = F.cross_entropy(outputs, targets)

        # --------------------
        # Distillation losses:
        # --------------------

        if t > 0 and ref_features is not None: 
            # print(ref_features.shape)
            pod_flat_loss = embeddings_similarity(ref_features, features)
            loss += pod_flat_loss
            ##############################################################################################
            # Add by NieX for pod_loss.
            # 这里 ref_features_mid和features_mid 就是你模型拿出来的中间层特征，分别是来自新旧模型分支1的。
            ref_features_att, _, _ = ref_features_mid[0]
            features_att, _, _ = features_mid[0]
            pod_spatial_loss = pod(
                ref_features_att[1:],
                features_att[1:],
                # memory_flags=memory_flags.bool(),
                # task_percent=(self._task + 1) / self._n_tasks,
                # **self._pod_spatial_config
            )
            loss += 3 * pod_spatial_loss
            ##############################################################################################

        return loss



    @staticmethod
    def warmup_luci_loss(outputs, targets):
        if type(outputs) == dict:
            # needed during train
            return torch.nn.functional.cross_entropy(outputs['wosigma'], targets)
        else:
            # needed during eval()
            return torch.nn.functional.cross_entropy(outputs, targets)

def nca(
    similarities,
    targets,
    class_weights=None,
    focal_gamma=None,
    scale=1,
    margin=0.6,
    exclude_pos_denominator=True,
    hinge_proxynca=False,
    memory_flags=None,
):
    """Compute AMS cross-entropy loss.

    Reference:
        * Goldberger et al.
          Neighbourhood components analysis.
          NeuriPS 2005.
        * Feng Wang et al.
          Additive Margin Softmax for Face Verification.
          Signal Processing Letters 2018.

    :param similarities: Result of cosine similarities between weights and features.
    :param targets: Sparse targets.
    :param scale: Multiplicative factor, can be learned.
    :param margin: Margin applied on the "right" (numerator) similarities.
    :param memory_flags: Flags indicating memory samples, although it could indicate
                         anything else.
    :return: A float scalar loss.
    """
    # print(similarities[0].shape)
    margins = torch.zeros_like(similarities)
    # print(margins.shape)
    # print(targets.shape)
    margins[torch.arange(margins.shape[0]), targets] = margin
    similarities = scale * (similarities - margin)

    if exclude_pos_denominator:  # NCA-specific
        similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.)

        loss = torch.mean(losses)
        return loss

    return F.cross_entropy(similarities, targets, weight=class_weights, reduction="mean")


def embeddings_similarity(features_a, features_b):
    return F.cosine_embedding_loss(
        features_a, features_b,
        torch.ones(features_a.shape[0]).to(features_a.device)
    )

def pod(
    list_attentions_a,
    list_attentions_b,
    collapse_channels="spatial",
    normalize=True,
    memory_flags=None,
    only_old=False,
    **kwargs
):
    """Pooled Output Distillation.

    Reference:
        * Douillard et al.
          Small Task Incremental Learning.
          arXiv 2020.

    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :param memory_flags: Integer flags denoting exemplars.
    :param only_old: Only apply loss to exemplars.
    :return: A float scalar loss.
    """
    assert len(list_attentions_a) == len(list_attentions_b)

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if only_old:
            a = a[memory_flags]
            b = b[memory_flags]
            if len(a) == 0:
                continue

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        if collapse_channels == "channels":
            a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(dim=1).view(b.shape[0], -1)
        elif collapse_channels == "width":
            a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
            b = b.sum(dim=2).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
            b = b.sum(dim=3).view(b.shape[0], -1)
        elif collapse_channels == "gap":
            a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
            b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
        elif collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(list_attentions_a)

class MultiCosineLinear(nn.Module):
    def __init__(self, in_features, out_features, features=None, targets=None, avg_weights_norm=None, proxy_per_class=10):
        super(MultiCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features*proxy_per_class, in_features))
        self.features = features
        self.targets = targets
        self.avg_weights_norm = avg_weights_norm
        self.proxy_per_class = proxy_per_class 
        self.distance = "neg_stable_cosine_distance"
        self.scaling = 1
        self.merging = "softmax"
        self.gamma = 1

        self.reset_parameters()
        self.init_parameters_kmeans()

    def init_parameters_kmeans(self):
        if self.features is not None:
            classes = self.targets.unique()
            for c in sorted(classes):
                c_index = self.targets == c
                c_features = self.targets[c_index]

                c_features_normalized = F.normalize(torch.from_numpy(features), p=2, dim=1)

                c_class_embeddings = torch.mean(c_features_normalized, dim=0)
                c_class_embeddings = F.normalize(c_class_embeddings, dim=0, p=2)

                clusterizer = KMeans(n_clusters=self.proxy_per_class)
                clusterizer.fit(c_features_normalized.numpy())

                for i, center in enumerate(clusterizer.cluster_centers_):
                    self.weight.data[c*proxy_per_class+i] = torch.tensor(center) * avg_weights_norm

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        # if self.sigma is not None:
        #     self.sigma.data.fill_(1)  # for initializaiton of sigma

    def forward(self, features):
        weights = self.weights
        if self.distance == "cosine":
            raw_similarities = cosine_similarity(features, weights)
        elif self.distance == "stable_cosine_distance":
            features = self.scaling * F.normalize(features, p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = stable_cosine_distance(features, weights)
        elif self.distance == "neg_stable_cosine_distance":
            features = self.scaling * F.normalize(features, p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = -1*stable_cosine_distance(features, weights)
        elif self.distance == "prelu_stable_cosine_distance":
            features = self.scaling * F.normalize(F.relu(features), p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = stable_cosine_distance(features, weights)
        elif self.distance == "prelu_neg_stable_cosine_distance":
            features = self.scaling * F.normalize(F.relu(features), p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = -1*stable_cosine_distance(features, weights)
        else:
            raise NotImplementedError("Unknown distance function {}.".format(self.distance))

        similarities = self._reduce_proxies(raw_similarities)

        return similarities

    def _reduce_proxies(self, similarities):
        # shape (batch_size, n_classes * proxy_per_class)
        n_classes = similarities.shape[1] / self.proxy_per_class
        assert n_classes.is_integer(), (similarities.shape[1], self.proxy_per_class)
        n_classes = int(n_classes)
        bs = similarities.shape[0]

        if self.merging == "mean":
            return similarities.view(bs, n_classes, self.proxy_per_class).mean(-1)
        elif self.merging == "softmax":
            simi_per_class = similarities.view(bs, n_classes, self.proxy_per_class)
            attentions = F.softmax(self.gamma * simi_per_class, dim=-1)  # shouldn't be -gamma?
            return (simi_per_class * attentions).sum(-1)
        elif self.merging == "max":
            return similarities.view(bs, n_classes, self.proxy_per_class).max(-1)[0]
        elif self.merging == "min":
            return similarities.view(bs, n_classes, self.proxy_per_class).min(-1)[0]
        else:
            raise ValueError("Unknown merging for multiple centers: {}.".format(self.merging))
        # out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        # if self.sigma is not None:
        #     out_s = self.sigma * out
        # else:
        #     out_s = out
        # if self.training:
        #     return {'wsigma': out_s, 'wosigma': out}
        # else:
        #     return out_s

class CosineClassifier(nn.Module):
    classifier_type = "cosine"

    def __init__(
        self,
        features_dim,
        device,
        *,
        proxy_per_class=10,
        distance="neg_stable_cosine_distance",
        merging="softmax",
        scaling=1,
        gamma=1.,
        use_bias=False,
        type=None,
        pre_fc=None,
        negative_weights_bias=None,
        train_negative_weights=False,
        eval_negative_weights=False
    ):
        super().__init__()

        self.n_classes = 0
        self._weights = nn.ParameterList([])
        self.bias = None
        self.features_dim = features_dim
        self.proxy_per_class = proxy_per_class
        self.device = device
        self.distance = distance
        self.merging = merging
        self.gamma = gamma

        self.negative_weights_bias = negative_weights_bias
        self.train_negative_weights = train_negative_weights
        self.eval_negative_weights = eval_negative_weights

        self._negative_weights = None
        self.use_neg_weights = True

        if isinstance(scaling, int) or isinstance(scaling, float):
            self.scaling = scaling
        else:
            logger.warning("Using inner learned scaling")
            self.scaling = FactorScalar(1.)

        if proxy_per_class > 1:
            logger.info("Using {} proxies per class.".format(proxy_per_class))

        if pre_fc is not None:
            self.pre_fc = nn.Sequential(
                nn.ReLU(inplace=True), nn.BatchNorm1d(self.features_dim),
                nn.Linear(self.features_dim, pre_fc)
            )
            self.features_dim = pre_fc
        else:
            self.pre_fc = None

        self._task_idx = 0

    def on_task_end(self):
        self._task_idx += 1
        if isinstance(self.scaling, nn.Module):
            self.scaling.on_task_end()

    def on_epoch_end(self):
        if isinstance(self.scaling, nn.Module):
            self.scaling.on_epoch_end()

    def forward(self, features):
        if hasattr(self, "pre_fc") and self.pre_fc is not None:
            features = self.pre_fc(features)

        weights = self.weights
        if self._negative_weights is not None and (
            self.training is True or self.eval_negative_weights
        ) and self.use_neg_weights:
            weights = torch.cat((weights, self._negative_weights), 0)

        if self.distance == "cosine":
            raw_similarities = distance_lib.cosine_similarity(features, weights)
        elif self.distance == "stable_cosine_distance":
            features = self.scaling * F.normalize(features, p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = distance_lib.stable_cosine_distance(features, weights)
        elif self.distance == "neg_stable_cosine_distance":
            features = self.scaling * F.normalize(features, p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = -distance_lib.stable_cosine_distance(features, weights)
        elif self.distance == "prelu_stable_cosine_distance":
            features = self.scaling * F.normalize(F.relu(features), p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = distance_lib.stable_cosine_distance(features, weights)
        elif self.distance == "prelu_neg_stable_cosine_distance":
            features = self.scaling * F.normalize(F.relu(features), p=2, dim=-1)
            weights = self.scaling * F.normalize(weights, p=2, dim=-1)

            raw_similarities = -distance_lib.stable_cosine_distance(features, weights)
        else:
            raise NotImplementedError("Unknown distance function {}.".format(self.distance))

        if self.proxy_per_class > 1:
            similarities = self._reduce_proxies(raw_similarities)
        else:
            similarities = raw_similarities

            if self._negative_weights is not None and self.negative_weights_bias is not None and\
               self.training is True:
                qt = self._negative_weights.shape[0]
                if isinstance(self.negative_weights_bias, float):
                    similarities[..., -qt:] = torch.clamp(
                        similarities[..., -qt:] - self.negative_weights_bias, min=0
                    )
                elif isinstance(
                    self.negative_weights_bias, str
                ) and self.negative_weights_bias == "min":
                    min_simi = similarities[..., :-qt].min(dim=1, keepdim=True)[0]
                    similarities = torch.min(
                        similarities,
                        torch.cat((similarities[..., :-qt], min_simi.repeat(1, qt)), dim=1)
                    )
                elif isinstance(
                    self.negative_weights_bias, str
                ) and self.negative_weights_bias == "max":
                    max_simi = similarities[..., :-qt].max(dim=1, keepdim=True)[0] - 1e-6
                    similarities = torch.min(
                        similarities,
                        torch.cat((similarities[..., :-qt], max_simi.repeat(1, qt)), dim=1)
                    )
                elif isinstance(self.negative_weights_bias,
                                str) and self.negative_weights_bias.startswith("top_"):
                    topk = int(self.negative_weights_bias.replace("top_", ""))
                    botk = min(qt - topk, qt)

                    indexes = (-similarities[..., -qt:]).topk(botk, dim=1)[1]
                    similarities[..., -qt:].scatter_(1, indexes, 0.)
                else:
                    raise NotImplementedError(f"Unknown {self.negative_weights_bias}.")

        return {"logits": similarities, "raw_logits": raw_similarities}

    def _reduce_proxies(self, similarities):
        # shape (batch_size, n_classes * proxy_per_class)
        n_classes = similarities.shape[1] / self.proxy_per_class
        assert n_classes.is_integer(), (similarities.shape[1], self.proxy_per_class)
        n_classes = int(n_classes)
        bs = similarities.shape[0]

        if self.merging == "mean":
            return similarities.view(bs, n_classes, self.proxy_per_class).mean(-1)
        elif self.merging == "softmax":
            simi_per_class = similarities.view(bs, n_classes, self.proxy_per_class)
            attentions = F.softmax(self.gamma * simi_per_class, dim=-1)  # shouldn't be -gamma?
            return (simi_per_class * attentions).sum(-1)
        elif self.merging == "max":
            return similarities.view(bs, n_classes, self.proxy_per_class).max(-1)[0]
        elif self.merging == "min":
            return similarities.view(bs, n_classes, self.proxy_per_class).min(-1)[0]
        else:
            raise ValueError("Unknown merging for multiple centers: {}.".format(self.merging))

    # ------------------
    # Weights management
    # ------------------

    def align_features(self, features):
        avg_weights_norm = self.weights.data.norm(dim=1).mean()
        avg_features_norm = features.data.norm(dim=1).mean()

        features.data = features.data * (avg_weights_norm / avg_features_norm)
        return features

    def add_custom_weights(self, weights, ponderate=None, **kwargs):
        if isinstance(ponderate, str):
            if ponderate == "weights_imprinting":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                weights = weights * avg_weights_norm
            elif ponderate == "align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_new_weights_norm = weights.data.norm(dim=1).mean()

                ratio = avg_weights_norm / avg_new_weights_norm
                weights = weights * ratio
            else:
                raise NotImplementedError(f"Unknown ponderation type {ponderate}.")

        self._weights.append(nn.Parameter(weights))
        self.to(self.device)

    def align_weights(self):
        """Align new weights based on old weights norm.

        # Reference:
            * Maintaining Discrimination and Fairness in Class Incremental Learning
              Zhao et al. 2019
        """
        if len(self._weights) == 1:
            return

        with torch.no_grad():
            old_weights = torch.cat([w for w in self.old_weights])

            old_norm = torch.mean(old_weights.norm(dim=1))
            new_norm = torch.mean(self.new_weights.norm(dim=1))

            self._weights[-1] = nn.Parameter((old_norm / new_norm) * self._weights[-1])

    def align_weights_i_to_j(self, indexes_i, indexes_j):
        with torch.no_grad():
            base_weights = self.weights[indexes_i]

            old_norm = torch.mean(base_weights.norm(dim=1))
            new_norm = torch.mean(self.weights[indexes_j].norm(dim=1))

            self.weights[indexes_j] = nn.Parameter((old_norm / new_norm) * self.weights[indexes_j])

    def align_inv_weights(self):
        """Align new weights based on old weights norm.

        # Reference:
            * Maintaining Discrimination and Fairness in Class Incremental Learning
              Zhao et al. 2019
        """
        with torch.no_grad():
            old_weights = torch.cat([w for w in self.old_weights])

            old_norm = torch.mean(old_weights.norm(dim=1))
            new_norm = torch.mean(self.new_weights.norm(dim=1))

            self._weights[-1] = nn.Parameter((new_norm / old_norm) * self._weights[-1])

    @property
    def weights(self):
        return torch.cat([clf for clf in self._weights])

    @property
    def new_weights(self):
        return self._weights[-1]

    @property
    def old_weights(self):
        if len(self._weights) > 1:
            return self._weights[:-1]
        return None

    def add_classes(self, n_classes):
        new_weights = nn.Parameter(torch.zeros(self.proxy_per_class * n_classes, self.features_dim))
        nn.init.kaiming_normal_(new_weights, nonlinearity="linear")

        self._weights.append(new_weights)

        self.to(self.device)
        self.n_classes += n_classes
        return self

    def add_imprinted_classes(
        self, class_indexes, inc_dataset, network, multi_class_diff="normal", type=None
    ):
        if self.proxy_per_class > 1:
            logger.info("Multi class diff {}.".format(multi_class_diff))

        weights_norm = self.weights.data.norm(dim=1, keepdim=True)
        avg_weights_norm = torch.mean(weights_norm, dim=0).cpu()

        new_weights = []
        for class_index in class_indexes:
            _, loader = inc_dataset.get_custom_loader([class_index])
            features, _ = utils.extract_features(network, loader)

            features_normalized = F.normalize(torch.from_numpy(features), p=2, dim=1)
            class_embeddings = torch.mean(features_normalized, dim=0)
            class_embeddings = F.normalize(class_embeddings, dim=0, p=2)

            if self.proxy_per_class == 1:
                new_weights.append(class_embeddings * avg_weights_norm)
            else:
                if multi_class_diff == "normal":
                    std = torch.std(features_normalized, dim=0)
                    for _ in range(self.proxy_per_class):
                        new_weights.append(torch.normal(class_embeddings, std) * avg_weights_norm)
                elif multi_class_diff == "kmeans":
                    clusterizer = KMeans(n_clusters=self.proxy_per_class)
                    clusterizer.fit(features_normalized.numpy())

                    for center in clusterizer.cluster_centers_:
                        new_weights.append(torch.tensor(center) * avg_weights_norm)
                else:
                    raise ValueError(
                        "Unknown multi class differentiation for imprinted weights: {}.".
                        format(multi_class_diff)
                    )

        new_weights = torch.stack(new_weights)
        self._weights.append(nn.Parameter(new_weights))

        self.to(self.device)
        self.n_classes += len(class_indexes)

        return self

    def set_negative_weights(self, negative_weights, ponderate=False):
        """Add weights that are used like the usual weights, but aren't actually
        parameters.

        :param negative_weights: Tensor of shape (n_classes * nb_proxy, features_dim)
        :param ponderate: Reponderate the negative weights by the existing weights norm, as done by
                          "Weights Imprinting".
        """
        logger.info("Add negative weights.")
        if isinstance(ponderate, str):
            if ponderate == "weights_imprinting":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                negative_weights = negative_weights * avg_weights_norm
            elif ponderate == "align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_negative_weights_norm = negative_weights.data.norm(dim=1).mean()

                ratio = avg_weights_norm / avg_negative_weights_norm
                negative_weights = negative_weights * ratio
            elif ponderate == "inv_align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_negative_weights_norm = negative_weights.data.norm(dim=1).mean()

                ratio = avg_negative_weights_norm / avg_weights_norm
                negative_weights = negative_weights * ratio
            else:
                raise NotImplementedError(f"Unknown ponderation type {ponderate}.")

        if self.train_negative_weights:
            self._negative_weights = nn.Parameter(negative_weights)
        else:
            self._negative_weights = negative_weights

def squared_euclidian_distance(a, b):
    return torch.cdist(a, b)**2


def cosine_similarity(a, b):
    return torch.mm(F.normalize(a, p=2, dim=-1), F.normalize(b, p=2, dim=-1).T)


def stable_cosine_distance(a, b, squared=True):
    """Computes the pairwise distance matrix with numerical stability."""
    mat = torch.cat([a, b])

    pairwise_distances_squared = torch.add(
        mat.pow(2).sum(dim=1, keepdim=True).expand(mat.size(0), -1),
        torch.t(mat).pow(2).sum(dim=0, keepdim=True).expand(mat.size(0), -1)
    ) - 2 * (torch.mm(mat, torch.t(mat)))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(pairwise_distances, (error_mask == False).float())

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(*pairwise_distances.size(), device=pairwise_distances.device)
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances[:a.shape[0], a.shape[0]:]
