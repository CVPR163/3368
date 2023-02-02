import torch
import warnings
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform

from torch.nn.parallel import DistributedDataParallel as DDP

# ----------------------------------------------------------------------------------------------
# Add by NieX.
# from .buffer import Buffer
import time
from torch import nn

from .Distill_loss_detach import Distill_Loss_detach

from collections import OrderedDict
# ----------------------------------------------------------------------------------------------

class Appr(Inc_Learning_Appr):
    """Class implementing the Incremental Classifier and Representation Learning (iCaRL) approach
    described in https://arxiv.org/abs/1611.07725
    Original code available at https://github.com/srebuffi/iCaRL
    """

    def __init__(self, model, device, l_alpha, l_beta, l_gamma, buffer_size, minibatch_size_1, minibatch_size_2, network, nepochs=160, lr=0.5, decay_mile_stone=[80,120], lr_decay=0.1, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, ddp=False, local_rank=0, logger=None, exemplars_dataset=None, 
                 lamb=1):
        super(Appr, self).__init__(model, device, nepochs, lr, decay_mile_stone, lr_decay, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, ddp, local_rank, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb

        # ----------------------------------------------------------------------------------------------
        # Add by NieX.
        self.buffer_size = buffer_size
        self.minibatch_size_1 = minibatch_size_1
        self.minibatch_size_2 = minibatch_size_2
        # ----------------------------------------------------------------------------------------------
        # Add in 10_20
        self.l_alpha = l_alpha
        self.l_beta = l_beta
        self.l_gamma = l_gamma
        # ----------------------------------------------------------------------------------------------
        # self.buffer = Buffer(self.buffer_size, self.device)
        self.network = network
        # ----------------------------------------------------------------------------------------------

        # iCaRL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: iCaRL is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def _get_optimizer(self):
        """Returns the optimizer"""
        if self.ddp:
            model = self.model.module
        else:
            model = self.model

        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        return optimizer

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4. " allowing iCaRL to balance between CE and distillation loss."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        return parser.parse_known_args(args)

    # Algorithm 1: iCaRL NCM Classify
    def classify(self, task, features, targets):
        # expand means to all batch images
        means = torch.stack(self.exemplar_means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        # expand all features to all classes
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get distances for all images to all exemplar class means -- nearest prototype
        dists = (features - means).pow(2).sum(1).squeeze()
        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task]
        offset = self.model.task_offset[task]
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def compute_mean_of_exemplars(self, trn_loader, transform):
        # change transforms to evaluation for this calculation
        with override_dataset_transform(self.exemplars_dataset, transform) as _ds:
            # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep same order
            icarl_loader = DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=False,
                                      num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            # extract features from the model for all train samples
            # Page 2: "All feature vectors are L2-normalized, and the results of any operation on feature vectors,
            # e.g. averages are also re-normalized, which we do not write explicitly to avoid a cluttered notation."
            extracted_features = []
            extracted_targets = []
            with torch.no_grad():
                self.model.eval()
                for images, targets in icarl_loader:
                    feats = self.model((images.to(self.device), images.to(self.device)), return_features=True)[1]
                    feats = feats[0:images.shape[0]]
                    # normalize
                    extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                    extracted_targets.extend(targets)
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            for curr_cls in np.unique(extracted_targets):
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                # add the exemplars to the set and normalize
                cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                self.exemplar_means.append(cls_feats_mean)

    def pre_train_process(self, t, trn_loader):
        if self.ddp:
            # ------------------------------------------------------------------------------------------------------------------------
            # Add for DDP by NieX.
            self.model = DDP(self.model.module, device_ids=[self.local_rank], broadcast_buffers=False, find_unused_parameters=True)
            # ------------------------------------------------------------------------------------------------------------------------

    # Algorithm 2: iCaRL Incremental Train
    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # remove mean of exemplars during training since Alg. 1 is not used during Alg. 2
        self.exemplar_means = []

        # Algorithm 3: iCaRL Update Representation
        # Alg. 3. "form combined training set", add exemplars to train_loader
        if t > 0:
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
            # trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
            #                                          batch_size=trn_loader.batch_size,
            #                                          shuffle=True,
            #                                          num_workers=trn_loader.num_workers,
            #                                          pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        # Algorithm 4: iCaRL ConstructExemplarSet and Algorithm 5: iCaRL ReduceExemplarSet
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform, self.ddp)

        # compute mean of exemplars
        self.compute_mean_of_exemplars(trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader, val_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later. This is different from the original approach, since they propose to
        #  extract the features and store them for future usage. However, when using data augmentation, it is easier to
        #  keep the model frozen and extract the features when needed.
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        if self.ddp:
            self.model_old.module.freeze_all()
        else:
            self.model_old.freeze_all()
        # ------------------------------------------------------------------------------------------------------
        # Add for dark branch by NieX.
        # Dark branch for inheriting parameter from br1 to br2.
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
        # ------------------------------------------------------------------------------------------------------

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # --------------------------------------------------------------------------------------------------------------
            # Add for Decom by NieX.
            images, targets = images.to(self.device), targets.to(self.device)

            # Distill_Loss_detach
            _, _, pod_features = self.model((images, images), return_features=True)
            loss = self.l_gamma * Distill_Loss_detach()(pod_features)

            # Forward old model
            outputs_old = None

            if t == 0:
                targets = torch.cat((targets, targets), dim=0)
                outputs, _, _ = self.model((images, images), return_features=True)
                loss = loss + self.criterion(t, outputs, targets.to(self.device), outputs_old)
            else:
                buf_outputs = []
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
                for i in range(len(outputs)):
                    buf_outputs.append(outputs[i][images.shape[0]:])
                    outputs[i] = outputs[i][0:images.shape[0]]

                buf_outputs = torch.cat([o for o in buf_outputs], dim=1)

                # loss += nn.CrossEntropyLoss()(buf_outputs, balanced_targets)
                loss = loss + self.l_beta*nn.CrossEntropyLoss()(buf_outputs, balanced_targets)

                outputs_old = self.model_old((images.to(self.device), images.to(self.device)))
                for i in range(len(outputs_old)):
                    outputs_old[i] = outputs_old[i][0:images.shape[0]]
                loss = loss + self.l_alpha*self.criterion(t, outputs, targets.to(self.device), outputs_old)

            # Forward current model
            # outputs = self.model(images.to(self.device))
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        # ----------------------------------------------------------------------------------------------

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                outputs_old = None
                if t > 0:
                    outputs_old = self.model_old((images.to(self.device), images.to(self.device)))
                    for i in range(len(outputs_old)):
                        outputs_old[i] = outputs_old[i][0:images.shape[0]]
                # Forward current model
                outputs, feats, _ = self.model((images.to(self.device), images.to(self.device)), return_features=True)
                for i in range(len(outputs)):
                    outputs[i] = outputs[i][0:images.shape[0]]
                feats = feats[0:images.shape[0]]
                # import pdb
                # pdb.set_trace()
                loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
                # during training, the usual accuracy is computed on the outputs
                if not self.exemplar_means:
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                else:
                    hits_taw, hits_tag = self.classify(t, feats, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # Algorithm 3: classification and distillation terms -- original formulation has no trade-off parameter (lamb=1)
    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""

        # Classification loss for new classes
        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        # Distillation loss for old classes
        if t > 0:
            # The original code does not match with the paper equation, maybe sigmoid could be removed from g
            g = torch.sigmoid(torch.cat(outputs[:t], dim=1))
            q_i = torch.sigmoid(torch.cat(outputs_old[:t], dim=1))
            loss += self.lamb * sum(torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y]) for y in
                                    range(sum(self.model.task_cls[:t])))
        return loss