import torch.nn as nn
import torch
import torch.nn.functional as F

class Distill_Loss_detach(nn.Module):

    def __init__(self):
        super(Distill_Loss_detach, self).__init__()
        self.MSE = torch.nn.MSELoss(reduction='mean')
        # self.CE = torch.nn.CrossEntropyLoss()
    
    def forward(self, pod_features):
        pod_features_1, pod_features_2 = pod_features
        pod_features_1_w, pod_features_1_h = pod_features_1
        pod_features_2_w, pod_features_2_h = pod_features_2
        for i in range (len(pod_features_1_w)):
            MSE_w = self.MSE(pod_features_1_w[i], pod_features_2_w[i].detach())
            MSE_h = self.MSE(pod_features_1_h[i], pod_features_2_h[i].detach())
        loss = MSE_w + MSE_h
        return loss