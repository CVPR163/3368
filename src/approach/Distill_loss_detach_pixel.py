import torch.nn as nn
import torch
import torch.nn.functional as F

class Distill_Loss_detach_pixel(nn.Module):

    def __init__(self):
        super(Distill_Loss_detach_pixel, self).__init__()
        self.MSE = torch.nn.MSELoss(reduction='mean')
        # self.CE = torch.nn.CrossEntropyLoss()
    
    def forward(self, pod_features):
        pod_features_1, pod_features_2 = pod_features
        pod_features_1_p, pod_features_1_w, pod_features_1_h = pod_features_1
        pod_features_2_p, pod_features_2_w, pod_features_2_h = pod_features_2
        for i in range (len(pod_features_1_p)):
            MSE = self.MSE(pod_features_1_p[i], pod_features_2_p[i].detach())
            print(pod_features_1_p[i].shape)
            print(pod_features_2_p[i].shape)
        loss = MSE
        return loss