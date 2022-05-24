from torch import nn, Tensor
import torch.nn.functional as F
import torch

import math

class ArcMarginProduct(nn.Module):
    def __init__(self, emb_size=128, out_feature=100, s=32.0, m=0.50, auto_s = False, device = torch.device('cpu')):
        super(ArcMarginProduct, self).__init__()
        self.device = device
        self.emb_size = emb_size
        self.out_features = out_feature
        if auto_s:
            self.s = math.sqrt(2) * math.log(out_feature - 1)
        else:
            self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_feature, emb_size))
        nn.init.xavier_normal_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward_old(self, x, targets):
        cosine = F.linear(x, F.normalize(self.weight), bias=None)
        cosine = cosine.clip(-1+1e-7, 1-1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
        

    def forward(self, x, targets):
        cosine = F.linear(x, F.normalize(self.weight), bias=None)
        cosine = cosine.clip(-1+1e-7, 1-1e-7)
        
        arc_cos = torch.acos(cosine)

        M = F.one_hot(targets, num_classes = self.out_features) * self.m
        arc_cos = arc_cos + M
        cos_theta_2 = torch.cos(arc_cos)
        
        output = cos_theta_2 * self.s

        return output

        
