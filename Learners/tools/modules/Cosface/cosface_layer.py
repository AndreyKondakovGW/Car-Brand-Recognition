from torch import nn, Tensor
import torch.nn.functional as F
import torch

import math

class CosMarginProduct(nn.Module):
    def __init__(self, emb_size=128, out_feature=100, s=32.0, m=0.35, device = torch.device('cpu')):
        super(CosMarginProduct, self).__init__()
        self.device = device
        self.emb_size = emb_size
        self.out_features = out_feature
        self.s = math.sqrt(2) * math.log(out_feature - 1)
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_feature, emb_size))
        nn.init.xavier_normal_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, targets):
        cosine = F.linear(x, F.normalize(self.weight), bias=None)
        cosine = cosine.clip(-1+1e-7, 1-1e-7)
        
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
        

    def forward_old(self, x, targets):
        cosine = F.linear(x, F.normalize(self.weight), bias=None)
        cosine = cosine.clip(-1+1e-7, 1-1e-7)
        
        M = F.one_hot(targets, num_classes = self.out_features) * self.m
        arc_cos = cosine - M
        
        output = arc_cos * self.s

        return output

        
