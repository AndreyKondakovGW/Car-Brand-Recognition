from torch.nn import Module, init
import torch.nn.functional as F
from torch import nn
import torch

class SoftMaxLayer(Module):
    def __init__(self,
                s: int,
                labels_num: int,
                emb_size: int = 256):
        super(SoftMaxLayer, self).__init__()
        self.s = s
        self.classifier = nn.Parameter(torch.FloatTensor(labels_num, emb_size))
    
    def _init_params(self):
        init.xavier_normal_(self.classifier)
        
    def forward(self, x, targets=None):
        x =  F.linear(x, F.normalize(self.classifier), bias=None)
        return x
    
    def get_centroids(self):
        return F.normalize(self.classifier)