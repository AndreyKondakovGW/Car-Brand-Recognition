from torch.nn import Module, Linear, Conv2d, Sequential, BatchNorm1d, Dropout, init,  AdaptiveAvgPool2d
from .cosface_layer import CosMarginProduct
from tools.ModelsConstructor import load_model
from omegaconf import DictConfig
from torch import tensor, device
import torch.nn.functional as F


class CosFaceEncoder(Module):
    def __init__(self, 
        base_cfg: DictConfig,
        labels_num: int,
        device = device('cpu'),
        arc_emb_size: int = 256,
        s: float = 32.0,
        m: float = 0.35,
        dropout = 0.4,
        criterion = F.cross_entropy,
        **kwargs):
        super(CosFaceEncoder, self).__init__()
                
        model_info = load_model(device, base_cfg)
        self.base = model_info['model']
        self.dropout = Dropout(dropout)
        self.fc = Linear(model_info['init_features'], arc_emb_size)
        self.bn = BatchNorm1d(arc_emb_size)
        self.cosface = CosMarginProduct(emb_size=arc_emb_size,out_feature = labels_num, s=s, m=m,device=device)
        self.criterion = criterion
        self.device = device
        
        self._init_params()

    def _init_params(self):
        init.xavier_normal_(self.fc.weight)
        init.constant_(self.fc.bias, 0)

    def forward(self, x, targets = None):
        #print(x[:1])
        x = self.base(x)
        #print(x[:1])
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        #print(x[:1])
        x = F.normalize(x)
        #print(x[:1])

        if targets is not None:
            logits = self.cosface(x, targets)
            #print(logits[:1])
            return logits
        return x
    
    def count_loss(self, x, targets):
        return self.criterion(x, targets)
        
    
    def gen_embs(self, x):
        x = self.base(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.normalize(x)
        
        return x
    
    def get_centroids(self):
        return F.normalize(self.cosface.weight)
    
    
    
    def warmup_base(self):
        weighted_layers = []
        for layer in list(self.base.modules()):
            if type(layer) is Conv2d:
                weighted_layers.append(layer)
        print()
        for child in weighted_layers:
            for p in child.parameters():
                p.requires_grad = True
