from torch.nn import Module, Conv2d, init
import torch.nn.functional as F

class EmbeddingModel(Module):
    def __init__(self,
        backbone: Module,
        losslayer: Module,
        fc_unit: Module,
        **kwargs):
        super(EmbeddingModel, self).__init__()
        
        
        self.backbone = backbone
        self.fc_unit = fc_unit
        self.losslayer = losslayer


    def forward(self, x, targets = None):
        x = self.backbone(x)
        x = self.fc_unit(x)
        x = F.normalize(x)
        
        if targets is not None:
            logits = self.losslayer(x, targets)
            return logits
        
        return x
    
    def get_centroids(self):
        #Векторы центроид классов
        return self.losslayer.get_centroids()
    
    def warmup_base(self):
        #Размораживат базовый слой
        weighted_layers = []
        for layer in list(self.base.modules()):
            if type(layer) is Conv2d:
                weighted_layers.append(layer)
        for child in weighted_layers:
            for p in child.parameters():
                p.requires_grad = True