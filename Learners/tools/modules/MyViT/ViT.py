from torch import nn
from .layers.main_layers import TransformerEncoder,ClassificationHead
from .layers.emb_layer import PatchEmbedding
class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                size: int = 224,
                depth: int = 6,
                n_classes: int = 100,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )