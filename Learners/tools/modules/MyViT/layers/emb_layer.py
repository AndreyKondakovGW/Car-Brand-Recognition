import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange
from einops import repeat

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            #Эквивалентно reshape(0, 2*3, 1)
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.nn.init.kaiming_normal_(torch.randn(1,1, emb_size)))
        self.positions = nn.Parameter(torch.nn.init.kaiming_normal_(torch.randn((img_size // patch_size) **2 + 1, emb_size)))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x