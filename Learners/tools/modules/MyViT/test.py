#from ViT import ViT
from torchsummary import summary
import torch
image_size = 224

if __name__ == "__main__": 
    x = torch.randn(1,1, 10)
    torch.nn.init.kaiming_normal_(x)
    print(x)
