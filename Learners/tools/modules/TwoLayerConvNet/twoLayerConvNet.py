from torch.nn import Module
from torch import nn

from tools.modules.Submodules import Flatten


class TwoLayerConvNet(Module):
    def __init__(self,
    channel1_size: int,
    channel2_size: int,
    labels_num: int,
    kernel_size: int,
    drop_out: int,
    size: int
    ): 
        super(TwoLayerConvNet, self).__init__()
        self.out_size = model_out_size = size // 4
        self.content = nn.Sequential(
            nn.Conv2d(3,channel1_size,(kernel_size, kernel_size), padding=kernel_size // 2),
            nn.Dropout(drop_out),
            nn.ReLU(),
            nn.Conv2d(channel1_size,channel2_size,(kernel_size, kernel_size),padding=kernel_size //2),
            nn.Dropout(drop_out),
            nn.ReLU(),
            nn.MaxPool2d(4),
            Flatten(),
            nn.Linear(channel2_size * model_out_size * model_out_size, labels_num))

    def forward(self, x, label=False):
        return self.content(x)