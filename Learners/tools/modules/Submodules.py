from torch.nn import Module

class Flatten(Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


#Фиктивный слой модели
class Identity(Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
