import torch
from torch import nn
from stn import STN_ON

class SVTR(nn.Module):
    def __init__(self):
        super(SVTR, self).__init__()
        self.transfrom = STN_ON()
        self.backbone = None
        self.neck = None
        self.head = None
    
    def forward(self, x):
        x = self.transfrom(x)
        return x

model = SVTR()
data = torch.randn(1, 3, 32, 100)

model.eval()
print(model(data).shape)