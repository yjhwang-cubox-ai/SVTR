import torch
from torch import nn
from stn import STN_ON
from svtrnet import SVTRNet
from rnn import SequenceEncoder

class SVTR(nn.Module):
    def __init__(self):
        super(SVTR, self).__init__()
        self.transfrom = STN_ON()
        self.backbone = SVTRNet()
        self.neck = SequenceEncoder(in_channels=192, encoder_type="reshape")
        self.head = None
    
    def forward(self, x):
        x = self.transfrom(x)
        x = self.backbone(x)
        x = self.neck(x)
        return x

model = SVTR()
data = torch.randn(1, 3, 100, 100)

model.eval()
print(model(data).shape)