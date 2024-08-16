import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_para_bias_attr(l2_decay, k):
    stdv = 1.0 / math.sqrt(k * 1.0)
    weight_initializer = torch.empty(k)
    bias_initializer = torch.empty(k)
    nn.init.uniform_(weight_initializer, -stdv, stdv)
    nn.init.uniform_(bias_initializer, -stdv, stdv)
    
    weight_attr = {
        'weight': nn.Parameter(weight_initializer.clone(), requires_grad=True),
        'weight_decay': l2_decay
    }
    bias_attr = {
        'bias': nn.Parameter(bias_initializer.clone(), requires_grad=True),
        'weight_decay': l2_decay
    }
    
    return [weight_attr, bias_attr]

class CTCHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        fc_decay=0.0004,
        mid_channels=None,
        return_feats=False,
        **kwargs,
    ):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            weight_attr, bias_attr = get_para_bias_attr(
                l2_decay=fc_decay, k=in_channels
            )
            self.fc = nn.Linear(in_channels, out_channels)
            nn.init.uniform_(self.fc.weight, -weight_attr['weight'].std().item(), weight_attr['weight'].std().item())
            nn.init.uniform_(self.fc.bias, -bias_attr['bias'].std().item(), bias_attr['bias'].std().item())
        else:
            weight_attr1, bias_attr1 = get_para_bias_attr(
                l2_decay=fc_decay, k=in_channels
            )
            self.fc1 = nn.Linear(in_channels, mid_channels)
            nn.init.uniform_(self.fc1.weight, -weight_attr1['weight'].std().item(), weight_attr1['weight'].std().item())
            nn.init.uniform_(self.fc1.bias, -bias_attr1['bias'].std().item(), bias_attr1['bias'].std().item())

            weight_attr2, bias_attr2 = get_para_bias_attr(
                l2_decay=fc_decay, k=mid_channels
            )
            self.fc2 = nn.Linear(mid_channels, out_channels)
            nn.init.uniform_(self.fc2.weight, -weight_attr2['weight'].std().item(), weight_attr2['weight'].std().item())
            nn.init.uniform_(self.fc2.bias, -bias_attr2['bias'].std().item(), bias_attr2['bias'].std().item())

        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, targets=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts
        if not self.training:
            predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result