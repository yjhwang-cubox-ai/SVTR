import torch
from torch import nn




class STN_ON(nn.Module):
    def __init__(
            self,
            in_channels,
            tps_inputsize,
            tps_outputsize,
            num_control,
            num_control_points,
            tps_margins,
            stn_activation,
    ):
        super(STN_ON, self).__init__()
        self.tps = None
        self.stn_head = None
        self.tps_inputsize = tps_inputsize
        self.out_channels = in_channels

    def forward(self, x):
        stn_input = nn.functional.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
        stn_img_feat, ctrl_points = self.stn_head(stn_input)
        x, _ = self.tps(x, ctrl_points)
        return x