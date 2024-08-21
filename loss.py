import torch
from torch import nn
import torch.nn.functional as F

class CTCLoss(nn.Module):
    def __init__(self, use_focal_loss=False):
        super().__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction="none")
        self.use_focal_loss = use_focal_loss
    
    def forward(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        predicts = predicts.transpose((1,0,2))
        N, B, _ = predicts.shape
        pred_lengths = torch.full(size=(B,), full_val=N, dtype=torch.int64)
        lebels = batch[1].to(torch.int32)
        lebel_length = batch[2].to(torch.int64)
        loss = self.loss_func(predicts, lebels, pred_lengths, lebel_length)
        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = 1- weight
            weight = weight ** 2
            loss = loss * weight
        loss = loss.mean()
        return {"loss": loss}