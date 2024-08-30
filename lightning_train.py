import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L

from stn import STN_ON
from svtrnet import SVTRNet
from rnn import SequenceEncoder
from ctc_head import CTCHead
from dataloader.dataset import TNGODataset

class LitSVTR(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.transform = STN_ON()
        self.backbone = SVTRNet()
        self.neck = SequenceEncoder(in_channels=192, encoder_type="reshape")
        self.head = CTCHead(in_channels=192, out_channels=228)
        self.criterion = torch.nn.CTCLoss(zero_infinity=True)

    def training_step(self, batch, batch_idx):
        image, label, label_length = batch['image'], batch['label'], batch['length']
        x = self.transform(image)
        x = self.backbone(x)
        x = self.neck(x)
        output = self.head(x)
        permuted_output = output[0].permute(1, 0, 2)
        N, B, _ = permuted_output.shape
        output_length = torch.tensor([N]*B, dtype=torch.long)
        loss = self.criterion(permuted_output, label, output_length, label_length)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2.5 / (10**4), weight_decay=0.05)
        return optimizer

def collate_fn(batch):
    return {
        'image': torch.stack([torch.tensor(x['image']) for x in batch]),
        'label': torch.stack([torch.tensor(x['label']) for x in batch]),
        'length': torch.stack([torch.tensor(x['length'], dtype=torch.int64) for x in batch])
}

def main():
    json_file = "/data/TNGoDataset/3_TNGo3_Text_final/CUBOX_VN_annotation.json"
    dataset = TNGODataset(json_path=json_file, mode='train')
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn, shuffle=True, drop_last=False)

    # model
    svtr = LitSVTR()

    # trian model
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model=svtr, train_dataloaders=dataloader)

if __name__ == '__main__':
    main()