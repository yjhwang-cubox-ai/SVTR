import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from stn import STN_ON
from svtrnet import SVTRNet
from rnn import SequenceEncoder
from ctc_head import CTCHead
from utils import CTCLabelConverter
# from loss import CTCLoss
from torch.nn import CTCLoss
from dataset import TNGODataset

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import json
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
JSONFFILE = "/data/CUBOX_VN_Recog_v7/CUBOX_VN_annotation_cleaned.json"
EPOCH = 100


class SVTR(nn.Module):
    def __init__(self):
        super(SVTR, self).__init__()
        self.transform = STN_ON()
        self.backbone = SVTRNet()
        self.neck = SequenceEncoder(in_channels=192, encoder_type="reshape")
        self.head = CTCHead(in_channels=192, out_channels=228)
    
    def forward(self, x):
        x = self.transform(x)
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


def main():
    model = SVTR()

    transforms_list = transforms.Compose([
        # transforms.Resize((64, 256)),
        transforms.ToTensor(),
    ])
    dataset = TNGODataset(json_path=JSONFFILE, transforms=transforms_list)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=False)
    
    criterion = CTCLoss(zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    model.to(DEVICE)
    model.train()
    for epoch in tqdm(range(EPOCH)):
        # e_loss = []
        for i, batch in enumerate(tqdm(dataloader)):            
            image, label, label_length = batch['image'], batch['label'], batch['length']
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            label_length = label_length.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(image)
            permuted_output = output.permute(1, 0, 2)
            N, B, _ = permuted_output.shape
            output_length = torch.tensor([N]*B, dtype=torch.int64)

            loss = criterion(permuted_output, label, output_length, label_length)
            
            loss.backward()
            
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss}")
            
            # e_loss.append(loss.item())
        
        # avg_loss = sum(e_loss) / len(e_loss)
        # print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")
            
            
            

if __name__ == "__main__":
    main()