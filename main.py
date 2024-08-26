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
from torch import nn
from dataloader.dataset import TNGODataset, TextDataset
import tqdm
from torchvision.transforms import ToPILImage
import math

import wandb
wandb.login()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# JSONFFILE = "/data/CUBOX_VN_Recog_v7/CUBOX_VN_annotation_cleaned.json"
JSONFFILE = "/data/TNGoDataset/3_TNGo3_Text_final/CUBOX_VN_annotation.json"
EPOCH = 50

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
    
    wandb.init(
        project="svtr-test",
        config={
            "epoch": 50,
            "batch_size": 128,
            "lr": 1e-4,
            })

    config = wandb.config
    
    dataset = TNGODataset(json_path=JSONFFILE)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=config.batch_size, 
                            collate_fn=collate_fn,
                            shuffle=True, 
                            drop_last=False)
    
    n_steps_per_epoch = math.ceil(len(dataloader.dataset) / config.batch_size)
    
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    model.to(DEVICE)
    model.train()
    example_ct = 0
    step_ct = 0    
    for epoch in range(EPOCH):
        # e_loss = []
        with tqdm.tqdm(dataloader, unit="it") as pbar:
            pbar.set_description(f"Epoch {epoch+1}")
            for step, batch in enumerate(pbar):
                image, label, label_length = batch['image'], batch['label'], batch['length']            
                image = image.to(DEVICE)
                label = label.to(DEVICE)
                label_length = label_length.to(DEVICE)
                optimizer.zero_grad()
                output = model(image)
                permuted_output = output[0].permute(1, 0, 2)
                N, B, _ = permuted_output.shape
                output_length = torch.tensor([N]*B, dtype=torch.long)

                loss = criterion(permuted_output, label, output_length, label_length)
                
                loss.backward()
                
                optimizer.step()
                # print(f"Epoch {epoch+1}, Loss: {loss}")
                example_ct += len(image)
                metrics = {"train/train_loss": loss,
                        "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                        "train/example_ct": example_ct}
                
                if step + 1 < n_steps_per_epoch:
                    wandb.log(metrics)
                
                step_ct += 1
    
    save_model_path = "svtr_vn222.pth"
    torch.save(model.state_dict(), save_model_path)
    
    wandb.finish()

def collate_fn(batch):
    return {
        'image': torch.stack([torch.tensor(x['image']) for x in batch]),
        'label': torch.stack([torch.tensor(x['label']) for x in batch]),
        'length': torch.stack([torch.tensor(x['length'], dtype=torch.int64) for x in batch])
}

if __name__ == "__main__":
    main()