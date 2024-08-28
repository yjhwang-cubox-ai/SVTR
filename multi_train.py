import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from stn import STN_ON
from svtrnet import SVTRNet
from rnn import SequenceEncoder
from ctc_head import CTCHead
# from loss import CTCLoss
from torch import nn
from dataloader.dataset import TNGODataset, TextDataset
import tqdm

import wandb
wandb.login()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class CombinedDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
    
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            idx -= len(dataset)
        raise IndexError("Index out of range")

def main(train_files, val_files):
    train_datasets = [TNGODataset(file, mode='train') for file in train_files]
    val_datasets = [TNGODataset(file, mode='test') for file in val_files]
    
    combined_train_dataset = CombinedDataset(*train_datasets)
    combined_val_dataset = CombinedDataset(*val_datasets)
    
    train_dataloader = DataLoader(dataset=combined_train_dataset,
                                  batch_size=256, 
                                  collate_fn=collate_fn,
                                  shuffle=True, 
                                  drop_last=False)
    val_dataloader = DataLoader(combined_val_dataset,
                                batch_size=128,
                                collate_fn=collate_fn,
                                shuffle=True, 
                                drop_last=False)
    
    model = SVTR().to(DEVICE)
    
    wandb.init(
        project="svtr-0827-fix-error",
        config={
            "epoch": 100,
            "batch_size": 256,
            "lr": 1e-4,
            "val_interval": 1,
            })

    config = wandb.config
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CTCLoss()
    
    model.train()
    
    for epoch in range(config.epoch):
        with tqdm.tqdm(train_dataloader, unit="it") as pbar:
            pbar.set_description(f"Epoch {epoch+1}")
            for step, batch in enumerate(pbar):
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                lengths = batch['length'].to(DEVICE)
                
                output = model(images)
                permuted_output = output[0].permute(1, 0, 2)
                N, B, _ = permuted_output.shape
                output_length = torch.tensor([N]*B, dtype=torch.long).to(DEVICE)
                
                loss = criterion(permuted_output, labels, output_length, lengths)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                metrics = {"train/train_loss": loss.item(),
                           "train/iter": (step + 1 + (len(train_dataloader) * epoch)) / len(train_dataloader)}
                
                if step + 1 < len(train_dataloader):
                    wandb.log(metrics)
                
        # Validation loop
        if (epoch + 1) % config.val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm.tqdm(val_dataloader):
                    images = batch['image'].to(DEVICE)
                    labels = batch['label'].to(DEVICE)
                    lengths = batch['length'].to(DEVICE)
                    
                    output = model(images)
                    permuted_output = output[0].permute(1, 0, 2)
                    N, B, _ = permuted_output.shape
                    output_length = torch.tensor([N]*B, dtype=torch.long).to(DEVICE)

                    loss = criterion(permuted_output, labels, output_length, lengths)
                    val_loss += loss.item()
            val_loss /= len(val_dataloader)
            wandb.log({"val/val_loss": val_loss})
            model.train()
            
    save_model_path = "svtr_vn_240828_1.pth"
    torch.save(model.state_dict(), save_model_path)
    
    wandb.finish()

def collate_fn(batch):
    return {
        'image': torch.stack([torch.tensor(x['image']) for x in batch]),
        'label': torch.stack([torch.tensor(x['label']) for x in batch]),
        'length': torch.stack([torch.tensor(x['length'], dtype=torch.int64) for x in batch])
}

if __name__ == "__main__":
    train_files = [
        "/data/TNGoDataset/3_TNGo3_Text_final/CUBOX_VN_annotation.json",
        "/data/TNGoDataset/4_TNGo4_Text_final/CUBOX_VN_annotation.json",
    ]
    val_files = [
        "/data/CUBOX_VN_Recog_v7/CUBOX_VN_annotation_cleaned.json",
    ]
    main(train_files, val_files)