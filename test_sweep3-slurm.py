import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from stn import STN_ON
from svtrnet import SVTRNet
from rnn import SequenceEncoder
from ctc_head import CTCHead
from rec_postprocess import CTCLabelDecode
from torch import nn
from dataloader.dataset import TNGODataset, TextDataset, SVTRRecResizeImg
import tqdm
import numpy as np
import cv2
import argsparse   

import math
from torch.optim.lr_scheduler import _LRScheduler

import wandb
wandb.login(key="53f960c86b81377b89feb5d30c90ddc6c3810d3a")
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

def main():
    wandb.init(config=wandb.config)

    train_datasets = TNGODataset(train_files, mode='train')
    val_datasets = TNGODataset(val_files, mode='test')
    
    train_dataloader = DataLoader(dataset=train_datasets,
                                  batch_size=wandb.config.batch_size, 
                                  collate_fn=collate_fn,
                                  shuffle=True, 
                                  drop_last=False)
    val_dataloader = DataLoader(val_datasets,
                                batch_size=wandb.config.batch_size,
                                collate_fn=collate_fn,
                                shuffle=True, 
                                drop_last=False)
    
    model = SVTR().to(DEVICE)

    # select optimizer
    if wandb.config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)
    elif wandb.config.optimizer == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.lr, weight_decay=0.01)    
    # select scheduler
    if wandb.config.scheduler == "step_lr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    elif wandb.config.scheduler == "cosine_annealing_warmup_restarts":
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=2, eta_max=0.1,  T_up=10, gamma=0.5)

    criterion = nn.CTCLoss()
    
    model.train()
    
    for epoch in range(wandb.config.epochs):
        with tqdm.tqdm(train_dataloader, unit="it") as pbar:
            pbar.set_description(f"Epoch {epoch+1}")
            e_losses = []
            for step, batch in enumerate(pbar):
                images = batch['image'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                lengths = batch['length'].to(DEVICE)
                
                output = model(images)
                permuted_output = output[0].permute(1, 0, 2)
                N, B, _ = permuted_output.shape
                output_length = torch.tensor([N]*B, dtype=torch.long).to(DEVICE)
                
                loss = criterion(permuted_output, labels, output_length, lengths)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()                
                e_losses.append(loss.item())
            e_loss = np.mean(e_losses)
            scheduler.step()

            wandb.log(
                {
                    "epoch": epoch,
                    "train/train_loss": e_loss
                }
            )
                
        # Validation loop
        if (epoch + 1) % 1 == 0:
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

                    # wandb image logging
                    img_names = ['testimg/word.jpg', 'testimg/word2.jpg', 'testimg/word3.jpg', 'testimg/word3.jpg']
                    resizer = SVTRRecResizeImg(image_shape=(3, 64, 256))
                    postprocessor = CTCLabelDecode(character_dict_path="dict/vietnam_dict.txt", use_space_char=False)
                    example_images = []
                    for img_name in img_names:
                        img = cv2.imread(img_name)
                        resize_img = resizer.resize_norm_img(img, image_shape=(3, 64, 256), padding=True)[0]
                        tensor_img = torch.tensor(resize_img).unsqueeze(0).to(DEVICE)
                        pred = model(tensor_img)[0]
                        text = postprocessor(pred.to('cpu'))[0][0]
                        conf = np.exp(postprocessor(pred.to('cpu'))[0][1])
                        example_images.append(wandb.Image(img, caption=f"Pred: {text}, Conf: {conf}"))
                    wandb.log({"examples": example_images})

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

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

if __name__ == "__main__":
    parser = argsparse.ArgumentParser()
    parser.add_argument("--sweep_id", required=True)
    parser.add_argument("--name", default="svtr_vn") 
    args = parser.parse_args()

    train_files = [
        "/purestorage/OCR/TNGoDataSet/1_TNGo_new_Text/annotation.json",        
        "/purestorage/OCR/TNGoDataSet/3_TNGo3_Text/annotation.json",
        "/purestorage/OCR/TNGoDataSet/4_TNGo4_Text/annotation.json",
        "/purestorage/OCR/TNGoDataSet/5_Employee_Text/annotation.json",
    ]
    val_files = [
        "/purestorage/OCR/CUBOX_VN_Recog_v2/CUBOX_VN_annotation.json",
    ]

    wandb.login()
    wandb.agent(args.sweep_id, function=main, count=5)