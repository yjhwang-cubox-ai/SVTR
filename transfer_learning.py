import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.profilers import SimpleProfiler
import wandb

from stn import STN_ON
from svtrnet import SVTRNet
from rnn import SequenceEncoder
from ctc_head import CTCHead
from dataloader.dataset import TNGODataset
from rec_postprocess import CTCLabelDecode

class ImagePredictionLogger(L.Callback):
    def __init__(self, val_samples, num_samples=5):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples['image'], val_samples['label']
    
    def on_validation_epoch_end(self, trainer, model):
        postprocessor = CTCLabelDecode(character_dict_path="dict/vietnam_dict.txt", use_space_char=False)
        val_imgs = self.val_imgs.to(model.device)
        # val_labels = postprocessor(self.val_labels.to('cpu'))
        pred = model(val_imgs)[0]
        # text = postprocessor(pred.to('cpu'))[0][0]
        pred_postprocessed = postprocessor(pred.to('cpu'))
        text = [item[0] for item in pred_postprocessed]
        conf = [np.exp(item[1]) for item in pred_postprocessed]
        
        trainer.logger.experiment.log({
            "examples": [
                wandb.Image(val_imgs[i], caption=f"Pred: {text[i]}, Conf: {conf[i]}")
                for i in range(self.num_samples)
            ]
        })

class LitSVTR(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.transform = STN_ON()
        self.backbone = SVTRNet()
        self.neck = SequenceEncoder(in_channels=384, encoder_type="reshape")
        self.head = CTCHead(in_channels=384, out_channels=228)
        self.criterion = torch.nn.CTCLoss(zero_infinity=True)        
        self.save_hyperparameters()

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
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, label, label_length = batch['image'], batch['label'], batch['length']
        x = self.transform(image)
        x = self.backbone(x)
        x = self.neck(x)
        output = self.head(x)
        permuted_output = output[0].permute(1, 0, 2)
        N, B, _ = permuted_output.shape
        output_length = torch.tensor([N]*B, dtype=torch.long)
        loss = self.criterion(permuted_output, label, output_length, label_length)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        image, label, label_length = batch['image'], batch['label'], batch['length']
        x = self.transform(image)
        x = self.backbone(x)
        x = self.neck(x)
        output = self.head(x)
        permuted_output = output[0].permute(1, 0, 2)
        N, B, _ = permuted_output.shape
        output_length = torch.tensor([N]*B, dtype=torch.long)
        loss = self.criterion(permuted_output, label, output_length, label_length)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2.5 / (10**4), weight_decay=0.05)
        return optimizer
    
    def forward(self, x):
        x = self.transform(x)
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

def collate_fn(batch):
    return {
        'image': torch.stack([torch.tensor(x['image']) for x in batch]),
        'label': torch.stack([torch.tensor(x['label']) for x in batch]),
        'length': torch.stack([torch.tensor(x['length'], dtype=torch.int64) for x in batch])
}

def main():
    wandb_logger = WandbLogger(project='pytorchlightning-transfer-large')    
    
    train_json_file = ["/data/TNGoDataset/3_TNGo3_Text_final/CUBOX_VN_annotation.json",
                       "/data/TNGoDataset/4_TNGo4_Text_final/CUBOX_VN_annotation.json",]
    test_json_file = ["/data/CUBOX_VN_Recog_v7/CUBOX_VN_annotation_cleaned.json"]
    
    train_datasets = TNGODataset(json_path=train_json_file, mode='train')
    # train_datasets = [TNGODataset(file, mode='train') for file in train_json_file]
    train_set_size = int(len(train_datasets) * 0.8)
    val_set_size = len(train_datasets) - train_set_size

    #split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_datasets, [train_set_size, val_set_size], generator=seed)
    val_dataset.dataset.mode = 'test'
    test_dataset = TNGODataset(json_path=test_json_file, mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True, drop_last=False, num_workers=5)
    val_dataloader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn, shuffle=False, drop_last=False, num_workers=5)
    test_dataloader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn, shuffle=False, drop_last=False, num_workers=5)

    # model
    svtr = LitSVTR()
    svtr.load_state_dict(torch.load('svtr_L_VN7_fine_e12.pth'))

    # trian model
    _profiler = SimpleProfiler(dirpath=".", filename="profile_logs")
    trainer = L.Trainer(max_epochs=1000,
                        callbacks=[
                            # EarlyStopping(monitor='val_loss', mode='min', patence=10),
                            ImagePredictionLogger(val_samples=next(iter(val_dataloader)), num_samples=5),
                            # ModelSummary(max_depth=-1)
                        ], 
                        profiler=_profiler,
                        logger=wandb_logger)

    trainer.fit(model=svtr, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # test model
    trainer.test(model=svtr, dataloaders=test_dataloader)

if __name__ == '__main__':
    main()