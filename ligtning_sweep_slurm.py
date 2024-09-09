import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.profilers import SimpleProfiler
import wandb
import argparse

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
        self.head = CTCHead(in_channels=192, out_channels=228)
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

def main(sweep_id):
    wandb_entity = "youngjun-hwang"
    wandb_project = "slurm-test"
    wandb.login(key="53f960c86b81377b89feb5d30c90ddc6c3810d3a")
    print("login!")


    def train():
        wandb.init()
        wandb_logger = WandbLogger(project=wandb_project, entity=wandb_entity)        
        
        train_datasets = TNGODataset(json_path=train_json_file, mode='train')
        # train_datasets = [TNGODataset(file, mode='train') for file in train_json_file]
        train_set_size = int(len(train_datasets) * 0.9)
        val_set_size = len(train_datasets) - train_set_size

        #split the train set into two
        seed = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(train_datasets, [train_set_size, val_set_size], generator=seed)
        val_dataset.dataset.mode = 'test'
        test_dataset = TNGODataset(json_path=test_json_file, mode='test')
        train_dataloader = DataLoader(train_dataset, batch_size=256, collate_fn=collate_fn, shuffle=True, drop_last=False, num_workers=5)
        val_dataloader = DataLoader(val_dataset, batch_size=256, collate_fn=collate_fn, shuffle=True, drop_last=False, num_workers=5)
        test_dataloader = DataLoader(test_dataset, batch_size=256, collate_fn=collate_fn, shuffle=False, drop_last=False, num_workers=5)

        # model
        svtr = LitSVTR()

        # trian model
        _profiler = SimpleProfiler(dirpath=".", filename="profile_logs")    
        trainer = L.Trainer(accelerator='gpu',
                            devices=1,
                            max_epochs=1000,
                            callbacks=[
                                # EarlyStopping(monitor='val_loss', mode='min', patience=10),
                                ImagePredictionLogger(val_samples=next(iter(val_dataloader)), num_samples=3),
                                # ModelSummary(max_depth=-1)
                            ], 
                            profiler=_profiler,
                            logger=wandb_logger,
                            enable_progress_bar=True)

        trainer.fit(model=svtr, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        # test model
        trainer.test(model=svtr, dataloaders=test_dataloader)
    
    wandb.agent(sweep_id, function=train, count=5, entity = wandb_entity, project=wandb_project)

if __name__ == '__main__':
    train_json_file = ["/purestorage/OCR/TNGoDataSet/1_TNGo_new_Text/annotation.json",
                       "/purestorage/OCR/TNGoDataSet/3_TNGo3_Text/annotation.json",
                       "/purestorage/OCR/TNGoDataSet/4_TNGo4_Text/annotation.json",
                       "/purestorage/OCR/TNGoDataSet/5_Employee_Text/annotation.json",]
    test_json_file = ["/purestorage/OCR/CUBOX_VN_Recog_v2/CUBOX_VN_annotation.json"]

    parser = argparse.ArgumentParser(description="Process sweep_id.")
    parser.add_argument("--sweep_id", type=str, default="The sweep ID to process")

    args = parser.parse_args()

    main(args.sweep_id)