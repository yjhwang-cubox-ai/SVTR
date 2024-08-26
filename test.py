import torch
from torch import nn
from stn import STN_ON
from svtrnet import SVTRNet
from rnn import SequenceEncoder
from ctc_head import CTCHead
from utils import CTCLabelConverter
from dataloader.dataset import TNGODataset, SVTRRecResizeImg
from rec_postprocess import CTCLabelDecode
import cv2
import numpy as np
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

def main():
    resizer = SVTRRecResizeImg(image_shape=(3, 64, 256))
    postprocessor = CTCLabelDecode(character_dict_path="dict/vietnam_dict.txt", use_space_char=False)
    model = SVTR()
    model.load_state_dict(torch.load("svtr_vn_3data.pth", weights_only=True))    
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        img = cv2.imread("testimg/48564_FRONT_00000019.jpg")
        data = resizer.resize_norm_img(img, image_shape=(3, 64, 256), padding=False)[0]
        tensor_img = torch.tensor(data).unsqueeze(0).to(DEVICE)
        pred = model(tensor_img)[0]
        text = postprocessor(pred.to('cpu'))[0][0]
        conf = np.exp(postprocessor(pred.to('cpu'))[0][1])
        
        print(f"text: {text}")
        print(f"pred: {conf}")
        
    

if __name__ == "__main__":
    main()