import torch
from torch import nn
from stn import STN_ON
from svtrnet import SVTRNet
from rnn import SequenceEncoder
from ctc_head import CTCHead
from dataloader.dataset import TNGODataset, SVTRRecResizeImg
from rec_postprocess import CTCLabelDecode
import cv2
import numpy as np
import json
from tqdm import tqdm

from metric import OneMinusNEDMetric, WordMetric, CharMetric
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
    
    # single gpu inference
    # model = load_multi_gpu_model("svtr_vn_240828_1.pth", model)
    with open('/data/CUBOX_VN_Recog_v7/CUBOX_VN_annotation_cleaned.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        test_data_list = data['data_list']

    model.load_state_dict(torch.load("svtr_vn_240828_2_45.pth", weights_only=True))
    # model.load_state_dict(torch.load("model/svtr_vn_3data.pth", weights_only=True))
    model.to(DEVICE)
    model.eval()
    infer_results = []
    with torch.no_grad():
        for test_data in tqdm(test_data_list):
            img_name = '/data/CUBOX_VN_Recog_v7/' + test_data['img_path']
            img = cv2.imread(img_name)
            data = resizer.resize_norm_img(img, image_shape=(3, 64, 256), padding=True)[0]
            tensor_img = torch.tensor(data).unsqueeze(0).to(DEVICE)
            pred = model(tensor_img)[0]
            gt_text = test_data['instances'][0]['text']
            text = postprocessor(pred.to('cpu'))[0][0]
            conf = np.exp(postprocessor(pred.to('cpu'))[0][1])

            infer_results.append({
                "img": img_name, 
                "gt_text": gt_text, 
                "pred_text": text, 
                "match": gt_text == text,
                "conf": conf
            })
    
    one_minus_ned = OneMinusNEDMetric()
    one_mainus_ned_results = one_minus_ned.compute_metrics(infer_results)
    
    word_metric = WordMetric()
    word_metric_results = word_metric.compute_metrics(infer_results)
    
    char_metric = CharMetric()
    char_metric_results = char_metric.compute_metrics(infer_results)
    
    print(one_mainus_ned_results)
    print(word_metric_results)
    print(char_metric_results)

if __name__ == "__main__":
    main()