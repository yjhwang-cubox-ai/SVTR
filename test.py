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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_multi_gpu_model(model_path, model):
    # Multi-GPU로 학습된 모델을 로드
    state_dict = torch.load(model_path)

    # 'module.' prefix를 제거한 새로운 state_dict 생성
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 'module.' prefix를 제거
        else:
            new_state_dict[k] = v

    # 모델에 state_dict를 로드
    model.load_state_dict(new_state_dict)
    return model

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
    with torch.no_grad():
        for test_data in test_data_list:
        # img = cv2.imread("testimg/48564_FRONT_00000019.jpg")
            img_name = '/data/CUBOX_VN_Recog_v7/' + test_data['img_path']
            img = cv2.imread(img_name)
            data = resizer.resize_norm_img(img, image_shape=(3, 64, 256), padding=True)[0]
            tensor_img = torch.tensor(data).unsqueeze(0).to(DEVICE)
            pred = model(tensor_img)[0]
            gt_text = test_data['instances'][0]['text']
            text = postprocessor(pred.to('cpu'))[0][0]
            conf = np.exp(postprocessor(pred.to('cpu'))[0][1])

            print(f"gt: {gt_text}")
            print(f"text: {text}")
            print(f"pred: {conf}")

if __name__ == "__main__":
    main()