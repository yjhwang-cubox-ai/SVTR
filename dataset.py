import os
import glob
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json

from label_ops import CTCLabelEncode
import cv2
import numpy as np
import math

class TNGODataset(Dataset):
    def __init__(self, json_path, character_dict_path="dict/vietnam_dict.txt", transforms=None):
        self.dir_path = os.path.dirname(json_path)
        with open(json_path, "r", encoding='utf8') as f:
            self.data_list = json.load(f)["data_list"]
        self.transforms = transforms
        self.encoder = CTCLabelEncode(max_text_length=30, character_dict_path=character_dict_path)
        self.resizer = SVTRRecResizeImg(image_shape=(3, 64, 256), padding=False)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.dir_path, self.data_list[index]["img_path"])
        text = self.data_list[index]["instances"][0]["text"]
        # print(img_path)
        img = cv2.imread(img_path)
        
        data = {'image': img, 'label': text}
        data = self.encoder(data)
        data = self.resizer(data)
        
        return data


class SVTRRecResizeImg:
    def __init__(self, image_shape, padding=False):
        self.image_shape = image_shape
        self.padding = padding    

    def __call__(self, data):
        img = data['image']

        norm_img, valid_ratio = self.resize_norm_img(img, self.image_shape,
                                                self.padding)
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data
    
    def resize_norm_img(self,
                    img,
                    image_shape,
                    padding=True,
                    interpolation=cv2.INTER_LINEAR):
        imgC, imgH, imgW = image_shape
        h = img.shape[0]
        w = img.shape[1]
        if not padding:
            resized_image = cv2.resize(
                img, (imgW, imgH), interpolation=interpolation)
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        valid_ratio = min(1.0, float(resized_w / imgW))
        return padding_im, valid_ratio




    
    
    
    
    


# 이미지 변환 정리
# transforms = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])


# data = TNGODataset("/data/CUBOX_VN_Recog_v7/CUBOX_VN_annotation_cleaned.json", transforms=transforms)
# dataloader = DataLoader(dataset=data,
#                         batch_size=16,
#                         shuffle=True,
#                         drop_last=False)

# for batch in dataloader:
#     img, text = batch
#     print(img.shape, text)
#     break