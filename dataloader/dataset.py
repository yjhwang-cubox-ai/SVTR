import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import json
import numpy as np

from dataloader.imaug.label_ops import CTCLabelEncode
import cv2
import yaml
import random
import math

from .imaug import transform, create_operators

class TNGODataset(Dataset):
    def __init__(self, json_path, mode, dataloader_config='dataloader/config.yml', character_dict_path="dict/vietnam_dict.txt"):
        self.data_list = []
        if isinstance(json_path, str):
            self.dir_path = os.path.dirname(json_path)
            with open(json_path, "r", encoding='utf8') as f:
                data_info = json.load(f)["data_list"]
            for data in data_info:
                data['img_path'] = os.path.join(os.path.dirname(path), data['img_path'])
            self.data_list = data_info
        elif isinstance(json_path, list):
            for path in json_path:
                self.dir_path = os.path.dirname(path)
                with open(path, "r", encoding='utf8') as f:
                    data_info = json.load(f)["data_list"]
                for data in data_info:
                    data['img_path'] = os.path.join(os.path.dirname(path), data['img_path'])
                self.data_list.extend(data_info)
        
        self.mode = mode
        self.config = self._load_config(dataloader_config)
        dataset_config = self.config['dataset']
        global_config = self.config['global']
        if self.mode == 'train':
            self.ops = create_operators(dataset_config['transforms_train'], global_config)
        elif self.mode == 'test':
            self.ops = create_operators(dataset_config['transforms_test'], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 1)

        self.encoder = CTCLabelEncode(max_text_length=30, character_dict_path=character_dict_path)
        # self.resizer = SVTRRecResizeImg(image_shape=(3, 64, 256), padding=False)
    
    def _load_config(self, file_path):
        _, ext = os.path.splitext(file_path)
        assert ext in ['.yml', '.yaml'], "only support yaml files for now"
        config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
        return config
        
    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, 'ext_data_num'):
                ext_data_num = getattr(op, 'ext_data_num')
                break
        
        ext_data = []
        while len(ext_data) < ext_data_num:
            idx = random.randint(0, len(self)-1)
            img_path = os.path.join(self.dir_path, self.data_list[idx]["img_path"])
            text = self.data_list[idx]["instances"][0]["text"]
            img = cv2.imread(img_path)
            data = {'image': img, 'label': text}
            # if data is None:
            #     continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, index):
        try:
            img_path = self.data_list[index]["img_path"]
            text = self.data_list[index]["instances"][0]["text"]
            # print(img_path)
            img = cv2.imread(img_path)
            
            data = {'image': img, 'label': text}
            data['ext_data'] = self.get_ext_data()
            outs = transform(data, self.ops)
            if outs is None:
                return self.__getitem__(random.randint(0, self.__len__()))
            return outs

        except Exception as e:
            print(f"예외 발생: {e}")            
            return self.__getitem__(random.randint(0, self.__len__()))
    
    def __len__(self):
        return len(self.data_list)

class TextDataset(Dataset):
    def __init__(self, text_path, character_dict_path="dict/vietnam_dict.txt", transforms=None):
        self.dir_path = os.path.dirname(text_path)        
        with open(text_path, "r", encoding='utf8') as f:
            self.data_list = f.readlines()
        self.transforms = transforms
        self.encoder = CTCLabelEncode(max_text_length=30, character_dict_path=character_dict_path)
        # self.resizer = SVTRRecResizeImg(image_shape=(3, 64, 256), padding=False)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        img_path = self.data_list[index].split("\t")[0]
        
        img_path_full = os.path.join(self.dir_path, img_path)
        text = self.data_list[index].split("\t")[1].strip('\n')
        # print(img_path)
        img = cv2.imread(img_path_full)
        
        data = {'image': img, 'label': text}
        # print(data)
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