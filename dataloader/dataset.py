import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json

from dataloader.imaug.label_ops import CTCLabelEncode
import cv2
import yaml
import random

from .imaug import transform, create_operators

def load_config(file_path):
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config

class TNGODataset(Dataset):
    def __init__(self, json_path, dataloader_config='dataloader/config.yml', character_dict_path="dict/vietnam_dict.txt"):
        self.dir_path = os.path.dirname(json_path)
        with open(json_path, "r", encoding='utf8') as f:
            self.data_list = json.load(f)["data_list"]

        config = load_config(dataloader_config)
        dataset_config = config['dataset']
        global_config = config['global']
        self.ops = create_operators(dataset_config['transforms'], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 1)

        self.encoder = CTCLabelEncode(max_text_length=30, character_dict_path=character_dict_path)
        # self.resizer = SVTRRecResizeImg(image_shape=(3, 64, 256), padding=False)
        
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
        img_path = os.path.join(self.dir_path, self.data_list[index]["img_path"])
        text = self.data_list[index]["instances"][0]["text"]
        # print(img_path)
        img = cv2.imread(img_path)
        
        data = {'image': img, 'label': text}
        data['ext_data'] = self.get_ext_data()
        outs = transform(data, self.ops)
        if outs is None:
            return self.__getitem__(random.randint(0, self.__len__()))
        return outs
    
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