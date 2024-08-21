import math
import cv2
import numpy as np
import random
from PIL import Image
import albumentations as A

from torchvision.transforms import Compose


class Albumentation(object):
    def __init__(self, **kwargs):
        self.transform = A.Compose([
        A.Blur(p = 0.1),
        A.MotionBlur(p = 0.1),
        A.GaussNoise(p = 0.1),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p = 0.2),
        A.CLAHE(p = 0.2),
        A.RandomBrightnessContrast(p=0.7),
        A.ISONoise(p = 0.3),
        A.RGBShift(p = 0.5),
        A.ChannelShuffle(p = 0.5),
        A.ShiftScaleRotate( shift_limit = 0.0, shift_limit_y = 0.01, scale_limit=0.0, rotate_limit=1, p=0.2),
        # A.ElasticTransform(alpha_affine=0.5, alpha=0.5, sigma=0, p = 0.1),
        # A.Perspective(p = 0.2),
        A.ToGray(p = 0.25),
        ])
    
    def __call__(self, data):
        img = data['image']
        transformed = self.transform(image=img)
        transformed_image = transformed['image']

        data['image'] = transformed_image
        return data

class RecConAug(object):
    def __init__(self,
                 prob=0.5,
                 image_shape=(32, 320, 3),
                 max_text_length=25,
                 ext_data_num=1,
                 **kwargs):
        self.ext_data_num = ext_data_num
        self.prob = prob
        self.max_text_length = max_text_length
        self.image_shape = image_shape
        self.max_wh_ratio = self.image_shape[1] / self.image_shape[0]

    def merge_ext_data(self, data, ext_data):
        ori_w = round(data['image'].shape[1] / data['image'].shape[0] *
                      self.image_shape[0])
        ext_w = round(ext_data['image'].shape[1] / ext_data['image'].shape[0] *
                      self.image_shape[0])
        data['image'] = cv2.resize(data['image'], (ori_w, self.image_shape[0]))
        ext_data['image'] = cv2.resize(ext_data['image'],
                                       (ext_w, self.image_shape[0]))
        data['image'] = np.concatenate(
            [data['image'], ext_data['image']], axis=1)
        data["label"] += ext_data["label"]
        return data

    def __call__(self, data):
        rnd_num = random.random()
        if rnd_num > self.prob:
            return data
        for idx, ext_data in enumerate(data["ext_data"]):
            if len(data["label"]) + len(ext_data[
                    "label"]) > self.max_text_length:
                break
            concat_ratio = data['image'].shape[1] / data['image'].shape[
                0] + ext_data['image'].shape[1] / ext_data['image'].shape[0]
            if concat_ratio > self.max_wh_ratio:
                break
            data = self.merge_ext_data(data, ext_data)
        data.pop("ext_data")
        return data

class RecAug(object):
    def __init__(self,
                 tia_prob=0.4,
                 crop_prob=0.6,
                 reverse_prob=0.2,
                 noise_prob=0.4,
                 jitter_prob=0.6,
                 blur_prob=0.6,
                 hsv_aug_prob=0.6,
                 **kwargs):
        self.tia_prob = tia_prob
        self.bda = BaseDataAugmentation(crop_prob, reverse_prob, noise_prob,
                                        jitter_prob, blur_prob, hsv_aug_prob)

    def __call__(self, data):
        img = data['image']
        h, w, _ = img.shape

        # tia
        if random.random() <= self.tia_prob:
            if h >= 20 and w >= 20:
                img = tia_distort(img, random.randint(3, 6))
                img = tia_stretch(img, random.randint(3, 6))
            img = tia_perspective(img)

        # bda
        data['image'] = img
        data = self.bda(data)
        return data

class BaseDataAugmentation(object):
    def __init__(self,
                 crop_prob=0.4,
                 reverse_prob=0.4,
                 noise_prob=0.4,
                 jitter_prob=0.4,
                 blur_prob=0.4,
                 hsv_aug_prob=0.4,
                 **kwargs):
        self.crop_prob = crop_prob
        self.reverse_prob = reverse_prob
        self.noise_prob = noise_prob
        self.jitter_prob = jitter_prob
        self.blur_prob = blur_prob
        self.hsv_aug_prob = hsv_aug_prob

    def __call__(self, data):
        img = data['image']
        h, w, _ = img.shape

        if random.random() <= self.crop_prob and h >= 20 and w >= 20:
            img = get_crop(img)

        if random.random() <= self.blur_prob:
            img = blur(img)

        if random.random() <= self.hsv_aug_prob:
            img = hsv_aug(img)

        if random.random() <= self.jitter_prob:
            img = jitter(img)

        if random.random() <= self.noise_prob:
            img = add_gasuss_noise(img)

        if random.random() <= self.reverse_prob:
            img = 255 - img

        data['image'] = img
        return data

class SVTRRecResizeImg(object):
    def __init__(self, image_shape, padding=True, **kwargs):
        self.image_shape = image_shape
        self.padding = padding

    def __call__(self, data):
        img = data['image']

        norm_img, valid_ratio = resize_norm_img(img, self.image_shape,
                                                self.padding)
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data


def resize_norm_img(img,
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