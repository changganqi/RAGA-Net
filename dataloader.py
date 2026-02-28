import os
import sys

import PIL.ImageShow
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import cv2
from torchvision import transforms
import torchvision
from natsort import natsorted
from util import *


# --- 修改1: 让 populate_list 接收可选的 snr_images_path ---
def populate_list(gt_images_path, hazy_images_path, snr_images_path=None):
    image_list_haze_index = natsorted(os.listdir(hazy_images_path))
    image_list_gt_index = natsorted(os.listdir(gt_images_path))

    # --- 修改2: 如果提供了SNR路径，也获取SNR文件列表 ---
    if snr_images_path and os.path.exists(snr_images_path):
        image_list_snr_index = natsorted(os.listdir(snr_images_path))
        # 确保文件数量一致
        assert len(image_list_haze_index) == len(image_list_snr_index), \
            f"Hazy images ({len(image_list_haze_index)}) and SNR maps ({len(image_list_snr_index)}) count mismatch!"
    else:
        image_list_snr_index = None

    image_dataset = []
    len_haze = len(image_list_haze_index)
    len_gt = len(image_list_gt_index)
    if len_haze != len_gt:
        # one gt image to many hazy image
        rep = len_haze // len_gt
        image_list_gt_index = [[_] * rep for _ in image_list_gt_index]
        image_list_gt_index = [i for j in image_list_gt_index for i in j]

    # --- 修改3: 在数据集中同时记录 gt, hazy 和 snr 的路径 ---
    for i in range(len(image_list_haze_index)):
        gt_path = os.path.join(gt_images_path, image_list_gt_index[i])
        haze_path = os.path.join(hazy_images_path, image_list_haze_index[i])

        # 如果有SNR图，则获取路径，否则为None
        snr_path = os.path.join(snr_images_path, image_list_snr_index[i]) if image_list_snr_index else None

        image_dataset.append((gt_path, haze_path, snr_path))

    train_list = image_dataset

    return train_list


def att(channal):
    cv2.normalize(channal, channal, 0, 255, cv2.NORM_MINMAX)
    M = np.ones(channal.shape, np.uint8) * 255
    img_new = cv2.subtract(M, channal)
    return img_new


def process(img):
    # img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # img = np.array(img)
    b, g, r = cv2.split(img)
    b = att(b)
    g = att(g)
    r = att(r)
    new_image = cv2.merge([r, g, b])
    return new_image


class dehazing_loader(data.Dataset):
    # --- 修改4: 在 __init__ 中接收 snr_images_path ---
    def __init__(self, orig_images_path, hazy_images_path, snr_images_path=None, mode='train', resize=None,
                 random_crop=None, base_resize=1):
        # --- 修改5: 将 snr_images_path 传递给 populate_list ---
        self.train_list = populate_list(orig_images_path, hazy_images_path, snr_images_path)
        self.val_list = populate_list(orig_images_path, hazy_images_path, snr_images_path)
        self.resize = resize
        self.random_crop = random_crop
        self.base_resize = base_resize

        seed = torch.random.seed()
        torch.random.manual_seed(seed)

        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(random_crop) if random_crop else torch.nn.Identity(),
            transforms.Resize(resize) if resize else torch.nn.Identity()
        ])
        self.trans_gt = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(random_crop) if random_crop else torch.nn.Identity(),
            transforms.Resize(resize) if resize else torch.nn.Identity()
        ])

    def __getitem__(self, index):

        # --- 修改6: 解包得到 snr 路径 ---
        data_clean_path, data_hazy_path, data_snr_path = self.data_list[index]
        file_name = os.path.split(data_clean_path)[-1]
        img_clean = Image.open(data_clean_path).convert('RGB')
        img_hazy = Image.open(data_hazy_path).convert('RGB')

        # --- 修改7: 加载SNR图（如果路径存在）---
        if data_snr_path:
            img_snr = Image.open(data_snr_path).convert('L')  # SNR图是单通道灰度图
        else:
            img_snr = None

        seed = 0
        if self.random_crop != None:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
        data_clean = self.trans(img_clean)

        if self.random_crop != None:
            torch.random.manual_seed(seed)
        data_hazy = self.trans(img_hazy)

        # --- 修改8: 对SNR图进行变换，或创建占位符 ---
        if img_snr:
            # 确保SNR图也使用相同的随机裁剪种子
            if self.random_crop != None:
                torch.random.manual_seed(seed)
            data_snr = self.trans(img_snr)
        else:
            # 如果没有SNR图，返回一个独特的、易于识别的占位符
            data_snr = torch.tensor([-1.0])

        if self.base_resize != 1:
            data_clean = base_resize(data_clean, base=self.base_resize)
            data_hazy = base_resize(data_hazy, base=self.base_resize)
            if img_snr:  # 如果有SNR图，也需要缩放
                data_snr = base_resize(data_snr, base=self.base_resize)

        # --- 修改9: 在返回值中加入 data_snr ---
        return data_clean, data_hazy, data_snr, file_name

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    pass