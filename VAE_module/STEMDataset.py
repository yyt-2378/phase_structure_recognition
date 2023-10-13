import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision.transforms as Trans

# 设置图片尺寸
img_size = 64
# 封装，对后面读取的图片的格式转换
tran = Trans.Compose([
    Trans.Grayscale(num_output_channels=1),  # 转为灰度图
    Trans.Resize((img_size, img_size)),      # 调整尺寸
    Trans.ToTensor()                         # 转为张量
])


class StemDataset(data.Dataset):
    def __init__(self, mode, dir):
        self.data_size = 0  # 数据集的大小
        self.img_list = []  # 用于存图
        self.img_label = []  # 标签
        self.trans = tran  # 转换的属性设置
        self.mode = mode  # 下面打开集的模式

        if self.mode == 'trainval':
            train_img_dir = dir + '/training/img/'  # 更新地址
            train_label_dir = dir + '/training/label/'
            for img_file in os.listdir(train_img_dir):  # 遍历
                self.img_list.append(train_img_dir + img_file)  # 存图
                self.data_size += 1
                label_x = img_file.split('.')[0] + '.png'
                self.img_label.append(train_label_dir + label_x)  # 存标签

        elif self.mode == 'validation':
            val_img_dir = dir + '/validation/img/'  # 更新地址
            val_label_dir = dir + '/validation/label/'
            for img_file in os.listdir(val_img_dir):  # 遍历
                self.img_list.append(val_img_dir + img_file)  # 存图
                self.data_size += 1
                label_x = img_file.split('.')[0] + '.png'
                self.img_label.append(val_label_dir + label_x)  # 存标签
        else:
            print("没有这个mode")

    def __getitem__(self, item):  # 获取数据
        if self.mode == 'trainval':
            img = Image.open(self.img_list[item])
            label_y = Image.open(self.img_label[item])
            label_tensor = self.trans(label_y)
            img_tensor = self.trans(img)
            return img_tensor, label_tensor  # 返回该图片的地址和标签
        elif self.mode == 'validation':
            img = Image.open(self.img_list[item])
            label_y = Image.open(self.img_label[item])
            img_tensor = self.trans(img)
            label_tensor = self.trans(label_y)
            return img_tensor, label_tensor  # 返回该图片的地址和标签
        else:
            print("None")

    def __len__(self):
        return self.data_size
