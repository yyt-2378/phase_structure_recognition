import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from PIL import Image
import zipfile


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class StemDataset(Dataset):
    def __init__(self, mode, dir, tran):
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


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "trainval" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
    
        train_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                              transforms.RandomHorizontalFlip(),
                                              # transforms.CenterCrop(64),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                            transforms.RandomHorizontalFlip(),
                                            # transforms.CenterCrop(64),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])
        
        self.train_dataset = StemDataset(
            mode='trainval',
            dir=self.data_dir,
            tran=train_transforms,
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = StemDataset(
            mode='validation',
            dir=self.data_dir,
            tran=val_transforms,
        )
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     