import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch


def get_kfold(dataset, k_fold, i_fold):
    validation_split = 1/k_fold
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    split_start = int(np.floor(validation_split * dataset_size * i_fold))
    split_end = int(np.floor(validation_split * dataset_size * (i_fold+1)))

    # 打乱索引
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    # 切片
    train_indices = indices[:split_start] + indices[split_end:]
    val_indices = indices[split_start: split_end]
    # 划分成两个数据集
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)

    return train_dataset, val_dataset



