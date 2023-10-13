import random
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class JointTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, label):
        seed = np.random.randint(2147483647)  # Create a random seed
        random.seed(seed)  # Apply this seed to img tranfsorms
        if self.transform is not None:
            image = self.transform(image)

        random.seed(seed)  # Apply this seed to target transforms
        if self.transform is not None:
            label = self.transform(label)

        return image, label


def build_transform(split, input_size, auto_augment, interpolation, mean, std,
                    horizontal_flip_prob=0.5, vertical_flip_prob=0.5, rotation_range=10,
                    translate=(0.1, 0.1), scale=None, shear=None, erase_prob=0.1):

    if split == "trainval":
        transform_list = [
            transforms.RandomResizedCrop(input_size, interpolation=interpolation),
            transforms.RandomHorizontalFlip(horizontal_flip_prob),
            transforms.RandomVerticalFlip(vertical_flip_prob),
            transforms.RandomRotation(rotation_range),
            transforms.RandomAffine(rotation_range, translate, scale, shear),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std),
            transforms.RandomErasing(erase_prob),
        ]
        if auto_augment:
            transform_list.insert(1, transforms.AutoAugment())
        transform = transforms.Compose(transform_list)
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(input_size[0] + 32),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )
    return JointTransform(transform)


def get_default_transform_joint():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    input_size = (128, 128)
    interpolation = InterpolationMode.BILINEAR
    train_transform = build_transform(split="trainval", input_size=input_size, auto_augment=False, interpolation=interpolation,
                                      mean=mean, std=std,
                                      horizontal_flip_prob=0.5, vertical_flip_prob=0.5, rotation_range=10,
                                      translate=(0.1, 0.1), scale=None, shear=None, erase_prob=0.1)
    val_transform = build_transform(split="val", input_size=input_size, auto_augment=False, interpolation=interpolation,
                                    mean=mean, std=std,
                                    horizontal_flip_prob=0.5, vertical_flip_prob=0.5, rotation_range=10,
                                    translate=(0.1, 0.1), scale=None, shear=None, erase_prob=0.1)

    test_transform = build_transform(split="test", input_size=input_size, auto_augment=False, interpolation=interpolation,
                                     mean=mean, std=std,
                                     horizontal_flip_prob=0.5, vertical_flip_prob=0.5, rotation_range=10,
                                     translate=(0.1, 0.1), scale=None, shear=None, erase_prob=0.1)
    return train_transform, val_transform, test_transform


def get_dummy_transform_joint():
    input_size = (224, 224)
    transform = transforms.Compose(
        [
            transforms.Resize(input_size[0] + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])
    return JointTransform(transform)
