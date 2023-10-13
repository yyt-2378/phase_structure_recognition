import torch
from torchvision import transforms
from PIL import Image
import numpy as np


def sliding_window_crop(image, crop_size):
    """滑动窗口裁剪图片。"""
    width, height = image.size
    crops = []
    positions = []

    for i in range(0, width, crop_size):
        for j in range(0, height, crop_size):
            if i + crop_size <= width and j + crop_size <= height:
                crop = image.crop((i, j, i + crop_size, j + crop_size))
                positions.append((i, j))
            elif i + crop_size > width and j + crop_size <= height:
                crop = image.crop((width - crop_size, j, width, j + crop_size))
                positions.append((width - crop_size, j))
            elif i + crop_size <= width and j + crop_size > height:
                crop = image.crop((i, height - crop_size, i + crop_size, height))
                positions.append((i, height - crop_size))
            else:
                crop = image.crop((width - crop_size, height - crop_size, width, height))
                positions.append((width - crop_size, height - crop_size))

            crops.append(crop)

    return crops, positions


def reconstruct_from_crops(crops, positions, original_size, crop_size):
    width, height = original_size
    output = np.zeros((height, width))
    count = np.zeros((height, width))

    for crop, (x, y) in zip(crops, positions):
        crop_np = np.array(crop)
        output[y:y + crop_size, x:x + crop_size] += crop_np
        count[y:y + crop_size, x:x + crop_size] += 1

    output /= count
    return Image.fromarray((output).astype(np.uint8))


def inference_for_large_image(image_path, model=None, crop_size=128):
    original_image = Image.open(image_path)
    crops, positions = sliding_window_crop(original_image, crop_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    reconstructed_crops = []
    with torch.no_grad():
        for crop in crops:
            crop_tensor = transform(crop).unsqueeze(0)
            # reconstructed, _, _ = model(crop_tensor)
            reconstructed_image = transforms.ToPILImage()(crop_tensor.squeeze())
            reconstructed_crops.append(reconstructed_image)

    reconstructed_image = reconstruct_from_crops(reconstructed_crops, positions, original_image.size, crop_size)

    return reconstructed_image


if __name__ == '__main__':
    # 示例
    reconstructed_image = inference_for_large_image("F:\\MOLECULE dataset\\label_annotation\\molecule_0.png")
    reconstructed_image.show()