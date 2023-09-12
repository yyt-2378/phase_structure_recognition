import os
import cv2
import numpy as np
import random


# 高斯噪声
def addGaussianNoise(image, percetage, seed=None):
    G_Noiseimg = image.copy()
    h, w = image.shape
    G_NoiseNum = int(percetage * h * w)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x, temp_y] = np.random.randn(1)[0]

    return G_Noiseimg


# 昏暗
def darker(image, percetage=0.9):
    image_copy = image.copy()
    h, w = image.shape
    # get darker
    for x in range(w):
        for y in range(h):
            image_copy[y, x] = int(image[y, x] * percetage)
    return image_copy


# 亮度
def brighter(image, percetage=1.2):
    image_copy = image.copy()
    h, w = image.shape
    # get brighter
    for x in range(w):
        for y in range(h):
            image_copy[y, x] = np.clip(int(image[y, x] * percetage), a_max=255, a_min=0)
    return image_copy


# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


# 翻转
def flip(image):
    flipped_image = np.fliplr(image)
    return flipped_image


# 中心裁剪
def center_crop(image, size):
    h, w = image.shape[:2]
    top = (h - size[0]) // 2
    left = (w - size[1]) // 2
    bottom = top + size[0]
    right = left + size[1]
    cropped_image = image[top:bottom, left:right, :]
    return cropped_image


if __name__ == '__main__':
    path = 'F:\\MOLECULE dataset\\training'  # 这里需要加一个反斜杠作为转义字符，是绝对路径
    output_path = 'F:\\STEM_img'
    os.makedirs(output_path, exist_ok=True)
    file_dir = os.listdir(path)  # os.listdir(path) 返回指定路径下所有文件和文件夹的名字，并存放于一个列表中
    for j in range(len(file_dir)):
        tar_path = path + '\\' + file_dir[j]  # 这里是文件夹中的子文件夹
        output_tar_path = output_path + '\\' + file_dir[j]  # 这里是文件夹中的子文件夹
        output_tar_path = output_tar_path.split('.')[0]
        img = cv2.imread(tar_path)
        # 写入
        rotate_angle = [30, 45, 60, 75, 90, 120, 135, 180]
        for angle in rotate_angle:
            # 中心裁剪
            rotated_img = rotate(img, angle)
            img_center_crop = center_crop(rotated_img, (128, 128))  # 指定裁剪尺寸
            cv2.imwrite(output_tar_path+f'_r{angle}.png', img_center_crop)

    out_file_dir = os.listdir(output_path)  # os.listdir(path) 返回指定路径下所有文件和文件夹的名字，并存放于一个列表中
    for j in range(len(out_file_dir)):
        tar_path = output_path + '\\' + out_file_dir[j]  # 这里是文件夹中的子文件夹
        output_tar_path = output_path + '\\' + out_file_dir[j]  # 这里是文件夹中的子文件夹
        output_tar_path = output_tar_path.split('.')[0]

        img = cv2.imread(tar_path, cv2.IMREAD_GRAYSCALE)
        # 镜像
        flipped_img = flip(img)
        cv2.imwrite(output_tar_path+'_fli.png', flipped_img)

        # 变亮、变暗
        img_darker = darker(img)
        cv2.imwrite(output_tar_path+'_darker.png', img_darker)
        img_brighter = brighter(img)
        cv2.imwrite(output_tar_path+'_brighter.png', img_brighter)

        blur = cv2.GaussianBlur(img, (3, 3), 1.5)
        # cv2.GaussianBlur(图像，卷积核，标准差）
        cv2.imwrite(output_tar_path+'_blur.png', blur)
