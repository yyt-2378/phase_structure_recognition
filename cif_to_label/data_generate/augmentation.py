import os
import cv2
import numpy as np
import random


def random_sample_from_list(input_list, num_samples, seed=3):
    """
    Randomly selects a specified number of elements from a list.

    Parameters:
        input_list (list): The list to sample from.
        num_samples (int): The number of elements to randomly select.

    Returns:
        list: A list of randomly selected elements.
    """
    random.seed(seed)  # 设置随机种子
    if num_samples > len(input_list):
        raise ValueError("Number of samples requested exceeds list size.")

    return random.sample(input_list, num_samples)


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


def sliding_window_crop(image, label, size, stride):
    """
    Crops the image and label using a sliding window approach.

    Parameters:
        image (numpy.ndarray): The input image.
        label (numpy.ndarray): The corresponding label.
        size (tuple): The desired output size in the format (height, width).
        stride (int): The stride of the sliding window.

    Returns:
        cropped_images (list of numpy.ndarray): List of cropped image segments.
        cropped_labels (list of numpy.ndarray): List of cropped label segments.
    """
    h, w = image.shape[:2]
    target_h, target_w = size

    if h < target_h or w < target_w:
        raise ValueError("Target size is larger than the input image size.")

    cropped_images = []
    cropped_labels = []

    for top in range(0, h - target_h + 1, stride):
        for left in range(0, w - target_w + 1, stride):
            bottom = top + target_h
            right = left + target_w

            # Crop the image
            cropped_img = image[top:bottom, left:right]

            # Crop the label with the same region as the image
            cropped_lbl = label[top:bottom, left:right]

            cropped_images.append(cropped_img)
            cropped_labels.append(cropped_lbl)

    return cropped_images, cropped_labels


def center_crop(image, label, size):
    """
    Randomly crops the image and label to the specified size.

    Parameters:
        image (numpy.ndarray): The input image.
        label (numpy.ndarray): The corresponding label.
        size (tuple): The desired output size in the format (height, width).

    Returns:
        cropped_image (numpy.ndarray): The cropped image.
        cropped_label (numpy.ndarray): The cropped label.
    """
    h, w = image.shape[:2]
    target_h, target_w = size

    if h < target_h or w < target_w:
        raise ValueError("Target size is larger than the input image size.")

    top = (h - size[0]) // 2
    left = (w - size[1]) // 2
    bottom = top + size[0]
    right = left + size[1]

    # Crop the image
    cropped_image = image[top:bottom, left:right]

    # Crop the label with the same region as the image
    cropped_label = label[top:bottom, left:right]

    return cropped_image, cropped_label


def zoom(image, label, scale_range=(1.0, 1.2)):
    """
    Resize the image and label by a random scaling factor within the specified range.

    Parameters:
        image (numpy.ndarray): The input image.
        label (numpy.ndarray): The corresponding label.
        scale_range (tuple): A tuple (min_scale, max_scale) specifying the scaling range.

    Returns:
        resized_image (numpy.ndarray): The resized image.
        resized_label (numpy.ndarray): The resized label.
    """
    min_scale, max_scale = scale_range
    scale_factor = random.uniform(min_scale, max_scale)

    # Resize the image
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    # Resize the label using nearest-neighbor interpolation to maintain label integrity
    resized_label = cv2.resize(label, None, fx=scale_factor, fy=scale_factor)

    return resized_image, resized_label


if __name__ == '__main__':
    path = 'F:\\MOLECULE dataset3\\training'  # Path to images
    label_path = 'F:\\MOLECULE dataset3\\label_annotation'  # Path to labels
    # cat_label_path = 'F:\\MOLECULE dataset2\\annotation'  # Path to labels
    output_path = 'F:\\final_data\\img3'  # Output path
    # output_cat_label_path = 'F:\\final_data\\cat_label4'  # Output path
    output_label_path = 'F:\\final_data\\label3'  # Output path
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_label_path, exist_ok=True)
    # os.makedirs(output_cat_label_path, exist_ok=True)

    file_dir = os.listdir(path)

    for j in range(len(file_dir)):
        tar_path = os.path.join(path, file_dir[j])
        label_tar_path = os.path.join(label_path, file_dir[j].split('.')[0]+'.png')

        # Load image and label
        img = cv2.imread(tar_path, cv2.IMREAD_GRAYSCALE)
        label_img = cv2.imread(label_tar_path, cv2.IMREAD_GRAYSCALE)

        # Rotate and center crop
        np.random.seed(3)
        rotate_angle = np.random.choice(range(181), 10, replace=False)
        rotate_angle.tolist()

        for angle in rotate_angle:
            rotated_img = rotate(img, angle)
            rotated_label = rotate(label_img, angle)

            img_crop, label_crop = center_crop(rotated_img, rotated_label, (128, 128))

            file_base_name = file_dir[j].split('.')[0]
            cv2.imwrite(os.path.join(output_path, f'{file_base_name}_r{angle}.png'), img_crop)
            cv2.imwrite(os.path.join(output_label_path, f'{file_base_name}_r{angle}.png'), label_crop)

    second_output_path = os.listdir(output_path)
    for second_augment_file in second_output_path:
        input_tar_path = output_path + '\\' + second_augment_file  # 这里是文件夹中的子文件夹
        input_label_tar_path = output_label_path + '\\' + second_augment_file
        output_tar_path = output_path + '\\' + second_augment_file  # 这里是文件夹中的子文件夹
        output_tar_label_path = output_label_path + '\\' + second_augment_file  # 这里是文件夹中的子文件夹
        output_tar_path = output_tar_path.split('.')[0]
        output_tar_label_path = output_tar_label_path.split('.')[0]

        img_second = cv2.imread(input_tar_path, cv2.IMREAD_GRAYSCALE)
        label_img_second = cv2.imread(input_label_tar_path, cv2.IMREAD_GRAYSCALE)

        # Flip
        flipped_img = flip(img_second)
        flipped_label = flip(label_img_second)

        cv2.imwrite(output_tar_path + '_fli.png', flipped_img)
        cv2.imwrite(output_tar_label_path + '_fli.png', flipped_label)

        # Brightness adjustment s
        img_darker = darker(img_second)
        img_brighter = brighter(img_second)

        cv2.imwrite(output_tar_path + '_darker.png', img_darker)
        cv2.imwrite(output_tar_path + '_brighter.png', img_brighter)
        cv2.imwrite(output_tar_label_path + '_darker.png', label_img_second)
        cv2.imwrite(output_tar_label_path + '_brighter.png', label_img_second)

        # Gaussian blur
        blur = cv2.GaussianBlur(img_second, (3, 3), 1.5)
        cv2.imwrite(output_tar_path + '_blur.png', blur)
        cv2.imwrite(output_tar_label_path + '_blur.png', label_img_second)

        # random zoom
        zoom_image, zoom_label = zoom(img_second, label_img_second)
        cv2.imwrite(output_tar_path + '_zoom.png', zoom_image)
        cv2.imwrite(output_tar_label_path + '_zoom.png', zoom_label)
