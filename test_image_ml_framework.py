import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
import yaml
import os
import copy
import random

# preprocess_model
from preprocess_model.image_preprocess_model import DIVAESR
# SR model
from preprocess_model.configs.option import args


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
    # Initialize the output and count arrays with three channels for RGB
    output = np.zeros((height, width, 3))
    count = np.zeros((height, width, 3))

    for crop, (x, y) in zip(crops, positions):
        crop_np = np.array(crop)
        # Update the output and count arrays for the RGB channels
        output[y:y + crop_size, x:x + crop_size] += crop_np
        count[y:y + crop_size, x:x + crop_size] += 1

    output /= count
    return Image.fromarray((output).astype(np.uint8))


def inference_for_large_image(image_path, model_weights, save_path, crop_size=128):
    # Load the VAE and SR model configuration
    sr_model_args = args
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    vae_model_args = config['model_params']

    # Define and load the model
    preprocess_model = DIVAESR(sr_model_args, vae_model_args)
    checkpoint = torch.load(model_weights, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace("model.", "")  # Remove the "model." prefix
        new_state_dict[new_key] = state_dict[key]
    preprocess_model.load_state_dict(new_state_dict)

    preprocess_model.eval()

    original_image = Image.open(image_path)
    crops, positions = sliding_window_crop(original_image, crop_size)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    reconstructed_crops = []
    reconstructed_crops_without_box = []
    i = 0
    with torch.no_grad():
        for crop in crops:
            i = i + 1
            crop_tensor = transform(crop).unsqueeze(0)
            _ = torch.zeros_like(crop_tensor)
            hr_label_tensor = torch.zeros([1, 1, 128, 128])
            results = preprocess_model.forward(crop_tensor, _, hr_label_tensor)
            vae_output, sr_output, output_label = results[1], results[2], results[3]
            # Save SR reconstructed image
            SR_reconstructed_image = sr_output.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze()
            SR_reconstructed_image = (SR_reconstructed_image - SR_reconstructed_image.min()) / (
                    SR_reconstructed_image.max() - SR_reconstructed_image.min())
            SR_reconstructed_image = (SR_reconstructed_image * 255).astype("uint8")
            SR_save_path = os.path.join(save_path, f"SR_reconstructed_img_{i}.png")
            Image.fromarray(SR_reconstructed_image, mode='L').save(SR_save_path)

            # Load the pre-trained Fast R-CNN model
            model = torchvision.models.detection.__dict__['fasterrcnn_resnet50_fpn'](num_classes=2, pretrained=False)
            state_dict = torch.load('D:\\project\\DIVAESR\\phase_structure_recognition\\faster_rcnn_checkpoints\\model.pth')
            model.load_state_dict(state_dict['model'])
            model.eval()

            # Load an image and its annotations
            # Replace with your own image and annotation loading code

            # Transform the image and annotations
            train_transforms = transforms.Compose([transforms.ToTensor()])
            faster_image = Image.open(SR_save_path).convert("RGB")
            imgae_without_box = copy.deepcopy(faster_image)
            image_tensor = train_transforms(faster_image)
            image_tensor = image_tensor.unsqueeze(0)

            # Make predictions
            with torch.no_grad():
                predictions = model(image_tensor)[0]
                print(predictions)

            # Display the image with predicted boxes (if score > 0.5)
            draw = ImageDraw.Draw(faster_image)
            for box, score in zip(predictions["boxes"], predictions["scores"]):
                if score > 0.85:
                    draw.rectangle([box[0], box[1], box[2], box[3]], outline="blue")
                    # draw.text((box[0], box[1]), f"Score: {score:.2f}", fill="red")

            # reconstructed_image = transforms.ToPILImage()(sr_output.squeeze())
            reconstructed_crops.append(faster_image)
            reconstructed_crops_without_box.append(imgae_without_box)

    reconstructed_image = reconstruct_from_crops(reconstructed_crops, positions, original_image.size, crop_size)
    reconstructed_image_without_box = reconstruct_from_crops(reconstructed_crops_without_box, positions,
                                                             original_image.size, crop_size)

    return reconstructed_image, reconstructed_image_without_box


if __name__ == '__main__':
    model_weights_path = [os.path.join('D:\\project\\DIVAESR\\phase_structure_recognition\\preprocess_model_weights', weight_i)
                          for weight_i in os.listdir('D:\\project\\DIVAESR\\phase_structure_recognition\\preprocess_model_weights')]
    image_folder_path = [os.path.join('D:\\project\\DIVAESR\\phase_structure_recognition\\test', im) for im in os.listdir('D:\\project\\DIVAESR\\phase_structure_recognition\\test')]
    for model_path in range(len(model_weights_path)):
        image_output_folder_path = os.path.join("C:\\Users\\yyt70\\Desktop\\test", str(model_path))
        model_weight = model_weights_path[model_path]
        os.makedirs(image_output_folder_path, exist_ok=True)
        for img_path in image_folder_path:
            file_basename = img_path.split('\\')[-1]
            file_base = file_basename.split('.png')[0]
            per_im_path = os.path.join(image_output_folder_path, file_base)
            os.makedirs(per_im_path, exist_ok=True)
            reconstructed_image, reconstructed_image_without_bbox = inference_for_large_image(img_path, model_weight,
                                                                                              per_im_path)
            reconstructed_image.save(os.path.join(per_im_path, 'large_test_image.png'))
            reconstructed_image_without_bbox.save(os.path.join(per_im_path, 'large_test_image_without_box.png'))
