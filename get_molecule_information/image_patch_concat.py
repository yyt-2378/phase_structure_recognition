import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
import yaml
import os
from PIL import ImageFilter

# preprocess_model
from preprocess_model.image_preprocess_model import DCVAESR
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


def apply_gaussian_blur(image, radius=2):
    """Apply Gaussian blur to the given image.

    Args:
        image (PIL.Image): The image to be blurred.
        radius (int, optional): The radius of the blur. Defaults to 2.

    Returns:
        PIL.Image: The blurred image.
    """
    return image.filter(ImageFilter.GaussianBlur(radius))


def save_crops(original_image, crops, positions, crop_size, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, (crop, (x, y)) in enumerate(zip(crops, positions)):
        crop_save_path = os.path.join(save_path, f"crop_{i+1}.png")
        crop.save(crop_save_path)

        original_crop = original_image.crop((x, y, x + crop_size, y + crop_size))
        original_crop_save_path = os.path.join(save_path, f"original_crop_{i+1}.png")
        original_crop.save(original_crop_save_path)


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
    preprocess_model = DCVAESR(sr_model_args, vae_model_args)
    checkpoint = torch.load(model_weights, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace("model.", "")  # Remove the "model." prefix
        new_state_dict[new_key] = state_dict[key]
    preprocess_model.load_state_dict(new_state_dict)

    preprocess_model.eval()

    original_image = Image.open(image_path)
    # original_image = original_image.resize((128, 128))
    crops, positions = sliding_window_crop(original_image, crop_size)
    save_crops(original_image, crops, positions, crop_size, save_path)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    reconstructed_crops = []
    i = 0
    with torch.no_grad():
        for crop in crops:
            i = i+1
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
            state_dict = torch.load('D:\\project\\phase_structure\\phase_structure_recognition\\checkpoints\\model_28.pth')
            model.load_state_dict(state_dict['model'])
            model.eval()

            # Load an image and its annotations
            # Replace with your own image and annotation loading code
            # Transform the image and annotations
            train_transforms = transforms.Compose([transforms.ToTensor()])
            faster_image = Image.open(SR_save_path).convert("RGB")
            image_tensor = train_transforms(faster_image)
            image_tensor = image_tensor.unsqueeze(0)

            # # Make predictions
            with torch.no_grad():
                predictions = model(image_tensor)[0]
                print(predictions)

            # Display the image with predicted boxes (if score > 0.5)
            draw = ImageDraw.Draw(faster_image)
            for box, score in zip(predictions["boxes"], predictions["scores"]):
                if score > 0.90:
                    draw.rectangle([box[0], box[1], box[2], box[3]], outline="blue", width=1)
                    # draw.text((box[0], box[1]), f"Score: {score:.2f}", fill="red")

            # reconstructed_image = transforms.ToPILImage()(sr_output.squeeze())
            reconstructed_crops.append(faster_image)

    reconstructed_image = reconstruct_from_crops(reconstructed_crops, positions, original_image.size, crop_size)

    return reconstructed_image


if __name__ == '__main__':
    model_weights_path = "F:\\new_logs\\VanillaVAE\\version_0\\checkpoints\\last.ckpt"
    img_path = 'F:\\rewrite_test_gauss'
    image_folder_path = [os.path.join(img_path, im_path) for im_path in os.listdir(img_path)]
    image_output_folder_path = "C:\\Users\\yyt70\\Desktop\\test_fin"
    os.makedirs(image_output_folder_path, exist_ok=True)
    # 示例
    for img_list in image_folder_path:
        img_name = img_list.split('\\')[-1]
        if img_name.endswith('.tiff'):
            img_base_name = img_name.split('.tiff')[0]
        else:
            img_base_name = img_name.split('.png')[0]
        per_image_output_folder = os.path.join(image_output_folder_path, img_base_name)
        os.makedirs(per_image_output_folder, exist_ok=True)
        reconstructed_image = inference_for_large_image(img_list, model_weights_path, per_image_output_folder)
        reconstructed_image.save(os.path.join(per_image_output_folder, 'large_test_image.png'))