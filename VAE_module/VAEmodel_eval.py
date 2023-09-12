import torch
from torchvision import transforms
from PIL import Image
from vae_models import VanillaVAE  # 你的 VAE 模型定义
import os
from scipy.ndimage import zoom


# 加载已训练的模型
model = VanillaVAE(in_channels=1, latent_dim=256)  # 根据你的模型参数创建模型实例
# 加载模型权重
checkpoint_path = "D:\\project\\deep_learning_recovery\\logs\\VanillaVAE\\version_9\\checkpoints\\epoch=181-step=9463.ckpt"  # 替换为实际的 .ckpt 文件路径
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # 加载 checkpoint，使用 map_location 参数指定设备
state_dict = checkpoint['state_dict']
new_state_dict = {}
for key in state_dict:
    new_key = key.replace("model.", "")  # 去掉前缀 "model."
    new_state_dict[new_key] = state_dict[key]
# 从 checkpoint 中提取模型权重
model.load_state_dict(new_state_dict)

model.eval()

# 图像预处理
tran = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转为灰度图
    transforms.Resize((128, 128)),      # 调整尺寸
    transforms.ToTensor()                         # 转为张量
])

# 图片文件夹路径
image_folder = "F:\\deep_learning_recovery\\STEM_VAE_Data\\validation\\img"  # 替换为图片文件夹的路径

# 遍历图片文件夹中的图片并进行推断
for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)
    image_tensor = tran(image).unsqueeze(0)  # 添加 batch 维度

    with torch.no_grad():
        label = torch.zeros_like(image_tensor)
        reconstructed_image_noise = model.forward(image_tensor, label)[0]

    # 在这里可以对重建图像进行后续处理或可视化
    # 例如，你可以将重建图像转换为 NumPy 数组并使用 Matplotlib 进行可视化

    # 保存重建图像
    reconstructed_image = image_tensor - reconstructed_image_noise
    reconstructed_image = reconstructed_image.squeeze(0).permute(1, 2, 0).cpu().numpy().squeeze()

    # upsampled_array = zoom(reconstructed_image, (2, 2), mode='nearest')
    upsampled_array = reconstructed_image
    upsampled_array = (upsampled_array - upsampled_array.min()) / (upsampled_array.max() - upsampled_array.min())

    save_path = os.path.join("F:\\output_folders", f"reconstructed_{image_file}")
    Image.fromarray((upsampled_array * 255).astype("uint8"), mode='L').save(save_path)

print("Testing complete.")
