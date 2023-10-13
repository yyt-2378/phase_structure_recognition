import os

import torch
from torchvision import transforms
import torchvision
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageDraw

# Load the pre-trained Fast R-CNN model
model = torchvision.models.detection.__dict__['fasterrcnn_resnet50_fpn'](num_classes=2, pretrained=False)
state_dict = torch.load('F:\\checkpoints\\model_39.pth')
model.load_state_dict(state_dict['model'])
model.eval()

# Load an image and its annotations
# Replace with your own image and annotation loading code
image_path = "D:\\project\\phase_structure\\phase_structure_recognition\\faster_rcnn_stem_dataset\\test"
output_path = 'F:\\faster_rcnn_output'
os.makedirs(output_path, exist_ok=True)

for img in os.listdir(image_path):
    # Load the image
    img_path = os.path.join(image_path, img)
    image = Image.open(img_path).convert("RGB")

    # Transform the image and annotations
    train_transforms = transforms.Compose([transforms.ToTensor()])
    image_tensor = train_transforms(image)
    image_tensor = image_tensor.unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        predictions = model(image_tensor)[0]
        print(predictions)

    # Display the image with predicted boxes (if score > 0.5)
    draw = ImageDraw.Draw(image)
    for box, score in zip(predictions["boxes"], predictions["scores"]):
        if score > 0.5:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="blue")
            draw.text((box[0], box[1]), f"Score: {score:.2f}", fill="blue")

        # Save the image with annotations
        output_img_path = os.path.join(output_path, img)
        image.save(output_img_path)

# 准备数据
# inputs = []
# img_path = os.path.join(val_images_path, os.listdir(val_images_path)[5])
# img = Image.open(img_path).convert("RGB")
# img_tensor = torch.from_numpy(np.array(img)/255.).permute(2,0,1).float()
# if torch.cuda.is_available():
#     img_tensor = img_tensor.cuda()
# inputs.append(img_tensor)
#
# model.eval()
#
# # 预测结果
# with torch.no_grad():
#     predictions = model(inputs)
#
# # 结果可视化
# vis_detection(img, predictions[0], list(idx2names.values()), min_score=0.8)
