# import torch
# import torchvision
# import torchvision.transforms as T
# import torchvision.datasets as datasets
# from torchvision.transforms import functional as F
# import cv2
# import random
#
# font = cv2.FONT_HERSHEY_SIMPLEX
#
# root = 'D:\\project\\phase_structure\\phase_structure_recognition\\faster_rcnn_stem_dataset\\hr_labels'
# annFile = 'D:\\project\\phase_structure\\phase_structure_recognition\\faster_rcnn_stem_dataset\\annotations\\annotation.json'
#
#
# # 定义 coco collate_fn
# def collate_fn_coco(batch):
#     return tuple(zip(*batch))
#
#
# # 创建 coco dataset
# coco_det = datasets.CocoDetection(root, annFile, transform=T.ToTensor())
# # 创建 Coco sampler
# sampler = torch.utils.data.RandomSampler(coco_det)
# batch_sampler = torch.utils.data.BatchSampler(sampler, 8, drop_last=True)
#
# # 创建 dataloader
# data_loader = torch.utils.data.DataLoader(coco_det, batch_sampler=batch_sampler, collate_fn=collate_fn_coco)
#
# # 可视化
# for imgs, labels in data_loader:
#     for i in range(len(imgs)):
#         bboxes = []
#         ids = []
#         img = imgs[i]
#         labels_ = labels[i]
#         for label in labels_:
#             bboxes.append([label['bbox'][0],
#                            label['bbox'][1],
#                            label['bbox'][0] + label['bbox'][2],
#                            label['bbox'][1] + label['bbox'][3]
#                            ])
#             ids.append(label['category_id'])
#
#         img = img.permute(1, 2, 0).numpy()
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         for box, id_ in zip(bboxes, ids):
#             x1 = int(box[0])
#             y1 = int(box[1])
#             x2 = int(box[2])
#             y2 = int(box[3])
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
#             cv2.putText(img, text=str(id_), org=(x1 + 5, y1 + 5), fontFace=font, fontScale=1,
#                         thickness=2, lineType=cv2.LINE_AA, color=(0, 255, 0))
#         cv2.imshow('test', img)
#         cv2.waitKey()


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pycocotools.coco import COCO
from PIL import Image
import os


def visualize_labels(coco, img_id):
    # Load image information
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(image_dir, img_info['file_name'])
    image = Image.open(img_path)

    # Load annotations for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    # Visualize the image
    plt.imshow(image)
    plt.axis('off')

    # Plot bounding boxes and labels
    for ann in annotations:
        bbox = ann['bbox']
        label = 'active molecule'
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(bbox[0], bbox[1] - 5, label, color='r', backgroundcolor='white')

    plt.show()

# Path to the COCO annotation file and image directory
annotation_file = 'D:\\project\\phase_structure\\phase_structure_recognition\\faster_rcnn_stem_dataset\\annotations\\annotation.json'
image_dir = 'D:\\project\\phase_structure\\phase_structure_recognition\\faster_rcnn_stem_dataset\\hr_labels'

# COCO instance for loading annotations
coco = COCO(annotation_file)

# Specify the image ID for visualization
img_id = 16  # Replace with the desired image ID

# Visualize labels for the specified image
visualize_labels(coco, img_id)
