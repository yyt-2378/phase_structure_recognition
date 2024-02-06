import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pycocotools.coco import COCO
from PIL import Image, ImageOps
import os


def visualize_labels(coco, img_id, output_dir):
    # Load image information
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(image_dir, img_info['file_name'])

    # Load the image
    image = Image.open(img_path).convert('L')  # Convert to grayscale

    # Load annotations for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    # Create a figure and axis to plot the image
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')  # Display the image in grayscale
    plt.axis('off')

    # Plot bounding boxes and labels
    for ann in annotations:
        bbox = ann['bbox']
        label = coco.loadCats(ann['category_id'])[0]['name']
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=4, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        # ax.text(bbox[0], bbox[1] - 5, label, color='r', fontsize=8, backgroundcolor='white')

    # Save the image with annotations
    output_path = os.path.join(output_dir, f"annotated_label_{img_info['file_name']}")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# Path to the COCO annotation file and image directory
annotation_file = 'F:\\trainval_annotation.json'
image_dir = 'F:\\final_version_stem_project\\phase_structure_recognition\\faster_rcnn_stem_dataset\\hr_labels'
output_directory = 'D:\\project\\phase_structure\\phase_structure_recognition\\get_molecule_information\\analysis_results'  # Directory to save annotated images

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# COCO instance for loading annotations
coco = COCO(annotation_file)

# Specify the image ID for visualization
img_id = 3535  # Replace with the desired image ID

# Visualize and save labels for the specified image
visualize_labels(coco, img_id, output_directory)

