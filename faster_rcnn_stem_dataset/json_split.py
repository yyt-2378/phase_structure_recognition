import json

# 路径替换为您的JSON文件的实际路径
trainval_annotation_path = '/root/autodl-tmp/phase_structure_recognition/faster_rcnn_stem_dataset/annotations/trainval_annotation.json'
test_annotation_path = '/root/autodl-tmp/phase_structure_recognition/faster_rcnn_stem_dataset/annotations/test_annotation.json'

def remove_zoom_annotations(file_path):
    # 加载JSON数据
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 如果JSON结构包含'images'和'annotations'，我们可以继续，否则输出错误
    if 'images' in data and 'annotations' in data:
        # 创建一个新列表，包含不含“zoom”的图片
        images_without_zoom = [img for img in data['images'] if 'zoom' not in img['file_name']]

        # 获取不包含“zoom”的图片ID集合
        valid_image_ids = set(img['id'] for img in images_without_zoom)

        # 基于有效的图片ID过滤annotations
        annotations_without_zoom = [ann for ann in data['annotations'] if ann['image_id'] in valid_image_ids]

        # 更新JSON数据
        data['images'] = images_without_zoom
        data['annotations'] = annotations_without_zoom

        # 将更新后的数据写回文件
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)  # 使用indent参数使JSON文件格式化，提高可读性
        print(f"Updated file: {file_path}")
    else:
        print(f"Error: The file {file_path} does not contain 'images' and/or 'annotations' keys.")

# 对trainval和test注解文件执行清理操作
remove_zoom_annotations(trainval_annotation_path)
remove_zoom_annotations(test_annotation_path)
