# -*- coding:utf-8 -*-

import json

json_file = "D:\\project\\deep_learning_recovery\\my_set\\annotations\\ann.json"
data = json.load(open(json_file, 'r'))

data_2 = {
    'categories': data['categories'],
    'images': [data['images'][0]]
}

annotation = []
imgID = data_2['images'][0]['id']
for ann in data['annotations']:
    if ann['image_id'] == imgID:
        annotation.append(ann)
data_2['annotations'] = annotation

json.dump(data_2, open(r'D:\\project\\deep_learning_recovery\\my_set\\single_person_kp.json', 'w'), indent=4)