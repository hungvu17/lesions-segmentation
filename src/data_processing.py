import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from detectron2.structures import BoxMode

def getAnnotations(id):
    '''
    Load annotations with segmentation mask and bounding box
    Input: image id
    Output: annotations
    '''
    anns_path = '/Users/minh/a2ds/data/dr_lesions'
    coco = COCO(os.path.join(anns_path, 'instances_default.json'))
    
    image = coco.loadImgs(id)
    anns_ids = coco.getAnnIds(id)

    lesions = []
    for anns_id in anns_ids:
        anns = coco.loadAnns(anns_id)
        segs = anns[0]['segmentation']
        category = anns[0]['category_id']
        bbox = []
        for seg in segs:
            X = seg[0::2]
            Y = seg[1::2]
            bbox = [min(X), min(Y), max(X), max(Y)]
            lesion = {
                'category_id': category - 1,
                'segmentation': [seg],
                'bbox': bbox,
                'bbox_mode': BoxMode.XYXY_ABS
            }
            lesions.append(lesion)

    annotation = {
        'file_name': filename,
        'image_id': id,
        'height': height,
        'width': width,
        'annotations': lesions
    }
    return annotation

if __name__ == '__main__':
    filenames = [x for x in os.listdir('{}/images'.format(root)) if x[:4] == 'CTEH']
    anns = []

    for id, filename in tqdm(enumerate(filenames)):
        ann = getAnnotations(id+1)
        anns.append(ann)
    with open('annotations.json', 'w') as annotations:
        json.dump(anns, annotations)
