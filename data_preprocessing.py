import os
import json
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO
from detectron2.structures import BoxMode

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='diabetic retinopathy lesions folder', default='./data')
parser.add_argument('--annotation', help='annotatation file', default='./data/instances_default.json')
parser.add_argument('--output', help='output directory', default='./data')

args = parser.parse_args()
annotation_path = args.annotation
data_path = args.data
output = args.output

def getAnnotations(id):
    '''
    Load annotations with segmentation mask and bounding box
    Input: image id
    Output: annotations
    '''
    coco = COCO(annotation_path)
    
    image = coco.loadImgs(id)
    anns_ids = coco.getAnnIds(id)


    filename = '{}/images/{}'.format(data_path, image[0]['file_name'])
    height, width = cv2.imread(filename).shape[:2]

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

def data_sampling(data):
    return data[:160], data[160:]

if __name__ == '__main__':
    filenames = [x for x in os.listdir('{}/images'.format(data_path)) if x[:4] == 'CTEH']
    anns = []

    for id, filename in tqdm(enumerate(filenames)):
        ann = getAnnotations(id+1)
        anns.append(ann)
    
    train_set, val_set = data_sampling(anns)
    with open('{}/annotations.json'.format(output), 'w') as annotations:
        json.dump(anns, annotations)
    with open('{}/train.json'.format(output), 'w') as annotations:
        json.dump(train_set, annotations)
    with open('{}/val.json'.format(output), 'w') as annotations:
        json.dump(val_set, annotations)
