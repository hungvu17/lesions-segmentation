# retinal-lesion-torch

Pytorch Implementation of Retinal Lesion Detection and Segmentation

Using detectron2

## Prerequisites 

1. pycocotools: [Installation](https://github.com/cocodataset/cocoapi)

2. Install packages*:
```
pip install -r requirments.txt
```
*Compatible with CUDA 10.1

## Data Structures

```
.
+-- images
|   +-- <CTEH-xxxxx.jpg>
|   +-- ...
+-- lesions
|   +-- hemorrhage
|   +-- <CTEH-xxxxx.jpg>
|   +-- ...
|   +-- exudates
|   +-- <CTEH-xxxxx.jpg>
|   +-- ...
|   +--  microaneurysms
|   +--  <CTEH-xxxxx.jpg>
|   +--  ...
+-- instances_default.json
```

## Procedure

1. Run **data_preprocessing** to generate **train_set.json** and **val_set.json**

   **Script**:
   python data_preprocessing --data path/to/data/folder --annotation path/to/annotation/file --output path/to/output/folder
   
   **Example**:
   ```
   python data_preprocessing --data ./data --annotation ./data/instances_default.json --output ./data
   ```
   



