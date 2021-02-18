# retinal-lesion-torch

Pytorch Implementation of Retinal Lesion Detection and Segmentation

Using detectron2

## I. Data processing

Converting CTEH DR lesions dataset into trainable dataset adapting Detectron2.

### Prerequisites 

pycocotools: [Installation](https://github.com/cocodataset/cocoapi)

### Procedures

#### Data Structures

```
.
+-- images
│		+-- <CTEH-xxxxx.jpg>
│		+-- ...
+-- lesions
│		+-- hemorrhage
│				+-- <CTEH-xxxxx.jpg>
│				+-- ...
│		+-- exudates
│				├── <CTEH-xxxxx.jpg>
│				├── ...
│		+-- microaneurysms
│				+-- <CTEH-xxxxx.jpg>
│				+-- ...
+-- preprocessing.ipynb
+-- dr_lesion.ipynb
+-- instances_default.json
```

1. Install pycocotools for preprocessing script

2. Run **preprocessing.ipynb** to generate **train_set.json** and **val_set.json**

   ```
   Input: instances_default.json
   OutputL: train_set.json and val_set.json
   ```

3. Run **dr_lesions.ipynb** * to train and evaluate data from **train_set.json** and **val_set.json**

   *If running **dr_lesions.ipynb** on Google Colab, please comply with the data structure described as above.





