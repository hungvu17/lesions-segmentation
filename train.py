import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

import numpy as np
import os, json, cv2, argparse, configparser
from src.utils import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='diabetic retinopathy lesions folder', default='./data')
parser.add_argument('--annotation', help='annotation file', default='./data/annotations.json')
parser.add_argument('--params', help='training params file', default='./params.ini')
parser.add_argument('--output', help='output folder', default='./output')

args = parser.parse_args()
annotation_path = args.annotation
data_path = args.data
config_path = args.config
output = args.output

params = ConfigParser()
params.read(config_path)

# Sampling and registering dataset
def config_detectron(params):
    model = params['MODEL']
    dataset = params['DATASET']
    solver = params['SOLVER']
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model['name']))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model['checkpoint_url'])
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = model['batch_size_per_image']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = model['classes']
    
    cfg.DATASETS.TRAIN = (dataset['train_set'],)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = dataset['workers']
    
    cfg.SOLVER.IMS_PER_BATCH = solver['ims_per_batch']
    cfg.SOLVER.BASE_LR = solver['base_lr']
    cfg.SOLVER.MAX_ITER = solver['iteration']   
    cfg.SOLVER.STEPS = []
    
    return cfg

def get_fold(index):
    dataset = Dataset('dr_lesions', annotation_path, data_path)
    folds = dataset.sampling()
    return folds[index]


if __name__ == '__main__':
    '''
    Train scripts
    Input:
        - annotation: preprocessed annotation files
        - data: data folder
        - config: training config
        - output: output folder
    '''
    # Setup Colab 
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.makedirs(output, exist_ok=True)
    
    # Load data fold
    train_set, _ = get_fold(0)
    
    # Register dataset to detectron2
    DatasetCatalog.register('dr_lesions_train', lambda d=d: train_set)
    MetadataCatalog.get('dr_lesions_train').set(thing_classes=["hemorrhage", "exudate", "microaneurysms"])
    dr_lesions_metadata = MetadataCatalog.get('dr_lesions_train')
    
    # Load detectron config
    config = config_detectron(params)
    
    # Train program
    trainer = DefaultTrainer(config) 
    trainer.resume_or_load(resume=False)
    trainer.train()

