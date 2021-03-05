import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog

import numpy as np
import os, json, cv2, argparse
from src.utils import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='diabetic retinopathy lesions folder', default='./data')
parser.add_argument('--annotation', help='annotation file', default='./data/annotations.json')
parser.add_argument('--config', help='training config file', default='./experiment/exp_1.json')
parser.add_argument('--output', help='output folder', default='./output')

args = parser.parse_args()
annotation_path = args.annotation
data_path = args.data
config_path = args.config
output = args.output

with open(config_path) as config_file:
    config = json.load(config_file)

def config_detectron(params):
    model = params['model']
    dataset = params['dataset']
    solver = params['solver']
    
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
    dataset = Dataset('dr_lesions', annotation_path, data_path)
    train_set, _ = dataset.sampling()[0]
    
    # Register dataset to detectron2
    DatasetCatalog.register('dr_lesions_train', lambda : train_set)
    MetadataCatalog.get('dr_lesions_train').set(thing_classes=config['dataset']['classes'])
    dr_lesions_metadata = MetadataCatalog.get('dr_lesions_train')
    
    # Load detectron config
    detectron2_config = config_detectron(config)
    
    # Train program
    trainer = DefaultTrainer(detectron2_config) 
    trainer.resume_or_load(resume=False)
    trainer.train()

