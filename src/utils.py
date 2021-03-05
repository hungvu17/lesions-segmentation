import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

class Dataset:
    def __init__(self, name, annotation_path=None, data_path=None):
        self.name = name
        self.annotation = annotation_path
        self.data_path = data_path
    
    def __get_dict(self):
        '''
        Load annotations from JSON file
        '''
        with open(self.annotation) as f:
            anns = json.load(f)
            return anns

    def __bin_to_dec(self, stats):
        '''
        Convert lesions binary list (LSB first) to decimal represented lesions combinations
        
        '''
        result = np.zeros((len(stats)), dtype=int)
        for idx, row in enumerate(stats):
            for order, element in enumerate(row):
                result[idx] += element*(2**order)
        return result

    def __key_to_lesion(self, key):
        '''
        Convert decimal key to lesions combinations
        '''
        lesion = {
            0: 'unlabeled',
            1: 'hemorrhage',
            2: 'exudate',
            3: 'hemorrhage_exudate',
            4: 'microaneurysms',
            5: 'hemorrhage_microaneurysms',
            6: 'exudate_microaneurysms',
            7: 'hemorrhage_exudate_microaneurysms'
        }
        return lesion.get(key)
    
    def histogram(self, data, titles=None):
        '''
        Plot histogram
        '''
        fig, axs = plt.subplots(1, 1,sharey=True, tight_layout=True)
        axs.hist(data, bins=np.arange(data.min(), data.max()+1))
        if titles is not None and len(titles) == 3:
            axs.set_title(titles[0])
            axs.set_xlabel(titles[1])
            axs.set_ylabel(titles[2])
            
        
    def sampling(self, scale=0.8, threshold = 8):
        '''
        Sample dataset by two rules:
            - Keep distribution of lesions combinations
            - Balance amount of lesions per class.
        Input: 
            - scale: splitting scale of train_set/test_set 
            - threshold: an integer separates a list of images into 2 subgroup: 'few lesions', 'many lesions'
        *threshold = 8 gives the minimum variance of the amount of lesions after folding.
        '''
        lesions_filenames = [x for x in os.listdir('{}/images'.format(self.data_path)) if x[:4] == 'CTEH']
        lesions_count = np.zeros((len(lesions_filenames),3), dtype=int)
        
        annotations = self.__get_dict()
        for idx, image in enumerate(annotations):
            lesions = image['annotations']
            image['file_name'] = os.path.join(self.data_path, 'images', image['file_name'].split('/')[-1])
            for lesion in lesions:
                lesions_count[idx, lesion['category_id']] += 1
                
        image_id = [annotation['image_id'] for annotation in annotations]
        count_mask = (lesions_count.sum(axis=1) >= threshold)*1
        ratio = int(scale/(1 - scale)) + 1
        skf = StratifiedKFold(n_splits=ratio)
        folds = []
        for train_index, test_index in skf.split(image_id, count_mask):
            train_set = np.array(annotations)[train_index]
            test_set = np.array(annotations)[test_index]
            folds.append([train_set, test_set])
            
        return folds
    
    def stats(self):
        '''
        General dataset statistics
        '''
        lesions_filenames = [x for x in os.listdir('{}/images'.format(self.data_path)) if x[:4] == 'CTEH']
        normal_filenames  = [x for x in os.listdir('{}/images/normal'.format(self.data_path)) if x[:4] == 'CTEH']
        
        lesions_count = np.zeros((len(lesions_filenames),3), dtype=int)
        annotations =  self.__get_dict()

        for idx, image in enumerate(annotations):
            lesions = image['annotations']
            for lesion in lesions:
                lesions_count[idx, lesion['category_id']] += 1
        
        lesions_combination = self.__bin_to_dec((lesions_count > 0)*1)
        unique, counts = np.unique(lesions_combination, return_counts=True)
        lesions_combination_dict = dict(zip((self.__key_to_lesion(key) for key in unique), counts))
        
        stats = {
            'number_of_lesions_images': len(lesions_filenames),
            'number_of_normal_images': len(normal_filenames),
            'number_of_hemorrhage_images': ((lesions_count > 0)*1).sum(axis=0)[0],
            'number_of_exudate_images': ((lesions_count > 0)*1).sum(axis=0)[1],
            'number_of_microaneurysms_images': ((lesions_count > 0)*1).sum(axis=0)[2],
            'number_of_hemorrhage': lesions_count.sum(axis=0)[0],
            'number_of_exudate': lesions_count.sum(axis=0)[1],
            'number_of_microaneurysms': lesions_count.sum(axis=0)[2],
            'combination': lesions_combination_dict
        }
        return stats  