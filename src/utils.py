import os
import json
import numpy as np
# from detectron2.data import MetadataCatalog, DatasetCatalog

class Dataset:
    def __init__(self, name, annotation_path, data_path=None):
        self.name = name
        self.annotation = annotation_path
        self.data_path = data_path
    
    def __get_dict(self):
        with open(self.annotation) as f:
            anns = json.load(f)
            return anns
	
#     def register(self):
#         DatasetCatalog.register("dr_lesions_" + self.name, self.__get_dict)
#         MetadataCatalog.get("dr_lesions_" + self.name).set(thing_classes=["hemorrhage", "exudate", "microaneurysms"])
#         dr_lesions_metadata = MetadataCatalog.get("dr_lesions_" + self.name)

    def __bin_to_dec(self, stats):
        result = np.zeros((len(stats)), dtype=int)
        for idx, row in enumerate(stats):
            for order, element in enumerate(row):
                result[idx] += element*(2**order)
        return result

    def __key_to_lesion(self, key):
        lesion = {
            0: 'no_lesion',
            1: 'hemorrhage',
            2: 'exudate',
            3: 'hemorrhage_exudate',
            4: 'microaneurysms',
            5: 'hemorrhage_microaneurysms',
            6: 'exudate_microaneurysms',
            7: 'hemorrhage_exudate_microaneurysms'
        }
        return lesion.get(key)
    
    def stats(self):
        lesions_filenames = [x for x in os.listdir('{}/images'.format(self.data_path)) if x[:4] == 'CTEH']
        normal_filenames  = [x for x in os.listdir('{}/images/normal'.format(self.data_path)) if x[:4] == 'CTEH']
        
        lesions_count = np.zeros((len(lesions_filenames),3), dtype=int)
        with open(self.annotation) as f:
            annotations = json.load(f)
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