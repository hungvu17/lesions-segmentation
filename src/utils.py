import os
import json
from detectron2.data import MetadataCatalog, DatasetCatalog

class Dataset:
    def __init__(self, name, annotation_path):
        self.name = name
        self.annotation = annotation_path
    def __get_dict(self):
        with open(self.annotation) as f:
            anns = json.load(f)
            return anns
	
    def register(self):
        DatasetCatalog.register("dr_lesions_" + self.name, self.__get_dict)
        MetadataCatalog.get("dr_lesions_" + self.name).set(thing_classes=["hemorrhage", "exudate", "microaneurysms"])
        dr_lesions_metadata = MetadataCatalog.get("dr_lesions_" + self.name)
