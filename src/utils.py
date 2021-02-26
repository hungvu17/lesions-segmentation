from detectron2.data import MetadataCatalog, DatasetCatalog

class Dataset:
    def register(dataset):
        DatasetCatalog.register("dr_lesions_" + dataset, lambda d=d: get_dict(dataset))
        MetadataCatalog.get("dr_lesions_" + dataset).set(thing_classes=["hemorrhage", "exudate", "microaneurysms"])
        dr_lesions_metadata = MetadataCatalog.get("dr_lesions_" + dataset)
