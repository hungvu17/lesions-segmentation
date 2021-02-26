from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from src.utils import Dataset

test_set = Dataset('test', './data/test.json')
test_set.register()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
