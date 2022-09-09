from utils.constants import *
from utils.detecting_birds import *

detector = Detector()
detector.downloadModel(EFFICIENTDET_D4_MODEL_URL)
detector.loadModel()