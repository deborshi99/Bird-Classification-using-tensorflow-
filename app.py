from utils.constants import *
from utils.detecting_birds import *

detector = Detector()
detector.downloadModel(EFFICIENTDET_D7_MODEL_URL)
detector.loadModel()
detector.predictImage("app/test_data/images.jpeg", threshold=0.5)