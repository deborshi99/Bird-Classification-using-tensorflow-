import cv2
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file
from .constants import LABELS, PRETRAINED_MODEL_FOLDER, MODEL_NAME, FILE_NAME

class Detector:
    def __init__(self):
        pass

    def downloadModel(self, modelURL):
        os.makedirs(PRETRAINED_MODEL_FOLDER, exist_ok=True)
        get_file(
            fname=FILE_NAME,
            origin=modelURL,
            cache_dir=PRETRAINED_MODEL_FOLDER,
            cache_subdir="checkpoints",
            extract=True
        )
        
    
    def loadModel(self):
        print(f"{MODEL_NAME} model is loading ")
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(
            os.path.join(
                PRETRAINED_MODEL_FOLDER,
                "checkpoints",
                MODEL_NAME,
                "saved_model"
            )
        )
        print(f"{MODEL_NAME} model has been loaded")

        





        
        