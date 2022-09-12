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

    def createBoundingBox(self, image, threshold):
        inputTensor = cv2.cvtColor(
            image, 
            cv2.COLOR_BGR2RGB
        )
        inputTensor = tf.convert_to_tensor(
            inputTensor,
            dtype=tf.uint8
        )
        inputTensor = inputTensor[tf.newaxis,...]

        detections = self.model(inputTensor)
        bboxes = detections["detection_boxes"][0].numpy()
        classIndexes = detections["detection_classes"][0].numpy().astype(np.int32)
        classScores = detections["detection_scores"][0].numpy()

        imH, imW, imC = image.shape
        bboxidx = tf.image.non_max_suppression(
            bboxes,
            classScores,
            max_output_size=50,
            iou_threshold=threshold,
            score_threshold=threshold
        )
        print(bboxidx)
        if len(bboxidx) != 0:
            for i in bboxidx:
                bbox = tuple(bboxes[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]

                classLabelText = LABELS[0]
                #classColor = self.colorsList[classIndex]

                displayText = f"{classLabelText}: {classConfidence}%"

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color = (255, 0, 0), thickness=1)
                cv2.putText(image, displayText, (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2) 
        return image

    def predictImage(self, imagePath, threshold=0.5):
        image = cv2.imread(imagePath)
        bboximage = self.createBoundingBox(image, threshold)
        #cv2.imwrite(self.modelName+"jpg", bboximage)

        cv2.imshow("Result", bboximage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





        
        