LABELS = ["birds"]

PRETRAINED_MODEL_FOLDER = "pretrained_model"

EFFICIENTDET_D4_MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
MODEL_NAME = EFFICIENTDET_D4_MODEL_URL.split("/")[-1].split(".")[0]
FILE_NAME = EFFICIENTDET_D4_MODEL_URL.split("/")[-1]