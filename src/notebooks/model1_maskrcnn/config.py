# -*- coding: utf-8 -*-
"""Initial model with Mask R-CNN: Config."""

import os
import sys
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_path + "/Mask_RCNN")
from mrcnn.config import Config


class GlobalConfig:
    """Config class for globals."""

    __conf = {
        'ROOT_DIR': '/projects/lungbox',
        'RANDOM_SEED': 42,
        'AWS_REGION': 'us-west-2',
        'AWS_ACCESS_KEY': '',
        'S3_BUCKET_NAME': 'lungbox',
        'S3_CLASS_INFO_KEY': 'data/raw/stage_1_detailed_class_info.csv',
        'S3_TRAIN_BOX_KEY': 'data/raw/stage_1_train_labels.csv',
        'S3_CLASS_INFO_PATH': 's3://lungbox/data/raw/stage_1_detailed_class_info.csv',
        'S3_TRAIN_BOX_PATH': 's3://lungbox/data/raw/stage_1_train_labels.csv',
        'S3_STAGE1_TRAIN_IMAGE_DIR': 'data/raw/stage_1_train_images',
        'S3_STAGE1_TEST_IMAGE_DIR': 'data/raw/stage_1_test_images',
        'MODEL_DIR': '/projects/lungbox/models',
        'S3_MODEL_DIR': 's3://lungbox/models',
        'TRAINING_DATA_MAX_SIZE': 25684
    }
    __setters = ['AWS_ACCESS_KEY']

    @staticmethod
    def get(name):
        """Get config by name."""
        return GlobalConfig.__conf[name]

    @staticmethod
    def set(name, value):
        """Set config if config is settable."""
        if name in GlobalConfig.__setters:
            GlobalConfig.__conf[name] = value
        else:
            raise NameError("Name not accepted in set() method")


class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia
    dataset. Overrides values in the base Config class.
    """

    # Give the configuration a recognizable name
    NAME = 'pneumonia'

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    BACKBONE = 'resnet101'

    NUM_CLASSES = 2  # background + 1 pneumonia classes

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 3
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.1

    STEPS_PER_EPOCH = 100

    def get_attrs(self):
        """Return all configs in one string."""
        return '\n'.join(
            "{:30} {}".format(a, getattr(self, a)) for a in dir(self)
            if not a.startswith("__")
            and not callable(getattr(self, a)))


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
