# -*- coding: utf-8 -*-
"""Initial model with Mask R-CNN: Config."""

import os
import sys
import numpy as np
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

        'LOCAL_CLASS_INFO_PATH': 'data/raw/stage_1_detailed_class_info.csv',
        'LOCAL_TRAIN_BOX_PATH': 'data/raw/stage_1_train_labels.csv',
        'LOCAL_STAGE1_TRAIN_IMAGE_DIR': 'data/raw/stage_1_train_images',
        'LOCAL_STAGE1_TEST_IMAGE_DIR': 'data/raw/stage_1_test_images',
        'LOCAL_DICOM_IMAGE_LIST_PATH': 'data/preprocessed/local_dicom_image_list.csv',

        'EXAMPLE_CLASS_INFO_PATH': 'data/example/example_detailed_class_info.csv',
        'EXAMPLE_TRAIN_BOX_PATH': 'data/example/example_train_labels.csv',
        'EXAMPLE_STAGE1_TRAIN_IMAGE_DIR': 'data/example/stage_1_train_images',
        'EXAMPLE_DICOM_IMAGE_LIST_PATH': 'data/example/example_dicom_image_list.csv',

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
    """
    MODEL 11
    Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    NAME = 'pneumonia'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2             # batch size = 1*2 = 2
    TRAIN_SAMPLE_SIZE = 2500*0.8
    VALID_SAMPLE_SIZE = 2500*0.2
    BATCH_SIZE = GPU_COUNT*IMAGES_PER_GPU
    BACKBONE = 'resnet101'
    NUM_CLASSES = 2                # background + 1 pneumonia classes
    MAX_GT_INSTANCES = 4           # up to 4 bounding boxes per image
    IMAGE_RESIZE_MODE = 'none'     # No resizing or padding. Return the image unchanged.
    IMAGE_MIN_DIM = 1024           # preserve original 1024 dimension
    IMAGE_MAX_DIM = 1024           # preserve original 1024 dimension
    IMAGE_CHANNEL_COUNT = 1        # grayscale
    MEAN_PIXEL = np.array([127.5]) # mean pixel intensity

    STEPS_PER_EPOCH = TRAIN_SAMPLE_SIZE // BATCH_SIZE
    VALIDATION_STEPS = VALID_SAMPLE_SIZE // BATCH_SIZE

    # How many anchors per image to use for RPN training
    # Set according to the Faster R-CNN paper, Section 3.1.3, Training RPNs.
    # default: 256 appears arbitrary. Treat this as a hyper-parameter.
    RPN_TRAIN_ANCHORS_PER_IMAGE = 512

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Non-max suppression threshold to filter RPN proposals.
    RPN_NMS_THRESHOLD = 0.9        # default: 0.7; increase to generate more proposals.

    # Training ROIs kept after non-maximum suppression
    POST_NMS_ROIS_TRAINING = 3200

    # Number of ROIs per image to feed to classifier/mask heads
    TRAIN_ROIS_PER_IMAGE = 256
    ROI_POSITIVE_RATIO = 0.33      # default: 0.33

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.002          # default: 0.001
    LEARNING_MOMENTUM = 0.9        # default: 0.9


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5 # positive class confidence threshold
    DETECTION_MAX_INSTANCES = 4    # up to 4 true boxes per image
    DETECTION_NMS_THRESHOLD = 0.1
