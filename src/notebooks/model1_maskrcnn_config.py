# -*- coding: utf-8 -*-
"""Initial model with Mask R-CNN: Config.

References:
    Intro to deep learning for medical imaging by MD.ai
    - https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-mdai-client-lib.ipynb
    - Retrieved 2018-09-20
    - Licensed under the Apache License, Version 2.0
"""

from libs.Mask_RCNN.mrcnn.config import Config


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
