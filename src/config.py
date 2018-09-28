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

    NAME = 'pneumonia'

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # Number of images to train with on each GPU.
    # A 12GB GPU can typically handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes.
    # Use the highest number that your GPU can handle for best performance.
    # p2.xlarge : GPU memory = 12GB
    # p2.8xlarge: GPU memory = 96GB
    # p3.2xlarge: GPU memory = 16GB
    # p3.8xlarge: GPU memory = 64GB
    IMAGES_PER_GPU = 2                 # default: 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 100             # default: 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 5              # default: 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = 'resnet50'

    # Number of classification classes (including background)
    NUM_CLASSES = 2                    # background + 1 pneumonia classes

    # Input image resizing
    IMAGE_RESIZE_MODE = 'none'         # default: square -- TODO: resizing currently causes problems
    IMAGE_MIN_DIM = 960                # default: 800  -- smaller cropper recommended
    IMAGE_MAX_DIM = 1024               # default: 1024 -- max size of DICOM images

    # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    IMAGE_CHANNEL_COUNT = 1            # grayscale

    # Image mean
    MEAN_PIXEL = np.array([127.5])

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # default: (32, 64, 128, 256, 512)

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 300         # default: 200 -- 1024/3, rounded down

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33          # default: 0.33

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]              # default: [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 4               # default: 100 -- up to 4 true boxes per image

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 4        # default: 100 -- up to 4 true boxes per image

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.67    # default: 0.7 -- lower to increase recall

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3      # default: 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001              # default: 0.001
    LEARNING_MOMENTUM = 0.9            # default: 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001              # default: 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,          # default: 1.
        "rpn_bbox_loss": 1.,           # default: 1.
        "mrcnn_class_loss": 1.,        # default: 1.
        "mrcnn_bbox_loss": 1.,         # default: 1.
        "mrcnn_mask_loss": 1.          # default: 1.
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True                # default: True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False                   # default: False, since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0           # default: 5.0

    def get_attrs(self):
        """Return all configs in one string."""
        return '\n'.join(
            "{:30} {}".format(a, getattr(self, a)) for a in dir(self)
            if not a.startswith("__")
            and not callable(getattr(self, a)))


class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
