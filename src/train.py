#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

"""
Train a lung opacity detector on the RSNA 2018 pneumonia challenge dataset.
Uses the official matterport's "official" implementation of Mask R-CNN.

Usage:
    python src/train.py \
        --data_source=example \
        --subset_size=2500 \
        --validation_split=0.2 \
        --epochs=100 \
        --learning_rate=0.002 \
        --model_dir='/projects/lungbox/models'
"""

import os
import sys
import time
from imgaug import augmenters as iaa

# Set up PATH
try:
    script_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_path = "/projects/lungbox/src"
    sys.path.append(script_path)
sys.path.append(script_path + "/Mask_RCNN")

# Set non-display matplotlib backend for scripting
# Avoids "backend not available" failure.
# Must be imported before any other imports that use matplotlib
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    # import matplotlib.pyplot as plt

# Set up logging
import logging
import logging.config
import yaml
with open('config/logging.yml', 'rt') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
    logger = logging.getLogger('__name__')

# Import modeling utils
import Mask_RCNN.mrcnn.model as modellib
from config import GlobalConfig
from config import DetectorConfig
from data import TrainingData

# Set manually since not properly picked up from system config
os.environ['AWS_REGION'] = GlobalConfig.get('AWS_REGION')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Runs Mask R-CNN for lung opacity detection in pneumonia.')
    parser.add_argument('--data_source', required=False,
                        help="One of ['example', 'local', or 's3']. Requires configuration in config.py. Default 'example'.")
    parser.add_argument('--subset_size', required=True, default=1000,
                        help='Number of instances to subset. Max 25684. Default 1000.')
    parser.add_argument('--validation_split', required=True, default=0.2,
                        help='Percentage of training data for validation. Stratifies on class. Range 0.00-0.99. Default 0.2.')
    parser.add_argument('--epochs', required=False, default=1,
                        help='Number of epochs to train. Default 1.')
    parser.add_argument('--learning_rate', required=False, default=0.002,
                        help='Learning rate. Default 0.002.')
    parser.add_argument('--use_augmentation', required=False, default=False,
                        help='Use image augmentation. Default false.')
    parser.add_argument('--model_dir', required=False,
                        default=GlobalConfig.get('MODEL_DIR'),
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=/projects/lungbox/models)')
    args = parser.parse_args()

    # Set up training data
    elapsed_start = time.perf_counter()
    data = TrainingData(subset_size=int(args.subset_size),
                        validation_split=float(args.validation_split))
    elapsed_end = time.perf_counter()
    logger.info('Data Setup Time: %ss' % round(elapsed_end - elapsed_start, 4))

    # Set up Mask R-CNN model
    elapsed_start = time.perf_counter()
    model = modellib.MaskRCNN(mode='training',
                              config=DetectorConfig(),
                              model_dir=args.model_dir)
    elapsed_end = time.perf_counter()
    logger.info('Model Setup Time: %ss' % round(elapsed_end - elapsed_start, 4))

    # Add image augmentation params
    if args.use_augmentation:
        augmentation = iaa.SomeOf((0, 5), [
            # iaa.Fliplr(0.5),

            # crop some of the images by 0-10% of their height/width
            iaa.Crop(percent=(0, 0.1)),

            # Apply affine transformations to some of the images
            # - scale to 90-110% of image height/width (each axis independently)
            # - translate by -10 to +10 relative to height/width (per axis)
            # - rotate by -10 to +10 degrees
            # - shear by -2 to +2 degrees
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-10, 10),
                shear=(-2, 2)),

            # Change brightness of images (90-110% of original value).
            iaa.Multiply((0.9, 1.1))
        ])
    else:
        augmentation = None

    # Train the model! (SLOW)
    # layers
    # * 'heads' = trains head branches (least memory)
    # * '3+'    = trains resnet stage 3 and up
    # * '4+'    = trains resnet stage 4 and up
    # * 'all'   = trains all layers (most memory)
    elapsed_start = time.perf_counter()
    model.train(train_dataset=data.get_dataset_train(),
                val_dataset=data.get_dataset_valid(),
                learning_rate=float(args.learning_rate),
                epochs=int(args.epochs),
                layers='all',
                augmentation=augmentation)
    elapsed_end = time.perf_counter()
    logger.info('Training Time: %ss' % round(elapsed_end - elapsed_start, 4))
