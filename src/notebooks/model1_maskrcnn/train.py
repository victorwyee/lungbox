#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

"""Initial model with Mask R-CNN: Training."""

import os
import sys
import time
from imgaug import augmenters as iaa

try:
    script_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_path = "/projects/lungbox/src/notebooks/model1_maskrcnn"
    sys.path.append(script_path)
sys.path.append(script_path + "/Mask_RCNN")

import Mask_RCNN.mrcnn.model as modellib
from config import GlobalConfig
from config import DetectorConfig
from data import TrainingData

# Set manually since not properly picked up from system config
os.environ['AWS_REGION'] = GlobalConfig.get('AWS_REGION')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset_size', help='Number of instances to subset. Max 25684.')
    parser.add_argument('--validation_split', help='Percentage of training data for validation. Range 0.00-0.99.')
    parser.add_argument('--epochs', help='Number of epochs to train.')
    parser.add_argument('--learning_rate', help='Learning rate. Default 0.001.')
    args = parser.parse_args()

    data = TrainingData(
        subset_size=int(args.subset_size),
        validation_split=float(args.validation_split))
    learning_rate = args.learning_rate if args.learning_rate else DetectorConfig().LEARNING_RATE

    # Set up Mask R-CNN model
    elapsed_start = time.perf_counter()
    model = modellib.MaskRCNN(
        mode='training',
        config=DetectorConfig(),
        model_dir=GlobalConfig.get('MODEL_DIR'))
    elapsed_end = time.perf_counter()
    print('Setup Time: %ss' % round(elapsed_end - elapsed_start, 4))

    # Add initial image augmentation params
    augmentation = iaa.SomeOf((0, 2), [
        # ia.Fliplr(0.5), # not recommended; human body not symmetric
        iaa.Flipud(0.5),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-10, 10),  # up to 10% rotations recommended
            shear=(-8, 8)
        ),
        iaa.Multiply((0.9, 1.1)),
        iaa.GaussianBlur(sigma=(0.0, 0.5))
    ])

    # Train the model! (SLOW)
    # layers
    # * 'heads' = trains head branches (least memory)
    # * '3+'    = trains resnet stage 3 and up
    # * '4+'    = trains resnet stage 4 and up
    # * 'all'   = trains all layers (most memory)
    elapsed_start = time.perf_counter()
    model.train(train_dataset=data.get_dataset_train(),
                val_dataset=data.get_dataset_valid(),
                learning_rate=learning_rate,
                epochs=args.epochs,
                layers='all',
                augmentation=augmentation)
    elapsed_end = time.perf_counter()
    print('Training Time: %ss' % round(elapsed_end - elapsed_start, 4))
