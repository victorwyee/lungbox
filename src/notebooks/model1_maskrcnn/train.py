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

# Set training constants
NUM_EPOCHS = 1   # 10
SUBSET_SIZE = 10 # GlobalConfig.get('TRAINING_DATA_MAX_SIZE')  # 25684

# Get data
data = TrainingData(subset_size=SUBSET_SIZE, validation_split=0.2)

# ---- LOAD MODEL --------------------------------------------------------------

# Set up Mask R-CNN model
elapsed_start = time.perf_counter()
model = modellib.MaskRCNN(
    mode='training',
    config=DetectorConfig(),
    model_dir=GlobalConfig.get('MODEL_DIR'))
elapsed_end = time.perf_counter()
print('Elapsed Time: %ss' % round(elapsed_end - elapsed_start, 4))

# Add initial image augmentation params
augmentation = iaa.SomeOf((0, 1), [
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-10, 10),
        shear=(-8, 8)
    ),
    iaa.Multiply((0.9, 1.1))
])

# ---- TRAIN MODEL -------------------------------------------------------------

# Train the model! (SLOW)
elapsed_start = time.perf_counter()
model.train(train_dataset=data.get_dataset_train(),
            val_dataset=data.get_dataset_valid(),
            learning_rate=DetectorConfig().LEARNING_RATE,
            epochs=NUM_EPOCHS,
            layers='all',
            augmentation=augmentation)
elapsed_end = time.perf_counter()
print('Elapsed Time: %ss' % round(elapsed_end - elapsed_start, 4))
