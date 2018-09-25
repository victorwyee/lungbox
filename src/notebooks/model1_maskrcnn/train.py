# -*- coding: utf-8 -*-
"""Initial model with Mask R-CNN: Training.

References:
    Mask R-CNN
    - https://github.com/matterport/Mask_RCNN
    - https://github.com/matterport/Mask_RCNN/blob/master/samples/coco/inspect_model.ipynb
    - Retrieved 2018-09-19
    - License: MIT License
    Intro to deep learning for medical imaging by MD.ai
    - https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-mdai-client-lib.ipynb
    - Retrieved 2018-09-20
    - License: Apache License, Version 2.0
"""

import os
import sys
import time
from imgaug import augmenters as iaa

sys.path.append(os.path.join('/projects/lungbox'))
sys.path.append(os.path.join('/projects/lungbox/libs/Mask_RCNN'))
import libs.Mask_RCNN.mrcnn.model as modellib
from configs.globals import GlobalConfig
from src.notebooks.model1_maskrcnn.config import DetectorConfig
from src.notebooks.model1_maskrcnn.data import TrainingData

# Set manually since not properly picked up from system config
os.environ['AWS_REGION'] = GlobalConfig.get('AWS_REGION')

# Set training constants
NUM_EPOCHS = 10
SUBSET_SIZE = GlobalConfig.get('TRAINING_DATA_MAX_SIZE')  # 25684

# Get data
data = TrainingData(subset_size=SUBSET_SIZE, validation_split=0.2)

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
