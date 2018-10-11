#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: infer.py

"""
Train a lung opacity detector on the RSNA 2018 pneumonia challenge dataset.
Uses the official matterport's "official" implementation of Mask R-CNN.

Usage:
    python src/infer.py \
        --image_path=/projects/lungbox/data/example/stage_1_train_images/00a85be6-6eb0-421d-8acf-ff2dc0007e8a.dcm \
        --h5_path=/projects/lungbox/models/pneumonia20181003T1228/mask_rcnn_pneumonia_0050.h5
        --model_dir='/projects/lungbox/models'
"""

import os
import sys
import time
import pydicom

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
from config import InferenceConfig

# Set manually since not properly picked up from system config
os.environ['AWS_REGION'] = GlobalConfig.get('AWS_REGION')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Runs Mask R-CNN for lung opacity detection in pneumonia.')
    parser.add_argument('--image_path', required=True,
                        help="Path to the input image")
    parser.add_argument('--h5_path', required=False,
                        default=GlobalConfig.get('MODEL_DIR') + GlobalConfig.get('LATEST_MODEL'),
                        metavar="/path/to/logs/",
                        help='Path to the Keras model')
    parser.add_argument('--model_dir', required=False,
                        default=GlobalConfig.get('MODEL_DIR'),
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=/projects/lungbox/models)')
    args = parser.parse_args()

    # Read the image
    elapsed_start = time.perf_counter()
    image = pydicom.read_file(args.image_path)
    elapsed_end = time.perf_counter()
    logger.info('Data Setup Time: %ss' % round(elapsed_end - elapsed_start, 4))

    # Set up Mask R-CNN model
    elapsed_start = time.perf_counter()
    infer_config = InferenceConfig()
    model = modellib.MaskRCNN(mode='training',
                              config=infer_config,
                              model_dir=args.model_dir)
    model.load_weights(args.h5_path, by_name=True)
    elapsed_end = time.perf_counter()
    logger.info('Model Setup Time: %ss' % round(elapsed_end - elapsed_start, 4))

    # Run inference!
    elapsed_start = time.perf_counter()
    r = model.detect([image], verbose=False)[0]
    elapsed_end = time.perf_counter()
    logger.info('Training Time: %ss' % round(elapsed_end - elapsed_start, 4))
