#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

"""Initial model with Mask R-CNN: Inference and Model Evaluation."""

import os
import sys
import time
import collections

try:
    script_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_path = "/projects/lungbox/src/notebooks/model1_maskrcnn"
    sys.path.append(script_path)
sys.path.append(script_path + "/Mask_RCNN")

import utils
import Mask_RCNN.mrcnn.model as modellib
from config import GlobalConfig
from config import InferenceConfig
from data import TrainingData

# Set manually since not properly picked up from system config
os.environ['AWS_REGION'] = GlobalConfig.get('AWS_REGION')

# Set evalaution constants
SUBSET_SIZE = 10  # GlobalConfig.get('TRAINING_DATA_MAX_SIZE')  # 25684

# Get data
data = TrainingData(subset_size=SUBSET_SIZE, validation_split=0.2)

# ---- LOAD MODEL --------------------------------------------------------------

model = modellib.MaskRCNN(
    mode='inference',
    config=InferenceConfig(),
    model_dir=GlobalConfig.get('MODEL_DIR'))

# Select the latest trained model
model_path = utils.find_latest_model(
    model_dir=model.model_dir,
    prefix='mask_rcnn',
    inference_config=InferenceConfig())

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# ---- MODEL EVALUATION --------------------------------------------------------

dataset_valid = data.get_dataset_valid()

# Compute metrics on a subset of metrics
# APs = modelutils.compute_batch_ap(
#    dataset=dataset_valid,
#    inference_config=InferenceConfig(),
#    image_ids=np.random.choice(dataset_valid.image_ids, 1))
# print("mAP @ IoU=50: ", np.mean(APs))

# Compute metrics on all test images
elapsed_start = time.perf_counter()
all_results = utils.compute_batch_metrics(
    dataset=dataset_valid,
    model=model,
    inference_config=InferenceConfig(),
    image_ids=dataset_valid.image_ids)
len(all_results)
class_counts = collections.Counter(v['class_result'] for k, v in all_results.items())
elapsed_end = time.perf_counter()
print(class_counts)
print('Elapsed Time: %ss' % round(elapsed_end - elapsed_start, 4))
