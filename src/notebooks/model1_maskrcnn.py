# -*- coding: utf-8 -*-
"""Initial model with Mask R-CNN.

References:
    Intro to deep learning for medical imaging by MD.ai
    - https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-mdai-client-lib.ipynb
    - Retrieved 2018-09-20
    - Licensed under the Apache License, Version 2.0
"""

import os
import sys
import random
import numpy as np
import time

# For image reading & manipulation
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import src.ingestion as ingest

# Import local matterport/Mask_RCNN lib
# TODO: Find better way to import a local library
sys.path.append(os.path.join('/projects/lungbox/libs/Mask_RCNN'))
from libs.Mask_RCNN.mrcnn.config import Config
import libs.Mask_RCNN.mrcnn.model as modellib
from libs.Mask_RCNN.mrcnn import visualize

# Import Mask CNN helper classes
from src.analysis.mask_rcnn import DetectorDataset

# Start streamlit notebook
import streamlit as st

# ---- CONFIG ------------------------------------------------------------------

# Read global config
from configs.globals import GlobalConfig
S3_BUCKET_NAME = GlobalConfig.get('S3_BUCKET_NAME')
S3_CLASS_INFO_KEY = GlobalConfig.get('S3_CLASS_INFO_KEY')
S3_TRAIN_BOX_KEY = GlobalConfig.get('S3_TRAIN_BOX_KEY')
S3_CLASS_INFO_PATH = GlobalConfig.get('S3_CLASS_INFO_PATH')
S3_TRAIN_BOX_PATH = GlobalConfig.get('S3_TRAIN_BOX_PATH')
S3_STAGE1_TRAIN_IMAGE_DIR = GlobalConfig.get('S3_STAGE1_TRAIN_IMAGE_DIR')
S3_STAGE1_TEST_IMAGE_DIR = GlobalConfig.get('S3_STAGE1_TEST_IMAGE_DIR')
MODEL_DIR = GlobalConfig.get('MODEL_DIR')

# Set manually since not properly picked up from system config
os.environ['AWS_REGION'] = GlobalConfig.get('AWS_REGION')

# Set other constants
DICOM_WIDTH = 1024
DICOM_HEIGHT = 1024
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ---- OVERVIEW ----------------------------------------------------------------

st.markdown('# Model 1: Mask R-CNN')
st.markdown('## Overview')
st.markdown(
    """
    This is a first pass at a bounding box predictor using
    [matterport\'s Mask R-CNN implementation](https://github.com/matterport/Mask_RCNN).
    """
)

# ---- DATA --------------------------------------------------------------------

st.markdown('## Data')
st.markdown(
    """
    * RSNA 2018 Challenge dataset
        * Subset of NIH CXR14 dataset of >100K classified x-rays
        * Training Data: 30K x-rays
        * Test Data: 4500 x-rays
        * 0-4 bounding boxes
        * Expert annotation by board-certified radiologists
            - Not perfect
    * 31% positive, 29% negative/normal, 40% negative/abnormal
    * Standard DICOM format
        * Already windowed & leveled
        * Already downsampled from 2000x2000 to 1024x1024
    """
)

# Create annotation dict
train_box_df = ingest.read_s3_df(
    bucket=S3_BUCKET_NAME, file_key=S3_TRAIN_BOX_KEY)
annotation_dict = ingest.parse_training_labels(
    train_box_df=train_box_df,
    train_image_dirpath=S3_STAGE1_TRAIN_IMAGE_DIR)
print(annotation_dict['0004cfab-14fd-4e49-80ba-63a80b6bddd6'])
print(annotation_dict['00436515-870c-4b36-a041-de91049b9ab4'])

# Get list of images
image_df = ingest.parse_dicom_image_list(bucket=S3_BUCKET_NAME)  # slow
print(image_df.head())

# Take initial subset of 1000 patients
st.markdown(
    """
    For our first iteration, we will use a small subset of 1000 images,
    with an 80/20 training/validation split.
    """
)
patient_id_subset = sorted(list(
    image_df[image_df['subdir'] == 'train']['patient_id'])[:1000])

# Split dataset for training and validation
sorted(patient_id_subset)
random.shuffle(patient_id_subset)

validation_split = 0.2
split_index = int((1 - validation_split) * len(patient_id_subset))

patient_id_train = patient_id_subset[:split_index]
patient_id_valid = patient_id_subset[split_index:]

print(len(patient_id_train), len(patient_id_valid))

# ---- MODEL -------------------------------------------------------------------

st.markdown('## Model\n')
st.markdown(' ')
st.markdown(
    """\n\n
    * FPN + ResNet50
        * (He et al. (2017). Mask R-CNN.)[https://arxiv.org/abs/1703.06870]
    * Existing implementations
        * https://github.com/facebookresearch/Detectron
        * https://github.com/matterport/Mask_RCNN
    """
)

# Non-optimal initial Mask R-CNN config
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


st.markdown('### Model Configuration')
MASKRCNN_CONFIG = DetectorConfig()
st.text(MASKRCNN_CONFIG.get_attrs())

# Prepare training and validation sets.
dataset_train = DetectorDataset(
    patient_ids=patient_id_train,
    annotation_dict=annotation_dict,
    orig_height=DICOM_HEIGHT,
    orig_width=DICOM_WIDTH,
    s3_bucket=S3_BUCKET_NAME)
dataset_valid = DetectorDataset(
    patient_ids=patient_id_valid,
    annotation_dict=annotation_dict,
    orig_height=DICOM_HEIGHT,
    orig_width=DICOM_WIDTH,
    s3_bucket=S3_BUCKET_NAME)
dataset_train.prepare()
dataset_valid.prepare()
assert len(dataset_train.image_ids) == len(patient_id_train)
assert len(dataset_valid.image_ids) == len(patient_id_valid)

st.markdown('### Example Training Labels')
st.json(dataset_train.image_info[:5])

# Inspect a single instance
st.markdown('### Example Instance')
image_id = 88 # random.randint(0, 100)
image_fp = dataset_train.image_reference(image_id)
image = dataset_train.load_image(image_id=image_id)
mask, class_ids = dataset_train.load_mask(image_id=image_id)
st.markdown(dataset_train.get_image_reference(image_id))
st.text(dataset_train.get_image_annotation(image_id))
st.text('Class IDs: %s' % np.array_str(class_ids))
st.text('Image Dimensions: %s' % ' '.join(str(a) for a in image.shape))

# Visualize instance
fig, axarr = plt.subplots(1, 2, sharex=False, sharey=True)
axarr[0].imshow(image[:, :, 0], cmap='gray')
axarr[0].axis('off')
masked = np.zeros(image.shape[:2])
for i in range(mask.shape[2]):
    masked += image[:, :, 0] * mask[:, :, i]
axarr[1].imshow(masked, cmap='gray')
axarr[1].axis('off')
st.pyplot(fig)

# Set up Mask R-CNN model
elapsed_start = time.perf_counter()
model = modellib.MaskRCNN(
    mode='training',
    config=MASKRCNN_CONFIG,
    model_dir=MODEL_DIR)
elapsed_end = time.perf_counter()
'Elapsed Time: %ss' % round(elapsed_end - elapsed_start, 4)

# Add initial image augmentation params
augmentation = iaa.SomeOf((0, 1), [
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    ),
    iaa.Multiply((0.9, 1.1))
])

# Train the model!
NUM_EPOCHS = 100
elapsed_start = time.perf_counter()
model.train(train_dataset=dataset_train,
            val_dataset=dataset_valid,
            learning_rate=MASKRCNN_CONFIG.LEARNING_RATE,
            epochs=NUM_EPOCHS,
            layers='all',
            augmentation=augmentation)  # slow
'Elapsed Time: %ss' % round(elapsed_end - elapsed_start, 4)

# Select the trained model
dir_names = next(os.walk(model.model_dir))[1]
key = MASKRCNN_CONFIG.NAME.lower()
dir_names = filter(lambda f: f.startswith(key), dir_names)
dir_names = sorted(dir_names)
if not dir_names:
    import errno
    raise IOError(
        errno.ENOENT,
        "Could not find model directory under {}".format(model.model_dir))
else:
    print(dir_names)

# Pick last directory
fps = []
for d in dir_names:
    dir_name = os.path.join(model.model_dir, d)
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        print('No weight files in {}'.format(dir_name))
    else:
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        fps.append(checkpoint)
model_path = sorted(fps)[-1]
print('Found model {}'.format(model_path))

# Conduct model inference
class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

INFERENCE_CONFIG = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(
    mode='inference',
    config=INFERENCE_CONFIG,
    model_dir=MODEL_DIR)

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Set color for class
def get_colors_for_class_ids(class_ids):
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors


# Show some examples of ground truth vs. predictions on the validation set
fig = plt.figure(figsize=(10, 30))

image_id = 6
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_valid, INFERENCE_CONFIG,
                           image_id, use_mini_mask=False)
visualize.display_instances(
    image=original_image,
    boxes=gt_bbox,
    masks=gt_mask,
    class_ids=gt_class_id,
    class_names=dataset_valid.class_names,
    figsize=(4, 4),
    colors=get_colors_for_class_ids(gt_class_id),
    ax=fig.axes)

results = model.detect([original_image], verbose=1)
r = results[0]
visualize.display_instances(
    image=original_image,
    boxes=r['rois'],
    masks=r['masks'],
    class_ids=r['class_ids'],
    class_names=dataset_valid.class_names,
    figsize=(4, 4),
    scores=r['scores'],
    colors=get_colors_for_class_ids(r['class_ids']),
    ax=fig.axes)
