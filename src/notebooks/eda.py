import os

# Analysis
import tensorflow as tf
import numpy as np
import pandas as pd

# DICOM
import pydicom
import gdcm
import pylab

import src.ingestion as ingest

import importlib
importlib.reload(ingest)

# Config
S3_BUCKET_NAME = 'lungbox'
os.environ['AWS_REGION'] = 'us-west-2'

S3_CLASS_INFO_KEY = 'data/raw/stage_1_detailed_class_info.csv'
S3_TRAIN_BOX_KEY = 'data/raw/stage_1_train_labels.csv'
S3_CLASS_INFO_PATH = 's3://' + S3_BUCKET_NAME + '/' + S3_CLASS_INFO_KEY
S3_TRAIN_BOX_PATH = 's3://' + S3_BUCKET_NAME + '/' + S3_TRAIN_BOX_KEY

S3_STAGE1_TRAIN_IMAGE_DIR = 'data/raw/stage_1_train_images'
S3_STAGE1_TEST_IMAGE_DIR = 'data/raw/stage_1_test_images'

# Check S3 connectivity
obj_paths = ingest.list_bucket_contents(S3_BUCKET_NAME)
assert len(obj_paths) == 26688

# Inspect DICOM list (slow)
image_df = ingest.parse_dicom_image_list(S3_BUCKET_NAME)
assert image_df['patient_id'].nunique() == 26684
image_df.describe()
image_df.head()

# Read label datasets
class_label_df = ingest.read_s3_df(bucket_name=S3_BUCKET_NAME,file_key=S3_CLASS_INFO_KEY)
train_box_df = ingest.read_s3_df(bucket_name=S3_BUCKET_NAME, file_key=S3_TRAIN_BOX_KEY)
print(class_label_df.iloc[:10, ])
print(train_box_df.iloc[:10, ])
