# -*- coding: utf-8 -*-

import os
import boto3
import numpy as np
import pandas as pd
import pathlib

# DICOM
import pydicom
import pylab
from io import BytesIO

# Config
S3_BUCKET_NAME = 'lungbox'
os.environ['AWS_REGION'] = 'us-west-2'

S3_CLASS_INFO_KEY = 'data/raw/stage_1_detailed_class_info.csv'
S3_TRAIN_BOX_KEY = 'data/raw/stage_1_train_labels.csv'
S3_CLASS_INFO_PATH = 's3://' + S3_BUCKET_NAME + '/' + S3_CLASS_INFO_KEY
S3_TRAIN_BOX_PATH = 's3://' + S3_BUCKET_NAME + '/' + S3_TRAIN_BOX_KEY

S3_STAGE1_TRAIN_IMAGE_DIR = 'data/raw/stage_1_train_images'
S3_STAGE1_TEST_IMAGE_DIR = 'data/raw/stage_1_test_images'

# Check S3 connectivity by listing  (slow)
obj_paths = []
s3 = boto3.resource('s3')
for obj in s3.Bucket(name=S3_BUCKET_NAME).objects.all():
    obj_paths.append(os.path.join(obj.bucket_name, obj.key))
assert len(obj_paths) == 26688

# Inspect DICOM list (slow)
image_df = pd.DataFrame(columns=['path', 'subdir', 'patient_id'])
for obj in s3.Bucket(name=S3_BUCKET_NAME).objects.all():
    path = os.path.join(obj.bucket_name, obj.key)
    if path.endswith('.dcm'):
        path_parts = path.split('/')
        image_df = image_df.append({
            'path': path,
            'subdir': path_parts[3].split('_')[2],
            'patient_id': path_parts[-1].split('.')[0]},
            ignore_index=True)
assert image_df['patient_id'].nunique() == 26684
image_df.describe()
image_df.head()

# Read label datasets
s3client = boto3.client('s3')
class_label_obj = s3client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_CLASS_INFO_KEY)
train_box_obj = s3client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_TRAIN_BOX_KEY)
class_label_df = pd.read_csv(class_label_obj['Body'])
train_box_df = pd.read_csv(train_box_obj['Body'])
print(class_label_df.iloc[:10, ])
print(train_box_df.iloc[:10, ])

# Inspect single patient label
patient_0 = train_box_df.loc[(train_box_df['Target'] != 0)].iloc[0]
print(patient_0)
print(patient_0['patientId'])

# Inspect DICOM metadata
dcm_obj = s3client.get_object(
    Bucket=S3_BUCKET_NAME,
    Key=S3_STAGE1_TRAIN_IMAGE_DIR + "/" + patient_0['patientId'] + ".dcm")
dcm_data = pydicom.read_file(BytesIO(dcm_obj['Body'].read()))

print(dcm_data)
im = dcm_data.pixel_array
print(type(im))
print(im.dtype)  # downsampled to 256 grayscale values
print(im.shape)  # resized from 2000x2000

# Visualize single DICOM
pylab.imshow(im, cmap=pylab.cm.gist_gray)
pylab.axis('off')

dcm_data = pydicom.read_file('/projects/lungbox/data/raw/stage_1_train_images/0a0f91dc-6015-4342-b809-d19610854a21.dcm')


# Get mean intensity of all images
mean_intensity = []
pathlist = pathlib.Path('/projects/lungbox/data/raw/stage_1_train_images').glob('**/*.dcm')
for path in pathlist:
    dcm_data = pydicom.read_file(str(path))
    im = dcm_data.pixel_array
    mean_intensity.append(im.mean())
    if (len(mean_intensity) == 10):
        break
np.mean(mean_intensity)  # 125.256578 -- close enough to 255/2 = 127.5

for obj in s3.Bucket(name=S3_BUCKET_NAME).objects.all():
    if obj.key.endswith('.dcm'):
        print(obj.key)
        dcm_obj = s3client.get_object(Bucket=S3_BUCKET_NAME, Key=obj.key)
        dcm_data = pydicom.read_file(BytesIO(dcm_obj['Body'].read()))
        im = dcm_data.pixel_array
        mean_intensity.append(im.mean())

im.mean()

# Parse patient labels and bounding boxes into dictionary
# parsed_df = ingest.parse_training_labels(
#     train_box_df=train_box_df,
#     train_image_dirpath=S3_STAGE1_TRAIN_IMAGE_DIR)
# print(parsed_df['0004cfab-14fd-4e49-80ba-63a80b6bddd6'])
# print(parsed_df['00436515-870c-4b36-a041-de91049b9ab4'])

# Visualize bounding boxes for single patientId
# ingest.draw(parsed_df=parsed_df,
#             patient_id='00436515-870c-4b36-a041-de91049b9ab4')

# Check that TensorFlow can read the S3 files
from tensorflow.python.lib.io import file_io
print(file_io.stat(S3_CLASS_INFO_PATH))

filenames = ["s3://lungbox/raw/stage_1_test_images/000924cf-0f8d-42bd-9158-1af53881a557.dcm",
             "s3://lungbox/raw/stage_1_test_images/000db696-cf54-4385-b10b-6b16fbb3f985.dcm",
             "s3://lungbox/raw/stage_1_test_images/000fe35a-2649-43d4-b027-e67796d412e0.dcm",
             "s3://lungbox/raw/stage_1_test_images/001031d9-f904-4a23-b3e5-2c088acd19c6.dcm",
             "s3://lungbox/raw/stage_1_test_images/0010f549-b242-4e94-87a8-57d79de215fc.dcm"]
dataset = tf.data.TFRecordDataset(filenames)
print(dataset)
