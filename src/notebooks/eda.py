import os

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

# Check S3 connectivity by listing  (slow)
obj_paths = ingest.list_bucket_contents(S3_BUCKET_NAME)
assert len(obj_paths) == 26688

# Inspect DICOM list (slow)
image_df = ingest.parse_dicom_image_list(S3_BUCKET_NAME)
assert image_df['patient_id'].nunique() == 26684
image_df.describe()
image_df.head()

# Read label datasets
class_label_df = ingest.read_s3_df(
    bucket_name=S3_BUCKET_NAME, file_key=S3_CLASS_INFO_KEY)
train_box_df = ingest.read_s3_df(
    bucket_name=S3_BUCKET_NAME, file_key=S3_TRAIN_BOX_KEY)
print(class_label_df.iloc[:10, ])
print(train_box_df.iloc[:10, ])

# Inspect single patient label
patient_0 = train_box_df.loc[(train_box_df['Target'] != 0)].iloc[0]
print(print(patient_0))
print(patient_0['patientId'])

# Inspect DICOM metadata
dcm_data = ingest.get_s3_dcm(bucket=S3_BUCKET_NAME,
                             dirpath=S3_STAGE1_TRAIN_IMAGE_DIR,
                             patient_id=patient_0['patientId'])
print(dcm_data)
im = dcm_data.pixel_array
print(type(im))
print(im.dtype)  # downsampled to 256 grayscale values
print(im.shape)  # resized from 2000x2000

# Visualize single DICOM
pylab.imshow(im, cmap=pylab.cm.gist_gray)
pylab.axis('off')

# Parse patient labels and bounding boxes into dictionary
parsed_df = ingest.parse_training_labels(
    train_box_df=train_box_df,
    train_image_dirpath=S3_STAGE1_TRAIN_IMAGE_DIR)
print(parsed_df['0004cfab-14fd-4e49-80ba-63a80b6bddd6'])
print(parsed_df['00436515-870c-4b36-a041-de91049b9ab4'])

# Visualize bounding boxes for single patientId
ingest.draw(parsed_df=parsed_df,
            patient_id='00436515-870c-4b36-a041-de91049b9ab4')

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
